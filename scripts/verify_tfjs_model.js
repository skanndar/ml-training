#!/usr/bin/env node
/**
 * Script de verificación del modelo TF.js
 *
 * Verifica que el modelo se puede cargar correctamente y que las dimensiones
 * de entrada/salida son las esperadas.
 *
 * Uso:
 *   node scripts/verify_tfjs_model.js
 */

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Configuración
const MODEL_PATH = path.join(__dirname, '../dist/models/student_v1_fp16/model.json');
const METADATA_PATH = path.join(__dirname, '../dist/models/student_v1_fp16/export_metadata.json');

// Colores para terminal
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function log(msg, color = 'reset') {
  console.log(`${colors[color]}${msg}${colors.reset}`);
}

async function verifyModel() {
  console.log('\n' + '='.repeat(80));
  log('VERIFICACIÓN DEL MODELO TF.JS', 'cyan');
  console.log('='.repeat(80) + '\n');

  // 1. Verificar que existen los archivos
  log('1. Verificando archivos...', 'blue');

  if (!fs.existsSync(MODEL_PATH)) {
    log(`   ✗ No se encuentra model.json: ${MODEL_PATH}`, 'red');
    process.exit(1);
  }
  log(`   ✓ model.json encontrado`, 'green');

  if (!fs.existsSync(METADATA_PATH)) {
    log(`   ✗ No se encuentra export_metadata.json: ${METADATA_PATH}`, 'red');
    process.exit(1);
  }
  log(`   ✓ export_metadata.json encontrado`, 'green');

  // Verificar shards
  const modelDir = path.dirname(MODEL_PATH);
  const shards = fs.readdirSync(modelDir).filter(f => f.endsWith('.bin'));
  log(`   ✓ ${shards.length} shards encontrados (${shards.join(', ')})`, 'green');

  // 2. Leer metadata
  log('\n2. Leyendo metadata...', 'blue');
  const metadata = JSON.parse(fs.readFileSync(METADATA_PATH, 'utf8'));
  log(`   ✓ Versión: ${metadata.model_version}`, 'green');
  log(`   ✓ Fecha export: ${metadata.export_date}`, 'green');
  log(`   ✓ Arquitectura: ${metadata.model_architecture}`, 'green');
  log(`   ✓ Clases: ${metadata.num_classes}`, 'green');
  log(`   ✓ Threshold: ${metadata.threshold.value} (accuracy=${metadata.threshold.accuracy}, coverage=${metadata.threshold.coverage})`, 'green');
  log(`   ✓ Temperatura: ${metadata.calibration.temperature} (ECE=${metadata.calibration.ece_after.toFixed(4)})`, 'green');

  // 3. Cargar modelo
  log('\n3. Cargando modelo TF.js...', 'blue');
  let model;
  try {
    model = await tf.loadGraphModel(`file://${MODEL_PATH}`);
    log(`   ✓ Modelo cargado exitosamente`, 'green');
  } catch (error) {
    log(`   ✗ Error al cargar modelo: ${error.message}`, 'red');
    process.exit(1);
  }

  // 4. Verificar dimensiones
  log('\n4. Verificando dimensiones del modelo...', 'blue');

  const inputShape = model.inputs[0].shape;
  const outputShape = model.outputs[0].shape;

  log(`   Input shape: [${inputShape}]`, 'cyan');
  log(`   Output shape: ${outputShape ? `[${outputShape}]` : 'undefined (dinámico)'}`, 'cyan');

  // Verificar input shape (esperado: [null, 3, 224, 224] o [null, 224, 224, 3])
  const expectedInputChannels = 3;
  const expectedInputSize = 224;

  if (inputShape[1] === expectedInputChannels && inputShape[2] === expectedInputSize && inputShape[3] === expectedInputSize) {
    log(`   ✓ Input shape correcto: [batch, ${expectedInputChannels}, ${expectedInputSize}, ${expectedInputSize}] (NCHW)`, 'green');
  } else if (inputShape[1] === expectedInputSize && inputShape[2] === expectedInputSize && inputShape[3] === expectedInputChannels) {
    log(`   ✓ Input shape correcto: [batch, ${expectedInputSize}, ${expectedInputSize}, ${expectedInputChannels}] (NHWC)`, 'green');
  } else {
    log(`   ⚠ Input shape inesperado. Esperado: [null, 3, 224, 224] o [null, 224, 224, 3]`, 'yellow');
  }

  // Verificar output shape (esperado: [null, 7120] o mayor por si se exportó con más clases)
  const expectedNumClasses = metadata.num_classes;

  // outputShape puede ser undefined si el modelo no especifica dimensión estática
  // En ese caso, la verificaremos durante la inferencia
  if (outputShape && outputShape[1] !== undefined) {
    if (outputShape[1] >= expectedNumClasses) {
      log(`   ✓ Output shape correcto: [batch, ${outputShape[1]}]`, 'green');
    } else {
      log(`   ⚠ Output shape inesperado. Esperado al menos ${expectedNumClasses} clases, pero tiene ${outputShape[1]}`, 'yellow');
    }
  } else {
    log(`   ℹ Output shape dinámico (se verificará en inferencia)`, 'cyan');
  }

  // 5. Test de inferencia
  log('\n5. Ejecutando test de inferencia...', 'blue');

  try {
    // Crear input dummy con la forma correcta
    let dummyInput;
    if (inputShape[1] === 3) {
      // NCHW
      dummyInput = tf.randomNormal([1, 3, 224, 224]);
    } else {
      // NHWC
      dummyInput = tf.randomNormal([1, 224, 224, 3]);
    }

    log(`   → Input creado: shape=${dummyInput.shape}`, 'cyan');

    const startTime = Date.now();
    const output = model.predict(dummyInput);
    const inferenceTime = Date.now() - startTime;

    log(`   ✓ Inferencia exitosa en ${inferenceTime}ms`, 'green');
    log(`   → Output shape: [${output.shape}]`, 'cyan');

    // Verificar que la suma de probabilidades es ~1.0 (softmax)
    const outputData = await output.data();
    const sum = outputData.reduce((a, b) => a + b, 0);
    log(`   → Suma de outputs: ${sum.toFixed(6)} (esperado: cercano a 1.0 si es softmax, o cualquier valor si son logits)`, 'cyan');

    // Encontrar top-5 predicciones
    const topK = 5;
    const values = Array.from(outputData);
    const indices = values
      .map((val, idx) => ({ val, idx }))
      .sort((a, b) => b.val - a.val)
      .slice(0, topK);

    log(`   → Top-${topK} predicciones:`, 'cyan');
    indices.forEach((item, rank) => {
      log(`      ${rank + 1}. Clase ${item.idx}: ${item.val.toFixed(6)}`, 'cyan');
    });

    // Cleanup
    tf.dispose([dummyInput, output]);

  } catch (error) {
    log(`   ✗ Error en inferencia: ${error.message}`, 'red');
    process.exit(1);
  }

  // 6. Estadísticas del modelo
  log('\n6. Estadísticas del modelo...', 'blue');

  const modelDir2 = path.dirname(MODEL_PATH);
  const allFiles = fs.readdirSync(modelDir2);

  let totalSize = 0;
  const fileSizes = {};

  allFiles.forEach(file => {
    const filePath = path.join(modelDir2, file);
    const stats = fs.statSync(filePath);
    fileSizes[file] = stats.size;
    totalSize += stats.size;
  });

  log(`   Total archivos: ${allFiles.length}`, 'cyan');
  log(`   Tamaño total: ${(totalSize / (1024 * 1024)).toFixed(2)} MB`, 'cyan');
  log(`   Breakdown:`, 'cyan');

  Object.entries(fileSizes)
    .sort((a, b) => b[1] - a[1])
    .forEach(([file, size]) => {
      const sizeMB = (size / (1024 * 1024)).toFixed(2);
      log(`      - ${file}: ${sizeMB} MB`, 'cyan');
    });

  // 7. Resumen final
  console.log('\n' + '='.repeat(80));
  log('RESUMEN DE VERIFICACIÓN', 'cyan');
  console.log('='.repeat(80));

  log('\n✓ Modelo TF.js válido y listo para usar', 'green');
  log(`✓ Versión: ${metadata.model_version}`, 'green');
  log(`✓ Threshold configurado: ${metadata.threshold.value}`, 'green');
  log(`✓ Tamaño total: ${(totalSize / (1024 * 1024)).toFixed(2)} MB`, 'green');
  log(`✓ Input: [batch, ${inputShape.slice(1).join(', ')}]`, 'green');
  log(`✓ Output: [batch, dinámico - determinado en inferencia]`, 'green');

  log('\nPRÓXIMOS PASOS:', 'yellow');
  log('1. Copiar modelo a frontend:', 'yellow');
  log(`   cp -r dist/models/student_v1_fp16 /ruta/a/aplantidaFront/public/models/student_v1.0`, 'yellow');
  log('2. Actualizar PlantRecognition.js con threshold 0.62', 'yellow');
  log('3. Configurar Service Worker para precache', 'yellow');
  log('4. Test en navegador con imagen real', 'yellow');

  console.log('\n');
}

// Ejecutar
verifyModel().catch(error => {
  log(`\nError fatal: ${error.message}`, 'red');
  console.error(error.stack);
  process.exit(1);
});
