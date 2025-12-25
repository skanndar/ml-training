#!/usr/bin/env python3
"""
Script para convertir SavedModel a TF.js en Google Colab.

INSTRUCCIONES:
1. Abrir nuevo notebook en https://colab.research.google.com
2. Copiar y pegar este código completo
3. Ejecutar la celda
4. Seguir las instrucciones para upload/download

Compatible con TensorFlow 2.17+ (disponible en Colab).
"""

# ==============================================================================
# PARTE 1: Instalación (ejecutar primero)
# ==============================================================================

print("=== Instalando dependencias ===")
import subprocess
import sys

# Instalar tensorflowjs (usa la versión de TF ya instalada en Colab)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tensorflowjs"])

print("✅ Dependencias instaladas")


# ==============================================================================
# PARTE 2: Upload del SavedModel (ejecutar después de Parte 1)
# ==============================================================================

print("\n=== Upload del SavedModel ===")
print("Comprimir el SavedModel localmente primero con:")
print("  cd /home/skanndar/SynologyDrive/local/aplantida/ml-training")
print("  zip -r saved_model.zip dist/models/student_v1_fp16_manual/saved_model/")
print("\nAhora sube saved_model.zip cuando se te pida:")

from google.colab import files
uploaded = files.upload()

# Descomprimir
import zipfile
import os

for filename in uploaded.keys():
    print(f"Descomprimiendo {filename}...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('.')
    print(f"✅ Descomprimido")

print("✅ SavedModel listo en: ./saved_model/")


# ==============================================================================
# PARTE 3: Conversión (ejecutar después de Parte 2)
# ==============================================================================

print("\n=== Conversión SavedModel → TF.js ===")

import tensorflowjs as tfjs

# Convertir con cuantización FP16
tfjs.converters.convert_tf_saved_model(
    saved_model_dir='./saved_model',
    output_dir='./student_v1_fp16',
    quantization_dtype_map={'float': 'float16'}  # TF.js usa dict en lugar de string
)

print("✅ Conversión completada")
print("Archivos generados:")

import glob
files_list = glob.glob('./student_v1_fp16/*')
for f in sorted(files_list):
    size = os.path.getsize(f) / (1024 * 1024)  # MB
    print(f"  - {os.path.basename(f):40s} ({size:6.2f} MB)")

total_size = sum(os.path.getsize(f) for f in files_list) / (1024 * 1024)
print(f"\n📦 Tamaño total: {total_size:.2f} MB")


# ==============================================================================
# PARTE 4: Download (ejecutar después de Parte 3)
# ==============================================================================

print("\n=== Comprimiendo resultado para download ===")

# Comprimir
import shutil
shutil.make_archive('student_v1_fp16', 'zip', './student_v1_fp16')

print("✅ Archivo comprimido: student_v1_fp16.zip")
print("Descargando...")

files.download('student_v1_fp16.zip')

print("\n" + "="*80)
print("✅ CONVERSIÓN COMPLETADA")
print("="*80)
print("\nPróximos pasos en tu máquina local:")
print("1. Descomprimir student_v1_fp16.zip")
print("2. Mover a: dist/models/student_v1_fp16/")
print("3. Copiar export_metadata.json:")
print("   cp dist/models/student_v1_fp16_manual/export_metadata.json \\")
print("      dist/models/student_v1_fp16/")
print("4. Verificar archivos:")
print("   ls -lh dist/models/student_v1_fp16/")
print("\nDeberías ver:")
print("  - model.json")
print("  - group1-shard*.bin (varios archivos)")
print("  - export_metadata.json")
print("\n🚀 Listo para integrar en la PWA!")
