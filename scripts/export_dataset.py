#!/usr/bin/env python3
"""
export_dataset.py - Phase 0: Export Dataset from MongoDB

Exports plant data from MongoDB to JSONL format for training.
Each line represents one image with its associated plant metadata.

Features:
- License filtering (permissive only for production)
- Geographic region classification
- URL validation (optional)
- Progress tracking

Usage:
    python scripts/export_dataset.py --output_dir ./data [--validate-urls] [--production-only]
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Permissive licenses for production
PERMISSIVE_LICENSES = {
    'CC0', 'CC-BY', 'CC-BY-SA',
    'public domain', 'Public Domain',
    'No known copyright', 'No copyright',
    'Apache-2.0', 'MIT', 'BSD'
}


def connect_to_mongodb():
    """Connect to MongoDB using environment variables."""
    load_dotenv()

    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    db_name = os.getenv('MONGODB_DB_NAME', 'qhopsDB')

    logger.info(f"Connecting to MongoDB...")

    client = MongoClient(mongo_uri)
    db = client[db_name]

    # Test connection
    try:
        client.admin.command('ping')
        logger.info("MongoDB connection successful")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        sys.exit(1)

    return db


def is_permissive_license(license_str: str) -> bool:
    """Check if a license is permissive (safe for commercial use)."""
    if not license_str:
        return False

    license_lower = license_str.lower()

    # Check for permissive licenses
    for perm in PERMISSIVE_LICENSES:
        if perm.lower() in license_lower:
            return True

    # Explicitly reject non-commercial
    if 'nc' in license_lower or 'non-commercial' in license_lower:
        return False

    # Explicitly reject no-derivatives
    if 'nd' in license_lower or 'no-deriv' in license_lower:
        return False

    return False


def classify_region(countries: list, continents: list) -> str:
    """Classify a plant into a geographic region."""

    EU_SW = {'ES', 'PT', 'FR', 'IT', 'AD', 'MC', 'SM', 'VA', 'GI'}
    EU_NORTH = {'DE', 'UK', 'GB', 'NL', 'BE', 'AT', 'CH', 'SE', 'NO', 'DK', 'FI', 'IE', 'PL', 'CZ'}
    EU_EAST = {'RO', 'BG', 'HU', 'SK', 'UA', 'BY', 'RU', 'HR', 'SI', 'RS', 'BA', 'MK', 'AL', 'ME', 'XK'}

    countries_set = set(countries)

    if countries_set & EU_SW:
        return 'EU_SW'
    elif countries_set & EU_NORTH:
        return 'EU_NORTH'
    elif countries_set & EU_EAST:
        return 'EU_EAST'
    elif 'Europe' in continents:
        return 'EU_OTHER'
    elif 'North America' in continents or any(c in countries_set for c in ['US', 'CA', 'MX']):
        return 'AMERICAS_NORTH'
    elif 'South America' in continents:
        return 'AMERICAS_SOUTH'
    elif 'Asia' in continents:
        return 'ASIA'
    elif 'Africa' in continents:
        return 'AFRICA'
    elif 'Oceania' in continents or 'Australia' in continents:
        return 'OCEANIA'
    else:
        return 'UNKNOWN'


def validate_url(url: str, timeout: int = 5) -> bool:
    """Validate that a URL is accessible."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200
    except Exception:
        return False


def validate_urls_batch(urls: list, max_workers: int = 10) -> dict:
    """Validate multiple URLs in parallel."""
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(validate_url, url): url for url in urls}

        for future in tqdm(as_completed(future_to_url), total=len(urls), desc="Validating URLs"):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception:
                results[url] = False

    return results


def export_to_jsonl(
    db,
    output_dir: str,
    min_images_per_species: int = 1,
    production_only: bool = False,
    validate_urls: bool = False,
    sample_url_validation: int = 0
):
    """
    Export plants from MongoDB to JSONL format.

    Each line is a JSON object representing one image:
    {
        "plant_id": "...",
        "latin_name": "Rosa canina",
        "common_name": "Wild Rose",
        "image_url": "https://...",
        "image_source": "inaturalist",
        "license": "CC-BY",
        "license_permissive": true,
        "region": "EU_SW",
        "country": "ES",
        "family": "Rosaceae",
        "genus": "Rosa"
    }
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plants = db.plants
    total_plants = plants.count_documents({})

    logger.info(f"Processing {total_plants} plants...")

    records = []
    stats = {
        'total_plants': total_plants,
        'plants_processed': 0,
        'plants_skipped': 0,
        'images_total': 0,
        'images_permissive': 0,
        'images_restrictive': 0,
        'images_by_source': Counter(),
        'images_by_region': Counter(),
        'urls_validated': 0,
        'urls_valid': 0,
        'urls_invalid': 0
    }

    # Create class mapping (plant_id -> class_index)
    class_mapping = {}
    class_index = 0

    for plant in tqdm(plants.find({}), total=total_plants, desc="Exporting plants"):
        plant_id = str(plant['_id'])
        latin_name = plant.get('latinName', '')
        common_name = plant.get('commonName', '')

        if not latin_name:
            stats['plants_skipped'] += 1
            continue

        # Get images from both 'images' array and legacy 'img' array
        structured_images = plant.get('images', [])
        legacy_images = plant.get('img', [])

        # Skip if not enough images
        total_images = len(structured_images) + len(legacy_images)
        if total_images < min_images_per_species:
            stats['plants_skipped'] += 1
            continue

        # Get geographic info
        distribution = plant.get('distribution', {})
        countries = distribution.get('countries', [])
        continents = distribution.get('continents', [])
        region = classify_region(countries, continents)
        country = countries[0] if countries else 'UNKNOWN'

        # Get taxonomy info
        taxonomy = plant.get('taxonomy', {})
        characteristics = plant.get('characteristics', {})
        family = taxonomy.get('family') or characteristics.get('family', '')
        genus = taxonomy.get('genus', '')

        # Assign class index
        if latin_name not in class_mapping:
            class_mapping[latin_name] = class_index
            class_index += 1

        plant_class_idx = class_mapping[latin_name]

        # Process structured images
        for img in structured_images:
            url = img.get('url', '')
            if not url:
                continue

            license_str = img.get('license', '')
            is_permissive = is_permissive_license(license_str)

            # Skip non-permissive if production_only
            if production_only and not is_permissive:
                stats['images_restrictive'] += 1
                continue

            source = img.get('source', 'unknown')

            record = {
                'plant_id': plant_id,
                'class_idx': plant_class_idx,
                'latin_name': latin_name,
                'common_name': common_name,
                'image_url': url,
                'image_source': source,
                'license': license_str,
                'license_permissive': is_permissive,
                'attribution': img.get('attribution', ''),
                'region': region,
                'country': country,
                'family': family,
                'genus': genus
            }

            records.append(record)
            stats['images_total'] += 1
            stats['images_by_source'][source] += 1
            stats['images_by_region'][region] += 1

            if is_permissive:
                stats['images_permissive'] += 1
            else:
                stats['images_restrictive'] += 1

        # Process legacy images (no license info)
        for img_url in legacy_images:
            if not img_url or not isinstance(img_url, str):
                continue

            record = {
                'plant_id': plant_id,
                'class_idx': plant_class_idx,
                'latin_name': latin_name,
                'common_name': common_name,
                'image_url': img_url,
                'image_source': 'legacy',
                'license': '',
                'license_permissive': False,  # Unknown = treat as restrictive
                'attribution': '',
                'region': region,
                'country': country,
                'family': family,
                'genus': genus
            }

            # Skip legacy images if production_only
            if not production_only:
                records.append(record)
                stats['images_total'] += 1
                stats['images_by_source']['legacy'] += 1
                stats['images_by_region'][region] += 1
                stats['images_restrictive'] += 1

        stats['plants_processed'] += 1

    logger.info(f"Extracted {len(records)} image records from {stats['plants_processed']} plants")

    # Optional URL validation (sample)
    if validate_urls and sample_url_validation > 0:
        logger.info(f"Validating sample of {sample_url_validation} URLs...")
        sample_urls = [r['image_url'] for r in records[:sample_url_validation]]
        validation_results = validate_urls_batch(sample_urls)

        stats['urls_validated'] = len(sample_urls)
        stats['urls_valid'] = sum(1 for v in validation_results.values() if v)
        stats['urls_invalid'] = stats['urls_validated'] - stats['urls_valid']

        logger.info(f"URL validation: {stats['urls_valid']}/{stats['urls_validated']} valid")

    # Write JSONL
    output_file = output_path / 'dataset_raw.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"Dataset exported to: {output_file}")

    # Write class mapping
    mapping_file = output_path / 'class_mapping.json'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)

    logger.info(f"Class mapping saved to: {mapping_file}")

    # Write stats
    stats['images_by_source'] = dict(stats['images_by_source'])
    stats['images_by_region'] = dict(stats['images_by_region'])
    stats['num_classes'] = len(class_mapping)
    stats['export_timestamp'] = datetime.now().isoformat()

    stats_file = output_path / 'export_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Export stats saved to: {stats_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("                   EXPORT SUMMARY")
    print("=" * 60)
    print(f"Total plants processed:  {stats['plants_processed']:,}")
    print(f"Plants skipped:          {stats['plants_skipped']:,}")
    print(f"Total images exported:   {stats['images_total']:,}")
    print(f"Permissive licenses:     {stats['images_permissive']:,}")
    print(f"Restrictive licenses:    {stats['images_restrictive']:,}")
    print(f"Number of classes:       {stats['num_classes']:,}")

    print("\nImages by source:")
    for source, count in stats['images_by_source'].items():
        print(f"  {source}: {count:,}")

    print("\nImages by region:")
    for region, count in stats['images_by_region'].items():
        print(f"  {region}: {count:,}")

    if stats['urls_validated'] > 0:
        print(f"\nURL validation ({stats['urls_validated']} sampled):")
        print(f"  Valid: {stats['urls_valid']:,}")
        print(f"  Invalid: {stats['urls_invalid']:,}")

    print("=" * 60)

    return records, stats


def main():
    parser = argparse.ArgumentParser(description='Export plant dataset from MongoDB')
    parser.add_argument('--output-dir', '-o', type=str,
                       default='./data',
                       help='Output directory for exported data')
    parser.add_argument('--min-images', type=int, default=1,
                       help='Minimum images per species to include')
    parser.add_argument('--production-only', action='store_true',
                       help='Only export images with permissive licenses')
    parser.add_argument('--validate-urls', action='store_true',
                       help='Validate image URLs (slow)')
    parser.add_argument('--sample-validation', type=int, default=100,
                       help='Number of URLs to sample for validation')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Connect to MongoDB
    db = connect_to_mongodb()

    # Export dataset
    records, stats = export_to_jsonl(
        db,
        output_dir=args.output_dir,
        min_images_per_species=args.min_images,
        production_only=args.production_only,
        validate_urls=args.validate_urls,
        sample_url_validation=args.sample_validation if args.validate_urls else 0
    )

    logger.info("Export complete!")

    return records, stats


if __name__ == '__main__':
    main()
