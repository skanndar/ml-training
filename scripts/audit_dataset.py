#!/usr/bin/env python3
"""
audit_dataset.py - Phase 0: Dataset Audit

Analyzes the MongoDB plant collection to understand:
- Total species and images
- Image distribution per species
- License distribution
- Image source distribution
- Geographic distribution
- Data quality issues

Usage:
    python scripts/audit_dataset.py [--output ./results/audit_report.json]
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Permissive licenses for production use
PERMISSIVE_LICENSES = {
    'CC0', 'CC-BY', 'CC-BY-SA',
    'public domain', 'Public Domain',
    'No known copyright', 'No copyright',
    'Apache-2.0', 'MIT', 'BSD'
}

# Known restrictive licenses (for filtering)
RESTRICTIVE_LICENSES = {
    'CC-BY-NC', 'CC-BY-NC-SA', 'CC-BY-NC-ND',
    'CC-BY-ND', 'GPL', 'AGPL',
    'All rights reserved', 'Copyright'
}


def connect_to_mongodb():
    """Connect to MongoDB using environment variables."""
    load_dotenv()

    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    db_name = os.getenv('MONGODB_DB_NAME', 'qhopsDB')

    logger.info(f"Connecting to MongoDB: {mongo_uri}")

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


def audit_plants_collection(db):
    """Audit the plants collection and return statistics."""

    plants = db.plants

    # Basic counts
    total_plants = plants.count_documents({})
    logger.info(f"Total plant documents: {total_plants}")

    if total_plants == 0:
        logger.error("No plants found in database!")
        return None

    # Initialize counters
    stats = {
        'timestamp': datetime.now().isoformat(),
        'total_species': total_plants,
        'total_images': 0,
        'images_per_species': {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'distribution': defaultdict(int)
        },
        'licenses': Counter(),
        'permissive_licenses_count': 0,
        'restrictive_licenses_count': 0,
        'unknown_licenses_count': 0,
        'sources': Counter(),
        'regions': Counter(),
        'countries': Counter(),
        'missing_fields': {
            'latinName': 0,
            'commonName': 0,
            'images': 0,
            'taxonomy': 0,
            'distribution': 0
        },
        'data_quality': {
            'species_with_no_images': 0,
            'species_with_few_images': 0,  # < 5 images
            'species_with_good_images': 0,  # 5-50 images
            'species_with_many_images': 0,  # > 50 images
        },
        'taxonomy_coverage': {
            'has_family': 0,
            'has_genus': 0,
            'has_species': 0
        },
        'sample_species': []  # First 10 for reference
    }

    image_counts = []

    # Iterate through all plants
    logger.info("Analyzing plants...")
    for i, plant in enumerate(tqdm(plants.find({}), total=total_plants, desc="Auditing plants")):

        # Check missing fields
        if not plant.get('latinName'):
            stats['missing_fields']['latinName'] += 1
        if not plant.get('commonName'):
            stats['missing_fields']['commonName'] += 1
        if not plant.get('images') and not plant.get('img'):
            stats['missing_fields']['images'] += 1
        if not plant.get('taxonomy'):
            stats['missing_fields']['taxonomy'] += 1
        if not plant.get('distribution'):
            stats['missing_fields']['distribution'] += 1

        # Count images (check both 'images' array and legacy 'img' array)
        images = plant.get('images', [])
        legacy_images = plant.get('img', [])

        num_images = len(images) + len(legacy_images)
        image_counts.append(num_images)
        stats['total_images'] += num_images

        # Categorize by image count
        if num_images == 0:
            stats['data_quality']['species_with_no_images'] += 1
        elif num_images < 5:
            stats['data_quality']['species_with_few_images'] += 1
        elif num_images <= 50:
            stats['data_quality']['species_with_good_images'] += 1
        else:
            stats['data_quality']['species_with_many_images'] += 1

        # Analyze structured images
        for img in images:
            # License analysis
            license_str = img.get('license', 'unknown')
            if license_str:
                stats['licenses'][license_str] += 1

                # Classify license
                is_permissive = any(perm.lower() in license_str.lower() for perm in PERMISSIVE_LICENSES)
                is_restrictive = any(restr.lower() in license_str.lower() for restr in RESTRICTIVE_LICENSES)

                if is_permissive:
                    stats['permissive_licenses_count'] += 1
                elif is_restrictive:
                    stats['restrictive_licenses_count'] += 1
                else:
                    stats['unknown_licenses_count'] += 1
            else:
                stats['unknown_licenses_count'] += 1

            # Source analysis
            source = img.get('source', 'unknown')
            stats['sources'][source] += 1

        # Geographic distribution
        distribution = plant.get('distribution', {})
        countries = distribution.get('countries', [])
        continents = distribution.get('continents', [])

        for country in countries:
            stats['countries'][country] += 1

        # Determine region
        region = classify_region(countries, continents)
        stats['regions'][region] += 1

        # Taxonomy coverage
        taxonomy = plant.get('taxonomy', {})
        if taxonomy.get('family'):
            stats['taxonomy_coverage']['has_family'] += 1
        if taxonomy.get('genus'):
            stats['taxonomy_coverage']['has_genus'] += 1
        if taxonomy.get('species'):
            stats['taxonomy_coverage']['has_species'] += 1

        # Sample species (first 10)
        if i < 10:
            stats['sample_species'].append({
                'latinName': plant.get('latinName', 'N/A'),
                'commonName': plant.get('commonName', 'N/A'),
                'num_images': num_images,
                'sources': list(set(img.get('source', 'unknown') for img in images))
            })

    # Calculate image distribution stats
    if image_counts:
        stats['images_per_species']['min'] = min(image_counts)
        stats['images_per_species']['max'] = max(image_counts)
        stats['images_per_species']['avg'] = sum(image_counts) / len(image_counts)

        # Distribution buckets
        for count in image_counts:
            if count == 0:
                bucket = '0'
            elif count < 5:
                bucket = '1-4'
            elif count < 10:
                bucket = '5-9'
            elif count < 20:
                bucket = '10-19'
            elif count < 50:
                bucket = '20-49'
            elif count < 100:
                bucket = '50-99'
            else:
                bucket = '100+'
            stats['images_per_species']['distribution'][bucket] += 1

    # Convert Counters to dicts for JSON serialization
    stats['licenses'] = dict(stats['licenses'].most_common(20))
    stats['sources'] = dict(stats['sources'])
    stats['regions'] = dict(stats['regions'])
    stats['countries'] = dict(stats['countries'].most_common(30))
    stats['images_per_species']['distribution'] = dict(stats['images_per_species']['distribution'])

    return stats


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


def print_report(stats: dict):
    """Print a human-readable report."""

    print("\n" + "=" * 70)
    print("                    APLANTIDA DATASET AUDIT REPORT")
    print("=" * 70)

    print(f"\nTimestamp: {stats['timestamp']}")

    print("\n--- BASIC STATISTICS ---")
    print(f"Total species:           {stats['total_species']:,}")
    print(f"Total images:            {stats['total_images']:,}")
    print(f"Avg images per species:  {stats['images_per_species']['avg']:.1f}")
    print(f"Min images per species:  {stats['images_per_species']['min']}")
    print(f"Max images per species:  {stats['images_per_species']['max']}")

    print("\n--- DATA QUALITY ---")
    print(f"Species with NO images:    {stats['data_quality']['species_with_no_images']:,}")
    print(f"Species with <5 images:    {stats['data_quality']['species_with_few_images']:,}")
    print(f"Species with 5-50 images:  {stats['data_quality']['species_with_good_images']:,}")
    print(f"Species with >50 images:   {stats['data_quality']['species_with_many_images']:,}")

    print("\n--- IMAGE DISTRIBUTION ---")
    for bucket, count in sorted(stats['images_per_species']['distribution'].items()):
        pct = count / stats['total_species'] * 100
        bar = '#' * int(pct / 2)
        print(f"  {bucket:>6} images: {count:>5} ({pct:>5.1f}%) {bar}")

    print("\n--- LICENSE DISTRIBUTION ---")
    print(f"Permissive licenses:   {stats['permissive_licenses_count']:,}")
    print(f"Restrictive licenses:  {stats['restrictive_licenses_count']:,}")
    print(f"Unknown licenses:      {stats['unknown_licenses_count']:,}")
    print("\nTop licenses:")
    for license_name, count in list(stats['licenses'].items())[:10]:
        print(f"  {license_name}: {count:,}")

    print("\n--- IMAGE SOURCES ---")
    for source, count in stats['sources'].items():
        pct = count / stats['total_images'] * 100 if stats['total_images'] > 0 else 0
        print(f"  {source}: {count:,} ({pct:.1f}%)")

    print("\n--- GEOGRAPHIC DISTRIBUTION ---")
    for region, count in stats['regions'].items():
        pct = count / stats['total_species'] * 100
        print(f"  {region}: {count:,} ({pct:.1f}%)")

    print("\n--- TOP COUNTRIES ---")
    for country, count in list(stats['countries'].items())[:15]:
        print(f"  {country}: {count:,}")

    print("\n--- MISSING FIELDS ---")
    for field, count in stats['missing_fields'].items():
        if count > 0:
            pct = count / stats['total_species'] * 100
            print(f"  Missing {field}: {count:,} ({pct:.1f}%)")

    print("\n--- TAXONOMY COVERAGE ---")
    for field, count in stats['taxonomy_coverage'].items():
        pct = count / stats['total_species'] * 100
        print(f"  {field}: {count:,} ({pct:.1f}%)")

    print("\n--- SAMPLE SPECIES ---")
    for sp in stats['sample_species']:
        print(f"  - {sp['latinName']} ({sp['commonName']}): {sp['num_images']} images from {sp['sources']}")

    # Production readiness assessment
    print("\n" + "=" * 70)
    print("                      PRODUCTION READINESS")
    print("=" * 70)

    issues = []
    warnings = []

    if stats['total_species'] < 1000:
        issues.append(f"Low species count ({stats['total_species']}). Target: 9000+")

    if stats['total_images'] < 100000:
        warnings.append(f"Moderate image count ({stats['total_images']}). Target: 900k+")

    if stats['data_quality']['species_with_no_images'] > stats['total_species'] * 0.1:
        issues.append(f"High % of species with no images ({stats['data_quality']['species_with_no_images']})")

    permissive_pct = stats['permissive_licenses_count'] / stats['total_images'] * 100 if stats['total_images'] > 0 else 0
    if permissive_pct < 50:
        warnings.append(f"Only {permissive_pct:.1f}% of images have permissive licenses")

    if stats['missing_fields']['latinName'] > 0:
        issues.append(f"{stats['missing_fields']['latinName']} species missing latinName")

    if issues:
        print("\nCRITICAL ISSUES:")
        for issue in issues:
            print(f"  [!] {issue}")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  [~] {warning}")

    if not issues and not warnings:
        print("\n[OK] Dataset appears ready for training!")
    elif not issues:
        print("\n[OK] Dataset can proceed with warnings noted above")
    else:
        print("\n[!!] Please address critical issues before training")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Audit Aplantida plant dataset')
    parser.add_argument('--output', '-o', type=str,
                       default='./results/audit_report.json',
                       help='Output path for JSON report')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Connect to MongoDB
    db = connect_to_mongodb()

    # Run audit
    logger.info("Starting dataset audit...")
    stats = audit_plants_collection(db)

    if stats is None:
        logger.error("Audit failed - no data found")
        sys.exit(1)

    # Print report
    print_report(stats)

    # Save JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info(f"Audit report saved to: {output_path}")

    return stats


if __name__ == '__main__':
    main()
