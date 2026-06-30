"""
Run the Legacy algorithm with hidden categories in parallel.
For each category (and category pair), hide the category from constraints
and run legacy, saving results to separate folders.

Usage:
    python run_legacy_hidden_parallel.py --instance instance_name_size
    python run_legacy_hidden_parallel.py --instance instance_name_size --num_draws 100
"""

import argparse
import csv
import numpy as np
import pandas as pd
import os
from itertools import combinations
from multiprocessing import Pool, cpu_count
from sortition_algorithms import run_stratification, read_in_features, read_in_people, Settings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance',   required=True,  help='Instance name, format should be instance_name_size')
    parser.add_argument('--num_draws',  type=int, default=1, help='Number of draws to run (default: 1)')
    return parser.parse_args()


def console_print(message):
    """Print with immediate flush so progress shows in real time."""
    print(message, flush=True)


def get_all_categories(categories_file):
    """
    Returns a list of all unique categories from categories.csv.
    """
    categories = []
    seen = set()
    with open(categories_file, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row['category']
            if cat not in seen:
                seen.add(cat)
                categories.append(cat)
    return categories


def get_columns_to_keep(categories_file, hidden_categories=None):
    """
    Reads categories.csv and returns a list of all unique category names
    plus 'nationbuilder_id', excluding any hidden categories.
    """
    if hidden_categories is None:
        hidden_categories = set()
    else:
        hidden_categories = set(hidden_categories)
    
    columns = ['nationbuilder_id']
    seen = set()
    with open(categories_file, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row['category']
            if cat not in seen and cat not in hidden_categories:
                seen.add(cat)
                columns.append(cat)
    return columns


def load_features(categories_file, assembly_size, hidden_categories=None):
    """
    Load categories.csv and convert to the format read_in_features expects,
    excluding any hidden categories.
    """
    if hidden_categories is None:
        hidden_categories = set()
    else:
        hidden_categories = set(hidden_categories)
    
    with open(categories_file, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            if r['category'] not in hidden_categories:
                rows.append({
                    'feature': r['category'],
                    'value':   r['name'],
                    'min':     r['min'],
                    'max':     r['max'],
                })
    
    features_head = ['feature', 'value', 'min', 'max']
    features, _, _ = read_in_features(features_head, rows, number_to_select=assembly_size)
    return features


def load_people(respondents_file, features, columns_to_keep):
    """
    Load respondents.csv and parse into the people structure read_in_people expects.
    """
    with open(respondents_file, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        people_rows = list(reader)

    people_head = list(people_rows[0].keys())

    settings = Settings(
        id_column           = 'nationbuilder_id',
        columns_to_keep     = columns_to_keep,
        selection_algorithm = 'legacy',
    )

    people, _ = read_in_people(people_head, people_rows, features, settings)
    return people, settings


def run_single_draw(args):
    """Worker function for a single draw with hidden categories."""
    draw_no, categories_file, respondents_file, assembly_size, columns_to_keep, hidden_categories = args
    
    # Reload inside worker to ensure thread/process safety
    try:
        features = load_features(categories_file, assembly_size, hidden_categories)
        people, settings = load_people(respondents_file, features, columns_to_keep)

        success, selected_panels, report = run_stratification(
            features,
            people,
            number_people_wanted = assembly_size,
            settings             = settings,
        )

        if success and selected_panels:
            selected_ids = frozenset(np.int64(int(pid)) for pid in selected_panels[0])
            return {'status': 'success', 'selected_ids': selected_ids}
        else:
            return {'status': 'failed', 'report': report}
    
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def run_draws_parallel(instance, num_draws, hidden_categories):
    """
    Run legacy with specified hidden categories using parallel processing.
    Returns (success, num_failed, output_file).
    """
    categories_file  = f'input-data/{instance}/categories.csv'
    respondents_file = f'input-data/{instance}/respondents.csv'
    
    # Create output directory and file
    if hidden_categories:
        hidden_str = '_'.join(sorted(hidden_categories))
        output_dir = f'intermediate_data/dropped_0/{instance}/hidden_big/{instance}_{hidden_str}'
    else:
        output_dir = f'intermediate_data/dropped_0/{instance}/rerun'
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/legacy_panels_output.csv'
    
    assembly_size = int(instance.split('_')[-1])
    columns_to_keep = get_columns_to_keep(categories_file, hidden_categories)

    hidden_str = ', '.join(hidden_categories) if hidden_categories else 'None'
    console_print(f'Hidden: [{hidden_str}] | Assembly: {assembly_size} | Draws: {num_draws}')

    # Verify features can be loaded
    try:
        _ = load_features(categories_file, assembly_size, hidden_categories)
    except Exception as e:
        console_print(f'  ERROR loading features: {e}')
        return False, 0, output_file

    # Prepare arguments for the pool
    task_args = [
        (i, categories_file, respondents_file, assembly_size, columns_to_keep, hidden_categories)
        for i in range(1, num_draws + 1)
    ]

    panels = []
    failed = 0

    # Run draws in parallel
    with Pool(cpu_count()) as pool:
        results = pool.map(run_single_draw, task_args)

    for res in results:
        if res['status'] == 'success':
            panels.append({
                'committees':    res['selected_ids'],
                'probabilities': 1.0 / num_draws,
            })
        else:
            failed += 1

    # Merge identical panels by summing their probabilities
    merged = {}
    for panel in panels:
        key = panel['committees']
        merged[key] = merged.get(key, 0.0) + panel['probabilities']

    merged_panels = [{'committees': k, 'probabilities': v} for k, v in merged.items()]

    success_msg = f'{len(merged_panels)} panels'
    if failed > 0:
        success_msg += f' ({failed} draws failed)'
    
    console_print(f'  → {success_msg}')

    if merged_panels:
        output_df = pd.DataFrame(merged_panels)
        output_df.index.name = None
        output_df.to_csv(output_file)
        return True, failed, output_file
    else:
        return False, failed, output_file


def run_configuration(args):
    """Worker function for running a single hidden category configuration."""
    instance, num_draws, hidden_categories = args
    
    hidden_str = ', '.join(hidden_categories) if hidden_categories else 'None'
    
    try:
        success, failed, output_file = run_draws_parallel(
            instance, 
            num_draws, 
            hidden_categories
        )
        return {
            'hidden_categories': hidden_categories,
            'success': success,
            'num_failed': failed,
            'output_file': output_file
        }
    except Exception as e:
        console_print(f'  [{hidden_str}] ERROR: {e}')
        return {
            'hidden_categories': hidden_categories,
            'success': False,
            'num_failed': 'ERROR',
            'output_file': 'N/A'
        }


def main():
    args = parse_args()
    instance = args.instance
    num_draws = args.num_draws
    
    categories_file = f'input-data/{instance}/categories.csv'
    
    # Get all unique categories
    all_categories = get_all_categories(categories_file)
    console_print(f'Instance: {instance} | Categories: {len(all_categories)} | Draws: {num_draws}')
    console_print(f'Using {cpu_count()} CPUs for parallel processing')
    
    # Build list of all configurations to run
    configurations = []
    
    # 1. Individual categories
    for category in all_categories:
        configurations.append((instance, num_draws, [category]))
    
    # # 2. Category pairs
    # category_pairs = list(combinations(all_categories, 2))
    # for cat1, cat2 in category_pairs:
    #     configurations.append((instance, num_draws, [cat1, cat2]))
    
    total_configs = len(configurations)
    console_print(f'\nTotal configurations to run: {total_configs}')
    console_print(f'  - Single categories: {len(all_categories)}')
    # console_print(f'  - Category pairs: {len(category_pairs)}')
    console_print(f'\nStarting parallel execution...\n')
    
    # Run all configurations in parallel
    # Note: Each configuration itself runs its draws in parallel,
    # may want to limit outer parallelism to avoid oversubscription ?
    # Using cpu_count() for outer parallelism since inner parallelism is also using cpu_count()
    
    results = []
    for i, config in enumerate(configurations, 1):
        console_print(f'\n[{i}/{total_configs}] Processing configuration...')
        result = run_configuration(config)
        results.append(result)
    
    # Print summary
    console_print(f'\n=== Summary ===')
    successful = sum(1 for r in results if r['success'])
    console_print(f'Completed: {successful}/{len(results)} configurations successful')
    
    # Print failures if any
    failures = [r for r in results if not r['success']]
    if failures:
        console_print(f'\nFailed configurations:')
        for r in failures:
            hidden_str = ', '.join(r['hidden_categories'])
            console_print(f'  - [{hidden_str}]')
    
    console_print('\nDone!')


if __name__ == '__main__':
    main()