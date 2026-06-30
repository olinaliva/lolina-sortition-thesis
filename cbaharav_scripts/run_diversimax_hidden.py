"""
Run the Diversimax algorithm with hidden categories.
For each category (and category pair), hide the category from constraints
and run diversimax, saving results to separate folders.

Usage:
    python run_diversimax_hidden.py --instance instance_name_size
    python run_diversimax_hidden.py --instance instance_name_size --num_draws 100
"""

import argparse
import csv
import numpy as np
import pandas as pd
import os
from itertools import combinations
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


def load_people(respondents_file, features, columns_to_keep, assembly_size):
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
        selection_algorithm = 'diversimax',
    )

    people, _ = read_in_people(people_head, people_rows, features, settings)
    return people, settings


def run_draws(instance, num_draws, hidden_categories):
    """
    Run diversimax with specified hidden categories.
    Returns (success, num_failed, output_file).
    """
    categories_file  = f'input-data/{instance}/categories.csv'
    respondents_file = f'input-data/{instance}/respondents.csv'
    
    # Create output directory and file
    if hidden_categories:
        hidden_str = '_'.join(sorted(hidden_categories))
        output_dir = f'intermediate_data/dropped_0/{instance}/hidden/{instance}_{hidden_str}'
    else:
        output_dir = f'intermediate_data/dropped_0/{instance}/rerun'
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/diversimax_panels_output.csv'
    
    assembly_size = int(instance.split('_')[-1])
    columns_to_keep = get_columns_to_keep(categories_file, hidden_categories)

    hidden_str = ', '.join(hidden_categories) if hidden_categories else 'None'
    console_print(f'Hidden: [{hidden_str}] | Assembly: {assembly_size} | Draws: {num_draws}')

    try:
        features = load_features(categories_file, assembly_size, hidden_categories)
    except Exception as e:
        console_print(f'ERROR loading features: {e}')
        return False, 0, output_file

    panels = []
    failed = 0

    for draw_no in range(1, num_draws + 1):
        try:
            people, settings = load_people(respondents_file, features, columns_to_keep, assembly_size)

            success, selected_panels, report = run_stratification(
                features,
                people,
                number_people_wanted = assembly_size,
                settings             = settings,
            )

            if success and selected_panels:
                selected_ids = frozenset(np.int64(int(pid)) for pid in selected_panels[0])
                panels.append({
                    'committees':    selected_ids,
                    'probabilities': 1.0 / num_draws,
                })
            else:
                failed = failed + 1
        
        except Exception as e:
            failed = failed + 1

    # Merge identical panels by summing their probabilities
    merged = {}
    for panel in panels:
        key = panel['committees']
        if key in merged:
            merged[key] = merged[key] + panel['probabilities']
        else:
            merged[key] = panel['probabilities']

    merged_panels = []
    for k, v in merged.items():
        merged_panels.append({'committees': k, 'probabilities': v})

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


def main():
    args = parse_args()
    instance = args.instance
    num_draws = args.num_draws
    
    categories_file = f'input-data/{instance}/categories.csv'
    
    # Get all unique categories
    all_categories = get_all_categories(categories_file)
    console_print(f'Instance: {instance} | Categories: {len(all_categories)} | Draws: {num_draws}')
    
    results = []
    
    # 1. Run with each single category hidden
    console_print(f'\n=== Hiding individual categories ({len(all_categories)}) ===')
    
    for category in all_categories:
        try:
            success, failed, output_file = run_draws(
                instance, 
                num_draws, 
                hidden_categories=[category]
            )
            results.append({
                'hidden_categories': [category],
                'success': success,
                'num_failed': failed,
                'output_file': output_file
            })
        except Exception as e:
            console_print(f'  [{category}] ERROR')
            results.append({
                'hidden_categories': [category],
                'success': False,
                'num_failed': f'ERROR',
                'output_file': 'N/A'
            })
    
    # # 2. Run with each pair of categories hidden
    # category_pairs = list(combinations(all_categories, 2))
    # console_print(f'\n=== Hiding category pairs ({len(category_pairs)}) ===')
    
    # for cat1, cat2 in category_pairs:
    #     try:
    #         success, failed, output_file = run_draws(
    #             instance, 
    #             num_draws, 
    #             hidden_categories=[cat1, cat2]
    #         )
    #         results.append({
    #             'hidden_categories': [cat1, cat2],
    #             'success': success,
    #             'num_failed': failed,
    #             'output_file': output_file
    #         })
    #     except Exception as e:
    #         console_print(f'  [{cat1}, {cat2}] ERROR')
    #         results.append({
    #             'hidden_categories': [cat1, cat2],
    #             'success': False,
    #             'num_failed': f'ERROR',
    #             'output_file': 'N/A'
    #         })
    
    # Print summary
    console_print(f'\n=== Summary ===')
    successful = sum(1 for r in results if r['success'])
    console_print(f'Completed: {successful}/{len(results)} configurations successful')
    
    console_print('\nDone!')


if __name__ == '__main__':
    main()