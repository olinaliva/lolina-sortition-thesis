"""
Run the Diversimax algorithm to select panels.
Calling this implementation of diversimax https://github.com/sortitionfoundation/sortition-algorithms/tree/main
Assembly size is inferred from the instance name (last underscore-separated token).
Columns to keep are read automatically from categories.csv plus nationbuilder_id.

Usage:
    python run_diversimax.py --instance instance_name_size
    python run_diversimax.py --instance instance_name_size --num_draws 100
"""

import argparse
import csv
import numpy as np
import pandas as pd
from sortition_algorithms import run_stratification, read_in_features, read_in_people, Settings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance',   required=True,  help='Instance name, format should be instance_name_size')
    parser.add_argument('--num_draws',  type=int, default=1, help='Number of draws to run (default: 1)')
    return parser.parse_args()


def console_print(message):
    """Print with immediate flush so progress shows in real time."""
    print(message, flush=True)


def get_columns_to_keep(categories_file):
    """
    Reads categories.csv and returns a list of all unique category names
    plus 'nationbuilder_id'.
    """
    columns = ['nationbuilder_id']
    seen = set()
    with open(categories_file, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row['category']
            if cat not in seen:
                seen.add(cat)
                columns.append(cat)
    return columns


def load_features(categories_file, assembly_size):
    """
    Load categories.csv and convert to the format read_in_features expects.
    categories.csv columns: category, name, min, max
    read_in_features expects: feature, value, min, max
    """
    with open(categories_file, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
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


def run_draws(instance, num_draws):
    categories_file  = f'input-data/{instance}/categories.csv'
    respondents_file = f'input-data/{instance}/respondents.csv'
    output_file      = f'intermediate_data/dropped_0/{instance}/diversimax_panels_output.csv'
    assembly_size    = int(instance.split('_')[-1])
    columns_to_keep  = get_columns_to_keep(categories_file)

    console_print(f'Instance:      {instance}')
    console_print(f'Assembly size: {assembly_size}')
    console_print(f'Num draws:     {num_draws}')
    console_print(f'Columns:       {columns_to_keep}')
    console_print(f'Starting {num_draws} draws with algorithm: diversimax')

    features = load_features(categories_file, assembly_size)

    panels = []
    failed = 0

    for draw_no in range(1, num_draws + 1):
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
            console_print(f'  Draw {draw_no}/{num_draws} done — {len(selected_ids)} people selected')
        else:
            failed = failed + 1
            if failed == 1:
                console_print(f'  Draw {draw_no}/{num_draws} FAILED — first failure report:')
                console_print(report)
            else:
                console_print(f'  Draw {draw_no}/{num_draws} FAILED — skipping')

    if failed > 0:
        console_print(f'  WARNING: {failed} draws failed and were skipped')

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

    console_print(f'  After merging: {len(merged_panels)} unique panels (from {len(panels)} successful draws)')

    output_df = pd.DataFrame(merged_panels)
    output_df.index.name = None
    output_df.to_csv(output_file)
    console_print(f'  Saved to {output_file}')


def main():
    args = parse_args()
    run_draws(args.instance, args.num_draws)
    console_print('Done!')


if __name__ == '__main__':
    main()