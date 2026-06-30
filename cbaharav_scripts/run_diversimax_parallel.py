import argparse
import csv
import numpy as np
import pandas as pd
import random
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

def get_columns_to_keep(categories_file):
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

def load_people_and_shuffle(respondents_file, features, columns_to_keep):
    """
    Load and shuffle inside the worker to ensure each draw 
    starts with a unique ordering.
    """
    with open(respondents_file, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        people_rows = list(reader)

    # uncomment to randomize people
    # random.shuffle(people_rows)
    
    people_head = list(people_rows[0].keys())
    settings = Settings(
        id_column           = 'nationbuilder_id',
        columns_to_keep     = columns_to_keep,
        selection_algorithm = 'diversimax',
    )

    people, _ = read_in_people(people_head, people_rows, features, settings)
    return people, settings

def run_single_draw(args):
    """Worker function for a single Diversimax draw."""
    draw_no, categories_file, respondents_file, assembly_size, columns_to_keep = args
    
    # Re-initialize for process safety
    features = load_features(categories_file, assembly_size)
    people, settings = load_people_and_shuffle(respondents_file, features, columns_to_keep)

    success, selected_panels, report = run_stratification(
        features,
        people,
        number_people_wanted = assembly_size,
        settings             = settings,
    )

    if success and selected_panels:
        selected_ids = frozenset(np.int64(int(pid)) for pid in selected_panels[0])
        print(f"  Draw {draw_no} done — {len(selected_ids)} people selected", flush=True)
        return {'status': 'success', 'selected_ids': selected_ids}
    else:
        # Note: Report can be large, we only return status if it fails
        return {'status': 'failed'}

def main():
    args = parse_args()
    instance = args.instance
    num_draws = args.num_draws
    
    categories_file  = f'input-data/{instance}/categories.csv'
    respondents_file = f'input-data/{instance}/respondents.csv'
    output_file      = f'intermediate_data/dropped_0/{instance}/diversimax_panels_output_big.csv'
    assembly_size    = int(instance.split('_')[-1])
    columns_to_keep  = get_columns_to_keep(categories_file)

    console_print(f'Instance:      {instance}')
    console_print(f'Assembly size: {assembly_size}')
    console_print(f'Num draws:     {num_draws}')
    console_print(f'Using {cpu_count()} CPU cores for parallel execution...')

    task_args = [
        (i, categories_file, respondents_file, assembly_size, columns_to_keep)
        for i in range(1, num_draws + 1)
    ]

    with Pool(cpu_count()) as pool:
        results = pool.map(run_single_draw, task_args)

    panels = []
    failed = 0
    for res in results:
        if res['status'] == 'success':
            panels.append({
                'committees':    res['selected_ids'],
                'probabilities': 1.0 / num_draws,
            })
        else:
            failed += 1

    if failed > 0:
        console_print(f'  WARNING: {failed} draws failed.')

    # Merge unique panels
    merged = {}
    for panel in panels:
        key = panel['committees']
        merged[key] = merged.get(key, 0.0) + panel['probabilities']

    merged_panels = [{'committees': k, 'probabilities': v} for k, v in merged.items()]

    console_print(f'  After merging: {len(merged_panels)} unique panels from {len(panels)} successes')

    output_df = pd.DataFrame(merged_panels)
    output_df.to_csv(output_file, index=False)
    console_print(f'  Saved to {output_file}')

if __name__ == '__main__':
    main()