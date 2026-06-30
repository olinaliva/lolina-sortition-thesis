"""
Run dual annealing optimization with hidden categories.
Uses scipy's dual_annealing with multiprocessing for parallel execution.

Usage:
    python run_dual_annealing_hidden.py --instance instance_name_size
    python run_dual_annealing_hidden.py --instance instance_name_size --num_draws 100
"""

import argparse
import csv
import numpy as np
import pandas as pd
import os
from itertools import combinations
from scipy.optimize import dual_annealing
from multiprocessing import Pool, cpu_count

# Configuration
RANDOM_SEED = 42
HOUSEHOLD_SWITCH = False
MAX_ITER = 5000
MAX_FUN = 100000


##############################################
# Utility
##############################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', required=True, help='Instance name, format: instance_name_size')
    parser.add_argument('--num_draws', type=int, default=1, help='Number of draws per configuration')
    parser.add_argument('--parallel_configs', action='store_true', 
                        help='Parallelize across configurations instead of draws')
    return parser.parse_args()


def console_print(message):
    """Print with immediate flush for real-time progress."""
    #relevant for shell scripts
    print(message, flush=True)


##############################################
# Category filtering functions
##############################################

def get_all_categories(categories_file):
    """Returns a list of all unique categories from categories.csv."""
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


##############################################
# Load and preprocess data
##############################################

def load_data(respondents_file, categories_file, hidden_categories=None):
    """
    Load respondents and categories CSVs, excluding hidden categories.
    
    Returns:
        volunteers, characteristics, INPUT_SIZE, categories_number
    """
    if hidden_categories is None:
        hidden_categories = set()
    else:
        hidden_categories = set(hidden_categories)
    
    volunteers = pd.read_csv(respondents_file)
    characteristics = pd.read_csv(categories_file)
    
    # Filter out hidden categories
    if hidden_categories:
        characteristics = characteristics[~characteristics['category'].isin(hidden_categories)].copy()
    
    characteristics = characteristics.rename(columns={'name': 'feature'})
    characteristics['min'] = characteristics['min'].astype(int)
    characteristics['max'] = characteristics['max'].astype(int)
    
    return volunteers, characteristics, len(volunteers), len(characteristics)


def build_feature_matrix(volunteers, characteristics):
    """Build boolean matrix of shape (INPUT_SIZE, categories_number)."""
    INPUT_SIZE = len(volunteers)
    categories_number = len(characteristics)
    feature_matrix = np.zeros((INPUT_SIZE, categories_number), dtype=np.int8)
    
    for k in range(categories_number):
        col = characteristics.iloc[k]['category']
        feature = characteristics.iloc[k]['feature']
        feature_matrix[:, k] = (volunteers[col] == feature).astype(np.int8)
    
    return feature_matrix


##############################################
# Objective Function
##############################################

class ObjectiveFunction:
    """Wrapper for dual_annealing objective function."""
    
    def __init__(self, volunteers, feature_matrix, characteristics, 
                 assembly_size, household_switch):
        self.volunteers = volunteers
        self.feature_matrix = feature_matrix
        self.characteristics = characteristics
        self.assembly_size = assembly_size
        self.household_switch = household_switch
        self.INPUT_SIZE = len(volunteers)
        
        self.mins = characteristics['min'].values
        self.maxs = characteristics['max'].values
        self.targets = (self.mins + self.maxs) / 2
        
        # Priority support
        self.priority = np.ones(len(characteristics))
        if 'priority' in characteristics.columns:
            self.priority = characteristics['priority'].values
        
        # Household IDs
        if 'HOUSEHOLD_ID' in volunteers.columns:
            self.household_ids = volunteers['HOUSEHOLD_ID'].to_numpy()
        else:
            self.household_ids = None
        
        # Convert to numpy arrays for speed
        self.feature_matrix = np.asarray(feature_matrix, dtype=np.int8)
        self.targets = np.asarray(self.targets)
    
    def __call__(self, x):
        """
        Evaluate energy for continuous vector x.
        Round to get integer indices.
        """
        # Round to valid indices
        indices = (np.round(x).astype(int) % self.INPUT_SIZE)
        
        # Check for duplicates
        if np.unique(indices).size != indices.size:
            return 1e9
        
        # Household constraint
        if self.household_switch and self.household_ids is not None:
            household_ids = self.household_ids[indices]
            if np.unique(household_ids).size != indices.size:
                return 1e9
        
        # Calculate energy
        counter = self.feature_matrix[indices, :].sum(axis=0)
        dist_from_target = counter - self.targets
        energy = np.sum(self.priority * (dist_from_target**2))
        
        return energy


##############################################
# Single draw runner
##############################################

def run_single_draw(args):
    """Run a single draw of dual annealing."""
    draw_no, volunteers, feature_matrix, characteristics, bounds, INPUT_SIZE, assembly_size = args
    
    console_print(f"  Draw {draw_no}...")
    # Random initial state
    x0 = np.random.choice(INPUT_SIZE, assembly_size, replace=False).astype(float)
    
    # Create objective function
    obj_func = ObjectiveFunction(
        volunteers, feature_matrix, characteristics,
        assembly_size, HOUSEHOLD_SWITCH
    )
    
    # Run dual annealing
    result = dual_annealing(
        obj_func,
        bounds=bounds,
        maxiter=MAX_ITER,
        maxfun=MAX_FUN,
        seed=RANDOM_SEED + draw_no,
        initial_temp=4000,
        no_local_search=True,
        x0=x0
    )
    
    # Extract selected indices
    indices = (np.round(result.x).astype(int) % INPUT_SIZE)
    selected_ids = frozenset(
        volunteers['nationbuilder_id'].iloc[indices].to_numpy(dtype=np.int64)
    )
    
    return {
        'committees': selected_ids,
        'probabilities': 1.0,
        'energy': result.fun,
        'draw_no': draw_no
    }


##############################################
# Run draws with hidden categories
##############################################

def run_draws(instance, num_draws, hidden_categories):
    """
    Run dual annealing with specified hidden categories.
    Returns (success, num_failed, output_file).
    """
    categories_file = f'input-data/{instance}/categories.csv'
    respondents_file = f'input-data/{instance}/respondents.csv'
    
    # Create output directory
    if hidden_categories:
        hidden_str = '_'.join(sorted(hidden_categories))
        output_dir = f'intermediate_data/dropped_0/{instance}/hidden_dual/{instance}_{hidden_str}'
    else:
        output_dir = f'intermediate_data/dropped_0/{instance}/rerun_dual'
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/DA_panels_output.csv'
    
    assembly_size = int(instance.split('_')[-1])
    
    hidden_str = ', '.join(hidden_categories) if hidden_categories else 'None'
    console_print(f'Hidden: [{hidden_str}] | Assembly: {assembly_size} | Draws: {num_draws}')
    
    # Load data
    try:
        volunteers, characteristics, INPUT_SIZE, categories_number = load_data(
            respondents_file, categories_file, hidden_categories
        )
    except Exception as e:
        console_print(f'ERROR loading data: {e}')
        return False, 0, output_file
    
    console_print(f'  Respondents: {INPUT_SIZE} | Category rows: {categories_number}')
    
    # Build feature matrix
    feature_matrix = build_feature_matrix(volunteers, characteristics)
    
    # Define bounds for optimization
    bounds = [(0, INPUT_SIZE - 1)] * assembly_size
    
    # Prepare arguments for parallel draws
    args_list = [
        (draw_no, volunteers, feature_matrix, characteristics, bounds, INPUT_SIZE, assembly_size)
        for draw_no in range(1, num_draws + 1)
    ]
    
    # Run draws in parallel
    try:
        with Pool(cpu_count()) as pool:
            results = pool.map(run_single_draw, args_list)
        
        # Create panels
        panels = [
            {
                'committees': r['committees'],
                'probabilities': 1.0 / num_draws
            }
            for r in results
        ]
        
        # Merge identical panels
        merged = {}
        for panel in panels:
            key = panel['committees']
            if key in merged:
                merged[key] += panel['probabilities']
            else:
                merged[key] = panel['probabilities']
        
        merged_panels = [{'committees': k, 'probabilities': v} for k, v in merged.items()]
        
        console_print(f'  → {len(merged_panels)} unique panels')
        
        # Save output
        if merged_panels:
            output_df = pd.DataFrame(merged_panels)
            output_df.index.name = None
            output_df.to_csv(output_file)
            return True, 0, output_file
        else:
            return False, num_draws, output_file
            
    except Exception as e:
        console_print(f'ERROR during optimization: {e}')
        return False, num_draws, output_file


##############################################
# Configuration runner (for parallel configs)
##############################################

def run_single_config(args):
    """Run a single configuration (hidden categories)."""
    instance, num_draws, hidden_categories = args
    try:
        success, failed, output_file = run_draws(instance, num_draws, hidden_categories)
        return {
            'hidden_categories': hidden_categories,
            'success': success,
            'num_failed': failed,
            'output_file': output_file
        }
    except Exception as e:
        hidden_str = ', '.join(hidden_categories) if hidden_categories else 'None'
        console_print(f'  [{hidden_str}] ERROR: {e}')
        return {
            'hidden_categories': hidden_categories,
            'success': False,
            'num_failed': 'ERROR',
            'output_file': 'N/A'
        }


##############################################
# Main
##############################################

def main():
    args = parse_args()
    instance = args.instance
    num_draws = args.num_draws
    parallel_configs = args.parallel_configs
    
    categories_file = f'input-data/{instance}/categories.csv'
    
    # Get all unique categories
    all_categories = get_all_categories(categories_file)
    console_print(f'Instance: {instance} | Categories: {len(all_categories)} | Draws: {num_draws}')
    console_print(f'Max iterations: {MAX_ITER} | Max function calls: {MAX_FUN}')
    console_print(f'CPUs available: {cpu_count()}')
    
    # Set random seed
    np.random.seed(RANDOM_SEED % (2**32))
    
    results = []
    
    # Prepare configurations (single categories)
    console_print(f'\n=== Running with individual categories hidden ({len(all_categories)}) ===')
    
    configs = [(instance, num_draws, [category]) for category in all_categories]
    
    if parallel_configs:
        # Parallelize across configurations
        console_print(f'Running {len(configs)} configurations in parallel...')
        with Pool(min(cpu_count(), len(configs))) as pool:
            results = pool.map(run_single_config, configs)
    else:
        # Run configurations sequentially (parallelizing draws within each)
        for category in all_categories:
            result = run_single_config((instance, num_draws, [category]))
            results.append(result)
    
    # Uncomment to also run category pairs
    # console_print(f'\n=== Running with category pairs hidden ===')
    # category_pairs = list(combinations(all_categories, 2))
    # pair_configs = [(instance, num_draws, list(pair)) for pair in category_pairs]
    # 
    # if parallel_configs:
    #     with Pool(min(cpu_count(), len(pair_configs))) as pool:
    #         pair_results = pool.map(run_single_config, pair_configs)
    #     results.extend(pair_results)
    # else:
    #     for cat1, cat2 in category_pairs:
    #         result = run_single_config((instance, num_draws, [cat1, cat2]))
    #         results.append(result)
    
    # Print summary
    console_print(f'\n=== Summary ===')
    successful = sum(1 for r in results if r['success'])
    console_print(f'Completed: {successful}/{len(results)} configurations successful')
    
    # Show failures if any
    failures = [r for r in results if not r['success']]
    if failures:
        console_print(f'\nFailed configurations:')
        for f in failures:
            hidden_str = ', '.join(f['hidden_categories']) if f['hidden_categories'] else 'None'
            console_print(f'  [{hidden_str}]: {f["num_failed"]}')
    
    console_print('\nDone!')

if __name__ == '__main__':
    main()