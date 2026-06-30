"""
Run the Simulated Annealing algorithm with hidden categories.
For each category (and category pair), hide the category from constraints
and run SA, saving results to separate folders.

Usage:
    python run_SA_hidden.py --instance instance_name_size
    python run_SA_hidden.py --instance instance_name_size --num_draws 100
"""

import argparse
import csv
import numpy as np
import pandas as pd
import os
from itertools import combinations
from simanneal import Annealer

# SA Configuration
SA_TEMPERATURE    = 5000.0
SA_MIN_TEMP       = 0.01
SA_MAX_STEPS      = 50000
SA_THRESHOLD_STOP = 0
RANDOM_SEED       = 42
HOUSEHOLD_SWITCH  = False


##############################################
# Utility
##############################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance',   required=True,  help='Instance name, format should be instance_name_size')
    parser.add_argument('--num_draws',  type=int, default=1, help='Number of draws to run (default: 1)')
    return parser.parse_args()


def console_print(message):
    """Print with immediate flush so progress shows in real time."""
    #relevant for shell scripts
    print(message, flush=True)


##############################################
# Category filtering functions
##############################################

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


##############################################
# Load and preprocess data
##############################################

def load_data(respondents_file, categories_file, hidden_categories=None):
    """
    Load respondents and categories CSVs, excluding hidden categories.
    
    Returns:
        volunteers        — DataFrame of respondents
        characteristics   — DataFrame of categories (filtered)
        INPUT_SIZE        — number of respondents
        categories_number — number of category rows (after filtering)
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
    characteristics['priority'] = 1.0

    return volunteers, characteristics, len(volunteers), len(characteristics)


##############################################
# Pre-compute lookup structures
##############################################

def build_feature_matrix(volunteers, characteristics):
    """
    Build a boolean matrix of shape (INPUT_SIZE, categories_number).
    feature_matrix[i, k] == 1 if respondent i belongs to category k.
    """
    INPUT_SIZE        = len(volunteers)
    categories_number = len(characteristics)

    feature_matrix = np.zeros((INPUT_SIZE, categories_number), dtype=np.int8)

    for k in range(categories_number):
        col     = characteristics.iloc[k]['category']
        feature = characteristics.iloc[k]['feature']
        feature_matrix[:, k] = (volunteers[col] == feature).astype(np.int8)

    return feature_matrix


##############################################
# Simulated Annealer class
##############################################

class ThresholdReached(Exception):
    """Raised inside energy() to stop the anneal early when threshold is hit."""
    pass


class SortitionAnnealer(Annealer):
    """
    Simulated annealer that selects assembly_size respondents
    whose demographic composition fits within the category min/max quotas
    as closely as possible.
    """

    def __init__(self, state, volunteers, feature_matrix, char_vectors,
                 household_switch, threshold_stop):
        super().__init__(state)
        self.volunteers       = volunteers
        self.feature_matrix   = feature_matrix
        self.char_vectors     = char_vectors
        self.household_switch = household_switch
        self.threshold_stop   = threshold_stop
        self.INPUT_SIZE       = len(volunteers)
        self.last_energy      = None

    def move(self):
        """Swap one randomly chosen panel member for a random respondent."""
        idx = np.random.randint(len(self.state))
        self.state[idx] = np.random.randint(0, self.INPUT_SIZE)

    def energy(self):
        """
        Compute the objective function value for the current panel.
        Penalizes the squared distance from the midpoint: (min + max) / 2.
        """
        v_indices = np.array(self.state) % self.INPUT_SIZE

        # Hard Constraints: Unique respondents and household rules
        if len(np.unique(v_indices)) != len(v_indices):
            return 99999999

        if self.household_switch:
            household_ids = self.volunteers['HOUSEHOLD_ID'].iloc[v_indices].values
            if len(np.unique(household_ids)) != len(v_indices):
                return 99999999

        # Calculate current counts for categories
        counter  = self.feature_matrix[v_indices, :].sum(axis=0)
        mins     = np.array(self.char_vectors['min'])
        maxs     = np.array(self.char_vectors['max'])
        priority = np.array(self.char_vectors['priority'])

        # Midpoint Target
        targets = (mins + maxs) / 2

        # Calculate Penalty
        dist_from_target = counter - targets
        ret = float(np.sum(priority * (dist_from_target**2)))

        self.last_energy = ret

        if ret <= self.threshold_stop:
            raise ThresholdReached(f"energy={ret}")

        return ret


##############################################
# Run draws with hidden categories
##############################################

def run_draws(instance, num_draws, hidden_categories):
    """
    Run simulated annealing with specified hidden categories.
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
    output_file = f'{output_dir}/SA_panels_output.csv'
    
    assembly_size = int(instance.split('_')[-1])
    
    hidden_str = ', '.join(hidden_categories) if hidden_categories else 'None'
    console_print(f'Hidden: [{hidden_str}] | Assembly: {assembly_size} | Draws: {num_draws}')

    try:
        volunteers, characteristics, INPUT_SIZE, categories_number = load_data(
            respondents_file, categories_file, hidden_categories
        )
    except Exception as e:
        console_print(f'ERROR loading data: {e}')
        return False, 0, output_file

    feature_matrix = build_feature_matrix(volunteers, characteristics)

    char_vectors = {
        'category': characteristics['category'].tolist(),
        'feature':  characteristics['feature'].tolist(),
        'min':      characteristics['min'].tolist(),
        'max':      characteristics['max'].tolist(),
        'priority': characteristics['priority'].tolist(),
    }

    panels = []
    failed = 0

    for draw_no in range(1, num_draws + 1):
        try:
            initial_panel = np.random.choice(INPUT_SIZE, assembly_size, replace=False).tolist()

            annealer = SortitionAnnealer(
                state            = initial_panel,
                volunteers       = volunteers,
                feature_matrix   = feature_matrix,
                char_vectors     = char_vectors,
                household_switch = HOUSEHOLD_SWITCH,
                threshold_stop   = SA_THRESHOLD_STOP,
            )

            annealer.Tmax    = SA_TEMPERATURE
            annealer.Tmin    = SA_MIN_TEMP
            annealer.steps   = SA_MAX_STEPS
            annealer.updates = 0

            try:
                best_state, best_energy = annealer.anneal()
            except ThresholdReached:
                best_state  = annealer.state
                best_energy = annealer.last_energy

            selected_indices = np.array(best_state) % INPUT_SIZE
            selected_ids     = frozenset(volunteers['nationbuilder_id'].iloc[selected_indices].to_numpy(dtype=np.int64))

            panels.append({
                'committees':    selected_ids,
                'probabilities': 1.0 / num_draws,
            })

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


##############################################
# Main
##############################################

def main():
    args = parse_args()
    instance = args.instance
    num_draws = args.num_draws
    
    categories_file = f'input-data/{instance}/categories.csv'
    
    # Get all unique categories
    all_categories = get_all_categories(categories_file)
    console_print(f'Instance: {instance} | Categories: {len(all_categories)} | Draws: {num_draws}')
    
    # Set random seed
    np.random.seed(RANDOM_SEED % (2**32))
    
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
            console_print(f'  [{category}] ERROR: {e}')
            results.append({
                'hidden_categories': [category],
                'success': False,
                'num_failed': 'ERROR',
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
    #         console_print(f'  [{cat1}, {cat2}] ERROR: {e}')
    #         results.append({
    #             'hidden_categories': [cat1, cat2],
    #             'success': False,
    #             'num_failed': 'ERROR',
    #             'output_file': 'N/A'
    #         })
    
    # Print summary
    console_print(f'\n=== Summary ===')
    successful = sum(1 for r in results if r['success'])
    console_print(f'Completed: {successful}/{len(results)} configurations successful')
    
    console_print('\nDone!')


if __name__ == '__main__':
    main()