"""
Simulated annealing for random selection of citizens' assemblies
https://bluedemocracy.pl/wp-content/uploads/2025/01/Simulated-Annealing_JoS.pdf
Translated from R Version 8.6
Uses simanneal library

Assembly size is inferred from the instance name (last underscore-separated token).

Usage:
    python run_SA.py --instance instance_name_size 
    python run_SA.py --instance instance_name_size --num_draws 100

NOTES:
    - the SA parameters did not any undergo optimization, guided by default parameters in the R version
    - no code to process "priority" since no datasets in analysis used it, 
        see original R implementation for details
    - penalizes the distance from midpoints, but has code to penalize distance from quotas (can be swapped)
    - respondents, categories, output file paths are all tweaked in main, by default
        respondents_file = f'input-data/{instance}/respondents.csv'
        categories_file  = f'input-data/{instance}/categories.csv'
        output_file      = f'intermediate_data/dropped_0/{instance}/SA_panels_output.csv'
    - assigns each panel 1/[number of panels] probability, 
        merges panels with the same set of people and sums their probabilities
    
"""

import argparse
import pandas as pd
import numpy as np
from simanneal import Annealer

SA_TEMPERATURE    = 5000.0  # starting temperature for simulated annealing
SA_MIN_TEMP       = 0.01    # ending temperature
SA_MAX_STEPS      = 50000   # max iterations per draw
SA_THRESHOLD_STOP = 0       # stop early if energy reaches this value (0 = perfect fit)
RANDOM_SEED       = 42      # for reproducibility
HOUSEHOLD_SWITCH  = False   # set True to prevent two people from the same household
                            # (requires a HOUSEHOLD_ID column in respondents.csv)


##############################################
# Utility
##############################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance',  required=True, help='Instance name, format should be instance_name_size')
    parser.add_argument('--num_draws', type=int, default=1, help='Number of draws to run (default: 1)')
    return parser.parse_args()


def console_print(message):
    """Print with immediate flush so progress shows in real time."""
    print(message, flush=True)


##############################################
# Load and preprocess data
##############################################

def load_data(respondents_file, categories_file):
    """
    Load respondents and categories CSVs.

    categories.csv columns: category, name, min, max
      - 'name' is the demographic feature value (e.g. 'Male', '18-24')
      - 'min'  is the minimum acceptable count in the panel
      - 'max'  is the maximum acceptable count in the panel

    Returns:
        volunteers        — DataFrame of respondents
        characteristics   — DataFrame of categories
        INPUT_SIZE        — number of respondents
        categories_number — number of category rows
    """
    volunteers = pd.read_csv(respondents_file)

    characteristics = pd.read_csv(categories_file)
    characteristics = characteristics.rename(columns={'name': 'feature'})
    characteristics['min'] = characteristics['min'].astype(int)
    characteristics['max'] = characteristics['max'].astype(int)
    characteristics['priority'] = 1.0 # nothing has priority so 1 by default

    return volunteers, characteristics, len(volunteers), len(characteristics)


##############################################
# Pre-compute lookup structures
##############################################

def build_feature_matrix(volunteers, characteristics):
    """
    Build a boolean matrix of shape (INPUT_SIZE, categories_number).
    feature_matrix[i, k] == 1 if respondent i belongs to category k.
    Pre-computing this makes the energy function fast.
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
    UPDATE: as close to (min+max)/2 to match the R implementation

    State:  a Python list of assembly_size integer row-indices into volunteers.
    Energy: sum of squared shortfalls/overages outside [min, max] for each
            category row, weighted by priority (all 1.0).
            0 means every category count is within its allowed range.
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

    # This one penalizes if outside quotas
    # def energy(self):
    #     """
    #     Compute the objective function value for the current panel.
    #     Lower is better; 0 means all category counts are within [min, max].

    #     Penalty per category row:
    #       - count < min: penalty = (min - count)^2
    #       - count > max: penalty = (count - max)^2
    #       - min <= count <= max: penalty = 0

    #     Returns 99999999 for duplicate respondents or same-household pairs
    #     (if HOUSEHOLD_SWITCH is True).
    #     """
    #     v_indices = np.array(self.state) % self.INPUT_SIZE

    #     if len(np.unique(v_indices)) != len(v_indices):
    #         return 99999999

    #     if self.household_switch:
    #         household_ids = self.volunteers['HOUSEHOLD_ID'].iloc[v_indices].values
    #         if len(np.unique(household_ids)) != len(v_indices):
    #             return 99999999

    #     counter  = self.feature_matrix[v_indices, :].sum(axis=0)
    #     mins     = np.array(self.char_vectors['min'])
    #     maxs     = np.array(self.char_vectors['max'])
    #     priority = np.array(self.char_vectors['priority'])

    #     below = np.maximum(0, mins - counter)
    #     above = np.maximum(0, counter - maxs)
    #     ret   = float(np.sum(priority * (below**2 + above**2)))

    #     self.last_energy = ret

    #     if ret <= self.threshold_stop:
    #         raise ThresholdReached(f"energy={ret}")

    #     return ret

    # This one penalizes not hitting the single value in quota midpoints
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
# Main
##############################################

def main():
    args = parse_args()
    instance   = args.instance
    num_draws  = args.num_draws
    assembly_size = int(instance.split('_')[-1])

    respondents_file = f'input-data/{instance}/respondents.csv'
    categories_file  = f'input-data/{instance}/categories.csv'
    output_file      = f'intermediate_data/dropped_0/{instance}/SA_panels_output.csv'
    # output_file      = f'intermediate_data/dropped_0/{instance}/SA_panels_output_big.csv'

    np.random.seed(RANDOM_SEED % (2**32))

    console_print(f'Instance:      {instance}')
    console_print(f'Assembly size: {assembly_size}')
    console_print(f'Num draws:     {num_draws}')

    console_print('Loading data...')
    volunteers, characteristics, INPUT_SIZE, categories_number = load_data(
        respondents_file, categories_file
    )
    console_print(f'  Respondents:   {INPUT_SIZE}')
    console_print(f'  Category rows: {categories_number}')

    feature_matrix = build_feature_matrix(volunteers, characteristics)

    char_vectors = {
        'category': characteristics['category'].tolist(),
        'feature':  characteristics['feature'].tolist(),
        'min':      characteristics['min'].tolist(),
        'max':      characteristics['max'].tolist(),
        'priority': characteristics['priority'].tolist(),
    }

    panels = []
    console_print(f'\nStarting {num_draws} draws (assembly size = {assembly_size})...')

    for draw_no in range(1, num_draws + 1):
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

        console_print(f'  Draw {draw_no}/{num_draws} done — energy: {best_energy}')


    # Merge identical panels by summing their probabilities
    # Panels with the same set of IDs get collapsed into one row,
    # with their probabilities summed. 
    merged = {}
    for panel in panels:
        key = panel['committees']  # frozenset, so hashable
        if key in merged:
            merged[key] += panel['probabilities']
        else:
            merged[key] = panel['probabilities']

    panels = [{'committees': k, 'probabilities': v} for k, v in merged.items()]
    console_print(f"  After merging: {len(panels)} unique panels")

    output_df = pd.DataFrame(panels)
    output_df.index.name = None
    output_df.to_csv(output_file)
    console_print(f'\nSaved {num_draws} panels to {output_file}')
    console_print('Done!')


if __name__ == '__main__':
    main()