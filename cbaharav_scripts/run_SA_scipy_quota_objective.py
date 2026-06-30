"""
Dual annealing optimizer for citizens' assemblies
Uses scipy's dual_annealing (similar to R's genSA)
"""

import argparse
import pandas as pd
import numpy as np
from scipy.optimize import dual_annealing
from multiprocessing import Pool, cpu_count

RANDOM_SEED = 42
HOUSEHOLD_SWITCH = False
# MAX_ITER = 200000  # Maximum iterations
MAX_ITER = 5000 # 200k is overkill and 5k should really be enough for a dual annealer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', required=True)
    parser.add_argument('--num_draws', type=int, default=1)
    return parser.parse_args()


def console_print(message):
    print(message, flush=True)


def load_data(respondents_file, categories_file):
    volunteers = pd.read_csv(respondents_file)
    characteristics = pd.read_csv(categories_file)
    characteristics = characteristics.rename(columns={'name': 'feature'})
    characteristics['min'] = characteristics['min'].astype(int)
    characteristics['max'] = characteristics['max'].astype(int)
    return volunteers, characteristics, len(volunteers), len(characteristics)


def build_feature_matrix(volunteers, characteristics):
    INPUT_SIZE = len(volunteers)
    categories_number = len(characteristics)
    feature_matrix = np.zeros((INPUT_SIZE, categories_number), dtype=np.int8)
    
    for k in range(categories_number):
        col = characteristics.iloc[k]['category']
        feature = characteristics.iloc[k]['feature']
        feature_matrix[:, k] = (volunteers[col] == feature).astype(np.int8)
    
    return feature_matrix


class ObjectiveFunction:
    """Wrapper to make the energy function work with scipy"""
    
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

        if 'HOUSEHOLD_ID' in volunteers.columns:
            self.household_ids = volunteers['HOUSEHOLD_ID'].to_numpy()
        else:
            self.household_ids = None

        self.feature_matrix = np.asarray(feature_matrix, dtype=np.int8)
        self.targets = np.asarray(self.targets)
        self.targets = (self.mins + self.maxs) / 2
        self.priority = np.ones(len(characteristics))  # Add priority support
        if 'priority' in characteristics.columns:
            self.priority = characteristics['priority'].values
    
    def __call__(self, x):
        """
        x is a continuous vector that we'll round to get integer indices.
        scipy's dual_annealing works on continuous spaces, so we convert.
        """
        # Round to valid indices
        indices = (np.round(x).astype(int) % self.INPUT_SIZE)
        
        # Check for duplicates
        if np.unique(indices).size != indices.size:
            return 1e9
        
        if self.household_switch and self.household_ids is not None:
            household_ids = self.household_ids[indices]
            if np.unique(household_ids).size != indices.size:
                return 1e9
        
        # Calculate energy
        counter = self.feature_matrix[indices, :].sum(axis=0)
        # # minimize maximum abs(count - target)/target
        # dist_from_target = abs(counter - self.targets)
        # energy = np.max(dist_from_target/self.targets)

        # Minimize maximum relative distance from target
        # dist_from_target = np.abs(counter - self.targets)
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     relative_dist = dist_from_target / self.targets
        #     relative_dist[self.targets == 0] = dist_from_target[self.targets == 0]  # Use absolute distance when target is 0
        # energy = np.max(relative_dist)


        # # minimize square distance from quotas
        dist_from_target = np.where(counter < self.mins, 
                                     self.mins - counter,
                                     np.where(counter > self.maxs,
                                              counter - self.maxs,
                                              0))
        energy = np.sum(dist_from_target**2)
        
        return energy


def main():
    args = parse_args()
    instance = args.instance
    num_draws = args.num_draws
    assembly_size = int(instance.split('_')[-1])

    respondents_file = f'input-data/{instance}/respondents.csv'
    categories_file = f'input-data/{instance}/categories.csv'
    output_file = f'intermediate_data/dropped_0/{instance}/SA_panels_output_scipyquota.csv'

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

    panels = []
    console_print(f'\nStarting {num_draws} draws (assembly size = {assembly_size})...')

    # Define bounds: each variable can be any respondent index
    bounds = [(0, INPUT_SIZE - 1)] * assembly_size

    obj_func = ObjectiveFunction(
        volunteers, feature_matrix, characteristics,
        assembly_size, HOUSEHOLD_SWITCH
    )
    
    args_list = [
        (draw_no, volunteers, feature_matrix, characteristics, bounds, INPUT_SIZE, assembly_size)
        for draw_no in range(1, num_draws + 1)
    ]

    with Pool(cpu_count()) as p:
        results = p.map(run_single_draw, args_list)

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

    panels = [{'committees': k, 'probabilities': v} for k, v in merged.items()]
    console_print(f"  After merging: {len(panels)} unique panels")

    output_df = pd.DataFrame(panels)
    output_df.index.name = None
    output_df.to_csv(output_file)
    console_print(f'\nSaved {num_draws} panels to {output_file}')
    console_print('Done!')

# def run_single_draw(args):
#     draw_no, obj_func, bounds, INPUT_SIZE, assembly_size, volunteers = args
def run_single_draw(args):
    draw_no, volunteers, feature_matrix, characteristics, bounds, INPUT_SIZE, assembly_size = args
    x0 = np.random.choice(INPUT_SIZE, assembly_size, replace=False).astype(float)
    obj_func = ObjectiveFunction(
        volunteers, feature_matrix, characteristics,
        assembly_size, HOUSEHOLD_SWITCH
    )
    result = dual_annealing(
        obj_func,
        bounds=bounds,
        maxiter=MAX_ITER,
        # maxfun=2000000,
        maxfun=100000,       # Safety limit, unlikely to hit and a drop down from 2M
        seed=RANDOM_SEED + draw_no,
        initial_temp=4000,
        no_local_search=True,
        x0=x0
    )

    indices = (np.round(result.x).astype(int) % INPUT_SIZE)

    selected_ids = frozenset(
        volunteers['nationbuilder_id'].iloc[indices].to_numpy(dtype=np.int64)
    )
    print(f"draw no: {draw_no}")
    
    return {
        'committees': selected_ids,
        'probabilities': 1.0,
        'energy': result.fun,
        'draw_no': draw_no
    }
if __name__ == '__main__':
    main()