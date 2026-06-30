import argparse
import pandas as pd
import numpy as np
from simanneal import Annealer
from multiprocessing import Pool, cpu_count

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
    parser.add_argument('--instance', required=True)
    parser.add_argument('--num_draws', type=int, default=1)
    return parser.parse_args()


def console_print(message):
    print(message, flush=True)


##############################################
# Load data
##############################################

def load_data(respondents_file, categories_file):
    volunteers = pd.read_csv(respondents_file)

    characteristics = pd.read_csv(categories_file)
    characteristics = characteristics.rename(columns={'name': 'feature'})
    characteristics['min'] = characteristics['min'].astype(int)
    characteristics['max'] = characteristics['max'].astype(int)
    characteristics['priority'] = 1.0

    return volunteers, characteristics, len(volunteers), len(characteristics)


##############################################
# Feature matrix
##############################################

def build_feature_matrix(volunteers, characteristics):
    INPUT_SIZE = len(volunteers)
    categories_number = len(characteristics)

    feature_matrix = np.zeros((INPUT_SIZE, categories_number), dtype=np.int8)

    for k in range(categories_number):
        col = characteristics.iloc[k]['category']
        feature = characteristics.iloc[k]['feature']
        feature_matrix[:, k] = (volunteers[col] == feature).astype(np.int8)

    return feature_matrix


##############################################
# Annealer
##############################################

class ThresholdReached(Exception):
    pass


class SortitionAnnealer(Annealer):

    def __init__(self, state, volunteers, feature_matrix, char_vectors,
                 household_switch, threshold_stop):
        super().__init__(state)

        self.volunteers       = volunteers
        self.feature_matrix   = feature_matrix
        self.household_switch = household_switch
        self.threshold_stop   = threshold_stop
        self.INPUT_SIZE       = len(volunteers)
        self.last_energy      = None

        # Precompute
        self.mins = np.array(char_vectors['min'])
        self.maxs = np.array(char_vectors['max'])
        self.targets = (self.mins + self.maxs) / 2
        self.priority = np.array(char_vectors['priority'])

        if 'HOUSEHOLD_ID' in volunteers.columns:
            self.household_ids = volunteers['HOUSEHOLD_ID'].to_numpy()
        else:
            self.household_ids = None


    def move(self):
        idx = np.random.randint(len(self.state))

        current_set = set(self.state)
        available = list(set(range(self.INPUT_SIZE)) - current_set)

        if available:
            self.state[idx] = np.random.choice(available)


    def energy(self):
        v_indices = self.state

        # uniqueness
        if np.unique(v_indices).size != len(v_indices):
            return 99999999

        # household constraint
        if self.household_switch and self.household_ids is not None:
            household_ids = self.household_ids[v_indices]
            if np.unique(household_ids).size != len(v_indices):
                return 99999999

        counter = self.feature_matrix[v_indices, :].sum(axis=0)
        dist = counter - self.targets

        ret = float(np.sum(self.priority * (dist ** 2)))

        self.last_energy = ret

        if ret <= self.threshold_stop:
            raise ThresholdReached

        return ret


##############################################
# Worker
##############################################

def run_single_draw(args):
    draw_no, INPUT_SIZE, assembly_size, volunteers, feature_matrix, char_vectors = args

    np.random.seed(RANDOM_SEED + draw_no)

    initial_panel = np.random.choice(INPUT_SIZE, assembly_size, replace=False).tolist()

    annealer = SortitionAnnealer(
        state=initial_panel,
        volunteers=volunteers,
        feature_matrix=feature_matrix,
        char_vectors=char_vectors,
        household_switch=HOUSEHOLD_SWITCH,
        threshold_stop=SA_THRESHOLD_STOP,
    )

    annealer.Tmax = SA_TEMPERATURE
    annealer.Tmin = SA_MIN_TEMP
    annealer.steps = SA_MAX_STEPS
    annealer.updates = 0

    try:
        best_state, best_energy = annealer.anneal()
    except ThresholdReached:
        best_state = annealer.state
        best_energy = annealer.last_energy

    selected_ids = frozenset(
        volunteers['nationbuilder_id'].iloc[best_state].to_numpy(dtype=np.int64)
    )

    return {
        'committees': selected_ids,
        'probabilities': 1.0,
        'energy': best_energy
    }


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
    output_file      = f'intermediate_data/dropped_0/{instance}/SA_panels_output_big.csv'

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

    console_print(f'\nStarting {num_draws} draws (parallel)...')

    args_list = [
        (draw_no, INPUT_SIZE, assembly_size, volunteers, feature_matrix, char_vectors)
        for draw_no in range(1, num_draws + 1)
    ]

    with Pool(cpu_count()) as p:
        results = p.map(run_single_draw, args_list)

    # normalize probabilities
    panels = [
        {
            'committees': r['committees'],
            'probabilities': 1.0 / num_draws
        }
        for r in results
    ]

    # merge identical panels
    merged = {}
    for panel in panels:
        key = panel['committees']
        merged[key] = merged.get(key, 0) + panel['probabilities']

    panels = [{'committees': k, 'probabilities': v} for k, v in merged.items()]

    console_print(f"After merging: {len(panels)} unique panels")

    output_df = pd.DataFrame(panels)
    output_df.to_csv(output_file)

    console_print(f'\nSaved {num_draws} panels to {output_file}')
    console_print('Done!')


if __name__ == '__main__':
    main()