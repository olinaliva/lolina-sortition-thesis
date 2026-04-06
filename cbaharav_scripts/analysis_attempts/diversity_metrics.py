import pandas as pd
import re
from itertools import combinations
import os
import numpy as np

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Set to 1 to include an instance, 0 to skip it
INSTANCES = {
    'geneva_30': 1,
    # 'some_other_instance':     0,
}

# Algorithms to run. Each entry: (output_subdirectroy, panel_distribution_file)
# Use {INSTANCE} as a placeholder for the instance name.
# File names are hardcoded so change it here if different naming convention used
ALGORITHMS = {
    'leximin':   'intermediate_data/dropped_0/{INSTANCE}/{INSTANCE}_m1000_leximin_opt_probabilities.csv',
    'goldilocks':'intermediate_data/dropped_0/{INSTANCE}/{INSTANCE}_m1000_goldilocks_p=50_gamma_1_scale=1_opt_probabilities.csv',
    'SA':        'intermediate_data/dropped_0/{INSTANCE}/SA_panels_output.csv',
    'diversimax':'intermediate_data/dropped_0/{INSTANCE}/diversimax_panels_output.csv',
    'entropy':   'intermediate_data/dropped_0/{INSTANCE}/entropy_panels_output.csv',
    'legacy':    'intermediate_data/dropped_0/{INSTANCE}/legacy_panels_output.csv',   
}

# Legacy is the only optional algorithm; all others will error if missing
OPTIONAL_ALGORITHMS = {'legacy'}

CATEGORIES_FILE_TEMPLATE = 'input-data/{INSTANCE}/categories.csv'
RESP_FILE_TEMPLATE       = 'input-data/{INSTANCE}/respondents.csv'
OUTPUT_DIR_TEMPLATE      = 'analysis_attempts/{INSTANCE}/{ALGORITHM}'


# -----------------------------------------------------------------------------
# PARSING
# -----------------------------------------------------------------------------

# Panel distribution is expected to be in a csv with columns
# ,committees,probabilities 
# and rows that look something like this
# 0,"frozenset({np.int64(2), np.int64(392), ..., np.int64(503)})",3.90679292004838e-10

def parse_frozenset(fs_str):
    """Extracts person IDs from a frozenset string like frozenset({np.int64(0), ...})"""
    if pd.isna(fs_str) or fs_str == "":
        return set()
    return set(map(int, re.findall(r'np\.int64\((\d+)\)', str(fs_str))))


# -----------------------------------------------------------------------------
# TYPE ROSTER + PREPROCESSING
# -----------------------------------------------------------------------------

def build_type_roster(df_resp, ATTR_COLUMNS):
    """
    Returns {attribute_tuple -> set of person IDs}.
    Each unique combination of attribute values across ATTR_COLUMNS is a 'type'.
    Maps each type to the set of pool members who have that combination.
    """
    type_roster = {}
    for person_id, row in df_resp.iterrows():
        t = []
        for col in ATTR_COLUMNS:
            t.append(row[col])
        t = tuple(t)
        if t not in type_roster:
            type_roster[t] = set()
        type_roster[t].add(person_id)
    return type_roster

def convert_to_types(committee_set, person_to_type):
    """
    Converts a set of person IDs into a {attribute_tuple -> count} dict.
    Represents each panel by how many of each person-type it contains
    rather than tracking individual person IDs.
    """
    type_counts = {}
    for pid in committee_set:
        t = person_to_type[pid]
        if t not in type_counts:
            type_counts[t] = 0
        type_counts[t] = type_counts[t] + 1
    return type_counts

# -----------------------------------------------------------------------------
# DIVERSITY COUNTING METRICS
# -----------------------------------------------------------------------------

def _weighted_percentile(values_and_probs, percentile):
    """
    Probability-weighted percentile: sort panels by value ascending, accumulate
    probability mass, return the value at which cumulative mass first reaches
    or exceeds the given percentile.
    """
    sorted_pairs = sorted(values_and_probs, key=lambda x: x[0])
    cumulative = 0.0
    for value, prob in sorted_pairs:
        cumulative += prob
        if cumulative >= percentile:
            return value
    return sorted_pairs[-1][0]  # floating point safety (in case percentile never sums to 1)


def _count_unique_singles(committee_types, ATTR_COLUMNS):
    """Number of distinct (attribute, value) pairs present on a panel."""
    seen = set()
    for t in committee_types:
        for col_idx, val in enumerate(t):
            seen.add((col_idx, val))
    return len(seen)


def _count_unique_pairs(committee_types, ATTR_COLUMNS):
    """Number of distinct two-attribute-value combinations present on a panel."""
    seen = set()
    n = len(ATTR_COLUMNS)
    for t in committee_types:
        for i, j in combinations(range(n), 2):
            seen.add((i, t[i], j, t[j]))
    return len(seen)


def _count_unique_trios(committee_types, ATTR_COLUMNS):
    """Number of distinct three-attribute-value combinations present on a panel."""
    seen = set()
    n = len(ATTR_COLUMNS)
    for t in committee_types:
        for i, j, k in combinations(range(n), 3):
            seen.add((i, t[i], j, t[j], k, t[k]))
    return len(seen)


def _count_unique_vectors(committee_types):
    """Number of distinct full attribute vectors (person types) present on a panel."""
    return len(committee_types)

def _max_possible_singles(type_roster, ATTR_COLUMNS, panel_size):
    """
    Pool ceiling for unique singles, capped by panel size.
    Each person on a panel contributes exactly n singles (one per attribute),
    so a panel of panel_size people can cover at most panel_size * n distinct singles.
    The true ceiling is min(that, how many distinct singles exist in the pool).
    """
    seen = set()
    for t in type_roster:
        for col_idx, val in enumerate(t):
            seen.add((col_idx, val))
    pool_ceiling = len(seen)
    panel_ceiling = panel_size * len(ATTR_COLUMNS)
    return min(pool_ceiling, panel_ceiling)

def _max_possible_pairs(type_roster, ATTR_COLUMNS, panel_size):
    """
    Pool ceiling for unique pairs, capped by panel size.
    Each person contributes C(n, 2) pairs, so a panel of panel_size people
    can cover at most panel_size * C(n, 2) distinct pairs.
    The true ceiling is min(that, how many distinct pairs exist in the pool).
    """
    seen = set()
    n = len(ATTR_COLUMNS)
    for t in type_roster:
        for i, j in combinations(range(n), 2):
            seen.add((i, t[i], j, t[j]))
    pool_ceiling = len(seen)
    pairs_per_person = len(list(combinations(range(n), 2)))
    panel_ceiling = panel_size * pairs_per_person
    return min(pool_ceiling, panel_ceiling)

def _max_possible_trios(type_roster, ATTR_COLUMNS, panel_size):
    """
    Pool ceiling for unique trios, capped by panel size.
    Each person contributes C(n, 3) trios, so a panel of panel_size people
    can cover at most panel_size * C(n, 3) distinct trios.
    The true ceiling is min(that, how many distinct trios exist in the pool).
    """
    seen = set()
    n = len(ATTR_COLUMNS)
    for t in type_roster:
        for i, j, k in combinations(range(n), 3):
            seen.add((i, t[i], j, t[j], k, t[k]))
    pool_ceiling = len(seen)
    trios_per_person = len(list(combinations(range(n), 3)))
    panel_ceiling = panel_size * trios_per_person
    return min(pool_ceiling, panel_ceiling)

def compute_diversity_count_metrics(df_probs, type_roster, ATTR_COLUMNS, panel_size):
    """
    For each of four diversity levels (singles, pairs, trios, vectors), computes
    8 summary metrics across the distribution of panels:
 
      max, min, p25, p50, p75:  distribution stats (probability-weighted percentiles)
      weighted_mean:             E[count] across the lottery
      avg_panel:                 count on the synthetic average panel
                                 (types with expected_count >= 1.0) / at least one "full seat"
      max_possible:              min(pool ceiling, panel_size * combos per person) for that level
    """
    counts_singles = []
    counts_pairs   = []
    counts_trios   = []
    counts_vectors = []
 
    for _, row in df_probs.iterrows():
        ct = row['committee_types']
        types_present = list(ct.keys())
        counts_singles.append(_count_unique_singles(types_present, ATTR_COLUMNS))
        counts_pairs.append(_count_unique_pairs(types_present, ATTR_COLUMNS))
        if len(ATTR_COLUMNS) >= 3:
            counts_trios.append(_count_unique_trios(types_present, ATTR_COLUMNS))
        else:
            counts_trios.append(0)
        counts_vectors.append(_count_unique_vectors(ct))
 
    probs = df_probs['probabilities'].tolist()
 
    # Build average panel: types with expected_count >= 1.0
    type_expected_counts = {}
    for (_, row), prob in zip(df_probs.iterrows(), probs):
        for t, cnt in row['committee_types'].items():
            if t not in type_expected_counts:
                type_expected_counts[t] = 0.0
            type_expected_counts[t] = type_expected_counts[t] + cnt * prob
 
    avg_panel_types = []
    for t, ec in type_expected_counts.items():
        if ec >= 1.0:
            avg_panel_types.append(t)
 
    avg_singles = _count_unique_singles(avg_panel_types, ATTR_COLUMNS)
    avg_pairs   = _count_unique_pairs(avg_panel_types, ATTR_COLUMNS)
    if len(ATTR_COLUMNS) >= 3:
        avg_trios = _count_unique_trios(avg_panel_types, ATTR_COLUMNS)
    else:
        avg_trios = 0
    avg_vectors = len(avg_panel_types)
 
    max_singles = _max_possible_singles(type_roster, ATTR_COLUMNS, panel_size)
    max_pairs   = _max_possible_pairs(type_roster, ATTR_COLUMNS, panel_size)
    if len(ATTR_COLUMNS) >= 3:
        max_trios = _max_possible_trios(type_roster, ATTR_COLUMNS, panel_size)
    else:
        max_trios = 0
    max_vectors = min(len(type_roster), panel_size)
 
    def _summarize(counts_list, avg_val, max_val):
        caps = []
        for i in range(len(counts_list)):
            caps.append((counts_list[i], probs[i]))
        weighted_mean = 0.0
        for c, p in caps:
            weighted_mean = weighted_mean + c * p
        result = {}
        result['max']           = max(counts_list)
        result['min']           = min(counts_list)
        result['p25']           = _weighted_percentile(caps, 0.25)
        result['p50']           = _weighted_percentile(caps, 0.50)
        result['p75']           = _weighted_percentile(caps, 0.75)
        result['weighted_mean'] = weighted_mean
        result['avg_panel']     = avg_val
        result['max_possible']  = max_val
        return result
 
    results = {}
    results['singles'] = _summarize(counts_singles, avg_singles, max_singles)
    results['pairs']   = _summarize(counts_pairs,   avg_pairs,   max_pairs)
    results['trios']   = _summarize(counts_trios,   avg_trios,   max_trios)
    results['vectors'] = _summarize(counts_vectors, avg_vectors, max_vectors)
    return results

# -----------------------------------------------------------------------------
# HAMMING DISTANCE DIVERSITY METRICS
# -----------------------------------------------------------------------------

def _hamming(type_a, type_b):
    """Hamming distance: number of attribute positions where two types differ."""
    distance = 0
    for i in range(len(type_a)):
        if type_a[i] != type_b[i]:
            distance = distance + 1
    return distance

def _panel_hamming_stats(types_and_weights):
    """
    Given a list of (type_tuple, weight) pairs, computes four Hamming-based
    diversity statistics over all distinct type pairs:
 
      mean_unweighted:  mean pairwise distance, each pair counted equally
      mean_weighted:    mean pairwise distance weighted by weight_a * weight_b
      sum_unweighted:   sum of pairwise distances, each pair counted equally
      sum_weighted:     sum of pairwise distances weighted by weight_a * weight_b
    """
    if len(types_and_weights) < 2:
        return {'mean_unweighted': 0.0, 'mean_weighted': 0.0,
                'sum_unweighted':  0.0, 'sum_weighted':  0.0}
 
    sum_uw, sum_w, total_w_pairs, n_pairs = 0.0, 0.0, 0.0, 0
 
    for i in range(len(types_and_weights)):
        for j in range(i + 1, len(types_and_weights)):
            t_a, w_a = types_and_weights[i]
            t_b, w_b = types_and_weights[j]
            d = _hamming(t_a, t_b)
            sum_uw        = sum_uw + d
            sum_w         = sum_w + d * w_a * w_b
            total_w_pairs = total_w_pairs + w_a * w_b
            n_pairs       = n_pairs + 1
 
    return {
        'mean_unweighted': sum_uw / n_pairs       if n_pairs > 0       else 0.0,
        'mean_weighted':   sum_w  / total_w_pairs if total_w_pairs > 0 else 0.0,
        'sum_unweighted':  sum_uw,
        'sum_weighted':    sum_w,
    }

def compute_hamming_diversity_metrics(df_probs, ATTR_COLUMNS):
    """
    For each panel, computes four Hamming-based diversity scores:
      mean_unweighted, mean_weighted, sum_unweighted, sum_weighted.
    For each score, reports 8 summary statistics across the panel distribution:
      max, min, p25, p50, p75, weighted_mean,
      avg_panel_unweighted (average panel, types weighted equally),
      avg_panel_weighted   (average panel, types weighted by expected_count).
    """
    panel_probs = df_probs['probabilities'].tolist()

    per_panel_mean_unweighted = []
    per_panel_mean_weighted   = []
    per_panel_sum_unweighted  = []
    per_panel_sum_weighted    = []

    for _, row in df_probs.iterrows():
        ct = row['committee_types']

        types_uw = []
        for t in ct:
            types_uw.append((t, 1))

        types_w = []
        for t, cnt in ct.items():
            types_w.append((t, cnt))

        stats_uw = _panel_hamming_stats(types_uw)
        stats_w  = _panel_hamming_stats(types_w)

        per_panel_mean_unweighted.append(stats_uw['mean_unweighted'])
        per_panel_sum_unweighted.append(stats_uw['sum_unweighted'])
        per_panel_mean_weighted.append(stats_w['mean_weighted'])
        per_panel_sum_weighted.append(stats_w['sum_weighted'])

    # Build average panel: types with expected_count >= 1.0
    type_expected_counts = {}
    for (_, row), prob in zip(df_probs.iterrows(), panel_probs):
        for t, cnt in row['committee_types'].items():
            if t not in type_expected_counts:
                type_expected_counts[t] = 0.0
            type_expected_counts[t] = type_expected_counts[t] + cnt * prob

    avg_panel_types = {}
    for t, ec in type_expected_counts.items():
        if ec >= 1.0:
            avg_panel_types[t] = ec

    avg_panel_uw_input = []
    for t in avg_panel_types:
        avg_panel_uw_input.append((t, 1))

    avg_panel_w_input = []
    for t, ec in avg_panel_types.items():
        avg_panel_w_input.append((t, ec))

    avg_uw = _panel_hamming_stats(avg_panel_uw_input)
    avg_w  = _panel_hamming_stats(avg_panel_w_input)

    def _summarize(values, uw_avg, w_avg):
        vp = []
        for i in range(len(values)):
            vp.append((values[i], panel_probs[i]))

        weighted_mean = 0.0
        for v, p in vp:
            weighted_mean = weighted_mean + v * p

        result = {}
        result['max']                  = max(values)
        result['min']                  = min(values)
        result['p25']                  = _weighted_percentile(vp, 0.25)
        result['p50']                  = _weighted_percentile(vp, 0.50)
        result['p75']                  = _weighted_percentile(vp, 0.75)
        result['weighted_mean']        = weighted_mean
        result['avg_panel_unweighted'] = uw_avg
        result['avg_panel_weighted']   = w_avg
        return result

    results = {}
    results['mean_unweighted'] = _summarize(per_panel_mean_unweighted, avg_uw['mean_unweighted'], avg_w['mean_unweighted'])
    results['mean_weighted']   = _summarize(per_panel_mean_weighted,   avg_uw['mean_weighted'],   avg_w['mean_weighted'])
    results['sum_unweighted']  = _summarize(per_panel_sum_unweighted,  avg_uw['sum_unweighted'],  avg_w['sum_unweighted'])
    results['sum_weighted']    = _summarize(per_panel_sum_weighted,    avg_uw['sum_weighted'],    avg_w['sum_weighted'])
    return results

# -----------------------------------------------------------------------------
# PER-INSTANCE-ALGORITHM RUNNER
# -----------------------------------------------------------------------------

def run_one(instance, algorithm, prob_file):
    """
    Runs diversity count and Hamming metrics for one (instance, algorithm) pair
    and saves the two output CSVs.
    """
    tag = f"[{instance} / {algorithm}]"
    output_dir = OUTPUT_DIR_TEMPLATE.format(INSTANCE=instance, ALGORITHM=algorithm)
    os.makedirs(output_dir, exist_ok=True)

    # Load shared inputs
    df_cat = pd.read_csv(CATEGORIES_FILE_TEMPLATE.format(INSTANCE=instance))
    ATTR_COLUMNS = df_cat['category'].unique().tolist()

    df_resp = pd.read_csv(RESP_FILE_TEMPLATE.format(INSTANCE=instance)).set_index('nationbuilder_id')
    panel_size = int(instance.split('_')[-1])  # infer from instance name, e.g. "geneva_30" -> 30

    df_probs = pd.read_csv(prob_file)
    df_probs['committee_set'] = df_probs['committees'].apply(parse_frozenset)

    # Convert panels to type representation
    type_roster = build_type_roster(df_resp, ATTR_COLUMNS)
    person_to_type = {pid: t for t, pids in type_roster.items() for pid in pids}
    df_probs['committee_types'] = df_probs['committee_set'].apply(
        lambda s: convert_to_types(s, person_to_type)
    )
    print(f"{tag} {len(type_roster)} types in pool of {len(df_resp)}, panel_size={panel_size}")

    # Diversity count metrics
    diversity_metrics = compute_diversity_count_metrics(df_probs, type_roster, ATTR_COLUMNS, panel_size)
    metric_names = ['max', 'min', 'p25', 'p50', 'p75', 'weighted_mean', 'avg_panel', 'max_possible']
    levels = ['singles', 'pairs', 'trios', 'vectors']
    diversity_csv = os.path.join(output_dir, f'{instance}_diversity_count_metrics.csv')
    pd.DataFrame(
        {level: {m: diversity_metrics[level][m] for m in metric_names} for level in levels}
    ).rename_axis('metric').to_csv(diversity_csv)
    print(f"{tag} Saved diversity count metrics -> {diversity_csv}")

    # Hamming distance metrics
    hamming_metrics = compute_hamming_diversity_metrics(df_probs, ATTR_COLUMNS)
    hamming_stat_names = ['max', 'min', 'p25', 'p50', 'p75', 'weighted_mean',
                          'avg_panel_unweighted', 'avg_panel_weighted']
    hamming_variants = ['mean_unweighted', 'mean_weighted', 'sum_unweighted', 'sum_weighted']
    hamming_csv = os.path.join(output_dir, f'{instance}_hamming_diversity_metrics.csv')
    pd.DataFrame(
        {variant: {s: hamming_metrics[variant][s] for s in hamming_stat_names}
         for variant in hamming_variants}
    ).rename_axis('metric').to_csv(hamming_csv)
    print(f"{tag} Saved Hamming diversity metrics  -> {hamming_csv}")


# -----------------------------------------------------------------------------
# BATCH RUNNER
# -----------------------------------------------------------------------------

def run_all():
    active_instances = [inst for inst, active in INSTANCES.items() if active]
    print(f"Running {len(active_instances)} instance(s) x {len(ALGORITHMS)} algorithm(s)...\n")

    for instance in active_instances:
        for algorithm, prob_template in ALGORITHMS.items():
            prob_file = prob_template.format(INSTANCE=instance)
            tag = f"[{instance} / {algorithm}]"

            # Handle optional algorithms (legacy): skip gracefully if file missing or empty
            if algorithm in OPTIONAL_ALGORITHMS:
                if not os.path.exists(prob_file):
                    print(f"{tag} WARNING: file not found, skipping. ({prob_file})")
                    continue
                if os.path.getsize(prob_file) == 0:
                    print(f"{tag} WARNING: file is empty, skipping. ({prob_file})")
                    continue

            try:
                run_one(instance, algorithm, prob_file)
            except Exception as e:
                print(f"{tag} ERROR: {e}")
                raise   # re-raise for non-optional algorithms so failures are visible

        print()  # blank line between instances

    print("All done.")


if __name__ == "__main__":
    run_all()