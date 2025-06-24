import numpy as np
import pandas as pd
import os
import random
import time
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

# --- Paths (relative to src/algorithms/GA/) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'Results_UPO-ABTS.xlsx')
INIT_GEN_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'initial_generations')
RESULTS_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'pso')

# --- Data and Utility Functions ---
def load_data(file_path):
    return pd.read_excel(file_path)

def is_in_hull(points, point):
    hull = ConvexHull(points)
    delaunay = Delaunay(hull.points[hull.vertices])
    return delaunay.find_simplex(point) >= 0

def interpolate_or_extrapolate_mean_specific_rate(points, values, point):
    if is_in_hull(points, point):
        mean_specific_rate = griddata(points, values, point, method='linear')
        if np.isnan(mean_specific_rate):
            mean_specific_rate = griddata(points, values, point, method='nearest')
    else:
        model = LinearRegression()
        model.fit(points, values)
        mean_specific_rate = model.predict([point])
    return mean_specific_rate[0] * random.uniform(0.87, 1.13)

data = load_data(DATA_PATH)
points = data[['pH', 'Salt Concentration', 'Cosubstrate Concentration', 'Organic Solvent Concentration', 'Temperature']].values
values = data['Mean Specific Rate [U/mg]'].values

def objective_function(params, points, values):
    return interpolate_or_extrapolate_mean_specific_rate(points, values, params)

def evaluate_population(population, points, values):
    return np.array(Parallel(n_jobs=-1)(delayed(objective_function)(params, points, values) for params in population))

def rank_selection(sorted_data, num_parents):
    return sorted_data.head(num_parents).values

def roulette_selection(data, num_parents):
    total_fitness = data['target'].sum()
    probabilities = data['target'] / total_fitness
    selected = data.sample(n=num_parents, weights=probabilities)
    return selected.values

def tournament_selection(data, num_parents, tournament_size=3):
    selected = []
    for _ in range(num_parents):
        tournament = data.sample(n=tournament_size)
        winner = tournament.loc[tournament['target'].idxmax()]
        selected.append(winner.values)
    return np.array(selected)

def cross_over_single_point(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def cross_over_uniform(parent1, parent2):
    mask = np.random.rand(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2

def cross_over_random(parents):
    return np.array([random.choice(parents[:, i]) for i in range(parents.shape[1])])

def mutate(value, var_range, mut_prob, mut_extent):
    if random.uniform(0, 1) < mut_prob:
        if mut_extent == 1:
            mutation = random.uniform(var_range[0], var_range[1])
            new_value = round(mutation, 1)
        elif mut_extent < 1:
            mutation = random.uniform(-mut_extent, mut_extent) * (var_range[1] - var_range[0])
            new_value = value + mutation
        return np.clip(new_value, var_range[0], var_range[1])
    return value

def clip_parameters(parameters, var_ranges):
    return [np.clip(param, var_range[0], var_range[1]) for param, var_range in zip(parameters, var_ranges)]

def iterate_ga(iteration, pop_size, var_names, var_ranges, parents, elite, mut_prob, mut_extent, cross_over_method, selection_method, dest_folder, gen_idx):
    if iteration == 0:
        data = pd.read_csv(os.path.join(dest_folder, f'initial_generation_{gen_idx}.csv'), delimiter=';')
    else:
        data = pd.read_csv(os.path.join(dest_folder, f'proposed_experiments_iter_{iteration}.csv'), delimiter=';')

    if 'target' not in data.columns or data['target'].isnull().all():
        data['target'] = evaluate_population(data[var_names].values, points, values)
        data.to_csv(os.path.join(dest_folder, f'proposed_experiments_iter_{iteration}.csv'), index=False, sep=';')

    sorted_data = data.sort_values(by='target', ascending=False)

    if selection_method == 'rank':
        selected_parents = rank_selection(sorted_data, parents)
    elif selection_method == 'roulette':
        selected_parents = roulette_selection(data, parents)
    elif selection_method == 'tournament':
        selected_parents = tournament_selection(data, parents)

    next_gen = np.zeros((pop_size, len(var_names)))
    unique_combinations = set()

    for c in range(pop_size):
        if c < elite:
            next_gen[c] = selected_parents[c]
        else:
            if cross_over_method == 'single_point':
                parent1, parent2 = random.sample(list(selected_parents), 2)
                child1, child2 = cross_over_single_point(parent1, parent2)
            elif cross_over_method == 'uniform':
                parent1, parent2 = random.sample(list(selected_parents), 2)
                child1, child2 = cross_over_uniform(parent1, parent2)
            elif cross_over_method == 'random':
                child1 = cross_over_random(selected_parents)
                child2 = cross_over_random(selected_parents)
            child1 = clip_parameters(child1, var_ranges)
            child2 = clip_parameters(child2, var_ranges)
            for i, _ in enumerate(var_names):
                child1[i] = mutate(child1[i], var_ranges[i], mut_prob, mut_extent)
                child2[i] = mutate(child2[i], var_ranges[i], mut_prob, mut_extent)
            if tuple(child1) not in unique_combinations:
                next_gen[c] = child1
                unique_combinations.add(tuple(child1))
            elif tuple(child2) not in unique_combinations:
                next_gen[c] = child2
                unique_combinations.add(tuple(child2))
            else:
                # fallback: re-use a parent
                next_gen[c] = selected_parents[random.randint(0, parents-1)]
    next_gen_df = pd.DataFrame(next_gen, columns=var_names)
    next_gen_df['target'] = np.nan
    next_gen_df.to_csv(os.path.join(dest_folder, f'proposed_experiments_iter_{iteration + 1}.csv'), index=False, sep=';')
    best_target = sorted_data.iloc[0]['target']
    return best_target

def main_ga_loop(pop_size, var_names, var_ranges, parents, elite, mut_prob, mut_extent, cross_over_method, selection_method, num_experiments, dest_folder, results_file):
    all_best_scores = []
    convergence_times = []
    iteration_counts = []
    for experiment in range(num_experiments):
        start_time = time.time()
        iteration = 0
        best_score_history = []
        best_score_overall = float('-inf')
        while True:
            best_score = iterate_ga(iteration, pop_size, var_names, var_ranges, parents, elite, mut_prob, mut_extent, cross_over_method, selection_method, dest_folder, experiment)
            best_score_history.append(best_score)
            if best_score > best_score_overall:
                best_score_overall = best_score
            if len(best_score_history) > 5:
                recent_scores = best_score_history[-5:]
                improvement = (recent_scores[-1] - recent_scores[0]) / (abs(recent_scores[0]) + 1e-10)
                if improvement < 0.05:
                    break
            iteration += 1
            if iteration > 100:
                break
        end_time = time.time()
        convergence_time = end_time - start_time
        convergence_times.append(convergence_time)
        iteration_counts.append(iteration)
        all_best_scores.append(best_score_overall)
    mean_best_score = np.mean(all_best_scores)
    std_best_score = np.std(all_best_scores)
    mean_convergence_time = np.mean(convergence_times)
    std_convergence_time = np.std(convergence_times)
    mean_iterations = np.mean(iteration_counts)
    std_iterations = np.std(iteration_counts)
    results_df = pd.DataFrame({
        'pop_size': [pop_size],
        'parents': [parents],
        'elite': [elite],
        'mut_prob': [mut_prob],
        'mut_extent': [mut_extent],
        'cross_over_method': [cross_over_method],
        'selection_method': [selection_method],
        'mean_best_score': [mean_best_score],
        'std_best_score': [std_best_score],
        'mean_convergence_time': [mean_convergence_time],
        'std_convergence_time': [std_convergence_time],
        'mean_iterations': [mean_iterations],
        'std_iterations': [std_iterations]
    })
    if not os.path.exists(results_file):
        results_df.to_csv(results_file, index=False)
    else:
        results_df.to_csv(results_file, mode='a', header=False, index=False)

def test_multiple_configurations(dest_folder, results_folder):
    pop_size = 8
    var_names = ['pH', 'Salt Concentration', 'Cosubstrate Concentration', 'Organic Solvent Concentration', 'Temperature']
    var_ranges = [(2.5, 8), (0, 500), (0, 10), (0, 30), (20, 60)]
    results_file = os.path.join(results_folder, 'ga_parameter_optimization_results.csv')
    num_experiments = 30
    elites = [0, 1, 2]
    mut_probs = [0.1, 0.2, 0.4, 0.8]
    mut_extents = [0.1, 0.3, 0.6, 1]
    cross_over_methods = ['random', 'single_point', 'uniform']
    selection_methods = ['rank', 'roulette', 'tournament']
    for elite in elites:
        for mut_prob in mut_probs:
            for mut_extent in mut_extents:
                for cross_over_method in cross_over_methods:
                    for selection_method in selection_methods:
                        print(f"Testing: elite={elite}, mut_prob={mut_prob}, mut_extent={mut_extent}, cross_over_method={cross_over_method}, selection_method={selection_method}")
                        main_ga_loop(
                            pop_size, var_names, var_ranges, 4, elite, mut_prob, mut_extent,
                            cross_over_method, selection_method, num_experiments, dest_folder, results_file
                        )

if __name__ == "__main__":
    test_multiple_configurations(INIT_GEN_FOLDER, RESULTS_FOLDER)