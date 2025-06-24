import numpy as np
import pandas as pd
import os
import random
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

# === File paths (adjust to your repo structure as needed) ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'Results_UPO-ABTS.xlsx')
INIT_GEN_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'initial_generations')
DEST_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'ga')
RESULTS_FILE = os.path.join(DEST_FOLDER, "ga_iterations_results.csv")

os.makedirs(DEST_FOLDER, exist_ok=True)

# === Data Loading and Interpolation Utilities ===
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

def objective_function(params, points, values):
    return interpolate_or_extrapolate_mean_specific_rate(points, values, params)

def evaluate_population(population, points, values):
    return np.array(Parallel(n_jobs=-1)(delayed(objective_function)(params, points, values) for params in population))

def rank_selection(sorted_data, num_parents):
    return sorted_data.head(num_parents).drop(columns='target').values

def roulette_selection(data, num_parents):
    total_fitness = data['target'].sum()
    probabilities = data['target'] / total_fitness
    selected = data.sample(n=num_parents, weights=probabilities)
    return selected.drop(columns='target').values

def tournament_selection(data, num_parents, tournament_size=3):
    selected = []
    for _ in range(num_parents):
        tournament = data.sample(n=tournament_size)
        winner = tournament.loc[tournament['target'].idxmax()]
        selected.append(winner.drop(labels='target').values)
    return np.array(selected)

def cross_over_single_point(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
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

def iterate_ga(iteration, pop_size, var_names, var_ranges, parents, elite, mut_prob, mut_extent, cross_over_method, selection_method, init_gen_folder, dest_folder, gen_idx):
    if iteration == 0:
        # Read from initial_generations folder
        data = pd.read_csv(os.path.join(init_gen_folder, f'initial_generation_{gen_idx}.csv'), delimiter=';')
    else:
        # Read from results/ga
        data = pd.read_csv(os.path.join(dest_folder, f'proposed_experiments_iter_{gen_idx}_{iteration}.csv'), delimiter=';')

    if 'target' not in data.columns or data['target'].isnull().all():
        data['target'] = evaluate_population(data[var_names].values, points, values)
        data.to_csv(os.path.join(dest_folder, f'proposed_experiments_iter_{gen_idx}_{iteration}.csv'), index=False, sep=';')

    sorted_data = data.sort_values(by='target', ascending=False)
    if selection_method == 'rank':
        selected_parents = rank_selection(sorted_data, parents)
    elif selection_method == 'roulette':
        selected_parents = roulette_selection(data, parents)
    elif selection_method == 'tournament':
        selected_parents = tournament_selection(data, parents)
    else:
        raise ValueError("Unknown selection method.")

    next_gen = np.zeros((pop_size, len(var_names)))
    unique_combinations = set()

    for c in range(pop_size):
        if c < elite:
            next_gen[c] = selected_parents[c]
        else:
            parent1, parent2 = random.sample(list(selected_parents), 2)
            if cross_over_method == 'single_point':
                child1, child2 = cross_over_single_point(parent1, parent2)
            elif cross_over_method == 'uniform':
                child1, child2 = cross_over_uniform(parent1, parent2)
            elif cross_over_method == 'random':
                child1 = cross_over_random(selected_parents)
                child2 = cross_over_random(selected_parents)
            else:
                raise ValueError("Unknown crossover method.")

            child1 = clip_parameters(child1, var_ranges)
            child2 = clip_parameters(child2, var_ranges)

            for i in range(len(var_names)):
                child1[i] = mutate(child1[i], var_ranges[i], mut_prob, mut_extent)
                child2[i] = mutate(child2[i], var_ranges[i], mut_prob, mut_extent)

            # Only add unique combinations
            if tuple(child1) not in unique_combinations:
                next_gen[c] = child1
                unique_combinations.add(tuple(child1))
            elif tuple(child2) not in unique_combinations:
                next_gen[c] = child2
                unique_combinations.add(tuple(child2))

    next_gen_df = pd.DataFrame(next_gen, columns=var_names)
    next_gen_df['target'] = np.nan
    next_gen_df.to_csv(os.path.join(dest_folder, f'proposed_experiments_iter_{gen_idx}_{iteration + 1}.csv'), index=False, sep=';')

    best_target = sorted_data.iloc[0]['target']
    return best_target

def main_ga_loop(pop_size, var_names, var_ranges, parents, elite, mut_prob, mut_extent, cross_over_method, selection_method, num_experiments, init_gen_folder, dest_folder, results_file):
    all_best_scores = []

    for experiment in range(num_experiments):
        iteration = 0
        best_score_history = []
        while iteration < 50:
            best_score = iterate_ga(
                iteration, pop_size, var_names, var_ranges, parents, elite, mut_prob, mut_extent,
                cross_over_method, selection_method, init_gen_folder, dest_folder, experiment
            )
            best_score_history.append(best_score)
            iteration += 1

        all_best_scores.append(best_score_history)
        experiment_df = pd.DataFrame({
            'experiment': experiment + 1,
            'iteration': range(1, 51),
            'best_score': best_score_history
        })
        if not os.path.exists(results_file):
            experiment_df.to_csv(results_file, index=False)
        else:
            experiment_df.to_csv(results_file, mode='a', header=False, index=False)
        print(f"Experiment {experiment + 1} completed and results saved.")

    all_best_scores = np.array(all_best_scores)
    mean_best_scores = np.mean(all_best_scores, axis=0)
    std_best_scores = np.std(all_best_scores, axis=0)
    summary_df = pd.DataFrame({
        'iteration': range(1, 51),
        'mean_best_score': mean_best_scores,
        'std_best_score': std_best_scores
    })
    summary_file = results_file.replace('ga_iterations_results.csv', 'ga_summary_results.csv')
    summary_df.to_csv(summary_file, index=False)
    print("Summary of mean and standard deviation for each iteration saved.")

# === Data Load for Population Evaluation ===
data = load_data(DATA_PATH)
points = data[['pH', 'Salt Concentration', 'Cosubstrate Concentration', 'Organic Solvent Concentration', 'Temperature']].values
values = data['Mean Specific Rate [U/mg]'].values

# === GA Config ===
pop_size = 8
parents = 4
elite = 2
mut_prob = 0.4
mut_extent = 0.3
cross_over_method = 'random'  # ['random', 'single_point', 'uniform']
selection_method = 'rank'       # ['rank', 'roulette', 'tournament']
var_names = ['pH', 'Salt Concentration', 'Cosubstrate Concentration', 'Organic Solvent Concentration', 'Temperature']
var_ranges = [(2.5, 8), (0, 500), (0, 10), (0, 30), (20, 60)]
num_experiments = 30

main_ga_loop(pop_size, var_names, var_ranges, parents, elite, mut_prob, mut_extent,
             cross_over_method, selection_method, num_experiments, INIT_GEN_FOLDER, DEST_FOLDER, RESULTS_FILE)
