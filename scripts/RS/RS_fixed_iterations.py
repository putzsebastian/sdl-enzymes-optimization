import numpy as np
import pandas as pd
import os
import time
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

# ----------- PATHS: Relative to repo structure ------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'Results_UPO-ABTS.xlsx')
INIT_GEN_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'initial_generations')
RESULTS_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'random_search')
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ------------ USER SETTINGS -------------
num_experiments = 30
num_samples = 8
num_iterations = 50
bounds = [(2.5, 8), (0, 500), (0, 10), (0, 30), (20, 60)]
precisions = [1, 0, 1, 0, 0]  # Decimal places for rounding

# ------------ DATA & UTILITIES -----------
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
    return mean_specific_rate * np.random.uniform(0.87, 1.13)

def load_initial_generation(init_gen_folder, gen_idx):
    file_path = os.path.join(init_gen_folder, f'initial_generation_{gen_idx}.csv')
    initial_df = pd.read_csv(file_path, delimiter=';')
    initial_positions = initial_df.iloc[:, :-1].values
    return initial_positions

def objective_function(params, points, values):
    return interpolate_or_extrapolate_mean_specific_rate(points, values, params)

def evaluate_population(population, points, values):
    return np.array(Parallel(n_jobs=-1)(delayed(objective_function)(params, points, values) for params in population))

def generate_random_samples(bounds, num_samples, tested_combinations, precisions):
    num_parameters = len(bounds)
    random_samples = np.zeros((num_samples, num_parameters))
    for i in range(num_samples):
        while True:
            sample = [np.random.uniform(low, high) for (low, high) in bounds]
            rounded_sample = [round(val, prec) for val, prec in zip(sample, precisions)]
            sample_tuple = tuple(rounded_sample)
            if sample_tuple not in tested_combinations:
                tested_combinations.add(sample_tuple)
                random_samples[i] = rounded_sample
                break
    return random_samples, tested_combinations

def iterate_random_search(iteration, results_folder, num_samples, bounds, tested_combinations, precisions, points, values):
    best_positions = np.load(os.path.join(results_folder, 'best_positions.npy'))
    best_scores = np.load(os.path.join(results_folder, 'best_scores.npy'))
    random_samples, tested_combinations = generate_random_samples(bounds, num_samples, tested_combinations, precisions)
    current_scores = evaluate_population(random_samples, points, values)
    for i in range(num_samples):
        new_score = current_scores[i]
        if new_score > best_scores[i]:
            best_positions[i] = random_samples[i]
            best_scores[i] = new_score
    np.save(os.path.join(results_folder, 'best_positions.npy'), best_positions)
    np.save(os.path.join(results_folder, 'best_scores.npy'), best_scores)
    return best_scores, tested_combinations

def main_random_search_loop(num_experiments, init_gen_folder, results_folder, num_samples, num_iterations, bounds, precisions):
    data = load_data(DATA_PATH)
    points = data[['pH', 'Salt Concentration', 'Cosubstrate Concentration', 'Organic Solvent Concentration', 'Temperature']].values
    values = data['Mean Specific Rate [U/mg]'].values
    all_best_scores = []

    for exp_num in range(num_experiments):
        start_time = time.time()
        tested_combinations = set()
        initial_positions = load_initial_generation(init_gen_folder, exp_num)
        np.save(os.path.join(results_folder, 'best_positions.npy'), initial_positions)
        np.save(os.path.join(results_folder, 'best_scores.npy'), np.full(num_samples, float('-inf')))
        best_score_history = []
        for iteration in range(num_iterations):
            best_scores, tested_combinations = iterate_random_search(
                iteration, results_folder, num_samples, bounds, tested_combinations, precisions, points, values)
            best_score_history.append(np.max(best_scores))
        end_time = time.time()
        print(f"Experiment {exp_num + 1}: Best score: {max(best_score_history)}, Time taken: {end_time - start_time:.2f} seconds")
        all_best_scores.append(best_score_history)
        experiment_df = pd.DataFrame({
            'experiment': exp_num + 1,
            'iteration': range(1, num_iterations + 1),
            'best_score': best_score_history
        })
        results_file = os.path.join(results_folder, 'random_search_iterations_results.csv')
        if not os.path.exists(results_file):
            experiment_df.to_csv(results_file, index=False)
        else:
            experiment_df.to_csv(results_file, mode='a', header=False, index=False)

    all_best_scores = np.array(all_best_scores)
    mean_best_scores = np.mean(all_best_scores, axis=0)
    std_best_scores = np.std(all_best_scores, axis=0)
    summary_df = pd.DataFrame({
        'iteration': range(1, num_iterations + 1),
        'mean_best_score': mean_best_scores,
        'std_best_score': std_best_scores
    })
    summary_file = os.path.join(results_folder, 'random_search_summary_results.csv')
    summary_df.to_csv(summary_file, index=False)
    print("Summary of mean and standard deviation for each iteration saved.")

# Run it!
main_random_search_loop(num_experiments, INIT_GEN_FOLDER, RESULTS_FOLDER, num_samples, num_iterations, bounds, precisions)
print("Experiment completed and results saved to 'random_search_summary_results.csv'")
