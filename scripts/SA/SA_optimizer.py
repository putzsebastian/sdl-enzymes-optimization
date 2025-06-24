import os
import numpy as np
import pandas as pd
import time
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

# --- PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'Results_UPO-ABTS.xlsx')
INIT_GEN_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'initial_generations')
RESULTS_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'sa')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_FOLDER, 'sa_optimization_results.csv')

# --- ALGORITHM PARAMETERS ---
var_names = ['pH', 'Salt Concentration', 'Cosubstrate Concentration', 'Organic Solvent Concentration', 'Temperature']
bounds = [(2.5, 8), (0, 500), (0, 10), (0, 30), (20, 60)]
precisions = [1, 0, 1, 1, 0]

# Sweep ranges:
initial_temperatures = [45000, 90000, 225000, 450000, 1800000]
cooling_rate_temperatures = [0.7, 0.8, 0.9, 0.95, 0.99]
cooling_rate_step_sizes = [0.7, 0.8, 0.9, 0.95, 0.99]
step_sizes = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
random_jump_probs = [0.0]
num_experiments = 30
max_iterations = 100

# --- DATA LOADING & OBJECTIVE ---
def load_data(file_path):
    return pd.read_excel(file_path)

def is_in_hull(points, point):
    hull = ConvexHull(points)
    delaunay = Delaunay(hull.points[hull.vertices])
    return delaunay.find_simplex(point) >= 0

def interpolate_or_extrapolate_mean_specific_rate(points, values, point):
    for i, (low, high) in enumerate(bounds):
        point[i] = np.clip(point[i], low, high)
    if is_in_hull(points, point):
        mean_specific_rate = griddata(points, values, point, method='linear')
        if np.isnan(mean_specific_rate):
            mean_specific_rate = griddata(points, values, point, method='nearest')
    else:
        model = LinearRegression()
        model.fit(points, values)
        mean_specific_rate = model.predict([point])[0]
    return mean_specific_rate * np.random.uniform(0.87, 1.13)

def objective_function(params, points, values):
    return interpolate_or_extrapolate_mean_specific_rate(points, values, params)

def evaluate_population(population, points, values):
    return np.array(Parallel(n_jobs=-1)(
        delayed(objective_function)(params, points, values) for params in population))

# --- SA STATE ---
def initialize_sa(run_folder, gen_idx, T):
    parameters = dict(zip(var_names, zip(bounds, precisions)))
    init_file = os.path.join(INIT_GEN_FOLDER, f'initial_generation_{gen_idx}.csv')
    initial_df = pd.read_csv(init_file, delimiter=';')
    initial_positions = initial_df[var_names].values
    np.save(os.path.join(run_folder, 'parameters.npy'), parameters)
    np.save(os.path.join(run_folder, 'current_positions.npy'), initial_positions)
    np.save(os.path.join(run_folder, 'current_scores.npy'), np.full(initial_positions.shape[0], float('-inf')))
    np.save(os.path.join(run_folder, 'best_positions.npy'), initial_positions)
    np.save(os.path.join(run_folder, 'best_scores.npy'), np.full(initial_positions.shape[0], float('-inf')))
    np.save(os.path.join(run_folder, 'temperature.npy'), np.array([T]))

def iterate_sa(T, cooling_rate_temperature, step_size, cooling_rate_step_size, run_folder, points, values, random_jump_prob=0.1):
    parameters = np.load(os.path.join(run_folder, 'parameters.npy'), allow_pickle=True).item()
    current_positions = np.load(os.path.join(run_folder, 'current_positions.npy'))
    current_scores = np.load(os.path.join(run_folder, 'current_scores.npy'))
    best_positions = np.load(os.path.join(run_folder, 'best_positions.npy'))
    best_scores = np.load(os.path.join(run_folder, 'best_scores.npy'))
    num_samples = current_positions.shape[0]
    new_positions = current_positions.copy()
    step_size *= cooling_rate_step_size
    for i in range(num_samples):
        for j, (param, (bnd, prec)) in enumerate(parameters.items()):
            step = (bnd[1] - bnd[0]) * step_size
            if np.random.rand() < random_jump_prob:
                new_positions[i, j] = np.round(np.random.uniform(bnd[0], bnd[1]), prec)
            else:
                new_positions[i, j] = np.round(current_positions[i, j] + np.random.uniform(-step, step), prec)
            new_positions[i, j] = np.clip(new_positions[i, j], bnd[0], bnd[1])
    new_scores = evaluate_population(new_positions, points, values)
    for i in range(num_samples):
        delta_score = new_scores[i] - current_scores[i]
        if delta_score > 0 or np.exp(delta_score / T) > np.random.rand():
            current_positions[i] = new_positions[i]
            current_scores[i] = new_scores[i]
        if new_scores[i] > best_scores[i]:
            best_positions[i] = new_positions[i]
            best_scores[i] = new_scores[i]
    T *= cooling_rate_temperature
    np.save(os.path.join(run_folder, 'current_positions.npy'), current_positions)
    np.save(os.path.join(run_folder, 'current_scores.npy'), current_scores)
    np.save(os.path.join(run_folder, 'best_positions.npy'), best_positions)
    np.save(os.path.join(run_folder, 'best_scores.npy'), best_scores)
    return best_scores, current_positions, T, step_size

# --- MAIN SA SWEEP LOOP ---
def main_sa_loop(initial_temperatures, cooling_rate_temperatures, cooling_rate_step_sizes, step_sizes, random_jump_probs, num_experiments, max_iterations, results_file, results_folder):
    data_interpolation = load_data(DATA_PATH)
    points = data_interpolation[var_names].values
    values = data_interpolation['Mean Specific Rate [U/mg]'].values
    results = []
    for T0 in initial_temperatures:
        for cr_temp in cooling_rate_temperatures:
            for cr_step in cooling_rate_step_sizes:
                for sjump in random_jump_probs:
                    for s0 in step_sizes:
                        all_best_scores = []
                        for gen_idx in range(num_experiments):
                            run_folder = os.path.join(results_folder, f'run_gen_{gen_idx}')
                            os.makedirs(run_folder, exist_ok=True)
                            T = T0
                            step_size = s0
                            initialize_sa(run_folder, gen_idx, T)
                            best_score_history = []
                            for iteration in range(max_iterations):
                                best_scores, _, T, step_size = iterate_sa(
                                    T, cr_temp, step_size, cr_step, run_folder, points, values, random_jump_prob=sjump)
                                best_score_history.append(np.max(best_scores))
                            all_best_scores.append(np.max(best_score_history))
                            # Per-experiment trajectory can be saved if desired!
                        results.append({
                            'initial_temperature': T0,
                            'cooling_rate_temperature': cr_temp,
                            'cooling_rate_step_size': cr_step,
                            'step_size': s0,
                            'random_jump_prob': sjump,
                            'mean_best_score': np.mean(all_best_scores),
                            'std_best_score': np.std(all_best_scores),
                            'num_experiments': num_experiments
                        })
                        print(f"Done: T0={T0}, crT={cr_temp}, crS={cr_step}, step={s0}, jump={sjump}")
    pd.DataFrame(results).to_csv(results_file, index=False)
    print(f"Full sweep done. Results saved to {results_file}")

# --- RUN ---
main_sa_loop(
    initial_temperatures,
    cooling_rate_temperatures,
    cooling_rate_step_sizes,
    step_sizes,
    random_jump_probs,
    num_experiments,
    max_iterations,
    RESULTS_FILE,
    RESULTS_FOLDER
)
print("Simulated Annealing parameter sweep complete.")
