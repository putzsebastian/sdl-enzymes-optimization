import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

# --- FILE PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'Results_UPO-ABTS.xlsx')
INIT_GEN_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'initial_generations')
RESULTS_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'sa')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_FOLDER, 'sa_iterations_results.csv')

# --- PARAMETERS ---
var_names = ['pH', 'Salt Concentration', 'Cosubstrate Concentration', 'Organic Solvent Concentration', 'Temperature']
bounds = [(2.5, 8), (0, 500), (0, 10), (0, 30), (20, 60)]
precisions = [1, 0, 1, 1, 0]

initial_temperature = 225000.0
cooling_rate = 0.7
step_size = 0.8
random_jump_prob = 0.0
num_experiments = 30
num_iterations = 50

# --- OBJECTIVE FUNCTION ---
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
        mean_specific_rate = model.predict([point])
    return mean_specific_rate[0] * np.random.uniform(0.87, 1.13)

def objective_function(params, points, values):
    return interpolate_or_extrapolate_mean_specific_rate(points, values, params)

def evaluate_population(population, points, values):
    return np.array(Parallel(n_jobs=-1)(
        delayed(objective_function)(params, points, values) for params in population))

# --- SA STATE INITIALIZATION ---
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

# --- ONE SA ITERATION ---
def iterate_sa(T, cooling_rate, step_size, run_folder, points, values, random_jump_prob=0.1):
    parameters = np.load(os.path.join(run_folder, 'parameters.npy'), allow_pickle=True).item()
    current_positions = np.load(os.path.join(run_folder, 'current_positions.npy'))
    current_scores = np.load(os.path.join(run_folder, 'current_scores.npy'))
    best_positions = np.load(os.path.join(run_folder, 'best_positions.npy'))
    best_scores = np.load(os.path.join(run_folder, 'best_scores.npy'))
    num_samples = current_positions.shape[0]
    new_positions = current_positions.copy()
    step_size *= cooling_rate  # Reduce the step size each iteration

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
        new_score = new_scores[i]
        delta_score = new_score - current_scores[i]
        if delta_score > 0 or np.exp(delta_score / T) > np.random.rand():
            current_positions[i] = new_positions[i]
            current_scores[i] = new_score
        if new_score > best_scores[i]:
            best_positions[i] = new_positions[i]
            best_scores[i] = new_score
    T *= cooling_rate
    np.save(os.path.join(run_folder, 'current_positions.npy'), current_positions)
    np.save(os.path.join(run_folder, 'current_scores.npy'), current_scores)
    np.save(os.path.join(run_folder, 'best_positions.npy'), best_positions)
    np.save(os.path.join(run_folder, 'best_scores.npy'), best_scores)
    return best_scores, current_positions, T, step_size

# --- MAIN SA FIXED ITERATION LOOP ---
def main_sa_loop(initial_temperature, cooling_rate, initial_step_size, random_jump_prob, num_experiments, num_iterations, results_folder, results_file):
    data_interpolation = load_data(DATA_PATH)
    points = data_interpolation[var_names].values
    values = data_interpolation['Mean Specific Rate [U/mg]'].values
    all_best_scores = []
    for gen_idx in range(num_experiments):
        run_folder = os.path.join(results_folder, f'run_gen_{gen_idx}')
        os.makedirs(run_folder, exist_ok=True)
        T = initial_temperature
        step_size = initial_step_size
        initialize_sa(run_folder, gen_idx, T)
        best_score_history = []
        for iteration in range(num_iterations):
            best_scores, _, T, step_size = iterate_sa(
                T, cooling_rate, step_size, run_folder, points, values, random_jump_prob=random_jump_prob)
            best_score_history.append(np.max(best_scores))
        all_best_scores.append(best_score_history)
        experiment_df = pd.DataFrame({
            'experiment': gen_idx + 1,
            'iteration': range(1, num_iterations + 1),
            'best_score': best_score_history
        })
        if not os.path.exists(results_file) or gen_idx == 0:
            experiment_df.to_csv(results_file, index=False)
        else:
            experiment_df.to_csv(results_file, mode='a', header=False, index=False)
        print(f"Experiment {gen_idx + 1}: Best overall score: {max(best_score_history)}")

    # Mean/std over all runs for plotting
    all_best_scores = np.array(all_best_scores)
    mean_best_scores = np.mean(all_best_scores, axis=0)
    std_best_scores = np.std(all_best_scores, axis=0)
    summary_df = pd.DataFrame({
        'iteration': range(1, num_iterations + 1),
        'mean_best_score': mean_best_scores,
        'std_best_score': std_best_scores
    })
    summary_file = results_file.replace('sa_iterations_results.csv', 'sa_summary_results.csv')
    summary_df.to_csv(summary_file, index=False)
    print("Summary of mean and std for each iteration saved.")

# --- RUN ---
main_sa_loop(
    initial_temperature,
    cooling_rate,
    step_size,
    random_jump_prob,
    num_experiments,
    num_iterations,
    RESULTS_FOLDER,
    RESULTS_FILE
)

print("Simulated Annealing process with fixed iterations completed and results saved.")
