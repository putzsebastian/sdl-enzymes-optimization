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
RESULTS_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'pso')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_FOLDER, 'pso_iterations_results.csv')
SUMMARY_FILE = os.path.join(RESULTS_FOLDER, 'pso_summary_results.csv')

# --- PSO SETTINGS ---
num_particles = 8
num_experiments = 30
max_iterations = 50

w_initial = 0.9
decay_factor = 0.95
c1 = 2.5
c2 = 1.0

var_names = ['pH', 'Salt Concentration', 'Cosubstrate Concentration', 'Organic Solvent Concentration', 'Temperature']
bounds = [(2.5, 8), (0, 500), (0, 10), (0, 30), (20, 60)]
precisions = [1, 0, 1, 1, 0]

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

# --- PSO STATE ---
def initialize_pso(dest_folder, gen_idx):
    parameters = dict(zip(var_names, zip(bounds, precisions)))
    # Load initial positions from pre-generated LHS samples
    init_file = os.path.join(INIT_GEN_FOLDER, f'initial_generation_{gen_idx}.csv')
    initial_df = pd.read_csv(init_file, delimiter=';')
    positions = initial_df[var_names].values
    velocities = np.random.uniform(-1, 1, (num_particles, len(var_names)))
    # Save PSO state
    np.save(os.path.join(dest_folder, 'positions.npy'), positions)
    np.save(os.path.join(dest_folder, 'velocities.npy'), velocities)
    np.save(os.path.join(dest_folder, 'personal_best_positions.npy'), positions.copy())
    np.save(os.path.join(dest_folder, 'personal_best_scores.npy'), np.full(num_particles, float('-inf')))
    np.save(os.path.join(dest_folder, 'global_best_position.npy'), positions[0].copy())
    np.save(os.path.join(dest_folder, 'global_best_score.npy'), float('-inf'))

def iterate_pso(w, c1, c2, dest_folder, points, values):
    positions = np.load(os.path.join(dest_folder, 'positions.npy'))
    velocities = np.load(os.path.join(dest_folder, 'velocities.npy'))
    personal_best_positions = np.load(os.path.join(dest_folder, 'personal_best_positions.npy'))
    personal_best_scores = np.load(os.path.join(dest_folder, 'personal_best_scores.npy'))
    global_best_position = np.load(os.path.join(dest_folder, 'global_best_position.npy'))
    global_best_score = np.load(os.path.join(dest_folder, 'global_best_score.npy'))
    for i in range(num_particles):
        r1 = np.random.rand(len(var_names))
        r2 = np.random.rand(len(var_names))
        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (personal_best_positions[i] - positions[i])
            + c2 * r2 * (global_best_position - positions[i])
        )
        positions[i] += velocities[i]
        # Apply bounds and rounding
        for j in range(len(var_names)):
            positions[i, j] = np.clip(positions[i, j], bounds[j][0], bounds[j][1])
            positions[i, j] = np.round(positions[i, j], precisions[j])
    new_scores = evaluate_population(positions, points, values)
    for i in range(num_particles):
        current_score = new_scores[i]
        if current_score > personal_best_scores[i]:
            personal_best_scores[i] = current_score
            personal_best_positions[i] = positions[i].copy()
    max_personal_best_score = np.max(personal_best_scores)
    best_index = np.argmax(personal_best_scores)
    if max_personal_best_score > global_best_score:
        global_best_score = max_personal_best_score
        global_best_position = personal_best_positions[best_index].copy()
    np.save(os.path.join(dest_folder, 'positions.npy'), positions)
    np.save(os.path.join(dest_folder, 'velocities.npy'), velocities)
    np.save(os.path.join(dest_folder, 'personal_best_positions.npy'), personal_best_positions)
    np.save(os.path.join(dest_folder, 'personal_best_scores.npy'), personal_best_scores)
    np.save(os.path.join(dest_folder, 'global_best_position.npy'), global_best_position)
    np.save(os.path.join(dest_folder, 'global_best_score.npy'), global_best_score)
    return global_best_score

# --- MAIN PSO LOOP (fixed iterations) ---
def main_pso_loop(num_particles, w_initial, decay_factor, c1, c2, num_experiments, max_iterations, results_file, summary_file, results_folder):
    data_interpolation = load_data(DATA_PATH)
    points = data_interpolation[var_names].values
    values = data_interpolation['Mean Specific Rate [1/s]'].values
    all_best_scores = []
    for gen_idx in range(num_experiments):
        run_folder = os.path.join(results_folder, f'run_gen_{gen_idx}')
        os.makedirs(run_folder, exist_ok=True)
        initialize_pso(run_folder, gen_idx)
        w = w_initial
        best_score_history = []
        for iteration in range(max_iterations):
            best_score = iterate_pso(w, c1, c2, run_folder, points, values)
            best_score_history.append(best_score)
            w *= decay_factor
        all_best_scores.append(best_score_history)
        # Save per-experiment results
        pd.DataFrame({
            'experiment': gen_idx+1,
            'iteration': range(1, max_iterations+1),
            'best_score': best_score_history
        }).to_csv(results_file, mode='a', header=(gen_idx==0), index=False)
        print(f"Experiment {gen_idx+1}/{num_experiments} completed.")
    # Summary stats (mean/std per iteration)
    all_best_scores = np.array(all_best_scores)
    mean_best_scores = np.mean(all_best_scores, axis=0)
    std_best_scores = np.std(all_best_scores, axis=0)
    summary_df = pd.DataFrame({
        'iteration': range(1, max_iterations+1),
        'mean_best_score': mean_best_scores,
        'std_best_score': std_best_scores
    })
    summary_df.to_csv(summary_file, index=False)
    print("Summary of mean and standard deviation for each iteration saved.")

# --- RUN ---
main_pso_loop(num_particles, w_initial, decay_factor, c1, c2, num_experiments, max_iterations, RESULTS_FILE, SUMMARY_FILE, RESULTS_FOLDER)
print(f"PSO fixed-iterations experiments complete. Results in:\n- {RESULTS_FILE}\n- {SUMMARY_FILE}")
