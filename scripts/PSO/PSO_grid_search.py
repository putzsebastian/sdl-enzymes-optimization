import os
import numpy as np
import pandas as pd
import time
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'Results_All.xlsx')
INIT_GEN_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'initial_generations')
RESULTS_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'pso')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_FOLDER, 'pso_gridsearch_summary.csv')

# --- Parameters ---
var_names = ['pH', 'Salt Concentration', 'Cosubstrate Concentration', 'Organic Solvent Concentration', 'Temperature']
bounds = [(2.5, 8), (0, 500), (0, 10), (0, 30), (20, 60)]
precisions = [1, 0, 1, 1, 0]

num_particles = 8
num_gens = 30   # Number of initial generations/experiments

# --- Data Loading ---
def load_data(file_path):
    return pd.read_excel(file_path)

def is_in_hull(points, point):
    hull = ConvexHull(points)
    delaunay = Delaunay(hull.points[hull.vertices])
    return delaunay.find_simplex(point) >= 0

def interpolate_or_extrapolate_mean_specific_rate(points, values, point):
    # Apply bounds
    for i, (low, high) in enumerate(bounds):
        if not (low <= point[i] <= high):
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
    return np.array(Parallel(n_jobs=-1)(delayed(objective_function)(params, points, values) for params in population))

# --- PSO Initialization ---
def initialize_pso(dest_folder, gen_idx):
    # Load initial positions from LHS
    init_file = os.path.join(INIT_GEN_FOLDER, f'initial_generation_{gen_idx}.csv')
    initial_df = pd.read_csv(init_file, delimiter=';')
    positions = initial_df[var_names].values
    velocities = np.random.uniform(-1, 1, (num_particles, len(var_names)))
    np.save(os.path.join(dest_folder, 'positions.npy'), positions)
    np.save(os.path.join(dest_folder, 'velocities.npy'), velocities)
    np.save(os.path.join(dest_folder, 'personal_best_positions.npy'), positions.copy())
    np.save(os.path.join(dest_folder, 'personal_best_scores.npy'), np.full(num_particles, float('-inf')))
    np.save(os.path.join(dest_folder, 'global_best_position.npy'), positions[0].copy())
    np.save(os.path.join(dest_folder, 'global_best_score.npy'), float('-inf'))

# --- PSO Iteration ---
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

    # Evaluate
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

    # Save state
    np.save(os.path.join(dest_folder, 'positions.npy'), positions)
    np.save(os.path.join(dest_folder, 'velocities.npy'), velocities)
    np.save(os.path.join(dest_folder, 'personal_best_positions.npy'), personal_best_positions)
    np.save(os.path.join(dest_folder, 'personal_best_scores.npy'), personal_best_scores)
    np.save(os.path.join(dest_folder, 'global_best_position.npy'), global_best_position)
    np.save(os.path.join(dest_folder, 'global_best_score.npy'), global_best_score)

    return global_best_score

# --- Main PSO Parameter Grid Search ---
def main_pso_loop(
        w_initial_values, decay_factor_values, c1_values, c2_values, 
        num_particles, num_gens, max_iterations, results_file, results_folder):
    data_interpolation = load_data(DATA_PATH)
    points = data_interpolation[var_names].values
    values = data_interpolation['Mean Specific Rate [U/mg]'].values

    for w_initial in w_initial_values:
        for decay_factor in decay_factor_values:
            for c1 in c1_values:
                for c2 in c2_values:
                    all_best_scores = []
                    for gen_idx in range(num_gens):
                        # Folder per run to avoid state conflicts
                        run_folder = os.path.join(results_folder, f'run_w{w_initial}_d{decay_factor}_c1{c1}_c2{c2}_gen{gen_idx}')
                        os.makedirs(run_folder, exist_ok=True)
                        initialize_pso(run_folder, gen_idx)
                        w = w_initial
                        best_score_history = []
                        for iteration in range(max_iterations):
                            best_score = iterate_pso(w, c1, c2, run_folder, points, values)
                            best_score_history.append(best_score)
                            w *= decay_factor
                        all_best_scores.append(best_score_history[-1])
                        # Save full history for this run
                        pd.DataFrame({'iteration': np.arange(max_iterations)+1, 'best_score': best_score_history}).to_csv(
                            os.path.join(run_folder, f'best_score_history.csv'), index=False)
                        print(f"Config: w={w_initial}, decay={decay_factor}, c1={c1}, c2={c2}, gen={gen_idx} | Best: {best_score_history[-1]:.4f}")

                    # Save summary row for this config
                    mean_best = np.mean(all_best_scores)
                    std_best = np.std(all_best_scores)
                    summary = pd.DataFrame([{
                        'w_initial': w_initial,
                        'decay_factor': decay_factor,
                        'c1': c1,
                        'c2': c2,
                        'particles': num_particles,
                        'gens': num_gens,
                        'max_iterations': max_iterations,
                        'mean_best_score': mean_best,
                        'std_best_score': std_best,
                    }])
                    header = not os.path.exists(results_file)
                    summary.to_csv(results_file, mode='a', header=header, index=False)
                    print(f"Summary saved for config: w={w_initial}, decay={decay_factor}, c1={c1}, c2={c2} | mean={mean_best:.3f} ± {std_best:.3f}")

# --- Grid/Values to Test ---
w_initial_values = [1.1, 0.9, 0.7, 0.5]
decay_factor_values = [0.95, 0.85, 0.75, 0.65]
c1_values = [1, 1.5, 2.0, 2.5]
c2_values = [1, 1.5, 2.0, 2.5]
max_iterations = 50
main_pso_loop(
    w_initial_values, decay_factor_values, c1_values, c2_values,
    num_particles, num_gens, max_iterations, RESULTS_FILE, RESULTS_FOLDER
)
print("PSO parameter grid search completed.")
