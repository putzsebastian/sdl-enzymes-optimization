import os
import numpy as np
import pandas as pd
import random
import time
import warnings
from itertools import product
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'Results_UPO-ABTS.xlsx')
INIT_GEN_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'initial_generations')
RESULTS_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'bo')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_FOLDER, 'bo_fixed_iters_summary.csv')

# --- Design Space ---
components = {
    'pH': np.round(np.arange(2.5, 8.1, 0.2), 1),
    'Salt Concentration': np.round(np.arange(0, 550, 50), 1),
    'Cosubstrate Concentration': np.round(np.arange(0.0, 10.5, 0.5), 1),
    'Organic Solvent Concentration': np.round(np.arange(0.0, 33, 3), 1),
    'Temperature': np.round(np.arange(20, 64, 4), 1)
}
points_array = np.array([list(v) for v in product(*components.values())])
var_names = list(components.keys())

# --- Data Utilities ---
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

# --- Acquisition Functions ---
def acquisition_function_ei(model, domain):
    y_pred, sigma = model.predict(domain, return_std=True)
    y_max = np.max(model.y_train_)
    Z = (y_pred - y_max) / sigma
    return (y_pred - y_max) * norm.cdf(Z) + sigma * norm.pdf(Z)

def acquisition_function_pi(model, domain, xi=0.01):
    y_pred, sigma = model.predict(domain, return_std=True)
    y_max = np.max(model.y_train_)
    Z = (y_pred - y_max - xi) / sigma
    return norm.cdf(Z)

def acquisition_function_ucb(model, domain, kappa=2.576):
    y_pred, sigma = model.predict(domain, return_std=True)
    return y_pred + kappa * sigma

# --- Kriging Believer (Batch BO) ---
class KrigingBeliever:
    def __init__(self, acq_func, batch_size, duplicates=False):
        self.acq_func = acq_func
        self.batch_size = batch_size
        self.duplicates = duplicates

    def run(self, model, obj):
        domain = obj['domain']
        selected_points = []
        all_selected_indices = []

        for _ in range(self.batch_size):
            acq_values = self.acq_func(model, domain)
            max_idx = np.argmax(acq_values)
            while not self.duplicates and max_idx in all_selected_indices:
                acq_values[max_idx] = -np.inf
                max_idx = np.argmax(acq_values)
            next_point = domain[max_idx]
            selected_points.append(next_point)
            all_selected_indices.append(max_idx)
            if not self.duplicates:
                domain = np.delete(domain, max_idx, axis=0)
            # "Fantasy" step: Add this as if it was already observed, using predicted mean
            model.fit(np.vstack([model.X_train_, next_point.reshape(1, -1)]),
                      np.append(model.y_train_, model.predict([next_point])))
        return np.array(selected_points)

# --- Main BO Loop (Fixed Iterations) ---
def bayesian_optimization_loop(
        gen_idx, points, values, acq_func_name='EI', kernel_name='Matern32', 
        iterations=50, batch_size=8):
    max_values_per_generation = []
    iteration_numbers = []

    # Kernel selection
    if kernel_name == 'Matern52':
        kernel = Matern(length_scale=[1]*points_array.shape[1], nu=2.5)
    elif kernel_name == 'Matern32':
        kernel = Matern(length_scale=[1]*points_array.shape[1], nu=1.5)
    elif kernel_name == 'Matern12':
        kernel = Matern(length_scale=[1]*points_array.shape[1], nu=0.5)
    elif kernel_name == 'RBF':
        kernel = RBF(length_scale=[1]*points_array.shape[1])
    elif kernel_name == 'RQ':
        kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
    else:
        raise ValueError(f"Unknown kernel name: {kernel_name}")

    if acq_func_name == 'EI':
        acq_func = acquisition_function_ei
    elif acq_func_name == 'PI':
        acq_func = acquisition_function_pi
    elif acq_func_name == 'UCB':
        acq_func = acquisition_function_ucb
    else:
        raise ValueError(f"Unknown acquisition function: {acq_func_name}")

    # Load initial generation
    init_file = os.path.join(INIT_GEN_FOLDER, f'initial_generation_{gen_idx}.csv')
    data = pd.read_csv(init_file, delimiter=';')
    if 'target' not in data.columns or data['target'].isnull().all():
        initial_points = data[var_names].values
        data['target'] = evaluate_population(initial_points, points, values)

    for i in range(iterations):
        X = data[var_names].values
        y = data['target'].values

        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
        gp.fit(X_scaled, y_scaled)

        param_grid_scaled = scaler_X.transform(points_array)
        obj = {'domain': param_grid_scaled, 'results': data, 'target': 'target', 'X': X_scaled, 'y': y_scaled}

        optimizer = KrigingBeliever(acq_func=acq_func, batch_size=batch_size)
        selected_points_scaled = optimizer.run(gp, obj)
        selected_points = scaler_X.inverse_transform(selected_points_scaled)
        top_df = pd.DataFrame(selected_points, columns=var_names)
        new_targets = evaluate_population(selected_points, points, values)

        new_data = top_df.copy()
        new_data['target'] = new_targets

        # Avoid duplicates
        mask = ~data[var_names].apply(tuple, axis=1).isin(top_df.apply(tuple, axis=1))
        if mask.any():
            data = pd.concat([data, new_data], ignore_index=True)

        max_value_current_iter = np.max(data['target'])
        max_values_per_generation.append(max_value_current_iter)
        iteration_numbers.append(i + 1)

        print(f"Gen {gen_idx} Iter {i+1}: Best value so far: {max_value_current_iter:.4f}")

    return data, max_values_per_generation

# --- Main Execution ---
if __name__ == "__main__":
    data_interpolation = load_data(DATA_PATH)
    points = data_interpolation[var_names].values
    values = data_interpolation['Mean Specific Rate [u/mg]'].values

    kernels = ['Matern32', 'Matern52', 'RBF', 'RQ']
    acq_funcs = ['EI', 'PI', 'UCB']
    batch_size = 8
    iterations = 50
    num_gens = 30

    for kernel_name in kernels:
        for acq_func_name in acq_funcs:
            all_best_scores = []
            for gen_idx in range(num_gens):
                final_data, max_scores = bayesian_optimization_loop(
                    gen_idx, points, values,
                    acq_func_name=acq_func_name,
                    kernel_name=kernel_name,
                    iterations=iterations,
                    batch_size=batch_size,
                )
                filename = f'bo_fixed_{acq_func_name}_{kernel_name}_gen_{gen_idx}.csv'
                final_data.to_csv(os.path.join(RESULTS_FOLDER, filename), index=False)
                all_best_scores.append(max_scores[-1])
                print(f"Saved: {filename}")

            # Save summary
            mean_best = np.mean(all_best_scores)
            std_best = np.std(all_best_scores)
            summary = pd.DataFrame([{
                'kernel': kernel_name,
                'acq_func': acq_func_name,
                'batch_size': batch_size,
                'iterations': iterations,
                'mean_best_score': mean_best,
                'std_best_score': std_best,
            }])
            header = not os.path.exists(RESULTS_FILE)
            summary.to_csv(RESULTS_FILE, mode='a', header=header, index=False)
            print(f"\n=== {kernel_name} + {acq_func_name} | mean_best: {mean_best:.3f} ± {std_best:.3f} ===")
