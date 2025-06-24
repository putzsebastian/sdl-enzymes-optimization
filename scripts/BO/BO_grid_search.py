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
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm
from joblib import Parallel, delayed

warnings.filterwarnings('ignore', category=ConvergenceWarning)

# --- Paths (relative to script) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'Results_UPO-ABTS.xlsx')
INIT_GEN_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'initial_generations')
RESULTS_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'bo')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_FOLDER, 'bo_optimization_results.csv')

# --- Design Space ---
components = {
    'pH': np.round(np.arange(2.5, 8.1, 0.1), 1),
    'Salt Concentration': np.round(np.arange(0, 525, 25), 1),
    'Cosubstrate Concentration': np.round(np.arange(0.0, 10.2, 0.2), 1),
    'Organic Solvent Concentration': np.round(np.arange(0.0, 32.5, 2.5), 1),
    'Temperature': np.round(np.arange(20, 62, 2), 1)
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

def acquisition_function_ucb(model, domain, kappa=2.576):
    y_pred, sigma = model.predict(domain, return_std=True)
    return y_pred + kappa * sigma

def acquisition_function_pi(model, domain, xi=0.01):
    y_pred, sigma = model.predict(domain, return_std=True)
    y_max = np.max(model.y_train_)
    Z = (y_pred - y_max - xi) / sigma
    return norm.cdf(Z)

# --- Strategies ---
class ThompsonSampling:
    def __init__(self, batch_size, duplicates=False, chunk_size=50000):
        self.batch_size = batch_size
        self.duplicates = duplicates
        self.chunk_size = chunk_size

    def run(self, model, obj):
        domain = obj['domain']
        selected_points = []
        all_selected_indices = []
        for start_idx in range(0, len(domain), self.chunk_size):
            domain_chunk = domain[start_idx:start_idx + self.chunk_size]
            y_sampled = model.sample_y(domain_chunk, random_state=np.random.randint(1e6))
            max_idx = np.argmax(y_sampled)
            while not self.duplicates and (start_idx + max_idx) in all_selected_indices:
                y_sampled[0, max_idx] = -np.inf
                max_idx = np.argmax(y_sampled)
            selected_points.append(domain_chunk[max_idx])
            all_selected_indices.append(start_idx + max_idx)
        return np.array(selected_points)

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
            model.fit(np.vstack([model.X_train_, next_point.reshape(1, -1)]),
                      np.append(model.y_train_, model.predict([next_point])))
        return np.array(selected_points)

class EpsilonGreedy:
    def __init__(self, epsilon=0.1, batch_size=1, duplicates=False):
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.duplicates = duplicates

    def run(self, model, obj):
        domain = obj['domain']
        selected_points = []
        all_selected_indices = []
        evaluated_points_set = set(map(tuple, obj['results'][var_names].values))
        for _ in range(self.batch_size):
            if random.random() < self.epsilon:
                next_idx = random.choice(range(len(domain)))
            else:
                y_pred = model.predict(domain)
                next_idx = np.argmax(y_pred)
            while not self.duplicates and (tuple(domain[next_idx]) in evaluated_points_set or next_idx in all_selected_indices):
                if random.random() < self.epsilon:
                    next_idx = random.choice(range(len(domain)))
                else:
                    y_pred[next_idx] = -np.inf
                    next_idx = np.argmax(y_pred)
            next_point = domain[next_idx]
            selected_points.append(next_point)
            all_selected_indices.append(next_idx)
            evaluated_points_set.add(tuple(next_point))
            if not self.duplicates:
                domain = np.delete(domain, next_idx, axis=0)
        return np.array(selected_points)

# --- Main BO Experiment Loop ---
def bayesian_optimization_loop(points, values, acq_func_name, strategy, kernel_name, gen_idx, iterations=100, batch_size=8, epsilon=0.1):
    max_values = []
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
        raise ValueError(f"Unknown kernel: {kernel_name}")
    # Acquisition function
    acq_funcs = {
        'EI': acquisition_function_ei,
        'PI': acquisition_function_pi,
        'UCB': acquisition_function_ucb
    }
    acq_func = acq_funcs.get(acq_func_name, None)
    # Load initial generation
    initial_file = os.path.join(INIT_GEN_FOLDER, f'initial_generation_{gen_idx}.csv')
    data = pd.read_csv(initial_file, delimiter=';')
    if 'target' not in data.columns:
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
        if strategy == 'thompson':
            optimizer = ThompsonSampling(batch_size=batch_size)
        elif strategy == 'kriging':
            optimizer = KrigingBeliever(acq_func=acq_func, batch_size=batch_size)
        elif strategy == 'epsilon_greedy':
            optimizer = EpsilonGreedy(epsilon=epsilon, batch_size=batch_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        selected_points_scaled = optimizer.run(gp, obj)
        selected_points = scaler_X.inverse_transform(selected_points_scaled)
        top_df = pd.DataFrame(selected_points, columns=var_names)
        new_targets = evaluate_population(selected_points, points, values)
        new_data = top_df.copy()
        new_data['target'] = new_targets
        data = pd.concat([data, new_data], ignore_index=True)
        max_value_current_iter = np.max(data['target'])
        max_values.append(max_value_current_iter)
    best_score = np.max(max_values)
    return data, best_score, max_values

# --- Main Run ---
if __name__ == "__main__":
    # Load data for interpolation
    data_interpolation = load_data(DATA_PATH)
    points = data_interpolation[var_names].values
    values = data_interpolation['Mean Specific Rate [U/mg]'].values

    kernels = ['Matern52', 'Matern32', 'Matern12', 'RBF', 'RQ']
    strategies = [
        {'strategy': 'kriging', 'acq_func_name': 'EI'},
        {'strategy': 'kriging', 'acq_func_name': 'PI'},
        {'strategy': 'kriging', 'acq_func_name': 'UCB'},
        {'strategy': 'thompson', 'acq_func_name': ''},
        {'strategy': 'epsilon_greedy', 'acq_func_name': ''}
    ]
    epsilon_values = [0.05, 0.1, 0.2, 0.3]
    num_initial_gens = 30

    # Prepare results summary
    if not os.path.exists(RESULTS_FILE):
        results_df = pd.DataFrame(columns=[
            'kernel', 'strategy', 'acq_func', 'epsilon',
            'mean_best_score', 'std_best_score'
        ])
        results_df.to_csv(RESULTS_FILE, index=False)

    for kernel_name in kernels:
        for strat in strategies:
            strategy_name = strat['strategy']
            acq_func_name = strat['acq_func_name']
            for epsilon in (epsilon_values if strategy_name == 'epsilon_greedy' else [None]):
                best_scores = []
                all_max_scores = []
                for gen_idx in range(num_initial_gens):
                    final_data, best_score, max_scores = bayesian_optimization_loop(
                        points=points,
                        values=values,
                        acq_func_name=acq_func_name,
                        strategy=strategy_name,
                        kernel_name=kernel_name,
                        gen_idx=gen_idx,
                        iterations=100,
                        batch_size=8,
                        epsilon=epsilon if epsilon else 0.1
                    )
                    # Save individual run
                    run_file = os.path.join(
                        RESULTS_FOLDER,
                        f'final_data_{strategy_name}_{acq_func_name}_{kernel_name}'
                        f'{f"_epsilon_{epsilon}" if epsilon else ""}_gen_{gen_idx}.csv'
                    )
                    final_data.to_csv(run_file, index=False)
                    best_scores.append(best_score)
                    all_max_scores.append(max_scores)
                    print(f"Run saved: {run_file}")

                # Save summary for this config
                mean_best = np.mean(best_scores)
                std_best = np.std(best_scores)
                result_row = {
                    'kernel': kernel_name,
                    'strategy': strategy_name,
                    'acq_func': acq_func_name,
                    'epsilon': epsilon,
                    'mean_best_score': mean_best,
                    'std_best_score': std_best
                }
                results_df = pd.DataFrame([result_row])
                results_df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
                print(f"Summary saved for {kernel_name} {strategy_name} {acq_func_name} {epsilon}")