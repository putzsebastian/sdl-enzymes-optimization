import numpy as np
import pandas as pd
from pyDOE2 import ccdesign
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
import os
import random

# -------------------- USER SETTINGS -------------------- #
total_runs = 30           # Number of independent RSM runs
rounds_per_run = 3        # Number of CCD rounds per run

# === Define all main paths relative to the script ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'Results_UPO-ABTS.xlsx')
RESULTS_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'rsm')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
output_excel = os.path.join(RESULTS_FOLDER, f"RSM_{total_runs}x{rounds_per_run}rounds.xlsx")

# -------------------- DATA & PARAMS -------------------- #
file_path = DATA_PATH

components = {
    'pH': [2.5, 8.0],
    'Salt Concentration': [0, 500],
    'Cosubstrate Concentration': [0.0, 10.0],
    'Organic Solvent Concentration': [0.0, 30.0],
    'Temperature': [20, 60]
}
param_names = list(components.keys())
initial_bounds = [tuple(components[k]) for k in param_names]
lower_bounds = np.array([b[0] for b in initial_bounds])
upper_bounds = np.array([b[1] for b in initial_bounds])

# -------------------- CORE FUNCTIONS -------------------- #
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
    return mean_specific_rate * random.uniform(0.87, 1.13)

def generate_ccd(center_point, lower_bounds, upper_bounds):
    ccd_design = ccdesign(len(components), center=(4, 4))
    scaled_ccd = lower_bounds + (ccd_design + 1) / 2 * (upper_bounds - lower_bounds)
    scaled_ccd = np.clip(scaled_ccd, lower_bounds, upper_bounds)
    scaled_ccd = np.vstack([scaled_ccd, center_point])
    return scaled_ccd

def adjust_bounds(optimal_params, lower_bounds, upper_bounds, initial_bounds):
    new_lower_bounds = np.zeros_like(optimal_params)
    new_upper_bounds = np.zeros_like(optimal_params)
    for i in range(len(optimal_params)):
        if np.isclose(optimal_params[i], lower_bounds[i]):
            new_lower_bounds[i] = max(initial_bounds[i][0], optimal_params[i] - (upper_bounds[i] - lower_bounds[i]) / 2)
            new_upper_bounds[i] = min(initial_bounds[i][1], optimal_params[i] + (upper_bounds[i] - lower_bounds[i]) / 2)
        elif np.isclose(optimal_params[i], upper_bounds[i]):
            new_lower_bounds[i] = max(initial_bounds[i][0], optimal_params[i] - (upper_bounds[i] - lower_bounds[i]) / 2)
            new_upper_bounds[i] = min(initial_bounds[i][1], optimal_params[i] + (upper_bounds[i] - lower_bounds[i]) / 2)
        else:
            new_range = (upper_bounds[i] - lower_bounds[i]) / 2
            new_lower_bounds[i] = max(initial_bounds[i][0], optimal_params[i] - new_range / 2)
            new_upper_bounds[i] = min(initial_bounds[i][1], optimal_params[i] + new_range / 2)
    return new_lower_bounds, new_upper_bounds

# -------------------- MAIN RSM LOOP -------------------- #
data = load_data(file_path)
points = data[param_names].values
values = data['Mean Specific Rate [U/mg]'].values

formula = (
    'response ~ (pH + Q("Salt Concentration") + Q("Cosubstrate Concentration") + '
    'Q("Organic Solvent Concentration") + Temperature)**2 + '
    'I(pH**2) + I(Q("Salt Concentration")**2) + I(Q("Cosubstrate Concentration")**2) + '
    'I(Q("Organic Solvent Concentration")**2) + I(Temperature**2)'
)

results_log = []

for run in range(1, total_runs + 1):
    print(f"\n--- Starting RSM Run {run} ---")
    center_point = np.mean([lower_bounds, upper_bounds], axis=0)
    current_bounds = (lower_bounds.copy(), upper_bounds.copy())
    combined_data = pd.DataFrame()

    for round_num in range(1, rounds_per_run + 1):
        scaled_ccd = generate_ccd(center_point, *current_bounds)
        ccd_df = pd.DataFrame(scaled_ccd, columns=param_names)
        ccd_df['response'] = [interpolate_or_extrapolate_mean_specific_rate(points, values, row)[0] for row in ccd_df.values]
        combined_data = pd.concat([combined_data, ccd_df], ignore_index=True)

        model = smf.ols(formula=formula, data=combined_data).fit()
        model_r2 = model.rsquared

        def objective_func(x):
            df = pd.DataFrame([x], columns=param_names)
            return -model.predict(df)[0]
        result = minimize(objective_func, center_point, bounds=list(zip(*current_bounds)), method='L-BFGS-B')
        optimal_params = result.x
        optimal_response = -result.fun
        optimum_actual = interpolate_or_extrapolate_mean_specific_rate(points, values, optimal_params)
        observed_max_response = max(ccd_df['response'].max(), optimum_actual[0])

        results_log.append({
            "Run": run,
            "Round": round_num,
            "Predicted Optimum Response": optimal_response,
            "Actual Optimum Response": optimum_actual[0],
            "Observed Maximum Response": observed_max_response,
            "R2": model_r2
        })

        current_bounds = adjust_bounds(optimal_params, *current_bounds, initial_bounds)
        center_point = np.clip(optimal_params, *current_bounds)

results_df = pd.DataFrame(results_log)
print("Summary of all runs:")
print(results_df)
results_df.to_excel(output_excel, index=False)
