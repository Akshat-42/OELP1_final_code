import numpy as np
import pandas as pd
import time
from keras.models import load_model
from joblib import load
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from forward_sim import air_sim, GRAVITY_VEC, BALL_MASS

# --- Configuration ---
MODEL_PATH = "cricket_trajectory_model.keras"
PREPROCESSOR_PATH = "scaler_X.joblib"
SCALER_Y_PATH = "scaler_y.joblib"
SAMPLE_SIZE = 4000
RESULTS_FILE = "model_test_results.txt"

V_BINS = [0, 22, 26, np.inf]
V_LABELS = ['Low', 'Medium', 'High']

W_MAG_BINS = [0, 150, 220, np.inf]
W_MAG_LABELS = ['Low', 'Medium', 'High']

W_ANGLE_BINS = [-np.inf, -10, 10, np.inf]
W_ANGLE_LABELS = ['Negative_Spin', 'Neutral_Spin', 'Positive_Spin']

def generate_random_test_data(n_samples):
    return pd.DataFrame({
        "p_x": np.random.uniform(-0.5, 0.5, n_samples),
        "p_y": np.random.uniform(-1.2, 1.2, n_samples),
        "p_z": np.random.uniform(2.0, 2.0, n_samples),
        "v_mag": np.random.uniform(18, 30, n_samples),
        "phi": np.random.uniform(0, 5, n_samples),
        "theta": np.random.uniform(-5, 5, n_samples),
        "w_mag": np.random.uniform(100, 260, n_samples),
        "w_angle": np.random.uniform(-45, 45, n_samples)
    })

def categorize_features(df):
    df['v_cat'] = pd.cut(df['v_mag'], bins=V_BINS, labels=V_LABELS, right=False)
    df['w_mag_cat'] = pd.cut(df['w_mag'], bins=W_MAG_BINS, labels=W_MAG_LABELS, right=False)
    # New Angle Cat
    df['w_angle_cat'] = pd.cut(df['w_angle'], bins=W_ANGLE_BINS, labels=W_ANGLE_LABELS)
    return df

def sim_worker(args):
    p_x, p_y, p_z, v_mag, phi, theta, w_mag, w_angle = args
    phi_rad = np.radians(phi)
    theta_rad = np.radians(theta)
    w_angle_rad = np.radians(w_angle)
    
    pos_in = np.array([p_x, p_y, p_z])
    v_z = v_mag * np.sin(phi_rad)
    v_xy = v_mag * np.cos(phi_rad)
    v_in = np.array([v_xy * np.cos(theta_rad), v_xy * np.sin(theta_rad), v_z])
    
    w_x = w_mag * np.sin(w_angle_rad)
    w_y = w_mag * np.cos(w_angle_rad)
    w_in = np.array([w_x, w_y, 0])
    
    return air_sim(pos_in, v_in, 60, GRAVITY_VEC, BALL_MASS, w_in)

def run_simulation_batch(df_inputs):
    cols = ['p_x', 'p_y', 'p_z', 'v_mag', 'phi', 'theta', 'w_mag', 'w_angle']
    tasks = df_inputs[cols].to_numpy()
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(sim_worker, tasks), total=len(tasks), desc="Simulating"))
    return pd.DataFrame(results, columns=['land_x', 'land_y'])

def main():
    start_time = time.time()
    try:
        model = load_model(MODEL_PATH)
        preprocessor = load(PREPROCESSOR_PATH)
        scaler_y = load(SCALER_Y_PATH)
    except Exception as e:
        print(f"Error loading model: {e}\nPlease run train_model.py first.")
        return

    print("Generating Test Data...")
    ground_truth = generate_random_test_data(SAMPLE_SIZE)
    ground_truth = categorize_features(ground_truth)
    
    print("Step 1: Simulating Ground Truth Landing Points...")
    landing_points = run_simulation_batch(ground_truth)
    
    model_input = pd.concat([
        landing_points, 
        ground_truth[['p_x', 'p_y', 'p_z']], 
        ground_truth[['v_cat', 'w_mag_cat', 'w_angle_cat']]
    ], axis=1)
    
    print("Step 2: AI Predicting Initial Conditions...")
    X_scaled = preprocessor.transform(model_input)
    y_pred = scaler_y.inverse_transform(model.predict(X_scaled))
    
    pred_df = pd.DataFrame(y_pred, columns=['v_mag', 'phi', 'theta', 'w_mag', 'w_angle'])
    
    pred_df['p_x'] = ground_truth['p_x'].values
    pred_df['p_y'] = ground_truth['p_y'].values
    pred_df['p_z'] = ground_truth['p_z'].values
    
    print("Step 3: Validating Predictions...")
    val_landing = run_simulation_batch(pred_df)
    
    error_x = val_landing['land_x'] - landing_points['land_x']
    error_y = val_landing['land_y'] - landing_points['land_y']
    dist_error = np.sqrt(error_x**2 + error_y**2)

    stats = f"""
==================================================
TEST RESULTS ({SAMPLE_SIZE} samples)
==================================================
Mean Absolute Error X : {np.mean(np.abs(error_x)):.4f} m
Mean Absolute Error Y : {np.mean(np.abs(error_y)):.4f} m
Mean Distance Error   : {np.mean(dist_error):.4f} m
==================================================
    """
    print(stats)
    with open(RESULTS_FILE, "w") as f:
        f.write(stats)

    print(f"Completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()