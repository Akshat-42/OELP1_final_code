import numpy as np
import csv
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- Constants ---
BALL_MASS = 0.160  # kg
BALL_RADIUS = 0.035  # m
GRAVITY_VEC = np.array([0, 0, -9.81]) * BALL_MASS  # N
COR = 0.5  # Coefficient of restitution
CF = 0.35  # Friction coefficient
DCF = 0.45  # Drag coefficient
RHO = 1.293  # Air density
BALL_AREA = np.pi * BALL_RADIUS**2

# --- Physics Functions ---

def bounce_calc(v, spin_vect):
    """Calculates velocity vector change after bounce."""
    vz_prev = v[2]
    v[2] = -v[2] * COR

    v_surf = np.cross([0, 0, BALL_RADIUS], spin_vect)

    if np.linalg.norm(v_surf) > 0:
        max_friction_impulse_mag = CF * BALL_MASS * (1 + COR) * vz_prev
        impulse_stop_slip = np.linalg.norm(v_surf) * BALL_MASS
        friction_impulse_mag = min(max_friction_impulse_mag, impulse_stop_slip)
        impulse_unit_vector = -v_surf / np.linalg.norm(v_surf)
        friction_impulse = impulse_unit_vector * friction_impulse_mag
        v += friction_impulse / BALL_MASS

    return v

def air_sim(pos_in, v_in, sps, gf, mb, w_in, return_trajectory=False, simulate_bounce=False):
    """Simulates the air trajectory of the ball."""
    curr_pos = np.array(pos_in, dtype=float)
    curr_vel = np.array(v_in, dtype=float)
    spin_vect = np.array(w_in, dtype=float)
    
    trajectory = []
    sample = 0
    t_step = 1.0 / sps
    spin_mag = np.linalg.norm(spin_vect)

    if return_trajectory:
        trajectory.append({
            "sample": sample,
            "time": 0.0,
            "position": curr_pos.tolist(),
            "velocity": curr_vel.tolist()
        })

    # Phase 1: Flight
    while curr_pos[2] > BALL_RADIUS:
        v_norm = np.linalg.norm(curr_vel)
        drag_f = -0.5 * RHO * BALL_AREA * DCF * curr_vel * v_norm
        drag_acc = drag_f / mb

        if spin_mag > 0 and v_norm > 0:
            spin_ratio = (spin_mag * BALL_RADIUS) / v_norm
            Cl = 0.54 * (spin_ratio ** 0.4)
            lift_f = Cl * 0.5 * RHO * BALL_AREA * np.cross(spin_vect, curr_vel) * v_norm / spin_mag
            lift_acc = lift_f / mb
        else:
            lift_acc = np.zeros(3)

        curr_acc = drag_acc + (gf / mb) + lift_acc
        curr_vel += curr_acc * t_step
        curr_pos += curr_vel * t_step
        sample += 1
        
        if return_trajectory:
            trajectory.append({
                "sample": sample,
                "time": sample * t_step,
                "position": curr_pos.tolist(),
                "velocity": curr_vel.tolist()
            })

    impact_point = curr_pos.tolist()

    # Phase 2: Bounce (Optional)
    if simulate_bounce:
        curr_vel = bounce_calc(curr_vel, spin_vect)
        while curr_pos[0] < 20.12 and curr_pos[0] > -5:
            v_norm = np.linalg.norm(curr_vel)
            drag_f = -0.5 * RHO * BALL_AREA * DCF * curr_vel * v_norm
            curr_acc = (drag_f + gf) / mb
            curr_vel += curr_acc * t_step
            curr_pos += curr_vel * t_step
            sample += 1
            
            if return_trajectory:
                trajectory.append({
                    "sample": sample,
                    "time": sample * t_step,
                    "position": curr_pos.tolist(),
                    "velocity": curr_vel.tolist()
                })
            if curr_pos[2] <= BALL_RADIUS: break

    return trajectory if return_trajectory else impact_point[0:2]

# --- Multiprocessing Helper ---

def worker_task(args):
    # Unpack args
    p_x, p_y, p_z, v_mag, phi, theta, w_mag, w_angle = args
    
    phi_rad = np.radians(phi)
    theta_rad = np.radians(theta)
    w_angle_rad = np.radians(w_angle)
    
    loc_in = np.array([p_x, p_y, p_z])
    
    # Velocity Vector
    v_z = v_mag * np.sin(phi_rad)
    v_xy = v_mag * np.cos(phi_rad)
    v_x = v_xy * np.cos(theta_rad)
    v_y = v_xy * np.sin(theta_rad)
    v_in = np.array([v_x, v_y, v_z])
    
    # Spin Vector
    w_x = w_mag * np.sin(w_angle_rad)
    w_y = w_mag * np.cos(w_angle_rad)
    spin_vect = np.array([w_x, w_y, 0])
    
    p_f = air_sim(loc_in, v_in, 60, GRAVITY_VEC, BALL_MASS, spin_vect, return_trajectory=False)
    
    return [p_x, p_y, p_z, v_mag, phi, theta, w_mag, w_angle, p_f]

def generate_random_inputs(n_samples=50000):
    """Generates random training data instead of grid search."""
    # Using float32 for consistency
    return np.column_stack([
        np.random.uniform(-0.5, 0.5, n_samples),   # p_x
        np.random.uniform(-1.2, 1.2, n_samples),   # p_y
        np.random.uniform(2.0, 2.0, n_samples),    # p_z (Fixed height usually)
        np.random.uniform(18, 30, n_samples),      # v_mag
        np.random.uniform(0, 5, n_samples),        # phi
        np.random.uniform(-5, 5, n_samples),       # theta
        np.random.uniform(100, 260, n_samples),    # w_mag
        np.random.uniform(-45, 45, n_samples)      # w_angle
    ])

if __name__ == '__main__':
    N_SAMPLES = 200000 # Large dataset for better learning
    print(f"Generating {N_SAMPLES} random training samples...")
    print(f"Using {cpu_count()} cores.")
    
    # Generate random inputs
    inputs = generate_random_inputs(N_SAMPLES)
    
    final_pts_file = r"Dataset/test_pts_new.csv"
    
    start_time = time.time()
    with open(final_pts_file, "w+", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["p_x", "p_y", "p_z", "v_mag", "phi", "theta", "w_mag", "w_angle", "p_f"])
        
        with Pool(cpu_count()) as pool:
            # Map the random inputs to the worker
            results = list(tqdm(pool.imap(worker_task, inputs, chunksize=100), total=N_SAMPLES))
            writer.writerows(results)

    print(f"----Completed in {time.time() - start_time:.2f} seconds----")