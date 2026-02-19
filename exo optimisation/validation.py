#%% ------------------------------------------------------------------------------------------------------------
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['MUJOCO_GL'] = 'egl'

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Load your training parameters
class PARAMS:
    EXPERIMENT_NAME = "exo_optimization_try_20251217_1700"  # UPDATE THIS to your experiment
    
    class DATASET:
        PATH = "data/exo_data2_07_04_copy.csv"
        DT = 0.02
        MAX_SAMPLES = 100000

    class SIM:
        PATH = "exoskeleton_m10_ankle_mpc.xml"
        INTEGRATOR = mujoco.mjtIntegrator.mjINT_EULER
        ITERATIONS = 5

JOINT_NAMES = [
    'left_hip_aa', 'left_hip_fe', 'left_knee', 
    'L_Front_Linear', 'L_Back_Linear',
    'right_hip_aa', 'right_hip_fe', 'right_knee',
    'R_Front_Linear', 'R_Back_Linear'
]

ACTUATED_QPOS_IDX = jnp.array([0, 1, 2, 4, 7, 11, 12, 13, 15, 18])
ACTUATED_QVEL_IDX = jnp.array([0, 1, 2, 4, 7, 11, 12, 13, 15, 18])
#%% ------------------------------------------------------------------------------------------------------------

# Load trained parameters
results_path = os.path.join("exo_experiments", PARAMS.EXPERIMENT_NAME, "checkpoints", "final_results.joblib")
results = joblib.load(results_path)

best_params = results['best_params']
loss_hist = results['loss_hist']

print("=== LOADED TRAINING RESULTS ===")
print(f"Best loss: {results['best_loss']:.6f}")
print(f"Epochs trained: {results['epochs_trained']}")
print(f"\nOptimized parameters:")
for param_name in ['armature', 'damping', 'frictionloss']:
    if param_name in best_params:
        print(f"\n{param_name.upper()}:")
        for i, joint in enumerate(JOINT_NAMES):
            print(f"  {joint}: {best_params[param_name][i]:.4f}")
#%% ------------------------------------------------------------------------------------------------------------
# Load test data (use data NOT seen during training)
def load_test_data(csv_path, downsample_factor=20, start_idx=0, num_samples=5000):
    """Load a subset of data for testing."""
    df = pd.read_csv(csv_path)
    
    if downsample_factor > 1:
        df = df.iloc[::downsample_factor].reset_index(drop=True)
    
    # Use data from a different time range than training
    df = df.iloc[start_idx:start_idx + num_samples].copy()
    
    n_samples = len(df)
    n_joints = len(JOINT_NAMES)
    
    qpos_actuated = np.zeros((n_samples, n_joints))
    qvel_actuated = np.zeros((n_samples, n_joints))
    qctrl_raw = np.zeros((n_samples, n_joints))
    
    for i, joint in enumerate(JOINT_NAMES):
        qpos_actuated[:, i] = df[f'{joint}_pos'].values
        qvel_actuated[:, i] = df[f'{joint}_vel'].values
        qctrl_raw[:, i] = (df[f'{joint}_torque'].values) / 100.0  # Scale control inputs
    
    # Convert linear actuator positions
    linear_actuator_indices = [3, 4, 8, 9]
    for idx in linear_actuator_indices:
        qpos_actuated[:, idx] = (qpos_actuated[:, idx] - 40.0) / 1000.0
        qvel_actuated[:, idx] = qvel_actuated[:, idx] / 1000.0
    
    return qpos_actuated, qvel_actuated, qctrl_raw, df

# Load test data (different from training set)
test_qpos, test_qvel, test_ctrl, test_df = load_test_data(
    PARAMS.DATASET.PATH, 
    downsample_factor=20,
    start_idx=0,  # Use beginning of dataset
    num_samples=5000
)

print(f"\nTest data loaded: {test_qpos.shape[0]} samples")
print(f"  qpos shape: {test_qpos.shape}")
print(f"  qvel shape: {test_qvel.shape}")
print(f"  ctrl shape: {test_ctrl.shape}")


#%% ------------------------------------------------------------------------------------------------------------
# Setup MuJoCo model with optimized and default parameters
mj_model_optimized = mujoco.MjModel.from_xml_path(PARAMS.SIM.PATH)
mj_model_optimized.opt.timestep = PARAMS.DATASET.DT
mj_model_optimized.opt.iterations = PARAMS.SIM.ITERATIONS
mj_model_optimized.opt.integrator = PARAMS.SIM.INTEGRATOR
mj_model_optimized.opt.disableflags = mj_model_optimized.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT

# Apply optimized parameters
for i in range(len(JOINT_NAMES)):
    if 'armature' in best_params:
        mj_model_optimized.dof_armature[ACTUATED_QVEL_IDX[i]] = float(best_params['armature'][i])
    if 'damping' in best_params:
        mj_model_optimized.dof_damping[ACTUATED_QVEL_IDX[i]] = float(best_params['damping'][i])
    if 'frictionloss' in best_params:
        mj_model_optimized.dof_frictionloss[ACTUATED_QVEL_IDX[i]] = float(best_params['frictionloss'][i])

# Create default model for comparison
mj_model_default = mujoco.MjModel.from_xml_path(PARAMS.SIM.PATH)
mj_model_default.opt.timestep = PARAMS.DATASET.DT
mj_model_default.opt.iterations = PARAMS.SIM.ITERATIONS
mj_model_default.opt.integrator = PARAMS.SIM.INTEGRATOR
mj_model_default.opt.disableflags = mj_model_default.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT

print("Models created with optimized and default parameters")
#%% ------------------------------------------------------------------------------------------------------------
# MUJOCO VISUALIZATION - Compare optimized model vs ground truth
import mujoco.viewer

def visualize_comparison(model_optimized, model_default, qpos0_actuated, qvel0_actuated, 
                        ctrl_sequence, ground_truth_qpos, n_steps=200):
    """
    Visualize optimized model, default model, and ground truth side by side.
    Press TAB to switch between models.
    Press SPACE to pause/resume.
    Press RIGHT ARROW to step forward when paused.
    """
    
    # Create data instances
    data_opt = mujoco.MjData(model_optimized)
    data_def = mujoco.MjData(model_default)
    
    # Initialize both
    for data in [data_opt, data_def]:
        data.qpos[:] = model_optimized.qpos0
        data.qvel[:] = 0.0
        for i in range(len(JOINT_NAMES)):
            data.qpos[ACTUATED_QPOS_IDX[i]] = qpos0_actuated[i]
            data.qvel[ACTUATED_QVEL_IDX[i]] = qvel0_actuated[i]
    
    # Launch viewer for optimized model
    print("\n=== VISUALIZATION INSTRUCTIONS ===")
    print("Now showing: OPTIMIZED model (blue wireframe)")
    print("Close window to see DEFAULT model")
    print("Controls:")
    print("  - LEFT MOUSE: Rotate view")
    print("  - RIGHT MOUSE: Move view")
    print("  - SCROLL: Zoom")
    print("  - SPACE: Pause/Resume")
    print("  - RIGHT ARROW: Step forward (when paused)")
    print("  - ESC: Close and continue")
    
    # Store trajectories
    opt_trajectory = []
    def_trajectory = []
    
    step = [0]  # Use list to allow modification in nested function
    
    with mujoco.viewer.launch_passive(model_optimized, data_opt) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90
        
        # Enable visualization of reference trajectory
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        
        while viewer.is_running() and step[0] < min(n_steps, len(ctrl_sequence)):
            # Apply control
            data_opt.ctrl[:] = ctrl_sequence[step[0]]
            data_def.ctrl[:] = ctrl_sequence[step[0]]
            
            # Step simulation
            mujoco.mj_step(model_optimized, data_opt)
            mujoco.mj_step(model_default, data_def)
            
            # Store trajectories
            opt_qpos = [data_opt.qpos[idx] for idx in ACTUATED_QPOS_IDX]
            def_qpos = [data_def.qpos[idx] for idx in ACTUATED_QPOS_IDX]
            opt_trajectory.append(opt_qpos)
            def_trajectory.append(def_qpos)
            
            # Update viewer
            viewer.sync()
            step[0] += 1
            
            # Control playback speed
            import time
            time.sleep(PARAMS.DATASET.DT)  # Real-time playback
    
    print(f"\nOptimized model visualization complete ({step[0]} steps)")
    
    # Reset and show default model
    step[0] = 0
    print("\nNow showing: DEFAULT model (red wireframe)")
    
    with mujoco.viewer.launch_passive(model_default, data_def) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90
        
        # Reset data
        data_def.qpos[:] = model_default.qpos0
        data_def.qvel[:] = 0.0
        for i in range(len(JOINT_NAMES)):
            data_def.qpos[ACTUATED_QPOS_IDX[i]] = qpos0_actuated[i]
            data_def.qvel[ACTUATED_QVEL_IDX[i]] = qvel0_actuated[i]
        
        while viewer.is_running() and step[0] < min(n_steps, len(ctrl_sequence)):
            data_def.ctrl[:] = ctrl_sequence[step[0]]
            mujoco.mj_step(model_default, data_def)
            viewer.sync()
            step[0] += 1
            
            import time
            time.sleep(PARAMS.DATASET.DT)
    
    print(f"\nDefault model visualization complete ({step[0]} steps)")
    
    return np.array(opt_trajectory), np.array(def_trajectory)
# %%

# Select a test trajectory to visualize
traj_idx = 0  # Change this to view different trajectories
start_idx = traj_idx * 1000

qpos0 = test_qpos[start_idx]
qvel0 = test_qvel[start_idx]
ctrl_seq = test_ctrl[start_idx:start_idx + 200]  # 4 seconds
ground_truth = test_qpos[start_idx:start_idx + 200]

print(f"\n=== STARTING VISUALIZATION ===")
print(f"Trajectory {traj_idx}, starting at sample {start_idx}")
print(f"Duration: {len(ctrl_seq) * PARAMS.DATASET.DT:.1f} seconds")

# Visualize comparison
opt_traj, def_traj = visualize_comparison(
    mj_model_optimized,
    mj_model_default,
    qpos0, qvel0,
    ctrl_seq,
    ground_truth,
    n_steps=200
)

# %%
