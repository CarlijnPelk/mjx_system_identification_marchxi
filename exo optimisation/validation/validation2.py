#%%
import mujoco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. SETUP
# Load your model (supports .xml MJCF or .urdf)
model = mujoco.MjModel.from_xml_path('/media/carlijn/500GB/mjx_sysid-main/exo optimisation/exo.xml')
data = mujoco.MjData(model)

# Load your real measured data (Time, Torque_Knee, Torque_Hip, etc.)
# Assume data columns: ['time', 'torque_knee', 'real_angle_knee']
df = pd.read_csv('/media/carlijn/500GB/mjx_sysid-main/exo optimisation/data/2025-11-26-11-26-comp.csv') 

# Simulation parameters
dt = 0.001 #2000Hz Simulation timestep (must match your data frequency or be smaller)
model.opt.timestep = dt

# Joint names in order
JOINT_NAMES = [
    'left_hip_aa', 'left_hip_fe', 'left_knee',
    'L_Front_Linear', 'L_Back_Linear',  # Changed to match CSV columns
    'right_hip_aa', 'right_hip_fe', 'right_knee',
    'R_Front_Linear', 'R_Back_Linear'
]

# These are the qpos/actuator indices in the MuJoCo model
ACTUATED_QPOS_IDX = [0, 1, 2, 4, 7, 11, 12, 13, 15, 18]

# Number of actuators should match JOINT_NAMES
n_joints = len(JOINT_NAMES)
assert model.nu == n_joints, f"Model has {model.nu} actuators but {n_joints} joint names defined"

print(f"Model: {model.nu} actuators")
print(f"Actuator names from XML: {[model.actuator(i).name for i in range(model.nu)]}")
print(f"Simulating {len(df)} timesteps at {dt*1000:.1f}ms per step")
#%%

simulated_angles = []

# 2. SIMULATION LOOP
for i, row in df.iterrows():
    # A. Apply Real Torque
    for actuator_idx in range(n_joints):
        joint_name = JOINT_NAMES[actuator_idx]
        
        # Get torque from CSV 
        torque = row[f'{joint_name}_torque']
        
        # Apply to actuator
        data.ctrl[actuator_idx] = torque / 100.0  # Scale as needed

    # B. Step the physics
    mujoco.mj_step(model, data)
    
    # C. Record the result (Joint Position)
    # Extract positions for actuated joints only
    current_qpos = [data.qpos[qpos_idx] for qpos_idx in ACTUATED_QPOS_IDX]
    simulated_angles.append(current_qpos)
    
    # Print progress
    if i % 50000 == 0:
        print(f"Step {i}/{len(df)}")

# 3. VISUALIZATION / VALIDATION
simulated_angles = np.array(simulated_angles)

# Time vector
time = np.arange(len(simulated_angles)) * dt

# Plot each joint
fig, axes = plt.subplots(n_joints, 1, figsize=(12, 3*n_joints))
if n_joints == 1:
    axes = [axes]

for i, joint_name in enumerate(JOINT_NAMES):
    # Get real data from CSV
    real_pos = df[f'{joint_name}_pos'].values[:len(simulated_angles)]
    
    # Plot comparison
    axes[i].plot(time, real_pos, label='Real', color='blue', linewidth=2)
    axes[i].plot(time, simulated_angles[:, i], label='Simulated', 
                 color='orange', linestyle='--', linewidth=2, alpha=0.8)
    axes[i].set_title(f'Joint: {joint_name}')
    axes[i].set_ylabel('Position')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()
#%%

# 4. REAL-TIME VISUALIZATION WITH MUJOCO VIEWER
import mujoco.viewer

import time

# We use "launch_passive" so we can manually step the simulation in our own loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    print("Simulation started. Press Space in viewer to pause/play (if configured).")

        # 2. SIMULATION LOOP
    for i, row in df.iterrows():
        # A. Apply Real Torque
        for actuator_idx in range(n_joints):
            joint_name = JOINT_NAMES[actuator_idx]
            
            # Get torque from CSV 
            torque = row[f'{joint_name}_torque']
            
            # Apply to actuator
            data.ctrl[actuator_idx] = torque   # Scale as needed
        
        # --- B. PHYSICS STEP ---
        mujoco.mj_step(model, data)
        
        # --- C. VISUALIZATION ---
        # Update the 3D view
        viewer.sync()
        
        # --- D. TIMING ---
        # Sleep for 'dt' seconds so the visualizer plays at roughly real-time speed.
        # Without this, the simulation will finish instantly.
        time.sleep(dt/100)

        # Check if user closed the window
        if not viewer.is_running():
            break

print("Simulation finished.")
# %%
