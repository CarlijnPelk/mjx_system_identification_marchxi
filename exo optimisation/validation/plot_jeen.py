import time
import mujoco
import mujoco.viewer
import csv
import numpy as np
import matplotlib.pyplot as plt

m = mujoco.MjModel.from_xml_path('/media/carlijn/500GB/mjx_sysid-main/exo optimisation/models/rotational_position.xml')
d = mujoco.MjData(m)

def sine_wave(t, frequency=0.5, amplitude=1.0):
    return amplitude * np.sin(np.pi * frequency * t)
f_name = "/media/carlijn/500GB/mjx_sysid-main/exo optimisation/data/0.5Hz_RotPositionfilter.csv"

with open(f_name, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    array = np.array(data, dtype=float)
    print(array[:,0])
t_act = array[:,0] - array[0,0]  # Use actual timestamps, normalize to start at 0
encoder_pos = array[:,1]
x = np.arange(encoder_pos.shape[0])
pos_setpoint = array[:,2]
time_step = 0.02
total_time = 30 # seconds
num_steps = int(total_time / time_step)
amplitude = np.max(pos_setpoint)
time_step = m.opt.timestep
total_time = 30 # seconds
num_steps = int(total_time / time_step)
t = np.linspace(0, total_time, num_steps)

def sine_wave(t, frequency=0.5, amplitude=1.0):
    return amplitude * np.sin(np.pi * frequency * t)

sine = sine_wave(t, frequency=0.5, amplitude=amplitude)
qpos = []
ctrl = []

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  counter = 0
  keyframe_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
  while counter < 1500:
    step_start = time.time()
    d.ctrl[0] = sine[counter] # reference position for the joint
    # d.ctrl[1] = (q_ref-d.qpos[0]) * kp #reference velocity is error of position multiplied by kp
    mujoco.mj_step(m, d)
    qpos.append(d.qpos[0])
    ctrl.append(d.ctrl[0])
    viewer.sync()
    counter += 1
t = np.linspace(0, total_time, counter)
plt.plot(t,qpos, label='Joint Position')
plt.plot(t,ctrl, label='Control Input')
plt.plot(t_act,encoder_pos, label='Encoder Position')
plt.xlabel('Time (s)')
plt.ylabel('Qpos')
plt.legend()
plt.show()