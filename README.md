# Mujoco System Identification
A framework for system identification using the **MJX differentiable simulator**.

## Features
- Estimates built-in **MuJoCo parameters** (e.g., mass, inertia, control gains, and more).
- Captures **complex non-linear behaviors** using **neural networks**.

## Usage Workflow
1. **Prepare Your System Model**
   - Ensure you have an **XML model** of your system.

2. **Collect Real-World Data**
   - Perform real-world experiments:
     - Record **state sequences** to identify your system (e.g., joint positions, joint velocities, main body position, and velocity).
     - Log the **control actions** applied to your system.

3. **Run System Identification**
   - **Set up your dataset**.
   - **Select parameters** to optimize.
   - **Run the optimization process**.

4. **Validate Optimized Parameters**
   - See **[this example](examples/result_simulation.ipynb)** for validation.

---

## Installation

### Create a Virtual Environment
```
python3.11 -m venv venv
source venv/bin/activate
```
### Install Dependencies
```
pip install mujoco
pip install mujoco_mjx
pip install brax
pip install -U "jax[cuda12]"
pip install pandas
```
### Install the Framework

```
pip install -e .
```


## Usage Examples
- [Dataset Generation](examples/dataset_generation.ipynb)
    - Learn how to create a dataset in the required format.
- [Parameter Optimization](examples/optimization.ipynb)
    - Define custom model parameters.
    - Optimize parameters and validate the obtained results.
- [Result Validation](examples/result_simulation.ipynb)
    - Validate optimized parameters using a modified model.xml file.
    - Generate a video showcasing model performance.

---

## Exo Model (Quick Use)
This repo includes example scripts and notebooks to identify parameters for an exoskeleton model.

1. **Provide an exo model XML**
   - Use [exo optimisation/exo.xml](exo%20optimisation/exo.xml) or your own MuJoCo XML.
2. **Record data as an MCAP file**
   - You need a real experiment recording in `.mcap` format (ROS 2 bag).
3. **Convert MCAP to CSV**
   - Run [exo optimisation/convert_rosbag_to_csv.py](exo%20optimisation/convert_rosbag_to_csv.py) to produce a CSV dataset.
4. **Run optimization**
   - Open [exo optimisation/optimization.ipynb](exo%20optimisation/optimization.ipynb) or [exo optimisation/optimization_arm_damp.ipynb](exo%20optimisation/optimization_arm_damp.ipynb) and update the dataset path and model path.
5. **Validate results**
   - Not working yet
