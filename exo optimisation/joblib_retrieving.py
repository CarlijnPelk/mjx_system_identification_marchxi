import joblib 
import os
import sys

# Add parent directory to path to import JOINT_NAMES if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Joint names for display
JOINT_NAMES = [
    'left_hip_aa', 'left_hip_fe', 'left_knee', 
    'L_Front_Slide', 'L_Back_Slide',
    'right_hip_aa', 'right_hip_fe', 'right_knee',
    'R_Front_Slide', 'R_Back_Slide'
]

# Joint type categorization
JOINT_CATEGORIES = {
    'hip_aa': [0, 5],      # left_hip_aa, right_hip_aa
    'hip_fe': [1, 6],      # left_hip_fe, right_hip_fe
    'knee': [2, 7],        # left_knee, right_knee
    'linear': [3, 4, 8, 9] # L_Front_Slide, L_Back_Slide, R_Front_Slide, R_Back_Slide
}

# Try multiple possible path configurations
script_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    # Path 1: Relative to script location
    os.path.join(script_dir, "exo_experiments", "exo_optimization_try_20251217_1700", "checkpoints", "final_results.joblib"),
]

# Find the correct path
path = None
for p in possible_paths:
    if os.path.exists(p):
        path = p
        print(f"✓ Found checkpoint at: {path}\n")
        break

if path is None:
    print("❌ ERROR: Could not find checkpoint file!")
    print(f"\nSearched in:")
    for p in possible_paths:
        print(f"  - {p}")
    
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    
    # List available experiments
    exp_dir = os.path.join(script_dir, "exo_experiments")
    if os.path.exists(exp_dir):
        print(f"\nAvailable experiments in {exp_dir}:")
        for exp in os.listdir(exp_dir):
            exp_path = os.path.join(exp_dir, exp)
            if os.path.isdir(exp_path):
                checkpoint_dir = os.path.join(exp_path, "checkpoints")
                if os.path.exists(checkpoint_dir):
                    files = os.listdir(checkpoint_dir)
                    print(f"  - {exp}/ ({len(files)} files)")
    sys.exit(1)

# Load checkpoint
try:
    loaded_checkpoint = joblib.load(path)
    print("=== CHECKPOINT LOADED SUCCESSFULLY ===\n")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    sys.exit(1)

# Inspect the checkpoint structure
print("Checkpoint contents:")
for key in loaded_checkpoint.keys():
    value = loaded_checkpoint[key]
    if hasattr(value, 'shape'):
        print(f"  - {key}: {type(value).__name__} with shape {value.shape}")
    elif isinstance(value, (list, dict)):
        print(f"  - {key}: {type(value).__name__} with {len(value)} items")
    else:
        print(f"  - {key}: {type(value).__name__}")

print("\n" + "="*50)

# Display training summary
if 'epochs_trained' in loaded_checkpoint:
    print(f"\nEpochs trained: {loaded_checkpoint['epochs_trained']}")
if 'best_loss' in loaded_checkpoint:
    print(f"Best loss achieved: {loaded_checkpoint['best_loss']:.6f}")
if 'loss_hist' in loaded_checkpoint:
    initial_loss = loaded_checkpoint['loss_hist'][0] if len(loaded_checkpoint['loss_hist']) > 0 else None
    if initial_loss:
        improvement = (initial_loss - loaded_checkpoint['best_loss']) / initial_loss * 100
        print(f"Initial loss: {initial_loss:.6f}")
        print(f"Improvement: {improvement:.1f}%")

# Display optimized parameters
if 'best_params' in loaded_checkpoint:
    print("\n=== OPTIMIZED PARAMETERS ===")
    best_params = loaded_checkpoint['best_params']
    
    for param_name in ['armature', 'damping', 'frictionloss']:
        if param_name in best_params:
            print(f"\n{param_name.upper()}:")
            param_values = best_params[param_name]
            
            # Handle both numpy arrays and JAX arrays
            if hasattr(param_values, '__iter__'):
                for i, (joint, value) in enumerate(zip(JOINT_NAMES, param_values)):
                    print(f"  {joint:20s}: {float(value):8.4f}")
            else:
                print(f"  Value: {param_values}")

# Calculate and display mean values per joint type
if 'best_params' in loaded_checkpoint:
    import numpy as np
    
    print("\n" + "="*50)
    print("\n=== MEAN VALUES PER JOINT TYPE ===")
    best_params = loaded_checkpoint['best_params']
    
    for param_name in ['armature', 'damping', 'frictionloss']:
        if param_name in best_params:
            print(f"\n{param_name.upper()} MEANS:")
            param_values = best_params[param_name]
            
            if hasattr(param_values, '__iter__'):
                # Convert to numpy for easier indexing
                param_array = np.array(param_values)
                
                for joint_type, indices in JOINT_CATEGORIES.items():
                    values = param_array[indices]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    print(f"  {joint_type:10s}: mean={mean_val:8.4f}, std={std_val:8.4f}")
                    print(f"              values: {[f'{v:.4f}' for v in values]}")

# Display final parameters (if different from best)
if 'final_params' in loaded_checkpoint and 'best_params' in loaded_checkpoint:
    final_params = loaded_checkpoint['final_params']
    best_params = loaded_checkpoint['best_params']
    
    # Check if they're different
    params_differ = False
    for param_name in best_params.keys():
        if param_name in final_params:
            if hasattr(best_params[param_name], '__iter__') and hasattr(final_params[param_name], '__iter__'):
                import numpy as np
                if not np.allclose(best_params[param_name], final_params[param_name]):
                    params_differ = True
                    break
    
    if params_differ:
        print("\n⚠️  Note: Final parameters differ from best parameters")
        print("    (This can happen if early stopping occurred)")

# Plot loss history if available
if 'loss_hist' in loaded_checkpoint:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        loss_hist = loaded_checkpoint['loss_hist']
        epochs = np.arange(len(loss_hist))
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss_hist, linewidth=2)
        if 'best_loss' in loaded_checkpoint:
            plt.axhline(y=loaded_checkpoint['best_loss'], color='r', linestyle='--', 
                       label=f"Best: {loaded_checkpoint['best_loss']:.6f}")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale to see details
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(path), "loss_history_review.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Loss plot saved to: {plot_path}")
        
        plt.show()
    except ImportError:
        print("\n⚠️  matplotlib not available - skipping plot")
    except Exception as e:
        print(f"\n⚠️  Could not create plot: {e}")

# Export parameters to readable format
export_path = os.path.join(os.path.dirname(path), "optimized_parameters.txt")
try:
    import numpy as np
    
    with open(export_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("OPTIMIZED EXOSKELETON PARAMETERS\n")
        f.write("="*60 + "\n\n")
        
        if 'best_loss' in loaded_checkpoint:
            f.write(f"Best Loss: {loaded_checkpoint['best_loss']:.6f}\n")
        if 'epochs_trained' in loaded_checkpoint:
            f.write(f"Epochs Trained: {loaded_checkpoint['epochs_trained']}\n")
        
        f.write("\n" + "="*60 + "\n")
        
        if 'best_params' in loaded_checkpoint:
            best_params = loaded_checkpoint['best_params']
            
            # Per-joint values
            for param_name in ['armature', 'damping', 'frictionloss']:
                if param_name in best_params:
                    f.write(f"\n{param_name.upper()}:\n")
                    f.write("-" * 40 + "\n")
                    param_values = best_params[param_name]
                    
                    if hasattr(param_values, '__iter__'):
                        for joint, value in zip(JOINT_NAMES, param_values):
                            f.write(f"{joint:20s}: {float(value):8.4f}\n")
                    else:
                        f.write(f"Value: {param_values}\n")
            
            # Mean values per joint type
            f.write("\n" + "="*60 + "\n")
            f.write("\nMEAN VALUES PER JOINT TYPE:\n")
            f.write("="*60 + "\n")
            
            for param_name in ['armature', 'damping', 'frictionloss']:
                if param_name in best_params:
                    f.write(f"\n{param_name.upper()}:\n")
                    f.write("-" * 40 + "\n")
                    param_values = best_params[param_name]
                    
                    if hasattr(param_values, '__iter__'):
                        param_array = np.array(param_values)
                        
                        for joint_type, indices in JOINT_CATEGORIES.items():
                            values = param_array[indices]
                            mean_val = np.mean(values)
                            std_val = np.std(values)
                            
                            f.write(f"{joint_type:10s}: mean={mean_val:8.4f}, std={std_val:8.4f}\n")
                            f.write(f"            values: {[f'{v:.4f}' for v in values]}\n")
    
    print(f"✓ Parameters exported to: {export_path}")
    
except Exception as e:
    print(f"⚠️  Could not export parameters: {e}")

print("\n" + "="*50)
print("Done!")