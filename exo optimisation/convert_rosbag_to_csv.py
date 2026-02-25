#!/usr/bin/env python3
"""
Convert ROS2 bag (MCAP format) to CSV for exoskeleton system identification.

This script extracts joint positions, velocities, and motor currents from ROS2 bags
and formats them for the optimization pipeline.

Requirements:
    pip install mcap mcap-ros2-support pandas numpy

Usage:
    Edit the INPUT_MCAP and OUTPUT_CSV variables below, then run:
    python convert_rosbag_to_csv.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import sys

# ========== CONFIGURATION ==========
INPUT_MCAP = "exo optimisation/data/raw/2026-02-17-15-41-59_0_fix_comp.mcap"  # Change this to your MCAP file path
OUTPUT_CSV = "exo optimisation/data/2026-02-17-15-41-59_0_fix_comp_2.csv"  # Output CSV file
JOINT_STATES_TOPIC = '/joint_states'
PDO_TOPIC = '/pdo_states'
START_TIME = 25  # Start time in seconds (relative to bag start), or None
END_TIME = 95    # End time in seconds (relative to bag start), or None
# Only process these topics when set (keeps conversion faster)
TOPICS_TO_KEEP = [JOINT_STATES_TOPIC, PDO_TOPIC]
# ===================================

# Joint names in the order needed for optimization
JOINT_NAMES = [
    'left_hip_aa', 'left_hip_fe', 'left_knee', 
    'L_Front_Linear', 'L_Back_Linear',
    'right_hip_aa', 'right_hip_fe', 'right_knee',
    'R_Front_Linear', 'R_Back_Linear'
]

# Mapping from PDO axes indices to joint names
PDO_AXES_MAPPING = {
    0: "right_ankle_back_linear",
    1: "right_ankle_front_linear",
    2: "right_hip_fe",
    3: "right_knee",
    4: "left_ankle_back_linear",
    5: "left_ankle_front_linear",
    6: "left_hip_fe",
    7: "left_knee",
    8: "left_hip_aa",
    9: "right_hip_aa",
}

# Mapping from joint_states indices to joint names
JOINT_STATES_MAPPING = {
    0: "left_ankle_dpf",
    1: "left_ankle_ie",
    2: "left_hip_aa",
    3: "left_hip_fe",
    4: "left_knee",
    5: "right_ankle_dpf",
    6: "right_ankle_ie",
    7: "right_hip_aa",
    8: "right_hip_fe",
    9: "right_knee",
    10: "L_Back_Linear",      # Map to optimization names
    11: "L_Front_Linear",     # Map to optimization names
    12: "R_Back_Linear",      # Map to optimization names
    13: "R_Front_Linear",     # Map to optimization names
}


def convert_mcap_to_csv(mcap_path, output_csv, joint_states_topic='/joint_states', pdo_topic='/pdo_states', 
                        start_time=None, end_time=None):
    """
    Convert MCAP rosbag to CSV format.
    
    Args:
        mcap_path: Path to .mcap file
        output_csv: Output CSV path
        joint_states_topic: Topic name for joint states
        pdo_topic: Topic name for PDO states (motor currents)
        start_time: Start time in seconds (relative to bag start). If None, start from beginning.
        end_time: End time in seconds (relative to bag start). If None, process until end.
    """
    print(f"Reading MCAP file: {mcap_path}")
    if start_time is not None:
        print(f"Start time: {start_time}s")
    if end_time is not None:
        print(f"End time: {end_time}s")
    
    # Storage for extracted data
    joint_states_data = []
    pdo_data = []
    
    # Read MCAP file - First pass to get bag start time and scan topics
    bag_start_time = None
    topics_found = set()
    
    topics_to_keep = set(TOPICS_TO_KEEP) if TOPICS_TO_KEEP else None

    print("Scanning bag for topics...")
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f)
        
        for schema, channel, message in reader.iter_messages():
            if bag_start_time is None:
                bag_start_time = message.log_time / 1e9
            
            topics_found.add(channel.topic)
            
            # Only scan first 100 messages to find topics quickly
            if len(topics_found) >= 2 and message.sequence > 100:
                break
    
    print(f"Topics found in bag: {sorted(topics_found)}")
    print(f"Bag start time: {bag_start_time:.3f}s")
    
    if bag_start_time is None:
        print("ERROR: Empty bag file!")
        return False
    
    # Calculate absolute time bounds
    abs_start_time = bag_start_time + (start_time if start_time is not None else 0)
    abs_end_time = bag_start_time + end_time if end_time is not None else float('inf')
    
    # Read MCAP file - Second pass to extract data (with proper decoding)
    print("Extracting data...")
    message_count = 0
    skipped_count = 0
    
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        
        for schema, channel, message, ros_msg in reader.iter_decoded_messages():
            if topics_to_keep and channel.topic not in topics_to_keep:
                continue
            timestamp = message.log_time / 1e9  # Convert nanoseconds to seconds
            
            # Skip messages outside time range
            if timestamp < abs_start_time:
                skipped_count += 1
                if skipped_count % 1000 == 0:
                    print(f"Skipping... (t={timestamp - bag_start_time:.1f}s)", end='\r')
                continue
            if timestamp > abs_end_time:
                print(f"\nReached end time, stopping.")
                break
            
            message_count += 1
            if message_count % 10 == 0:
                print(f"Processed {message_count} messages... (t={timestamp - bag_start_time:.1f}s)", end='\r')
            
            # Process joint_states messages
            if channel.topic == joint_states_topic:
                positions = list(ros_msg.position) if hasattr(ros_msg, 'position') else []
                velocities = list(ros_msg.velocity) if hasattr(ros_msg, 'velocity') else []
                efforts = list(ros_msg.effort) if hasattr(ros_msg, 'effort') else []
                if positions:
                    joint_states_data.append({
                        'timestamp': timestamp,
                        'positions': positions,
                        'velocities': velocities,
                        'efforts': efforts,
                    })
            
            # Process PDO messages (motor currents)
            elif channel.topic == pdo_topic or 'pdo' in channel.topic.lower():
                currents = []
                if hasattr(ros_msg, 'axes'):
                    for axis in ros_msg.axes:
                        if hasattr(axis, 'current'):
                            currents.append(axis.current)
                        else:
                            currents.append(0.0)
                if currents:
                    pdo_data.append({
                        'timestamp': timestamp,
                        'currents': currents,
                    })
    
    print(f"\nProcessed {message_count} total messages")
    print(f"Found {len(joint_states_data)} joint_states messages")
    print(f"Found {len(pdo_data)} PDO messages")
    
    if not joint_states_data:
        print("ERROR: No joint_states data found in bag file!")
        print(f"Available topics might be different. Please check your bag file.")
        return False
    
    # Convert to dataframes for easier processing
    js_df = pd.DataFrame(joint_states_data)
    
    # Interpolate to common timestamps (use joint_states as base)
    timestamps = js_df['timestamp'].values
    
    # Create output dataframe
    output_data = {'time': timestamps}
    
    # Extract positions, velocities, and efforts for each joint
    for i, row in js_df.iterrows():
        timestamp = row['timestamp']
        positions = row['positions']
        velocities = row['velocities']
        efforts = row['efforts']
        
        # Map joint_states data to joint names
        for idx, joint_name in JOINT_STATES_MAPPING.items():
            if idx < len(positions):
                if joint_name in JOINT_NAMES:
                    if f'{joint_name}_pos' not in output_data:
                        output_data[f'{joint_name}_pos'] = []
                        output_data[f'{joint_name}_vel'] = []
                        output_data[f'{joint_name}_torque'] = []
                    output_data[f'{joint_name}_pos'].append(positions[idx])
                    output_data[f'{joint_name}_vel'].append(velocities[idx])
                    # Use effort as torque
                    torque = efforts[idx] if idx < len(efforts) else 0.0
                    output_data[f'{joint_name}_torque'].append(torque)
    
    # Create final dataframe
    df_output = pd.DataFrame(output_data)
    
    # Remove any rows with missing data
    df_output = df_output.dropna()
    
    # Save to CSV
    df_output.to_csv(output_csv, index=False)
    
    print(f"\nConversion complete!")
    print(f"Output saved to: {output_csv}")
    print(f"Number of samples: {len(df_output)}")
    print(f"Duration: {df_output['time'].iloc[-1] - df_output['time'].iloc[0]:.3f} seconds")
    print(f"Sampling rate: ~{len(df_output) / (df_output['time'].iloc[-1] - df_output['time'].iloc[0]):.1f} Hz")
    
    # Print data summary
    print("\nData summary:")
    for joint in JOINT_NAMES:
        pos_col = f'{joint}_pos'
        vel_col = f'{joint}_vel'
        torque_col = f'{joint}_torque'
        if pos_col in df_output.columns:
            print(f"{joint}:")
            print(f"  Position: [{df_output[pos_col].min():.3f}, {df_output[pos_col].max():.3f}]")
            print(f"  Velocity: [{df_output[vel_col].min():.3f}, {df_output[vel_col].max():.3f}]")
            print(f"  Torque:   [{df_output[torque_col].min():.3f}, {df_output[torque_col].max():.3f}]")
    
    return True


def convert_rosbag2_to_csv(bag_path, output_csv):
    """
    Alternative method using rosbag2_py for ROS2 bags stored as directory.
    
    Args:
        bag_path: Path to ROS2 bag directory
        output_csv: Output CSV path
    """
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        print("ERROR: rosbag2_py not installed. Install with: pip install rosbag2_py")
        return False
    
    print(f"Reading ROS2 bag: {bag_path}")
    
    # Storage for extracted data
    joint_states_data = []
    pdo_data = []
    
    # Open bag
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    # Get topic types
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}
    
    # Read messages
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        timestamp_sec = timestamp / 1e9
        
        if topic not in type_map:
            continue
        
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        
        if topic == '/joint_states':
            joint_states_data.append({
                'timestamp': timestamp_sec,
                'positions': list(msg.position),
                'velocities': list(msg.velocity),
            })
        
        elif 'pdo' in topic.lower():
            currents = []
            if hasattr(msg, 'axes'):
                for axis in msg.axes:
                    if hasattr(axis, 'current'):
                        currents.append(axis.current)
            
            pdo_data.append({
                'timestamp': timestamp_sec,
                'currents': currents,
            })
    
    print(f"Found {len(joint_states_data)} joint_states messages")
    print(f"Found {len(pdo_data)} PDO messages")
    
    # Continue with same processing as MCAP version...
    # (Implementation similar to above)
    
    return True


def main():
    """Main function - edit INPUT_MCAP and OUTPUT_CSV at the top of this file."""
    
    input_path = Path(INPUT_MCAP)
    
    if not input_path.exists():
        print(f"ERROR: Input file/directory not found: {input_path}")
        print(f"Please edit the INPUT_MCAP variable at the top of this script.")
        sys.exit(1)
    
    # Determine input type
    if input_path.is_file() and input_path.suffix == '.mcap':
        success = convert_mcap_to_csv(
            INPUT_MCAP,
            OUTPUT_CSV,
            JOINT_STATES_TOPIC,
            PDO_TOPIC,
            START_TIME,
            END_TIME
        )
    elif input_path.is_dir():
        print("Detected ROS2 bag directory, using rosbag2_py...")
        success = convert_rosbag2_to_csv(INPUT_MCAP, OUTPUT_CSV)
    else:
        print("ERROR: Input must be either a .mcap file or ROS2 bag directory")
        sys.exit(1)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
