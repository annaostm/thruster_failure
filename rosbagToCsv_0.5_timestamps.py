import rosbag
import pandas as pd
import numpy as np
import os

def quaternion_to_yaw(qx, qy, qz, qw):
    """
    Convert quaternion (qx, qy, qz, qw) to yaw angle in degrees.
    """
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return np.degrees(yaw)  # Convert from radians to degrees

def extract_combined_data_from_rosbag(bag_path, output_csv, failure_actuator, failure_time):
    bag = rosbag.Bag(bag_path, 'r')
    nav_data = []
    imu_data = []
    yaw_data = []
    rc_data = []
    
    first_timestamp = None  # To normalize timestamps
    max_force = 500  # Newtons
    max_torque = 200  # Newton-meters
    scale = 1000  # Controller input scale
    
    for topic, msg, t in bag.read_messages():
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            raw_timestamp = msg.header.stamp.to_sec()
            if raw_timestamp == 0.0:
                raw_timestamp = t.to_sec()  # Fallback to ROS message timestamp if invalid
        else:
            raw_timestamp = t.to_sec()  # Use ROS message timestamp
        
        if first_timestamp is None:
            first_timestamp = raw_timestamp  # Store the first timestamp only once
        elif raw_timestamp < first_timestamp:  
            print(f"Warning: New timestamp {raw_timestamp} is earlier than first timestamp {first_timestamp}!")
        
        relative_timestamp = raw_timestamp - first_timestamp  # Normalize time
        
        if topic == '/navigation/twist_body':
            nav_data.append({
                'timestamp': relative_timestamp,
                'linear_velocity_x': msg.twist.linear.x,
                'linear_velocity_y': msg.twist.linear.y
            })
        
        elif topic == '/imu/data':
            imu_data.append({
                'timestamp': relative_timestamp,
                'linear_acceleration_x': msg.linear_acceleration.x,
                'linear_acceleration_y': msg.linear_acceleration.y
            })
        
        elif topic == '/navigation/pose':  # Extract yaw from quaternion pose
            try:
                qx = msg.pose.orientation.x
                qy = msg.pose.orientation.y
                qz = msg.pose.orientation.z
                qw = msg.pose.orientation.w
                yaw = quaternion_to_yaw(qx, qy, qz, qw)  # Convert quaternion to yaw
                yaw_data.append({'timestamp': relative_timestamp, 'yaw': yaw})
            except AttributeError:
                pass  # Suppress errors if missing data
        
        elif topic == '/rc_state':
            try:
                right_x = msg.right_stick.x_axis / scale
                right_y = msg.right_stick.y_axis / scale
                left_y = msg.left_stick.y_axis / scale
                
                force_x = (right_x * abs(right_x)) * max_force
                force_y = (right_y * abs(right_y)) * max_force
                torque_n = left_y * max_torque
                
                rc_data.append({'timestamp': relative_timestamp, 'force_x': force_x, 'force_y': force_y, 'torque_n': torque_n})
            except AttributeError:
                pass  # Suppress attribute errors
    
    bag.close()
    
    df_nav = pd.DataFrame(nav_data)
    df_imu = pd.DataFrame(imu_data)
    df_yaw = pd.DataFrame(yaw_data)
    df_rc = pd.DataFrame(rc_data)
    
    if not df_rc.empty:
        df_rc.sort_values('timestamp', inplace=True)  # Ensure timestamps are sorted
    
    if not df_nav.empty and not df_imu.empty:
        df_combined = pd.merge_asof(df_nav.sort_values('timestamp'), df_imu.sort_values('timestamp'), on='timestamp', direction='nearest')
        
        if not df_yaw.empty:
            df_combined = pd.merge_asof(df_combined.sort_values('timestamp'), df_yaw.sort_values('timestamp'), on='timestamp', direction='nearest')
            df_combined['yaw_rate'] = df_combined['yaw'].diff() / df_combined['timestamp'].diff()
            df_combined['yaw_rate'].fillna(0, inplace=True)  # Replace NaN with 0 for the first entry
        else:
            df_combined['yaw_rate'] = 0
        
        if not df_rc.empty:
            print("First few rows of RC Data:")
            print(df_rc.head())
            df_combined = pd.merge_asof(df_combined.sort_values('timestamp'), df_rc, on='timestamp', direction='backward')
            df_combined.insert(0, 'id', range(1, len(df_combined) + 1))  # Add ID column
            print("First few rows of Combined Data:")
            print(df_combined[['id', 'timestamp', 'force_x', 'force_y', 'torque_n']].head())
        else:
            df_combined['force_x'] = df_combined['force_y'] = df_combined['torque_n'] = 0
        
        for actuator in ['/actuator_ref_1', '/actuator_ref_2', '/actuator_ref_3', '/actuator_ref_4']:
            df_combined[actuator] = 0
        
        df_combined[failure_actuator] = (df_combined['timestamp'] >= failure_time).astype(int)
        df_combined['fault'] = df_combined[['/actuator_ref_1', '/actuator_ref_2', '/actuator_ref_3', '/actuator_ref_4']].sum(axis=1).clip(upper=1)
        
        # ---- New Changes for 0.5s Resampling ----
        df_combined['timestamp'] = df_combined['timestamp'].round(3)  # Round to milliseconds
        df_combined = df_combined.drop_duplicates(subset='timestamp', keep='last')  # Keep the last value for each timestamp
        
        # Ensure timestamp is in datetime format for resampling
        df_combined['timestamp'] = pd.to_timedelta(df_combined['timestamp'], unit='s')
        
        # Resample every 0.5 seconds, keeping the most recent values
        if not df_combined.empty:
            df_combined = df_combined.set_index('timestamp').resample('0.5S').ffill().reset_index()
        else:
            print("Warning: DataFrame is empty before resampling!")

        # Convert timestamp back to float seconds
        df_combined['timestamp'] = df_combined['timestamp'].dt.total_seconds()

        # Remove yaw column as requested
        df_combined = df_combined.drop(columns=['yaw'], errors='ignore')

        df_combined.reset_index(drop=True, inplace=True)
        df_combined.to_csv(output_csv, index=False)
        print(f"Saved combined CSV: {output_csv}")
    else:
        print("Error: One or more DataFrames are empty. No CSV file created.")

# Usage - Specify the actuator that failed and the failure time
bag_file = "/workspace/Bager/autoteaming_exp1a_1802025.bag"
output_csv_file = "/workspace/Thruster Data 0.5 Timestamps/1a_180225_data_0.5.csv"
failure_actuator = '/actuator_ref_1'  # Change this per experiment
failure_time = 44.5  # Change this per experiment

extract_combined_data_from_rosbag(bag_file, output_csv_file, failure_actuator, failure_time)
