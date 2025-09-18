import numpy as np
import pandas as pd
import os
import logging
import time
import argparse
import joblib
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ID mapping dictionary
IDS = {'group1_order1_user0': 1, 'group1_order1_user1': 2, 'group1_order1_user10': 3, 'group1_order1_user11': 4, 'group1_order1_user12': 5, 
       'group1_order1_user13': 6, 'group1_order1_user14': 7, 'group1_order1_user2': 8, 'group1_order1_user3': 9, 'group1_order1_user4': 10,
       'group1_order1_user5': 11, 'group1_order1_user6': 12, 'group1_order1_user7': 13, 'group1_order1_user8': 14, 'group1_order1_user9': 15, 
       'group1_order2_user0': 16, 'group1_order2_user1': 17, 'group1_order2_user10': 18, 'group1_order2_user11': 19, 'group1_order2_user12': 20, 
       'group1_order2_user13': 21, 'group1_order2_user14': 22, 'group1_order2_user2': 23, 'group1_order2_user3': 24, 'group1_order2_user4': 25, 
       'group1_order2_user5': 26, 'group1_order2_user6': 27, 'group1_order2_user7': 28, 'group1_order2_user8': 29, 'group1_order2_user9': 30, 
       'group2_order1_user0': 31, 'group2_order1_user1': 32, 'group2_order1_user10': 33, 'group2_order1_user11': 34, 'group2_order1_user12': 35, 
       'group2_order1_user13': 36, 'group2_order1_user14': 37, 'group2_order1_user2': 38, 'group2_order1_user3': 39, 'group2_order1_user4': 40, 
       'group2_order1_user5': 41, 'group2_order1_user6': 42, 'group2_order1_user7': 43, 'group2_order1_user8': 44, 'group2_order1_user9': 45, 
       'group2_order2_user0': 46, 'group2_order2_user1': 47, 'group2_order2_user10': 48, 'group2_order2_user11': 49, 'group2_order2_user12': 50, 
       'group2_order2_user13': 51, 'group2_order2_user14': 52, 'group2_order2_user2': 53, 'group2_order2_user3': 54, 'group2_order2_user4': 55, 
       'group2_order2_user5': 56, 'group2_order2_user6': 57, 'group2_order2_user7': 58, 'group2_order2_user8': 59, 'group2_order2_user9': 60}

# Create reverse mapping for faster lookup
IDS_REVERSE = {v: k for k, v in IDS.items()}

class RealisticIntruderGenerator:
    """
    Generates realistic intruder data by modifying raw VR movement data based on
    physical attributes and movement patterns of target users.
    """
    
    def __init__(self, raw_data_path, output_path, scalers_dir='./models/scalers'):
        """
        Initialize the intruder generator.
        
        Args:
            raw_data_path: Path to directory containing raw movement data files
            output_path: Path to save generated intruder features
            scalers_dir: Path to directory containing saved scalers
        """
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.scalers_dir = scalers_dir
        self.user_raw_data = {}  
        self.user_raw_data_fast = {} 
        self.user_raw_data_slow = {}  
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Create subdirectories
        self.phys_output_dir = os.path.join(output_path, "physical")
        self.mov_output_dir = os.path.join(output_path, "movement")
        self.combined_output_dir = os.path.join(output_path, "combined")
        
        for directory in [self.phys_output_dir, self.mov_output_dir, self.combined_output_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def _extract_id(self, filepath):
        """Extract user ID from file path."""
        parts = filepath.split(os.sep)
        for part in parts:
            if 'group' in part and 'order' in part and 'user' in part:
                return part
        return None
    
    def load_raw_data(self, game_type='fast'):
        """
        Load raw movement data for all users of a specific game type.
        
        Args:
            game_type: 'fast' or 'slow' indicating the game speed
        """
        # Check if we've already loaded this game type
        if game_type == 'fast' and self.user_raw_data_fast:
            self.user_raw_data = self.user_raw_data_fast
            logger.info(f"Using cached fast game data ({len(self.user_raw_data)} users)")
            return self.user_raw_data
        elif game_type == 'slow' and self.user_raw_data_slow:
            self.user_raw_data = self.user_raw_data_slow
            logger.info(f"Using cached slow game data ({len(self.user_raw_data)} users)")
            return self.user_raw_data
            
        logger.info(f"Loading raw {game_type} movement data...")
        start_time = time.time()
        
        # Find all relevant movement files
        movement_files = []
        for path, _, files in os.walk(self.raw_data_path):
            for file in files:
                if f'_{game_type}_movement.csv' in file:
                    movement_files.append(os.path.join(path, file))
        
        # Load files sequentially
        self.user_raw_data = {}  
        for file_path in tqdm(movement_files, desc="Loading files"):
            try:
                # Look for the groupX_orderY_userZ pattern in the path
                found_id = False
                parts = file_path.split(os.sep)
                for part in parts:
                    if part.startswith('group') and 'order' in part and 'user' in part:
                        user_id = part
                        found_id = True
                        break
                
                if not found_id:
                    # Try to extract from filename if not found in path
                    filename = os.path.basename(file_path)
                    if "group" in filename and "order" in filename and "user" in filename:
                        # Extract user ID from filename
                        for part in filename.split('_'):
                            if part.startswith('group') or part.startswith('order') or part.startswith('user'):
                                if len(part) > 4:  # Avoid just 'user' or 'group'
                                    user_id = part
                                    found_id = True
                                    break
                
                if not found_id:
                    logger.warning(f"Could not extract user ID from {file_path}")
                    continue
                
                # Only load necessary columns
                cols_to_use = ["time", "HeadPosX", "HeadPosY", "HeadPosZ", 
                              "HeadOrientationX", "HeadOrientationY", "HeadOrientationZ", "HeadOrientationW",
                              "LeftTouchPosX", "LeftTouchPosY", "LeftTouchPosZ", 
                              "LeftTouchOrientationX", "LeftTouchOrientationY", "LeftTouchOrientationZ", "LeftTouchOrientationW",
                              "RightTouchPosX", "RightTouchPosY", "RightTouchPosZ",
                              "RightTouchOrientationX", "RightTouchOrientationY", "RightTouchOrientationZ", "RightTouchOrientationW"]
                
                raw_data = pd.read_csv(file_path, usecols=cols_to_use)
                self.user_raw_data[user_id] = raw_data
                
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
        
        # Save the data to the appropriate cache
        if game_type == 'fast':
            self.user_raw_data_fast = self.user_raw_data.copy()
        elif game_type == 'slow':
            self.user_raw_data_slow = self.user_raw_data.copy()
            
        # Log some debug information about loaded users
        logger.info(f"Loaded raw movement data for {len(self.user_raw_data)} users in {time.time() - start_time:.2f} seconds")
        
        # Print some sample user IDs
        sample_ids = list(self.user_raw_data.keys())[:5] if self.user_raw_data else []
        logger.info(f"Sample user IDs in self.user_raw_data: {sample_ids}")
        
        # Check if any of the loaded user IDs match the IDS dictionary
        matching_ids = [user_id for user_id in self.user_raw_data.keys() if user_id in IDS]
        logger.info(f"Found {len(matching_ids)} users that match IDS mapping keys")
        if matching_ids:
            logger.info(f"Sample matching IDs: {matching_ids[:5]}")
        
        return self.user_raw_data
    
    def extract_physical_attributes(self, user_id):
        """Extract physical attributes from a user's raw movement data."""
        if user_id not in self.user_raw_data:
            logger.error(f"No raw data available for user {user_id}")
            return None
            
        raw_data = self.user_raw_data[user_id]
        
        # Use vectorized operations for better performance
        head_pos = raw_data[['HeadPosX', 'HeadPosY', 'HeadPosZ']].values
        left_touch_pos = raw_data[['LeftTouchPosX', 'LeftTouchPosY', 'LeftTouchPosZ']].values
        right_touch_pos = raw_data[['RightTouchPosX', 'RightTouchPosY', 'RightTouchPosZ']].values
        
        # Calculate physical attributes
        avg_head_height = np.mean(head_pos[:, 1])
        
        # Vector differences
        left_arm_vec = left_touch_pos - head_pos
        right_arm_vec = right_touch_pos - head_pos
        shoulder_vec = left_touch_pos - right_touch_pos
        
        # Distances
        left_arm_length = np.mean(np.sqrt(np.sum(left_arm_vec**2, axis=1)))
        right_arm_length = np.mean(np.sqrt(np.sum(right_arm_vec**2, axis=1)))
        shoulder_width = np.mean(np.sqrt(np.sum(shoulder_vec**2, axis=1)))
        
        # Play area 
        play_area_width = raw_data['HeadPosX'].max() - raw_data['HeadPosX'].min()
        play_area_depth = raw_data['HeadPosZ'].max() - raw_data['HeadPosZ'].min()
        
        # Head orientation
        head_orientation = raw_data[['HeadOrientationX', 'HeadOrientationY', 'HeadOrientationZ']].mean().values
        
        physical_attributes = {
            'height': avg_head_height,
            'left_arm_length': left_arm_length,
            'right_arm_length': right_arm_length,
            'shoulder_width': shoulder_width,
            'play_area_width': play_area_width,
            'play_area_depth': play_area_depth,
            'head_tilt_x': head_orientation[0],
            'head_tilt_y': head_orientation[1],
            'head_tilt_z': head_orientation[2]
        }
        
        return physical_attributes
    
    def extract_movement_patterns(self, user_id):
        """Extract movement patterns from a user's raw movement data."""
        if user_id not in self.user_raw_data:
            logger.error(f"No raw data available for user {user_id}")
            return None
            
        raw_data = self.user_raw_data[user_id]
        
        # Pre-compute time differences
        time_diffs = np.diff(raw_data['time'].values)
        # Handle potential zero time differences
        time_diffs = np.where(time_diffs == 0, 1e-6, time_diffs)
        
        # Position arrays
        head_pos = raw_data[['HeadPosX', 'HeadPosY', 'HeadPosZ']].values
        left_touch_pos = raw_data[['LeftTouchPosX', 'LeftTouchPosY', 'LeftTouchPosZ']].values
        right_touch_pos = raw_data[['RightTouchPosX', 'RightTouchPosY', 'RightTouchPosZ']].values
        
        # Calculate position differences (for velocities)
        head_pos_diff = np.diff(head_pos, axis=0)
        left_pos_diff = np.diff(left_touch_pos, axis=0)
        right_pos_diff = np.diff(right_touch_pos, axis=0)
        
        # Calculate velocities
        head_velocity = np.sqrt(np.sum(head_pos_diff**2, axis=1)) / time_diffs
        left_velocity = np.sqrt(np.sum(left_pos_diff**2, axis=1)) / time_diffs
        right_velocity = np.sqrt(np.sum(right_pos_diff**2, axis=1)) / time_diffs
        
        # Calculate accelerations
        head_accel = np.diff(head_velocity) / time_diffs[:-1]
        left_accel = np.diff(left_velocity) / time_diffs[:-1]
        right_accel = np.diff(right_velocity) / time_diffs[:-1]
        
        # Calculate jerk (derivative of acceleration)
        if len(head_accel) > 1:
            head_jerk = np.diff(head_accel) / time_diffs[:-2]
            left_jerk = np.diff(left_accel) / time_diffs[:-2]
            right_jerk = np.diff(right_accel) / time_diffs[:-2]
            
            head_jerk_mean = np.mean(np.abs(head_jerk)) if len(head_jerk) > 0 else 0
            left_jerk_mean = np.mean(np.abs(left_jerk)) if len(left_jerk) > 0 else 0
            right_jerk_mean = np.mean(np.abs(right_jerk)) if len(right_jerk) > 0 else 0
        else:
            head_jerk_mean = left_jerk_mean = right_jerk_mean = 0
        
        movement_patterns = {
            'avg_head_speed': np.mean(head_velocity),
            'avg_left_speed': np.mean(left_velocity),
            'avg_right_speed': np.mean(right_velocity),
            'head_speed_var': np.var(head_velocity),
            'left_speed_var': np.var(left_velocity),
            'right_speed_var': np.var(right_velocity),
            'head_jerk': head_jerk_mean,
            'left_jerk': left_jerk_mean,
            'right_jerk': right_jerk_mean,
        }
        
        return movement_patterns
    
    def create_physical_attribute_intruder(self, base_user_id, target_user_id, adaptation_factor=0.8):
        """Create an intruder with physical attributes adjusted to match a target user."""
        if base_user_id not in self.user_raw_data or target_user_id not in self.user_raw_data:
            logger.error(f"Missing raw data for users {base_user_id} or {target_user_id}")
            return None
            
        # Get physical attributes
        base_attributes = self.extract_physical_attributes(base_user_id)
        target_attributes = self.extract_physical_attributes(target_user_id)
        
        if base_attributes is None or target_attributes is None:
            return None
            
        # Create a copy of base user's data
        base_data = self.user_raw_data[base_user_id].copy()
        
        # Calculate scaling factors
        height_scale = 1 + adaptation_factor * ((target_attributes['height'] / base_attributes['height']) - 1)
        left_arm_scale = 1 + adaptation_factor * ((target_attributes['left_arm_length'] / base_attributes['left_arm_length']) - 1)
        right_arm_scale = 1 + adaptation_factor * ((target_attributes['right_arm_length'] / base_attributes['right_arm_length']) - 1)
        
        # Apply height transformation
        base_data.loc[:, 'HeadPosY'] *= height_scale
        
        # Get positions as NumPy arrays for faster computation
        head_pos = base_data[['HeadPosX', 'HeadPosY', 'HeadPosZ']].values
        left_pos = base_data[['LeftTouchPosX', 'LeftTouchPosY', 'LeftTouchPosZ']].values
        right_pos = base_data[['RightTouchPosX', 'RightTouchPosY', 'RightTouchPosZ']].values
        
        # Calculate vectors and scale them
        left_vec = left_pos - head_pos
        right_vec = right_pos - head_pos
        
        # Scale vectors
        left_vec_scaled = left_vec * left_arm_scale
        right_vec_scaled = right_vec * right_arm_scale
        
        # Calculate new positions
        new_left_pos = head_pos + left_vec_scaled
        new_right_pos = head_pos + right_vec_scaled
        
        # Update the DataFrame with new positions
        base_data.loc[:, ['LeftTouchPosX', 'LeftTouchPosY', 'LeftTouchPosZ']] = new_left_pos
        base_data.loc[:, ['RightTouchPosX', 'RightTouchPosY', 'RightTouchPosZ']] = new_right_pos
        
        # Adjust head orientation
        for axis in ['X', 'Y', 'Z']:
            base_attr = base_attributes[f'head_tilt_{axis.lower()}']
            target_attr = target_attributes[f'head_tilt_{axis.lower()}']
            delta = target_attr - base_attr
            base_data[f'HeadOrientation{axis}'] += adaptation_factor * delta
        
        return base_data
    
    def create_movement_pattern_intruder(self, base_user_id, target_user_id, adaptation_factor=0.8, raw_data=None):
        """Create an intruder with movement patterns adjusted to match a target user."""
        # Use provided raw_data if available, otherwise load from base_user_id
        if raw_data is not None:
            base_data = raw_data.copy()
            # Need to compute base patterns from the provided data
            base_velocity_head = np.diff(base_data[['HeadPosX', 'HeadPosY', 'HeadPosZ']].values, axis=0)
            base_velocity_left = np.diff(base_data[['LeftTouchPosX', 'LeftTouchPosY', 'LeftTouchPosZ']].values, axis=0)
            base_velocity_right = np.diff(base_data[['RightTouchPosX', 'RightTouchPosY', 'RightTouchPosZ']].values, axis=0)
            
            time_diffs = np.diff(base_data['time'].values)
            time_diffs = np.where(time_diffs == 0, 1e-6, time_diffs)
            
            base_speed_head = np.sqrt(np.sum(base_velocity_head**2, axis=1)) / time_diffs
            base_speed_left = np.sqrt(np.sum(base_velocity_left**2, axis=1)) / time_diffs
            base_speed_right = np.sqrt(np.sum(base_velocity_right**2, axis=1)) / time_diffs
            
            # Extract jerkiness
            if len(base_speed_head) > 2:
                head_accel = np.diff(base_speed_head) / time_diffs[:-1]
                left_accel = np.diff(base_speed_left) / time_diffs[:-1]
                right_accel = np.diff(base_speed_right) / time_diffs[:-1]
                
                if len(head_accel) > 1:
                    head_jerk = np.diff(head_accel) / time_diffs[:-2]
                    left_jerk = np.diff(left_accel) / time_diffs[:-2]
                    right_jerk = np.diff(right_accel) / time_diffs[:-2]
                    
                    base_patterns = {
                        'avg_head_speed': np.mean(base_speed_head),
                        'avg_left_speed': np.mean(base_speed_left),
                        'avg_right_speed': np.mean(base_speed_right),
                        'head_jerk': np.mean(np.abs(head_jerk)) if len(head_jerk) > 0 else 0,
                        'left_jerk': np.mean(np.abs(left_jerk)) if len(left_jerk) > 0 else 0,
                        'right_jerk': np.mean(np.abs(right_jerk)) if len(right_jerk) > 0 else 0,
                    }
                else:
                    base_patterns = {
                        'avg_head_speed': np.mean(base_speed_head),
                        'avg_left_speed': np.mean(base_speed_left),
                        'avg_right_speed': np.mean(base_speed_right),
                        'head_jerk': 0, 'left_jerk': 0, 'right_jerk': 0
                    }
            else:
                base_patterns = {
                    'avg_head_speed': np.mean(base_speed_head) if len(base_speed_head) > 0 else 0,
                    'avg_left_speed': np.mean(base_speed_left) if len(base_speed_left) > 0 else 0,
                    'avg_right_speed': np.mean(base_speed_right) if len(base_speed_right) > 0 else 0,
                    'head_jerk': 0, 'left_jerk': 0, 'right_jerk': 0
                }
        else:
            if base_user_id not in self.user_raw_data:
                logger.error(f"Missing raw data for user {base_user_id}")
                return None
                
            base_data = self.user_raw_data[base_user_id].copy()
            base_patterns = self.extract_movement_patterns(base_user_id)
        
        if target_user_id not in self.user_raw_data:
            logger.error(f"Missing raw data for target user {target_user_id}")
            return None
            
        target_patterns = self.extract_movement_patterns(target_user_id)
        
        # Calculate scaling factors
        head_speed_scale = 1 + adaptation_factor * ((target_patterns['avg_head_speed'] / (base_patterns['avg_head_speed'] + 1e-6)) - 1)
        left_speed_scale = 1 + adaptation_factor * ((target_patterns['avg_left_speed'] / (base_patterns['avg_left_speed'] + 1e-6)) - 1)
        right_speed_scale = 1 + adaptation_factor * ((target_patterns['avg_right_speed'] / (base_patterns['avg_right_speed'] + 1e-6)) - 1)
        
        # Get position arrays
        head_pos = base_data[['HeadPosX', 'HeadPosY', 'HeadPosZ']].values
        left_pos = base_data[['LeftTouchPosX', 'LeftTouchPosY', 'LeftTouchPosZ']].values
        right_pos = base_data[['RightTouchPosX', 'RightTouchPosY', 'RightTouchPosZ']].values
        
        # Scale movements 
        new_head_pos = np.zeros_like(head_pos)
        new_left_pos = np.zeros_like(left_pos)
        new_right_pos = np.zeros_like(right_pos)
        
        # First point remains unchanged
        new_head_pos[0] = head_pos[0]
        new_left_pos[0] = left_pos[0]
        new_right_pos[0] = right_pos[0]
        
        # Generate jerk parameters only once
        head_jerk_ratio = target_patterns['head_jerk'] / (base_patterns.get('head_jerk', 1) + 1e-6)
        left_jerk_ratio = target_patterns['left_jerk'] / (base_patterns.get('left_jerk', 1) + 1e-6)
        right_jerk_ratio = target_patterns['right_jerk'] / (base_patterns.get('right_jerk', 1) + 1e-6)
        
        jerk_factor = adaptation_factor * 0.3
        head_jerk_std = head_jerk_ratio * jerk_factor
        left_jerk_std = left_jerk_ratio * jerk_factor
        right_jerk_std = right_jerk_ratio * jerk_factor
        
        # Add noise only if jerk is significant
        add_head_noise = head_jerk_std > 0.01
        add_left_noise = left_jerk_std > 0.01
        add_right_noise = right_jerk_std > 0.01
        
        # Rebuild trajectory with scaled velocities
        for i in range(1, len(base_data)):
            # Scale velocities
            head_velocity = head_pos[i] - head_pos[i-1]
            left_velocity = left_pos[i] - left_pos[i-1]
            right_velocity = right_pos[i] - right_pos[i-1]
            
            scaled_head_velocity = head_velocity * head_speed_scale
            scaled_left_velocity = left_velocity * left_speed_scale
            scaled_right_velocity = right_velocity * right_speed_scale
            
            # Add jerk noise for non-edge points
            if i > 1 and i < len(base_data) - 1 and np.random.random() < 0.7: 
                if add_head_noise:
                    head_jerk_noise = np.random.normal(0, head_jerk_std, 3)
                    scaled_head_velocity += head_jerk_noise
                
                if add_left_noise:
                    left_jerk_noise = np.random.normal(0, left_jerk_std, 3)
                    scaled_left_velocity += left_jerk_noise
                
                if add_right_noise:
                    right_jerk_noise = np.random.normal(0, right_jerk_std, 3)
                    scaled_right_velocity += right_jerk_noise
            
            # Calculate new positions
            new_head_pos[i] = new_head_pos[i-1] + scaled_head_velocity
            new_left_pos[i] = new_left_pos[i-1] + scaled_left_velocity
            new_right_pos[i] = new_right_pos[i-1] + scaled_right_velocity
        
        # Update the DataFrame with new positions
        base_data.loc[:, ['HeadPosX', 'HeadPosY', 'HeadPosZ']] = new_head_pos
        base_data.loc[:, ['LeftTouchPosX', 'LeftTouchPosY', 'LeftTouchPosZ']] = new_left_pos
        base_data.loc[:, ['RightTouchPosX', 'RightTouchPosY', 'RightTouchPosZ']] = new_right_pos
        
        return base_data
    
    def extract_features(self, raw_data, time_window=10):
        """Extract features from raw movement data for use in authentication systems."""
        if raw_data is None or raw_data.empty:
            logger.error("No valid data provided for feature extraction")
            return None
            
        df = raw_data.copy()
        
        # Calculate time differences - vectorized
        time_values = df['time'].values
        df['Δt'] = np.concatenate([np.diff(time_values), [np.diff(time_values)[-1]]])
        
        # Avoid division by zero
        df.loc[df['Δt'] <= 0, 'Δt'] = 0.01  
        
        # List of segments to process
        segments = ['Head', 'LeftTouch', 'RightTouch']
        
        # Vectorized velocity and acceleration calculations
        for segment in segments:
            # Position columns and their values
            pos_cols = [f'{segment}PosX', f'{segment}PosY', f'{segment}PosZ']
            pos_values = df[pos_cols].values
            
            # Calculate position differences
            pos_diff = np.zeros_like(pos_values)
            pos_diff[:-1] = np.diff(pos_values, axis=0)
            pos_diff[-1] = pos_diff[-2]  
            
            # Calculate velocities
            vel_cols = [f'Velocity_{segment}PosX', f'Velocity_{segment}PosY', f'Velocity_{segment}PosZ']
            df[vel_cols] = pos_diff / df['Δt'].values[:, np.newaxis]
            
            # Calculate velocity magnitude
            df[f'{segment}_Velocity'] = np.sqrt(np.sum(df[vel_cols].values**2, axis=1))
            
            # Calculate accelerations
            acc_cols = [f'Accel_{segment}PosX', f'Accel_{segment}PosY', f'Accel_{segment}PosZ']
            vel_diff = np.zeros_like(pos_diff)
            vel_diff[:-1] = np.diff(df[vel_cols].values, axis=0)
            vel_diff[-1] = vel_diff[-2]  # Duplicate last value
            df[acc_cols] = vel_diff / df['Δt'].values[:, np.newaxis]
            
            # Orientation velocity and acceleration
            ori_cols = [f'{segment}OrientationX', f'{segment}OrientationY', f'{segment}OrientationZ']
            ori_values = df[ori_cols].values
            
            # Calculate orientation differences
            ori_diff = np.zeros_like(ori_values)
            ori_diff[:-1] = np.diff(ori_values, axis=0)
            ori_diff[-1] = ori_diff[-2]  
            
            # Calculate orientation velocities
            ori_vel_cols = [f'{segment}_OrientationVelocityX', f'{segment}_OrientationVelocityY', f'{segment}_OrientationVelocityZ']
            df[ori_vel_cols] = ori_diff / df['Δt'].values[:, np.newaxis]
            
            # Calculate orientation accelerations
            ori_acc_cols = [f'{segment}_OrientationAccelX', f'{segment}_OrientationAccelY', f'{segment}_OrientationAccelZ']
            ori_vel_diff = np.zeros_like(ori_diff)
            ori_vel_diff[:-1] = np.diff(df[ori_vel_cols].values, axis=0)
            ori_vel_diff[-1] = ori_vel_diff[-2]
            df[ori_acc_cols] = ori_vel_diff / df['Δt'].values[:, np.newaxis]
        
        # Calculate distances between body parts
        head_pos = df[['HeadPosX', 'HeadPosY', 'HeadPosZ']].values
        left_touch_pos = df[['LeftTouchPosX', 'LeftTouchPosY', 'LeftTouchPosZ']].values
        right_touch_pos = df[['RightTouchPosX', 'RightTouchPosY', 'RightTouchPosZ']].values
        
        # Distance calculations
        df['distance_LeftTouch_to_Head'] = np.sqrt(np.sum((left_touch_pos - head_pos)**2, axis=1))
        df['distance_RightTouch_to_Head'] = np.sqrt(np.sum((right_touch_pos - head_pos)**2, axis=1))
        df['distance_LeftTouch_to_RightTouch'] = np.sqrt(np.sum((left_touch_pos - right_touch_pos)**2, axis=1))
        
        # Angle calculations - more complex so handle special cases
        def safe_norm(v):
            # Handle zero vectors by adding a tiny value
            norms = np.sqrt(np.sum(v**2, axis=1))
            return np.where(norms > 1e-10, norms, 1e-10)[:, np.newaxis] 
        
        def angle_between(v1, v2):
            v1_u = v1 / safe_norm(v1)
            v2_u = v2 / safe_norm(v2)
            dot_product = np.sum(v1_u * v2_u, axis=1)
            # Clamp to valid range for arccos
            dot_product = np.clip(dot_product, -1.0, 1.0)
            return np.arccos(dot_product)
        
        left_to_head = left_touch_pos - head_pos
        right_to_head = right_touch_pos - head_pos
        left_to_right = left_touch_pos - right_touch_pos
        
        df['angle_LeftTouch_to_Head'] = angle_between(left_to_head, right_to_head)
        df['angle_RightTouch_to_Head'] = angle_between(right_to_head, left_to_head)
        df['angle_LeftTouch_to_RightTouch'] = angle_between(left_to_right, head_pos - right_touch_pos)
        
        # Process data by time windows
        df['time_interval'] = (df['time'] // time_window).astype(int)
        max_interval = time_window * 6
        df = df[df['time_interval'] < max_interval]
        
        # Group and compute statistics more efficiently
        result_stats = []
        
        # Get numerical columns for stats (exclude time_interval)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        num_cols.remove('time_interval')
        
        for interval, interval_df in df.groupby('time_interval'):
            if interval_df.empty:
                continue
                
            # Calculate statistics for all columns at once
            interval_stats = interval_df[num_cols].describe().transpose()
            # Drop count which isn't useful
            interval_stats = interval_stats.drop(columns=['count'])
            
            # Convert to flat format
            stats = pd.DataFrame({f"{col}_{stat}": interval_stats.loc[col, stat] 
                                for col in interval_stats.index 
                                for stat in interval_stats.columns}, index=[0])
            
            # Add metadata
            stats['time_interval'] = int(interval / 6) + 1
            
            result_stats.append(stats)
        
        if not result_stats:
            logger.error("No valid time intervals found in data")
            return None
            
        # Combine all interval statistics
        result_df = pd.concat(result_stats, ignore_index=True)
        
        # Apply MinMax scaling using the saved scalers
        try:
            # Try to load the scaler - use movement scaler for all movement-based intruders
            scaler_path = os.path.join(self.scalers_dir, 'movement_scaler.joblib')
            if os.path.exists(scaler_path):
                logger.info(f"Applying MinMax scaling using saved scaler: {scaler_path}")
                scaler = joblib.load(scaler_path)
                
                # Get only numeric columns to scale (exclude metadata columns)
                meta_cols = ['time_interval']
                feature_cols = [col for col in result_df.columns if col not in meta_cols and 
                                pd.api.types.is_numeric_dtype(result_df[col])]
                
                # Make sure all columns from the scaler are present, add missing ones with zeros
                for col in scaler.feature_names_in_:
                    if col not in feature_cols and col not in meta_cols:
                        result_df[col] = 0
                
                # Apply scaling to the feature columns
                result_df_features = result_df[scaler.feature_names_in_].copy()
                scaled_features = scaler.transform(result_df_features)
                result_df[scaler.feature_names_in_] = scaled_features
                
                logger.info(f"Successfully scaled features using saved scaler")
            else:
                logger.warning(f"Scaler not found at {scaler_path}, skipping scaling")
        except Exception as e:
            logger.error(f"Error applying scaling: {str(e)}")
            # Continue without scaling if there's an error
        
        return result_df
    
    def _get_valid_users_in_range(self, target_user_id, game_type):
        """Get valid users in the same game range as the target."""
        if target_user_id not in IDS:
            logger.error(f"Invalid target user ID: {target_user_id}")
            return []
            
        target_numeric_id = IDS[target_user_id]
        
        # Determine valid ID range based on game type and target ID
        if target_numeric_id <= 30:  # Group 1 (IDs 1-30)
            valid_range = range(1, 31)
        else:  # Group 2 (IDs 31-60)
            valid_range = range(31, 61)
        
        # Get user IDs from the valid range that exist in our data
        valid_users = []
        for id_num in valid_range:
            if id_num != target_numeric_id:  # Exclude the target user
                user_id = IDS_REVERSE.get(id_num)
                if user_id in self.user_raw_data:
                    valid_users.append(user_id)
        
        return valid_users
    
    def _save_intruder_features(self, features, output_file, base_user_id, target_user_id, 
                            attribute_type, adaptation_factor):
        """Save intruder features to a CSV file with proper metadata."""
        if features is None:
            logger.error(f"Cannot save None features for {base_user_id} -> {target_user_id}")
            return False
        
        # Create metadata dictionary
        metadata = {
            'base_user_id': IDS.get(base_user_id, -1),
            'target_user_id': IDS.get(target_user_id, -1),
            'attribute_type': attribute_type,
            'adaptation_factor': adaptation_factor
        }
        
        # Add metadata all at once to avoid fragmentation
        features = features.copy()  # Create a copy to avoid fragmentation
        for key, value in metadata.items():
            features[key] = value
        
        # Save to CSV
        try:
            features.to_csv(output_file, index=False)
            logger.info(f"Saved intruder features to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving to {output_file}: {str(e)}")
            return False

    def generate_single_intruder(self, base_user_id, target_user_id, attribute_type='physical', 
                                adaptation_factor=0.8, intruder_index=0):
        """Generate a single intruder dataset and save it to CSV."""
        start_time = time.time()
        
        # Define output directory based on attribute type
        if attribute_type == 'physical':
            output_dir = self.phys_output_dir
        elif attribute_type == 'movement':
            output_dir = self.mov_output_dir
        else:  # combined
            output_dir = self.combined_output_dir
        
        # Create output filename
        base_id_num = IDS.get(base_user_id, -1)
        target_id_num = IDS.get(target_user_id, -1)
        output_file = os.path.join(
            output_dir, 
            f"intruder_{game_type}_base{base_id_num}_target{target_id_num}_{attribute_type}_{adaptation_factor:.2f}_{intruder_index}.csv"
        )
        
        # Skip if file already exists
        if os.path.exists(output_file):
            logger.info(f"Skipping existing file: {output_file}")
            return output_file
        
        # Generate intruder based on attribute type
        try:
            if attribute_type == 'physical':
                modified_data = self.create_physical_attribute_intruder(
                    base_user_id, target_user_id, adaptation_factor
                )
            elif attribute_type == 'movement':
                modified_data = self.create_movement_pattern_intruder(
                    base_user_id, target_user_id, adaptation_factor
                )
            else:  # combined
                # First modify physical attributes
                phys_modified = self.create_physical_attribute_intruder(
                    base_user_id, target_user_id, adaptation_factor
                )
                # Then modify movement patterns
                modified_data = self.create_movement_pattern_intruder(
                    None, target_user_id, adaptation_factor, raw_data=phys_modified
                )
            
            if modified_data is not None:
                # Extract features with scaling applied
                features = self.extract_features(modified_data)
                
                # Save features to CSV
                success = self._save_intruder_features(
                    features, output_file, base_user_id, target_user_id, 
                    attribute_type, adaptation_factor
                )
                
                elapsed = time.time() - start_time
                logger.info(f"Generated {attribute_type} intruder: {base_user_id} -> {target_user_id} "
                            f"(adapt={adaptation_factor:.2f}) in {elapsed:.2f}s")
                return output_file if success else None
            else:
                logger.error(f"Failed to generate {attribute_type} intruder: {base_user_id} -> {target_user_id}")
                return None
        except Exception as e:
            logger.error(f"Error generating {attribute_type} intruder {base_user_id} -> {target_user_id}: {str(e)}")
            return None
    
    def generate_intruders(self, game_type='fast', num_target_users=3, num_intruders_per_target=5,
                        phys_adapt_range=(0.7, 0.9), move_adapt_range=(0.7, 0.9), comb_adapt_range=(0.7, 0.9)):
        """
        Generate intruder datasets based on different attributes.
        
        Args:
            game_type: 'fast' or 'slow' indicating the game speed
            num_target_users: Number of target users to generate intruders for
            num_intruders_per_target: Number of intruders to generate per target user
            phys_adapt_range: Tuple (min, max) for physical adaptation factor range
            move_adapt_range: Tuple (min, max) for movement adaptation factor range
            comb_adapt_range: Tuple (min, max) for combined adaptation factor range
            
        Returns:
            List of paths to saved intruder feature CSVs
        """
        # Load raw data if not already loaded
        if not self.user_raw_data:
            self.load_raw_data(game_type)
            
        # Print the keys from IDS for debugging
        logger.info(f"IDS dictionary contains {len(IDS)} entries")
        logger.info(f"Sample IDS keys: {list(IDS.keys())[:5]}")
        
        # Find valid users
        valid_users = [user_id for user_id in self.user_raw_data.keys() if user_id in IDS]
        logger.info(f"Found {len(valid_users)} valid users")
        
        if len(valid_users) < 2:
            # If we don't have enough valid users, try to fix ID format issues
            logger.warning("Not enough valid users found. Attempting to fix ID format issues...")
            
            # Try reconstructing correct IDs
            fixed_user_data = {}
            
            # First, check if we need to extract 'groupX_orderY_userZ' from paths
            for user_id, data in self.user_raw_data.items():
                # Try to extract the user ID part
                for id_key in IDS.keys():
                    # Check if the key is part of the path
                    if id_key in user_id:
                        fixed_user_data[id_key] = data
                        logger.info(f"Fixed: {user_id} -> {id_key}")
                        break
                
                # If we couldn't find a match, try to rebuild the ID
                if user_id not in fixed_user_data:
                    parts = user_id.split(os.sep)
                    for part in parts:
                        if 'group' in part and 'order' in part and 'user' in part:
                            for id_key in IDS.keys():
                                if part == id_key:
                                    fixed_user_data[id_key] = data
                                    logger.info(f"Matched: {user_id} -> {id_key}")
                                    break
            
            # Update user_raw_data with fixed IDs
            self.user_raw_data = fixed_user_data
            
            # Recheck for valid users
            valid_users = [user_id for user_id in self.user_raw_data.keys() if user_id in IDS]
            logger.info(f"After fixing: Found {len(valid_users)} valid users")
        
        # Final check
        if len(valid_users) < 2:
            logger.error(f"Not enough valid users available. Found {len(valid_users)} users.")
            return []
        
        # Select target users
        target_users = np.random.choice(valid_users, size=min(num_target_users, len(valid_users)), replace=False)
        
        # Track all output files
        output_files = []
        total_start_time = time.time()
        
        # Process intruders sequentially
        total_intruders = 0
        for target_idx, target_user in enumerate(target_users):
            logger.info(f"Processing target user {target_user} (ID: {IDS.get(target_user)}) ({target_idx+1}/{len(target_users)})")
            
            # Get valid base users for this target
            valid_base_users = self._get_valid_users_in_range(target_user, game_type)
            
            if not valid_base_users:
                logger.warning(f"No valid base users found for target {target_user}")
                continue
            
            # Select base users (intruders)
            base_users = np.random.choice(
                valid_base_users, 
                size=min(num_intruders_per_target, len(valid_base_users)), 
                replace=False
            )
            
            # Create a progress bar for this target's intruders
            intruder_types = ['physical', 'movement', 'combined']
            total_for_target = 2 * len(base_users) + min(len(base_users), max(1, num_intruders_per_target//2))
            
            with tqdm(total=total_for_target, desc=f"Target {IDS.get(target_user, -1)} intruders") as pbar:
                # Generate physical attribute intruders
                for i, base_user in enumerate(base_users):
                    # Use the provided adaptation factor range
                    adaptation_factor = np.random.uniform(phys_adapt_range[0], phys_adapt_range[1])
                    output_file = self.generate_single_intruder(
                        base_user, target_user, 'physical', adaptation_factor, i
                    )
                    if output_file:
                        output_files.append(output_file)
                        total_intruders += 1
                    pbar.update(1)
                
                # Generate movement pattern intruders
                for i, base_user in enumerate(base_users):
                    # Use the provided adaptation factor range
                    adaptation_factor = np.random.uniform(move_adapt_range[0], move_adapt_range[1])
                    output_file = self.generate_single_intruder(
                        base_user, target_user, 'movement', adaptation_factor, i
                    )
                    if output_file:
                        output_files.append(output_file)
                        total_intruders += 1
                    pbar.update(1)
                
                # Generate combined attribute intruders
                for i, base_user in enumerate(base_users[:max(1, num_intruders_per_target//2)]):
                    # Use the provided adaptation factor range for combined
                    adaptation_factor = np.random.uniform(comb_adapt_range[0], comb_adapt_range[1])
                    output_file = self.generate_single_intruder(
                        base_user, target_user, 'combined', adaptation_factor, i
                    )
                    if output_file:
                        output_files.append(output_file)
                        total_intruders += 1
                    pbar.update(1)
        
        total_time = time.time() - total_start_time
        logger.info(f"Generated {total_intruders} intruders in {total_time:.2f} seconds")
        return output_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate realistic VR intruder data')
    parser.add_argument('--raw_data_path', type=str, default='./data/raw/Raw_traffic_and_movement_data/',
                        help='Path to raw movement data files')
    parser.add_argument('--output_path', type=str, default='./data/processed/intruders',
                        help='Path to save generated intruder features')
    parser.add_argument('--scalers_dir', type=str, default='./models/scalers',
                        help='Path to directory containing saved scalers')
    parser.add_argument('--game_types', type=str, default='both', choices=['fast', 'slow', 'both'],
                        help='Game type(s) to process: fast, slow, or both')
    parser.add_argument('--num_targets', type=int, default=60,
                        help='Number of target users to generate intruders for')
    parser.add_argument('--num_intruders', type=int, default=10,
                        help='Number of intruders to generate per target user')
    parser.add_argument('--phys_adapt_min', type=float, default=0.85,
                        help='Minimum physical adaptation factor (0.0-1.0)')
    parser.add_argument('--phys_adapt_max', type=float, default=1.0,
                        help='Maximum physical adaptation factor (0.0-1.0)')
    parser.add_argument('--move_adapt_min', type=float, default=0.85,
                        help='Minimum movement adaptation factor (0.0-1.0)')
    parser.add_argument('--move_adapt_max', type=float, default=1.0,
                        help='Maximum movement adaptation factor (0.0-1.0)')
    parser.add_argument('--comb_adapt_min', type=float, default=0.85,
                        help='Minimum combined adaptation factor (0.0-1.0)')
    parser.add_argument('--comb_adapt_max', type=float, default=1.0,
                        help='Maximum combined adaptation factor (0.0-1.0)')
    
    args = parser.parse_args()
    


















































































































































































































    
    generator = RealisticIntruderGenerator(args.raw_data_path, args.output_path, args.scalers_dir)


















































































































































































































    
    all_output_files = []


















































































































































































































    
    


















































































































































































































    
    # Determine which game types to process
    game_types = []
    if args.game_types == 'both':
        game_types = ['fast', 'slow']
    else:
        game_types = [args.game_types]
    
    # Process each game type
    for game_type in game_types:
        logger.info(f"Processing game type: {game_type}")
        output_files = generator.generate_intruders(
            game_type=game_type,
            num_target_users=args.num_targets, 
            num_intruders_per_target=args.num_intruders,
            phys_adapt_range=(args.phys_adapt_min, args.phys_adapt_max),
            move_adapt_range=(args.move_adapt_min, args.move_adapt_max),
            comb_adapt_range=(args.comb_adapt_min, args.comb_adapt_max)
        )
        all_output_files.extend(output_files)
    
    if all_output_files:
        print(f"Successfully generated {len(all_output_files)} intruders.")
        print(f"Output saved to: {args.output_path}")
    else:
        print("Failed to generate intruders. Check the logs for details.")

   