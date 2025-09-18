# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def get_all_files(data_path):
    """Retrieve all files in the specified directory."""
    all_files = []
    for path, _, files in os.walk(data_path):
        all_files.extend([os.path.join(path, name) for name in files])
    return all_files

def extract_ids(filepath):
    """Extract IDs from the file path."""
    parts = filepath.split(os.sep)
    for part in parts:
        if 'group' in part and 'order' in part and 'user' in part:
            return part


def feature_engineering(df, data_type):
    if data_type == "movement":
        df = df.drop(columns=["TouchButtons", "LeftIndexTrigger", "RightIndexTrigger", "LeftHandTrigger", "RightHandTrigger"])
        df['Δt'] = df['time'].diff().shift(-1)
        df.loc[df.index[-1], 'Δt'] = df['Δt'].iloc[-2]
        
        segments = ['Head', 'LeftTouch', 'RightTouch']
        for segment in segments:
            pos_cols = [f'{segment}PosX', f'{segment}PosY', f'{segment}PosZ']
            vel_cols = [f'Velocity_{segment}PosX', f'Velocity_{segment}PosY', f'Velocity_{segment}PosZ']
            acc_cols = [f'Accel_{segment}PosX', f'Accel_{segment}PosY', f'Accel_{segment}PosZ']
            
            df[vel_cols] = df[pos_cols].diff().div(df['Δt'], axis=0)
            df[f'{segment}_Velocity'] = np.linalg.norm(df[vel_cols], axis=1)
            df[acc_cols] = df[vel_cols].diff().div(df['Δt'], axis=0)

            ori_vel_cols = [f'{segment}_OrientationVelocityX', f'{segment}_OrientationVelocityY', f'{segment}_OrientationVelocityZ']
            ori_acc_cols = [f'{segment}_OrientationAccelX', f'{segment}_OrientationAccelY', f'{segment}_OrientationAccelZ']
            ori_cols = [f'{segment}OrientationX', f'{segment}OrientationY', f'{segment}OrientationZ']
            
            df[ori_vel_cols] = df[ori_cols].diff().div(df['Δt'], axis=0)
            df[ori_acc_cols] = df[ori_vel_cols].diff().div(df['Δt'], axis=0)

        # Calculate distances and angles
        head_pos = df[['HeadPosX', 'HeadPosY', 'HeadPosZ']].values
        left_touch_pos = df[['LeftTouchPosX', 'LeftTouchPosY', 'LeftTouchPosZ']].values
        right_touch_pos = df[['RightTouchPosX', 'RightTouchPosY', 'RightTouchPosZ']].values

        df['distance_LeftTouch_to_Head'] = np.linalg.norm(left_touch_pos - head_pos, axis=1)
        df['distance_RightTouch_to_Head'] = np.linalg.norm(right_touch_pos - head_pos, axis=1)
        df['distance_LeftTouch_to_RightTouch'] = np.linalg.norm(left_touch_pos - right_touch_pos, axis=1)

        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1, axis=1)[:, np.newaxis]
            v2_u = v2 / np.linalg.norm(v2, axis=1)[:, np.newaxis]
            return np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0))

        df['angle_LeftTouch_to_Head'] = angle_between(left_touch_pos - head_pos, right_touch_pos - head_pos)
        df['angle_RightTouch_to_Head'] = angle_between(right_touch_pos - head_pos, left_touch_pos - head_pos)
        df['angle_LeftTouch_to_RightTouch'] = angle_between(left_touch_pos - right_touch_pos, head_pos - right_touch_pos)

    if data_type == "traffic":
        # Traffic data processing remains unchanged
        df['Δt'] = df['time'].diff().fillna(0)
        df['size_cumsum'] = df.groupby('direction')['size'].cumsum()

        ul_dl_diff = df.groupby('direction')['size'].cumsum().diff().fillna(0)
        df['ul_dl_ratio'] = ul_dl_diff.replace({0: np.nan}).apply(lambda x: x if x > 0 else -1/x)
        df['packet_burst'] = (df['direction'].shift(-1) != df['direction']).cumsum()

        df['size_rate'] = df['size'] / df['Δt'].replace({0: np.nan})
        df['burstiness'] = df['size_rate'].rolling(window=10).apply(lambda x: np.std(x) / np.mean(x))

        df['packet_count'] = df.groupby('direction').cumcount() + 1

    df.drop(columns=["Δt"], inplace=True)
    return df

def process_data(filepaths, time_window, data_type):
    result_stat = []

    for filepath in tqdm(filepaths, desc=f'Processing {data_type} data'):
        df = pd.read_csv(filepath)
        df = feature_engineering(df, data_type)

        df['time_interval'] = (df['time'] // 10).astype(int)
        max_interval = time_window * 6
        df = df[df['time_interval'] < max_interval]

        for interval, interval_df in df.groupby('time_interval'):
            if interval_df.empty:
                continue

            num_stats = interval_df.drop(columns = ["time_interval"]).describe().transpose().drop(columns=['count'])
            stats = num_stats.stack().to_frame().T

            try:
                cat_stats = interval_df.describe(include='object').transpose()
                for col in cat_stats.index:
                    mode_val = interval_df[col].mode().iloc[0] if not interval_df[col].mode().empty else np.nan
                    cat_counts = interval_df[col].value_counts()
                    cat_stats.loc[col, 'mode'] = mode_val
                    for cat, count in cat_counts.items():
                        cat_name = str(cat).replace(" ", "_").replace("/", "_").lower()
                        cat_stats.loc[col, f'count_{cat_name}'] = count
                stats = pd.concat([stats, cat_stats], axis=1)
            except:
                pass

            ids = extract_ids(filepath)
            stats['time_interval'] = int(interval / 6) + 1 
            stats['ID'] = ids

            result_stat.append(stats)

    result_stat = pd.concat(result_stat, ignore_index=True)
    new_columns = ['_'.join(col) if isinstance(col, tuple) else col for col in result_stat.columns]
    result_stat.columns = new_columns
    result_stat.reset_index(drop=True, inplace=True)

    return result_stat

# Input and output file paths
INPUT_FILEPATH = "./data/raw/Raw_traffic_and_movement_data/"
OUTPUT_FILEPATH = "./data/processed/"
TIME_WINDOW = 10

def main():
    """Runs data processing scripts to extract features from raw data."""
    logger = logging.getLogger(__name__)
    logger.info('Making final statistical summary dataset from raw data')

    # Get all files related to the participants' data
    all_files = get_all_files(INPUT_FILEPATH)

    # Separate traffic data and movement data paths
    traffic_data = [x for x in all_files if '_traffic.csv' in x]
    movement_data = [x for x in all_files if '_movement.csv' in x]

    # Process movement data
    logger.info('Processing Fast Movement Data')
    movement_fast_stat = process_data([filepath for filepath in movement_data if 'fast' in filepath], TIME_WINDOW, "movement")
    movement_fast_stat.to_csv(OUTPUT_FILEPATH + 'movement_fast_stat.csv', index=False)

    logger.info('Processing Slow Movement Data')
    movement_slow_stat = process_data([filepath for filepath in movement_data if 'slow' in filepath], TIME_WINDOW, "movement")
    movement_slow_stat.to_csv(OUTPUT_FILEPATH + 'movement_slow_stat.csv', index=False)

    # Process traffic data
    logger.info('Processing Fast Traffic Data')
    traffic_fast_stat = process_data([filepath for filepath in traffic_data if 'fast' in filepath], TIME_WINDOW, "traffic")
    traffic_fast_stat.to_csv(OUTPUT_FILEPATH + 'traffic_fast_stat.csv', index=False)

    logger.info('Processing Slow Traffic Data')
    traffic_slow_stat = process_data([filepath for filepath in traffic_data if 'slow' in filepath], TIME_WINDOW, "traffic")
    traffic_slow_stat.to_csv(OUTPUT_FILEPATH + 'traffic_slow_stat.csv', index=False)

    logger.info('Save processed data')

if __name__ == '__main__':
    main()
