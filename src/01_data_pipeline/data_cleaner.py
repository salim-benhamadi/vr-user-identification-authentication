import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Dictionary mapping participant IDs to numerical values
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

class Preprocessing:
    """
    A class to perform data preprocessing steps including fixing feature naming, handling missing values,
    feature scaling, label encoding, and matching columns between datasets.
    """

    @staticmethod
    def fix_feature_naming(df):
        """
        Fix feature naming by removing leading and trailing spaces and renaming specific columns.

        Args:
            df (DataFrame): Dataframe to fix feature naming.

        Returns:
            DataFrame: Dataframe with fixed feature naming.
        """
        df.columns = df.columns.str.strip()
        if "ID_" in df.columns:
            df.rename(columns={"ID_": "ID", "time_interval_": "time_interval"}, inplace=True)
        return df

    @staticmethod
    def find_non_varying_variables(df):
        """
        Find non-varying variables in a dataframe.

        Args:
            df (DataFrame): DataFrame to find non-varying variables.

        Returns:
            DataFrame: DataFrame containing non-varying variables and their variability percentages.
        """
        non_varying_columns = []
        variability_percentage = []
        
        for column in df.columns:
            unique_count = df[column].nunique()
            total_count = len(df[column])
            variability = unique_count / total_count * 100
            
            if unique_count == 1:
                non_varying_columns.append(column)
                variability_percentage.append(variability)
        
        result_df = pd.DataFrame({'Variable': non_varying_columns, 'Variability Percentage': variability_percentage})
        return result_df

    @staticmethod
    def drop_non_varying_variables(df):
        """
        Drop non-varying variables in a dataframe.

        Args:
            df (DataFrame): DataFrame to drop non-varying variables.

        Returns:
            DataFrame: DataFrame with non-varying variables dropped.
        """
        non_varying_columns = []

        for column in df.columns:
            unique_count = df[column].nunique()
            if unique_count == 1:
                non_varying_columns.append(column)

        df = df.drop(columns=non_varying_columns)
        return df

    @staticmethod
    def missing_columns(dataframe):
        """
        Returns a DataFrame containing missing column names and percent of missing values.

        Args:
            dataframe (DataFrame): DataFrame to check for missing columns.

        Returns:
            DataFrame: DataFrame containing missing column names and their percentage of missing values.
        """
        missing_values = dataframe.isnull().sum().sort_values(ascending=False)
        missing_values_pct = 100 * missing_values / len(dataframe)
        concat_values = pd.concat([missing_values, missing_values / len(dataframe), missing_values_pct.round(1)],
                                  axis=1)
        concat_values.columns = ['Missing Count', 'Missing Count Ratio', 'Missing Count %']
        return concat_values[concat_values.iloc[:, 1] != 0]

    @staticmethod
    def match_columns(training_set, testing_set):
        """
        Matches column count between training and testing sets.

        Args:
            training_set (DataFrame): DataFrame representing the training set.
            testing_set (DataFrame): DataFrame representing the testing set.

        Returns:
            DataFrame: Testing set with matched columns.
        """
        for column in training_set.columns:
            if column not in testing_set.columns:
                testing_set[column] = 0
        for column in testing_set.columns:
            if column not in training_set.columns:
                testing_set = testing_set.drop(columns=[column])
        return testing_set

    @staticmethod
    def scaling(df, scaler=None):
        """
        Perform feature scaling on numeric columns using MinMaxScaler.

        Args:
            df (DataFrame): DataFrame to perform feature scaling.
            scaler (MinMaxScaler): Scaler to fit or transform.

        Returns:
            DataFrame, MinMaxScaler: Scaled DataFrame and fitted scaler.
        """
        numeric_cols = df.select_dtypes(include=['number']).columns.difference(["time_interval"])
        if not scaler:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[numeric_cols])
        else:
            scaled_data = scaler.transform(df[numeric_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)
        for col in df.columns:
            if col not in numeric_cols:
                scaled_df[col] = df[col].values
        return scaled_df, scaler

    @staticmethod
    def encoding(df, encoders=None):
        """
        Perform label encoding on categorical columns.

        Args:
            df (DataFrame): DataFrame to perform label encoding.
            encoders (dict): Dictionary of existing LabelEncoders for each column.

        Returns:
            DataFrame, dict: DataFrame with encoded categorical features and dictionary of fitted encoders.
        """
        if encoders is None:
            encoders = {}
            
        for col in df.select_dtypes('object').columns.difference(["ID"]):
            if col not in encoders:
                encoders[col] = LabelEncoder()
                df[col] = encoders[col].fit_transform(df[col].astype(str))
            else:
                # Try to use existing classes, handle unseen labels
                try:
                    df[col] = encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen labels by fitting a new encoder and storing it
                    encoders[col] = LabelEncoder()
                    df[col] = encoders[col].fit_transform(df[col].astype(str))
                    
        return df, encoders

    @staticmethod
    def prepare_label(df):
        """
        Prepare labels by converting participant IDs to numerical values.

        Args:
            df (DataFrame): DataFrame containing participant IDs.

        Returns:
            DataFrame: DataFrame with prepared labels.
        """
        df['ID'] = df['ID'].apply(lambda x: x.split('/')[4] if '/' in x else x)
        df['ID'] = df['ID'].apply(lambda x: IDS[x] - 1)
        df = df.reindex(sorted(df.columns), axis=1)
        return df


def process_data(input_paths, output_paths, scalers_dir='../models/scalers'):
    """
    Process all datasets applying the preprocessing pipeline.
    
    Args:
        input_paths (dict): Dictionary with paths to input data files
        output_paths (dict): Dictionary with paths to output processed files
        scalers_dir (str): Directory to save the scalers and encoders
        
    Returns:
        dict: Information about processed data shapes
    """
    # Create scalers directory if it doesn't exist
    if not os.path.exists(scalers_dir):
        os.makedirs(scalers_dir)
    
    print("Loading data...")
    # Load datasets
    mov_fast = pd.read_csv(input_paths['movement_fast'])
    mov_slow = pd.read_csv(input_paths['movement_slow'])
    traffic_fast = pd.read_csv(input_paths['traffic_fast'])
    traffic_slow = pd.read_csv(input_paths['traffic_slow'])
    
    print("Fixing feature naming...")
    # Fix feature naming
    mov_fast = Preprocessing.fix_feature_naming(mov_fast)
    mov_slow = Preprocessing.fix_feature_naming(mov_slow)
    traffic_fast = Preprocessing.fix_feature_naming(traffic_fast)
    traffic_slow = Preprocessing.fix_feature_naming(traffic_slow)
    
    print("Checking for non-varying variables...")
    # Print non-varying variables info (optional)
    print("Movement Fast - Non-varying variables:")
    print(Preprocessing.find_non_varying_variables(mov_fast))
    print("\nMovement Slow - Non-varying variables:")
    print(Preprocessing.find_non_varying_variables(mov_slow))
    print("\nTraffic Fast - Non-varying variables:")
    print(Preprocessing.find_non_varying_variables(traffic_fast))
    print("\nTraffic Slow - Non-varying variables:")
    print(Preprocessing.find_non_varying_variables(traffic_slow))
    
    print("Dropping non-varying variables...")
    # Drop non-varying variables
    mov_fast = Preprocessing.drop_non_varying_variables(mov_fast)
    mov_slow = Preprocessing.drop_non_varying_variables(mov_slow)
    traffic_fast = Preprocessing.drop_non_varying_variables(traffic_fast)
    traffic_slow = Preprocessing.drop_non_varying_variables(traffic_slow)
    
    print("Checking for missing values...")
    # Check for missing values (optional)
    print("Movement Fast - Missing values:")
    print(Preprocessing.missing_columns(mov_fast))
    print("\nMovement Slow - Missing values:")
    print(Preprocessing.missing_columns(mov_slow))
    print("\nTraffic Fast - Missing values:")
    print(Preprocessing.missing_columns(traffic_fast))
    print("\nTraffic Slow - Missing values:")
    print(Preprocessing.missing_columns(traffic_slow))
    
    print("Matching columns...")
    # Match columns before scaling
    mov_fast = Preprocessing.match_columns(mov_slow, mov_fast)
    traffic_fast = Preprocessing.match_columns(traffic_slow, traffic_fast)
    
    print("Scaling features...")
    # Scale features
    mov_slow, mov_scaler = Preprocessing.scaling(mov_slow)
    mov_fast, _ = Preprocessing.scaling(mov_fast, mov_scaler)
    traffic_slow, traffic_scaler = Preprocessing.scaling(traffic_slow)
    traffic_fast, _ = Preprocessing.scaling(traffic_fast, traffic_scaler)
    
    # Export scalers
    print("Exporting scalers...")
    joblib.dump(mov_scaler, os.path.join(scalers_dir, 'movement_scaler.joblib'))
    joblib.dump(traffic_scaler, os.path.join(scalers_dir, 'traffic_scaler.joblib'))
    print(f"Scalers saved to {scalers_dir}")
    
    print("Encoding categorical variables...")
    # Label encoding with encoder export
    encoders = {}
    mov_slow, mov_encoders = Preprocessing.encoding(mov_slow)
    encoders['movement'] = mov_encoders
    
    mov_fast, _ = Preprocessing.encoding(mov_fast, mov_encoders)
    
    traffic_slow, traffic_encoders = Preprocessing.encoding(traffic_slow)
    encoders['traffic'] = traffic_encoders
    
    traffic_fast, _ = Preprocessing.encoding(traffic_fast, traffic_encoders)
    
    # Export encoders
    print("Exporting encoders...")
    joblib.dump(encoders, os.path.join(scalers_dir, 'label_encoders.joblib'))
    print(f"Encoders saved to {scalers_dir}")
    
    print("Matching columns again...")
    # Match columns after transformations
    mov_fast = Preprocessing.match_columns(mov_slow, mov_fast)
    traffic_fast = Preprocessing.match_columns(traffic_slow, traffic_fast)
    
    print("Preparing labels...")
    # Prepare labels
    mov_fast = Preprocessing.prepare_label(mov_fast)
    mov_slow = Preprocessing.prepare_label(mov_slow)
    traffic_fast = Preprocessing.prepare_label(traffic_fast)
    traffic_slow = Preprocessing.prepare_label(traffic_slow)
    
    print("Saving processed data...")
    # Save processed data
    mov_fast.to_csv(output_paths['movement_fast'], index=False)
    mov_slow.to_csv(output_paths['movement_slow'], index=False)
    traffic_fast.to_csv(output_paths['traffic_fast'], index=False)
    traffic_slow.to_csv(output_paths['traffic_slow'], index=False)
    
    print("Data processing complete!")
    
    # Save IDS mapping for reference
    ids_df = pd.DataFrame(list(IDS.items()), columns=['User_ID', 'Numeric_ID'])
    ids_df.to_csv(os.path.join(scalers_dir, 'ids_mapping.csv'), index=False)
    
    # Return information about processed data
    return {
        'movement_fast': mov_fast.shape,
        'movement_slow': mov_slow.shape,
        'traffic_fast': traffic_fast.shape,
        'traffic_slow': traffic_slow.shape
    }


if __name__ == "__main__":
    # Define input and output paths
    input_paths = {
        'movement_fast': './data/processed/movement_fast_stat.csv',
        'movement_slow': './data/processed/movement_slow_stat.csv',
        'traffic_fast': './data/processed/traffic_fast_stat.csv',
        'traffic_slow': './data/processed/traffic_slow_stat.csv'
    }
    
    output_paths = {
        'movement_fast': './data/processed/movement_fast_stat_cleaned.csv',
        'movement_slow': './data/processed/movement_slow_stat_cleaned.csv',
        'traffic_fast': './data/processed/traffic_fast_stat_cleaned.csv',
        'traffic_slow': './data/processed/traffic_slow_stat_cleaned.csv'
    }
    
    # Create a directory for the scalers and encoders
    scalers_dir = './models/scalers'
    
    # Process all datasets and export transformers
    shapes = process_data(input_paths, output_paths, scalers_dir)
    
    # Print final shapes
    print("\nFinal data shapes:")
    for dataset, shape in shapes.items():
        print(f"{dataset}: {shape}")
        
    print(f"\nAll transformation objects saved to {scalers_dir}")
    print("Saved objects:")
    print("- movement_scaler.joblib: MinMaxScaler for movement data")
    print("- traffic_scaler.joblib: MinMaxScaler for traffic data")
    print("- label_encoders.joblib: Dictionary of LabelEncoders for categorical variables")
    print("- ids_mapping.csv: Mapping between user IDs and numeric IDs")