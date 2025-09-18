import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

class HierarchicalUserAuthenticator:
    """
    Hierarchical User Authentication System using behavioral biometrics.
    This system uses a two-level approach:
    1. First level: Cluster users into groups
    2. Second level: Train user-specific models within each cluster
    
    This approach scales better as the number of users increases.
    """
    
    def __init__(self, num_clusters=None):
        """
        Initialize the hierarchical authenticator.
        
        Args:
            num_clusters: Number of clusters to use (if None, will be determined automatically)
        """
        self.num_clusters = num_clusters
        self.clusters = {}  # Dictionary to store cluster information
        self.user_to_cluster = {}  # Mapping from user_id to cluster_id
        self.cluster_models = {}  # Models for cluster assignment
        self.user_models = {}  # User-specific models within clusters
        self.pca_models = {}  # PCA models for dimensionality reduction
        self.scalers = {}  # Scalers for feature normalization
        self.user_centroids = {}  # Store user centroids in feature space
        self.performance_metrics = pd.DataFrame()
        
    def prepare_data(self, mov_slow, mov_fast, traffic_slow=None, traffic_fast=None):
        """
        Prepare data for hierarchical authentication by creating user datasets.
        
        Args:
            mov_slow: DataFrame with slow game movement data
            mov_fast: DataFrame with fast game movement data
            traffic_slow: Optional DataFrame with slow game traffic data
            traffic_fast: Optional DataFrame with fast game traffic data
            
        Returns:
            user_data: Dictionary mapping user IDs to their feature data
            all_user_ids: List of all user IDs
        """
        print("Preparing data for hierarchical authentication system...")
        
        # Use movement data as primary source
        data_slow = mov_slow.copy()
        data_fast = mov_fast.copy()
        
        # Get all unique user IDs
        all_user_ids = np.unique(np.concatenate([data_slow['ID'].unique(), data_fast['ID'].unique()]))
        print(f"Total unique users: {len(all_user_ids)}")
        
        # Create data dictionary to store per-user datasets
        user_data = {}
        
        for user_id in all_user_ids:
            # Extract data for this user
            user_data_slow = data_slow[data_slow['ID'] == user_id].drop(columns=['ID'])
            user_data_fast = data_fast[data_fast['ID'] == user_id].drop(columns=['ID'])
            
            # Store in dictionary
            user_data[user_id] = {
                'slow': user_data_slow,
                'fast': user_data_fast
            }
        
        self.user_data = user_data
        self.all_user_ids = all_user_ids
        
        return user_data, all_user_ids

    def get_user_centroids(self, game_type='slow'):
        """
        Calculate centroids for each user in feature space.
        
        Args:
            game_type: 'slow' or 'fast' indicating which game data to use
            
        Returns:
            centroids: DataFrame with user centroids
        """
        centroids = []
        
        for user_id in self.all_user_ids:
            # Get user data
            user_features = self.user_data[user_id][game_type]
            
            # Calculate centroid (mean of all feature vectors)
            centroid = user_features.mean().to_dict()
            centroid['user_id'] = user_id
            
            centroids.append(centroid)
        
        # Convert to DataFrame
        centroids_df = pd.DataFrame(centroids)
        
        return centroids_df
    
    def cluster_users(self, game_type='slow', method='kmeans', visualize=True):
        """
        Cluster users based on their behavioral patterns.
        
        Args:
            game_type: 'slow' or 'fast' indicating which game data to use
            method: Clustering method ('kmeans' or 'hierarchical')
            visualize: Whether to visualize the clusters
        """
        print(f"Clustering users based on {game_type} game data...")
        
        # Get user centroids
        centroids_df = self.get_user_centroids(game_type)
        self.user_centroids[game_type] = centroids_df
        
        # Extract features and user IDs
        user_ids = centroids_df['user_id'].values
        features = centroids_df.drop(columns=['user_id'])
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self.scalers[game_type] = scaler
        
        # Apply dimensionality reduction for visualization
        pca = PCA(n_components=min(10, features.shape[1]))
        features_pca = pca.fit_transform(features_scaled)
        self.pca_models[game_type] = pca
        
        # Determine optimal number of clusters if not specified
        if self.num_clusters is None:
            # Use simple rule: sqrt of number of users
            self.num_clusters = max(2, int(np.sqrt(len(user_ids))))
            print(f"Automatically determined {self.num_clusters} clusters")
        
        # Cluster the users
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features_pca)
            self.cluster_models[game_type] = kmeans
        else:
            # Add other clustering methods as needed
            raise ValueError("Only kmeans clustering is currently supported")
        
        # Store cluster information
        self.clusters[game_type] = {}
        
        for i in range(self.num_clusters):
            cluster_members = user_ids[cluster_labels == i]
            self.clusters[game_type][i] = {
                'members': cluster_members,
                'size': len(cluster_members)
            }
        
        # Create mapping from user_id to cluster_id
        self.user_to_cluster[game_type] = {}
        for i, user_id in enumerate(user_ids):
            self.user_to_cluster[game_type][user_id] = cluster_labels[i]
        
        # Print cluster sizes
        print("Cluster sizes:")
        for cluster_id, info in self.clusters[game_type].items():
            print(f"  Cluster {cluster_id}: {info['size']} users")
        
        # Visualize clusters if requested
        if visualize:
            self.visualize_clusters(game_type, features_pca, user_ids, cluster_labels)
        
        return self.clusters[game_type]
    
    def visualize_clusters(self, game_type, features_pca, user_ids, cluster_labels):
        """
        Visualize user clusters.
        
        Args:
            game_type: 'slow' or 'fast' indicating which game data to use
            features_pca: PCA-transformed features
            user_ids: Array of user IDs
            cluster_labels: Array of cluster labels
        """
        # Use t-SNE for better visualization in 2D
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features_pca)
        
        # Create DataFrame for plotting
        viz_df = pd.DataFrame({
            'x': features_2d[:, 0],
            'y': features_2d[:, 1],
            'user_id': user_ids,
            'cluster': cluster_labels
        })
        
        # Plot
        plt.figure(figsize=(14, 10))
        sns.scatterplot(x='x', y='y', hue='cluster', data=viz_df, palette='tab10', s=100)
        
        # Add user IDs as labels
        for i, row in viz_df.iterrows():
            plt.text(row['x'] + 0.1, row['y'] + 0.1, int(row['user_id']), fontsize=9)
        
        plt.title(f'User Clustering ({game_type.capitalize()} Game)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.show()
    
    def train_hierarchical_models(self, game_type='slow', test_size=0.2, balance_classes=True):
        """
        Train hierarchical authentication models.
        
        Args:
            game_type: 'slow' or 'fast' indicating which game data to use
            test_size: Proportion of data to use for testing
            balance_classes: Whether to balance classes for user-specific models
        """
        print(f"Training hierarchical authentication models for {game_type} game...")
        
        # Check if clustering has been performed
        if game_type not in self.clusters:
            raise ValueError(f"Must perform clustering for {game_type} game first")
        
        # Initialize model dictionaries
        if game_type not in self.user_models:
            self.user_models[game_type] = {}
        
        # Create performance metrics dataframe
        performance_metrics = []
        
        # For each cluster, train a classifier to identify users within that cluster
        for cluster_id, cluster_info in self.clusters[game_type].items():
            cluster_members = cluster_info['members']
            
            if len(cluster_members) <= 1:
                print(f"Skipping cluster {cluster_id} with only {len(cluster_members)} member")
                continue
            
            print(f"\nTraining models for cluster {cluster_id} with {len(cluster_members)} users...")
            
            # Train authentication models for each user in this cluster
            for user_id in cluster_members:
                # Get positive samples (genuine user)
                genuine_samples = self.user_data[user_id][game_type].copy()
                genuine_samples['is_genuine'] = 1
                
                # Get negative samples (other users in the same cluster)
                impostor_samples_list = []
                
                for other_id in cluster_members:
                    if other_id != user_id:
                        other_samples = self.user_data[other_id][game_type].copy()
                        impostor_samples_list.append(other_samples)
                
                if not impostor_samples_list:
                    print(f"Skipping user {user_id} as there are no other users in cluster {cluster_id}")
                    continue
                
                impostor_samples = pd.concat(impostor_samples_list)
                impostor_samples['is_genuine'] = 0
                
                # Balance the classes if requested
                if balance_classes:
                    if len(impostor_samples) > len(genuine_samples):
                        impostor_samples = impostor_samples.sample(n=len(genuine_samples), replace=False)
                    elif len(genuine_samples) > len(impostor_samples):
                        genuine_samples = genuine_samples.sample(n=len(impostor_samples), replace=False)
                
                # Combine positive and negative samples
                train_data = pd.concat([genuine_samples, impostor_samples])
                
                # Split features and target
                X = train_data.drop(columns=['is_genuine'])
                y = train_data['is_genuine']
                
                # Split into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Train model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Store the model
                self.user_models[game_type][user_id] = model
                
                # Evaluate the model
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Calculate confusion matrix
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Acceptance Rate
                frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Rejection Rate
                
                # Store metrics
                performance_metrics.append({
                    'User_ID': user_id,
                    'Cluster_ID': cluster_id,
                    'Game_Type': game_type.capitalize(),
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1,
                    'FAR': far,
                    'FRR': frr
                })
        
        # Convert to DataFrame and combine with existing metrics
        metrics_df = pd.DataFrame(performance_metrics)
        self.performance_metrics = pd.concat([self.performance_metrics, metrics_df], ignore_index=True)
        
        print(f"\nTraining completed for {game_type} game.")
        
        # Print summary
        cluster_summary = metrics_df.groupby('Cluster_ID')[['Accuracy', 'F1', 'FAR', 'FRR']].mean()
        print("\nAverage performance by cluster:")
        print(cluster_summary)
        
        overall_avg = metrics_df[['Accuracy', 'F1', 'FAR', 'FRR']].mean()
        print("\nOverall average performance:")
        print(overall_avg)
        
        return metrics_df
    
    def authenticate_user(self, claimed_user_id, gameplay_data, game_type='slow'):
        """
        Authenticate a user using the hierarchical approach.
        
        Args:
            claimed_user_id: The ID of the user trying to authenticate
            gameplay_data: DataFrame with gameplay features
            game_type: 'slow' or 'fast' indicating the game being played
            
        Returns:
            is_genuine: Boolean indicating if the user is genuine
            confidence: Confidence score of the prediction
        """
        # Check if the user is known
        if claimed_user_id not in self.all_user_ids:
            print(f"Error: User ID {claimed_user_id} not found in the system.")
            return False, 0.0
        
        # Check if we have models for this game type
        if game_type not in self.user_models:
            print(f"Error: No models available for {game_type} game.")
            return False, 0.0
        
        # First, determine which cluster this user belongs to
        cluster_id = self.user_to_cluster[game_type][claimed_user_id]
        
        # Get the user-specific model
        if claimed_user_id not in self.user_models[game_type]:
            print(f"Error: No model available for User ID {claimed_user_id}.")
            return False, 0.0
        
        model = self.user_models[game_type][claimed_user_id]
        
        # Make prediction
        prediction = model.predict(gameplay_data)[0]
        
        # Get confidence score
        confidence = model.predict_proba(gameplay_data)[0][1]
        
        is_genuine = bool(prediction)
        
        return is_genuine, confidence
    
    def test_hierarchical_authentication(self, game_type='slow', num_samples=5):
        """
        Test the hierarchical authentication system with all users.
        
        Args:
            game_type: 'slow' or 'fast' indicating which game to test
            num_samples: Number of samples to test per user
            
        Returns:
            results_df: DataFrame with test results
        """
        print(f"Testing hierarchical authentication system for {game_type} game...")
        
        results = []
        
        # For each user, test genuine and impostor attempts
        for true_user_id in self.all_user_ids:
            # Genuine authentication attempts (user as self)
            genuine_data = self.user_data[true_user_id][game_type].sample(
                n=min(num_samples, len(self.user_data[true_user_id][game_type]))
            )
            
            for i, (_, sample) in enumerate(genuine_data.iterrows()):
                sample_df = pd.DataFrame([sample])
                
                is_genuine, confidence = self.authenticate_user(
                    true_user_id, sample_df, game_type
                )
                
                results.append({
                    'True_User_ID': true_user_id,
                    'Claimed_User_ID': true_user_id,
                    'Sample_ID': i,
                    'Is_Genuine_Attempt': True,
                    'Authentication_Result': is_genuine,
                    'Confidence': confidence,
                    'Result_Type': 'True Positive' if is_genuine else 'False Negative'
                })
            
            # Impostor authentication attempts (other users as this user)
            # Choose other users, preferably from the same cluster
            cluster_id = self.user_to_cluster[game_type][true_user_id]
            cluster_members = self.clusters[game_type][cluster_id]['members']
            other_users_in_cluster = [uid for uid in cluster_members if uid != true_user_id]
            
            # If there are other users in the same cluster, use them
            # Otherwise sample from all other users
            if other_users_in_cluster:
                impostor_ids = np.random.choice(
                    other_users_in_cluster, 
                    size=min(num_samples, len(other_users_in_cluster)),
                    replace=len(other_users_in_cluster) < num_samples
                )
            else:
                other_users = [uid for uid in self.all_user_ids if uid != true_user_id]
                impostor_ids = np.random.choice(
                    other_users,
                    size=min(num_samples, len(other_users)),
                    replace=len(other_users) < num_samples
                )
            
            for i, impostor_id in enumerate(impostor_ids):
                impostor_data = self.user_data[impostor_id][game_type].sample(1)
                
                is_genuine, confidence = self.authenticate_user(
                    true_user_id, impostor_data, game_type
                )
                
                results.append({
                    'True_User_ID': impostor_id,
                    'Claimed_User_ID': true_user_id,
                    'Sample_ID': i,
                    'Is_Genuine_Attempt': False,
                    'Authentication_Result': is_genuine,
                    'Confidence': confidence,
                    'Result_Type': 'False Positive' if is_genuine else 'True Negative'
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate overall metrics
        tp = len(results_df[(results_df['Is_Genuine_Attempt'] == True) & (results_df['Authentication_Result'] == True)])
        tn = len(results_df[(results_df['Is_Genuine_Attempt'] == False) & (results_df['Authentication_Result'] == False)])
        fp = len(results_df[(results_df['Is_Genuine_Attempt'] == False) & (results_df['Authentication_Result'] == True)])
        fn = len(results_df[(results_df['Is_Genuine_Attempt'] == True) & (results_df['Authentication_Result'] == False)])
        
        total = len(results_df)
        accuracy = (tp + tn) / total
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"\nHierarchical Authentication Results ({game_type} game):")
        print(f"Total Tests: {total}")
        print(f"True Positives (Correctly Authenticated): {tp}")
        print(f"True Negatives (Correctly Rejected): {tn}")
        print(f"False Positives (Incorrectly Authenticated): {fp}")
        print(f"False Negatives (Incorrectly Rejected): {fn}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"False Acceptance Rate: {far:.4f}")
        print(f"False Rejection Rate: {frr:.4f}")
        
        # Test results by cluster
        print("\nResults by cluster:")
        for cluster_id in self.clusters[game_type]:
            cluster_members = self.clusters[game_type][cluster_id]['members']
            cluster_results = results_df[results_df['Claimed_User_ID'].isin(cluster_members)]
            
            if len(cluster_results) > 0:
                cluster_tp = len(cluster_results[(cluster_results['Is_Genuine_Attempt'] == True) & 
                                               (cluster_results['Authentication_Result'] == True)])
                cluster_tn = len(cluster_results[(cluster_results['Is_Genuine_Attempt'] == False) & 
                                               (cluster_results['Authentication_Result'] == False)])
                cluster_total = len(cluster_results)
                cluster_accuracy = (cluster_tp + cluster_tn) / cluster_total
                
                print(f"Cluster {cluster_id} ({len(cluster_members)} users): Accuracy = {cluster_accuracy:.4f}")
        
        return results_df
    
    def visualize_performance(self):
        """Visualize the performance of the hierarchical authentication system."""
        if len(self.performance_metrics) == 0:
            print("No performance metrics available. Train models first.")
            return
        
        # Performance by cluster
        plt.figure(figsize=(14, 10))
        
        # Plot F1 score by cluster
        plt.subplot(2, 2, 1)
        cluster_perf = self.performance_metrics.groupby(['Game_Type', 'Cluster_ID'])['F1'].mean().reset_index()
        sns.barplot(x='Cluster_ID', y='F1', hue='Game_Type', data=cluster_perf)
        plt.title('F1 Score by Cluster')
        plt.ylim(0, 1)
        
        # Plot FAR by cluster
        plt.subplot(2, 2, 2)
        cluster_perf = self.performance_metrics.groupby(['Game_Type', 'Cluster_ID'])['FAR'].mean().reset_index()
        sns.barplot(x='Cluster_ID', y='FAR', hue='Game_Type', data=cluster_perf)
        plt.title('False Acceptance Rate by Cluster')
        plt.ylim(0, 0.5)
        
        # Plot accuracy distribution
        plt.subplot(2, 2, 3)
        sns.histplot(data=self.performance_metrics, x='Accuracy', hue='Game_Type', kde=True)
        plt.title('Distribution of User Authentication Accuracies')
        plt.xlim(0, 1)
        
        # Plot FRR vs FAR
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=self.performance_metrics, x='FAR', y='FRR', hue='Game_Type', style='Cluster_ID')
        plt.title('Security Tradeoff: FRR vs FAR')
        plt.xlim(0, 0.5)
        plt.ylim(0, 0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Cluster sizes
        if 'slow' in self.clusters:
            game_type = 'slow'
            
            sizes = [info['size'] for info in self.clusters[game_type].values()]
            ids = list(self.clusters[game_type].keys())
            
            plt.figure(figsize=(14, 6))
            plt.bar(ids, sizes)
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of Users')
            plt.title('Cluster Sizes')
            plt.xticks(ids)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
        
    def compare_to_flat_model(self, game_type='slow', test_size=0.2):
        """
        Compare hierarchical model to flat (non-hierarchical) model.
        
        Args:
            game_type: 'slow' or 'fast' indicating which game data to use
            test_size: Proportion of data to use for testing
            
        Returns:
            DataFrame with comparison results
        """
        print("Comparing hierarchical model to flat model...")
        
        # Train a flat model where each user is compared against all others
        flat_results = []
        
        for user_id in self.all_user_ids:
            # Get positive samples (genuine user)
            genuine_samples = self.user_data[user_id][game_type].copy()
            genuine_samples['is_genuine'] = 1
            
            # Get negative samples (all other users)
            impostor_samples_list = []
            
            for other_id in self.all_user_ids:
                if other_id != user_id:
                    other_samples = self.user_data[other_id][game_type].copy()
                    impostor_samples_list.append(other_samples)
            
            if not impostor_samples_list:
                continue
                
            impostor_samples = pd.concat(impostor_samples_list).sample(
                n=len(genuine_samples), replace=len(pd.concat(impostor_samples_list)) < len(genuine_samples)
            )
            impostor_samples['is_genuine'] = 0
            
            # Combine positive and negative samples
            train_data = pd.concat([genuine_samples, impostor_samples])
            
            # Split features and target
            X = train_data.drop(columns=['is_genuine'])
            y = train_data['is_genuine']
            
            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Acceptance Rate
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Rejection Rate
            
            # Store metrics
            flat_results.append({
                'User_ID': user_id,
                'Model_Type': 'Flat',
                'Game_Type': game_type.capitalize(),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'FAR': far,
                'FRR': frr
            })
        
        # Get hierarchical results
        hierarchical_results = []
        
        for _, row in self.performance_metrics.iterrows():
            if row['Game_Type'].lower() == game_type:
                hierarchical_results.append({
                    'User_ID': row['User_ID'],
                    'Model_Type': 'Hierarchical',
                    'Game_Type': row['Game_Type'],
                    'Accuracy': row['Accuracy'],
                    'Precision': row['Precision'],
                    'Recall': row['Recall'],
                    'F1': row['F1'],
                    'FAR': row['FAR'],
                    'FRR': row['FRR']
                })
        
        # Combine results
        all_results = pd.DataFrame(flat_results + hierarchical_results)
        
        # Calculate averages
        avg_results = all_results.groupby('Model_Type')[['Accuracy', 'Precision', 'Recall', 'F1', 'FAR', 'FRR']].mean()
        
        print("\nAverage Performance Comparison:")
        print(avg_results)
        
        # Visualize comparison
        plt.figure(figsize=(14, 10))
        
        # Plot accuracy comparison
        plt.subplot(2, 2, 1)
        sns.boxplot(x='Model_Type', y='Accuracy', data=all_results)
        plt.title('Accuracy Comparison')
        plt.ylim(0, 1)
        
        # Plot F1 comparison
        plt.subplot(2, 2, 2)
        sns.boxplot(x='Model_Type', y='F1', data=all_results)
        plt.title('F1 Score Comparison')
        plt.ylim(0, 1)
        
        # Plot FAR comparison
        plt.subplot(2, 2, 3)
        sns.boxplot(x='Model_Type', y='FAR', data=all_results)
        plt.title('False Acceptance Rate Comparison')
        plt.ylim(0, 0.5)
        
        # Plot FRR comparison
        plt.subplot(2, 2, 4)
        sns.boxplot(x='Model_Type', y='FRR', data=all_results)
        plt.title('False Rejection Rate Comparison')
        plt.ylim(0, 0.5)
        
        plt.tight_layout()
        plt.show()
        
        return all_results

    def evaluate_scalability(self, max_users=100, step=10):
        """
        Evaluate how the hierarchical approach scales with increasing numbers of users.
        This is a simulated analysis since we may not have enough real users.
        
        Args:
            max_users: Maximum number of simulated users to test
            step: Step size for increasing user count
            
        Returns:
            DataFrame with scalability metrics
        """
        import time
        
        print("Evaluating scalability of hierarchical vs. flat approach...")
        
        results = []
        
        # Base feature set
        feature_names = list(self.user_data[self.all_user_ids[0]]['slow'].columns)
        
        # Create simulated users if needed
        num_real_users = len(self.all_user_ids)
        num_synthetic_users = max(0, max_users - num_real_users)
        
        all_users = list(self.all_user_ids)
        
        if num_synthetic_users > 0:
            print(f"Creating {num_synthetic_users} synthetic users for scalability testing...")
            
            # Start IDs after the highest real user ID
            start_id = int(max(self.all_user_ids)) + 1
            
            for i in range(num_synthetic_users):
                user_id = start_id + i
                all_users.append(user_id)
                
                # Create synthetic data by mixing real user data with noise
                base_user = np.random.choice(self.all_user_ids)
                base_data_slow = self.user_data[base_user]['slow'].copy()
                
                # Add noise to create unique pattern
                noise = np.random.normal(0, 0.2, size=base_data_slow.shape)
                synthetic_data_slow = base_data_slow + noise
                
                # Create fast game data similarly
                base_data_fast = self.user_data[base_user]['fast'].copy()
                noise = np.random.normal(0, 0.2, size=base_data_fast.shape)
                synthetic_data_fast = base_data_fast + noise
                
                # Store in user_data
                self.user_data[user_id] = {
                    'slow': synthetic_data_slow,
                    'fast': synthetic_data_fast
                }
        
        # Now test scalability at different user counts
        for num_users in range(step, min(len(all_users) + 1, max_users + 1), step):
            print(f"\nTesting with {num_users} users...")
            
            # Select subset of users
            user_subset = all_users[:num_users]
            
            # Test flat approach timing
            flat_train_times = []
            flat_auth_times = []
            
            start_time = time.time()
            
            # Simulate training time (simplified)
            # In a flat approach, each user model compares against all other users
            for user_id in user_subset[:min(10, len(user_subset))]:  # Limit to 10 users to save time
                # Time for one model training
                train_start = time.time()
                
                # Simulate model training (just calculating correlation to save time)
                user_data = self.user_data[user_id]['slow']
                user_data.corr()
                
                flat_train_times.append(time.time() - train_start)
            
            # Extrapolate training time for all users
            flat_total_train_time = sum(flat_train_times) / len(flat_train_times) * num_users
            
            # Simulate authentication time
            for _ in range(10):  # 10 authentication attempts
                auth_start = time.time()
                
                # In flat approach, just need to run one model
                user_id = np.random.choice(user_subset)
                user_data = self.user_data[user_id]['slow'].iloc[0:1]
                
                # But the model would be more complex as it distinguishes from more users
                # Add simulated complexity factor
                time.sleep(0.001 * (num_users / 100))  # Simulated increase in model complexity
                
                flat_auth_times.append(time.time() - auth_start)
            
            flat_avg_auth_time = sum(flat_auth_times) / len(flat_auth_times)
            
            # Test hierarchical approach timing
            hierarchy_train_times = []
            hierarchy_auth_times = []
            
            # Determine number of clusters based on user count
            num_clusters = max(2, int(np.sqrt(num_users)))
            
            # Simulate clustering time
            cluster_start = time.time()
            
            # Simulate clustering (just calculate centroids to save time)
            user_centroids = {}
            for user_id in user_subset:
                user_centroids[user_id] = self.user_data[user_id]['slow'].mean().values
            
            cluster_time = time.time() - cluster_start
            
            # Simulate model training time
            # First divide users into clusters
            cluster_sizes = []
            for i in range(num_clusters):
                # Roughly equal distribution among clusters
                start_idx = i * (num_users // num_clusters)
                end_idx = min(num_users, (i + 1) * (num_users // num_clusters))
                cluster_sizes.append(end_idx - start_idx)
            
            # For each cluster, train user models
            for cluster_size in cluster_sizes:
                # Time to train models for this cluster
                train_start = time.time()
                
                # Simulate model training for a few users in this cluster
                for _ in range(min(5, cluster_size)):
                    # Simpler models since comparing against fewer users
                    user_id = np.random.choice(user_subset)
                    user_data = self.user_data[user_id]['slow']
                    user_data.corr()
                
                # Extrapolate training time for this cluster
                cluster_train_time = (time.time() - train_start) / min(5, cluster_size) * cluster_size
                hierarchy_train_times.append(cluster_train_time)
            
            # Total training time includes clustering and model training
            hierarchy_total_train_time = cluster_time + sum(hierarchy_train_times)
            
            # Simulate authentication time
            for _ in range(10):  # 10 authentication attempts
                auth_start = time.time()
                
                # In hierarchical approach, first find cluster, then run model
                # Simulating this two-step process
                user_id = np.random.choice(user_subset)
                user_data = self.user_data[user_id]['slow'].iloc[0:1]
                
                # Simulate cluster identification
                time.sleep(0.001)  # Fixed time for cluster identification
                
                # Then run user model within cluster
                # Model complexity depends on cluster size, not total users
                avg_cluster_size = num_users / num_clusters
                time.sleep(0.001 * (avg_cluster_size / 100))  # Simulated model complexity
                
                hierarchy_auth_times.append(time.time() - auth_start)
            
            hierarchy_avg_auth_time = sum(hierarchy_auth_times) / len(hierarchy_auth_times)
            
            # Store results
            results.append({
                'Num_Users': num_users,
                'Num_Clusters': num_clusters,
                'Flat_Train_Time': flat_total_train_time,
                'Hierarchy_Train_Time': hierarchy_total_train_time,
                'Flat_Auth_Time': flat_avg_auth_time,
                'Hierarchy_Auth_Time': hierarchy_avg_auth_time,
                'Train_Speedup': flat_total_train_time / hierarchy_total_train_time if hierarchy_total_train_time > 0 else 0,
                'Auth_Speedup': flat_avg_auth_time / hierarchy_avg_auth_time if hierarchy_avg_auth_time > 0 else 0
            })
            
            print(f"Flat Approach - Training: {flat_total_train_time:.4f}s, Auth: {flat_avg_auth_time:.4f}s")
            print(f"Hierarchical Approach - Training: {hierarchy_total_train_time:.4f}s, Auth: {hierarchy_avg_auth_time:.4f}s")
            print(f"Speedup - Training: {flat_total_train_time / hierarchy_total_train_time:.2f}x, Auth: {flat_avg_auth_time / hierarchy_avg_auth_time:.2f}x")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Visualize scalability results
        plt.figure(figsize=(14, 10))
        
        # Plot training time comparison
        plt.subplot(2, 2, 1)
        plt.plot(results_df['Num_Users'], results_df['Flat_Train_Time'], marker='o', label='Flat')
        plt.plot(results_df['Num_Users'], results_df['Hierarchy_Train_Time'], marker='s', label='Hierarchical')
        plt.xlabel('Number of Users')
        plt.ylabel('Training Time (s)')
        plt.title('Training Time Scalability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot authentication time comparison
        plt.subplot(2, 2, 2)
        plt.plot(results_df['Num_Users'], results_df['Flat_Auth_Time'], marker='o', label='Flat')
        plt.plot(results_df['Num_Users'], results_df['Hierarchy_Auth_Time'], marker='s', label='Hierarchical')
        plt.xlabel('Number of Users')
        plt.ylabel('Authentication Time (s)')
        plt.title('Authentication Time Scalability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot speedup factors
        plt.subplot(2, 2, 3)
        plt.plot(results_df['Num_Users'], results_df['Train_Speedup'], marker='o', label='Training Speedup')
        plt.plot(results_df['Num_Users'], results_df['Auth_Speedup'], marker='s', label='Authentication Speedup')
        plt.xlabel('Number of Users')
        plt.ylabel('Speedup Factor (x)')
        plt.title('Speedup from Hierarchical Approach')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot number of clusters
        plt.subplot(2, 2, 4)
        plt.plot(results_df['Num_Users'], results_df['Num_Clusters'], marker='o')
        plt.xlabel('Number of Users')
        plt.ylabel('Number of Clusters')
        plt.title('Cluster Count vs. User Count')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return results_df

    def run_demo(self, game_type='slow'):
        """
        Run a complete demo of the hierarchical authentication system.
        
        Args:
            game_type: 'slow' or 'fast' indicating which game to demo
        """
        print("\n=== HIERARCHICAL AUTHENTICATION SYSTEM DEMO ===\n")
        
        # Step 1: Check if we have user data
        if not hasattr(self, 'user_data') or not self.user_data:
            print("Error: No user data available. Please prepare data first.")
            return
        
        # Step 2: Cluster users
        print("\nStep 1: Clustering Users...")
        self.cluster_users(game_type=game_type, visualize=True)
        
        # Step 3: Train hierarchical models
        print("\nStep 2: Training Hierarchical Models...")
        self.train_hierarchical_models(game_type=game_type)
        
        # Step 4: Test authentication
        print("\nStep 3: Testing Authentication Performance...")
        test_results = self.test_hierarchical_authentication(game_type=game_type)
        
        # Step 5: Visualize performance
        print("\nStep 4: Visualizing Performance Metrics...")
        self.visualize_performance()
        
        # Step 6: Compare to flat model
        print("\nStep 5: Comparing to Flat Authentication Model...")
        comparison = self.compare_to_flat_model(game_type=game_type)
        
        # Step 7: Evaluate scalability
        print("\nStep 6: Evaluating Scalability...")
        scalability = self.evaluate_scalability(max_users=min(100, len(self.all_user_ids)*2))
        
        print("\n=== DEMO COMPLETED ===\n")
        
        # Return all results
        return {
            'clusters': self.clusters[game_type],
            'performance': self.performance_metrics,
            'test_results': test_results,
            'comparison': comparison,
            'scalability': scalability
        }