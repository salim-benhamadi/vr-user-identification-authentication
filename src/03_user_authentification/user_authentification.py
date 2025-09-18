import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplot2tikz as tikzplotlib
matplotlib.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 12})
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from tqdm.notebook import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

fig_width = 14
fig_height = 8
SEED = 42

def prepare_auth_data(mov_forklift_simulator, mov_beat_saber, traffic_forklift_simulator, traffic_beat_saber, n_intruders=60, include_traffic=False):
    """Prepare data for authentication systems."""
    
    data_forklift_simulator = mov_forklift_simulator.copy()
    data_beat_saber = mov_beat_saber.copy()
    
    if include_traffic:
        traffic_forklift = traffic_forklift_simulator.copy()
        traffic_beat_saber = traffic_beat_saber.copy()
        
        merged_forklift = []
        merged_beat_saber = []
        
        for user_id in data_forklift_simulator['ID'].unique():
            user_mov = data_forklift_simulator[data_forklift_simulator['ID'] == user_id]
            user_traffic = traffic_forklift[traffic_forklift['ID'] == user_id]
            
            mov_features = user_mov.drop(columns=['ID']).reset_index(drop=True)
            traffic_features = user_traffic.drop(columns=['ID']).reset_index(drop=True)
            combined = pd.concat([mov_features, traffic_features], axis=1)
            combined['ID'] = user_id
            
            merged_forklift.append(combined)
        
        for user_id in data_beat_saber['ID'].unique():
            user_mov = data_beat_saber[data_beat_saber['ID'] == user_id]
            user_traffic = traffic_beat_saber[traffic_beat_saber['ID'] == user_id]
            
            mov_features = user_mov.drop(columns=['ID']).reset_index(drop=True)
            traffic_features = user_traffic.drop(columns=['ID']).reset_index(drop=True)
                
            combined = pd.concat([mov_features, traffic_features], axis=1)
            combined['ID'] = user_id
            
            merged_beat_saber.append(combined)

        if merged_forklift:
            data_forklift_simulator = pd.concat(merged_forklift, ignore_index=True)
        
        if merged_beat_saber:
            data_beat_saber = pd.concat(merged_beat_saber, ignore_index=True)
    
    forklift_user_ids = np.array(sorted(data_forklift_simulator['ID'].unique()))
    beat_saber_user_ids = np.array(sorted(data_beat_saber['ID'].unique()))
    
    forklift_datasets = {}
    beat_saber_datasets = {}
    
    for user_id in forklift_user_ids:
        user_data = data_forklift_simulator[data_forklift_simulator['ID'] == user_id].drop(columns=['ID'])
        
        genuine_samples = user_data.copy()
        genuine_samples['is_genuine'] = 1
        genuine_samples['original_id'] = user_id
        
        impostor_ids = np.array([id for id in forklift_user_ids if id != user_id])
        impostor_data = data_forklift_simulator[data_forklift_simulator['ID'] != user_id].sample(
            n=n_intruders, 
            replace=True if len(data_forklift_simulator[data_forklift_simulator['ID'] != user_id]) < len(genuine_samples) else False
        ).drop(columns=['ID'])
        
        impostor_data['is_genuine'] = 0
        impostor_data['original_id'] = np.random.choice(impostor_ids, size=len(impostor_data))
        
        user_data_combined = pd.concat([genuine_samples, impostor_data], ignore_index=True)
        
        forklift_datasets[user_id] = user_data_combined
    
    for user_id in beat_saber_user_ids:
        user_data = data_beat_saber[data_beat_saber['ID'] == user_id].drop(columns=['ID'])
        
        genuine_samples = user_data.copy()
        genuine_samples['is_genuine'] = 1
        genuine_samples['original_id'] = user_id
        
        impostor_ids = np.array([id for id in beat_saber_user_ids if id != user_id])
        impostor_data = data_beat_saber[data_beat_saber['ID'] != user_id].sample(
            n=n_intruders, 
            replace=True if len(data_beat_saber[data_beat_saber['ID'] != user_id]) < len(genuine_samples) else False
        ).drop(columns=['ID'])
        
        impostor_data['is_genuine'] = 0
        impostor_data['original_id'] = np.random.choice(impostor_ids, size=len(impostor_data))
        
        user_data_combined = pd.concat([genuine_samples, impostor_data], ignore_index=True)
        
        beat_saber_datasets[user_id] = user_data_combined
    
    return {
        'forklift_simulator': {
            'datasets': forklift_datasets,
            'user_ids': forklift_user_ids
        },
        'beat_saber': {
            'datasets': beat_saber_datasets,
            'user_ids': beat_saber_user_ids
        }
    }

def apply_smote_oversampling(data):
    """Apply SMOTE oversampling to balance classes."""
    X = data.drop(columns=['is_genuine', 'original_id', 'time_interval'])
    y = data['is_genuine']
    
    original_class_dist = np.bincount(y)
    minority_class = np.argmin(original_class_dist)
    majority_class = np.argmax(original_class_dist)
    
    minority_count = original_class_dist[minority_class]
    majority_count = original_class_dist[majority_class]
    
    try:
        smote = SMOTE(random_state=SEED)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        new_class_dist = np.bincount(y_resampled)
        new_sample_count = new_class_dist[minority_class]
        added_samples = new_sample_count - minority_count
        
        resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
        resampled_data['is_genuine'] = y_resampled
        resampled_data['original_id'] = data['original_id'].reset_index(drop=True).iloc[:len(resampled_data)]
        
        return resampled_data
    except Exception as e:
        return data

class GameAuthenticator:
    """Main authentication system class."""
    
    def __init__(self, authentication_data):
        self.forklift_datasets = authentication_data['forklift_simulator']['datasets']
        self.forklift_user_ids = authentication_data['forklift_simulator']['user_ids']
        
        self.beat_saber_datasets = authentication_data['beat_saber']['datasets']
        self.beat_saber_user_ids = authentication_data['beat_saber']['user_ids']
        
        self.forklift_models = {}
        self.beat_saber_models = {}
        self.forklift_test_sets = {}
        self.beat_saber_test_sets = {}
        
        self.performance_metrics = pd.DataFrame(columns=['User_ID', 'Game_Type', 'Model_Type', 'Accuracy', 'Precision', 
                                                        'Recall', 'F1', 'AUC', 'FAR', 'FRR'])
        self.best_models = {'forklift_simulator': {}, 'beat_saber': {}}
    
    def preprocess_data(self, data):
        """Preprocess data for training/testing."""
        processed_data = data.copy()
            
        X = processed_data.drop(columns=['time_interval','is_genuine', 'original_id'])
        y = processed_data['is_genuine']
        
        return X, y
    
    def train_multiple_models(self, filename="normal", test_size=0.2, use_time_intervals=True, compare_models=True, apply_smote=False):
        """Train and compare multiple authentication models."""
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=SEED),
            'LogisticRegression': LogisticRegression(random_state=SEED, max_iter=1000),
            'SVM': SVC(probability=True, random_state=SEED),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'GradientBoosting': GradientBoostingClassifier(random_state=SEED),
            'AdaBoost': AdaBoostClassifier(random_state=SEED),
            'GaussianNB': GaussianNB(),
            'DecisionTree': DecisionTreeClassifier(random_state=SEED),
            'MLP': MLPClassifier(hidden_layer_sizes=(100,), random_state=SEED, max_iter=1000),
            'XGBoost': XGBClassifier(random_state=SEED)
        }
        
        self._train_game_models(
            self.forklift_datasets, 
            self.forklift_user_ids, 
            self.forklift_models, 
            'forklift_simulator', 
            test_size, 
            use_time_intervals,
            compare_models,
            models,
            apply_smote=apply_smote,
            filename=filename
        )
        
        self._train_game_models(
            self.beat_saber_datasets, 
            self.beat_saber_user_ids, 
            self.beat_saber_models, 
            'beat_saber', 
            test_size, 
            use_time_intervals,
            compare_models,
            models,
            apply_smote=apply_smote,
            filename=filename
        )
        
        self._save_metrics_to_csv(filename)
        
    def _train_game_models(self, game_datasets, user_ids, model_dict, game_type, test_size, use_time_intervals, compare_models, models, apply_smote=False, filename="normal"):
        """Train models for a specific game."""
        models_dir = os.path.join('../../models', 'user_authentification', 'training_evaluation', game_type)
        os.makedirs(models_dir, exist_ok=True)
        
        dataset_metrics = []
        
        for user_id in tqdm(user_ids):
            model_dict[user_id] = {}
            self.best_models[game_type][user_id] = None
            
            user_models_dir = os.path.join(models_dir, f'user_{user_id}')
            os.makedirs(user_models_dir, exist_ok=True)
            
            data = game_datasets[user_id]
            
            all_intruder_ids = np.array([id for id in user_ids if id != user_id])
            np.random.seed(SEED)
            test_intruders_count = int(len(all_intruder_ids) * 0.1)
            test_only_intruders = np.random.choice(all_intruder_ids, size=test_intruders_count, replace=False)
            train_intruders = np.array([id for id in all_intruder_ids if id not in test_only_intruders])
            
            if 'time_interval' in data.columns and use_time_intervals:
                time_intervals = data['time_interval'].unique()
                train_intervals = time_intervals[:int(len(time_intervals) * (1-test_size))]
                test_intervals = time_intervals[int(len(time_intervals) * (1-test_size)):]
                
                train_data = data[data['time_interval'].isin(train_intervals)]
                test_data = data[data['time_interval'].isin(test_intervals)]
            else:
                genuine_data = data[data['is_genuine'] == 1]
                train_genuine, test_genuine = train_test_split(genuine_data, test_size=test_size, random_state=SEED, shuffle=True)
                
                impostor_data = data[data['is_genuine'] == 0]
                test_impostor = impostor_data[impostor_data['original_id'].isin(test_only_intruders)]
                
                impostor_from_train_ids = impostor_data[impostor_data['original_id'].isin(train_intruders)]
                
                if len(impostor_from_train_ids) > 0:
                    train_impostor, extra_test_from_train = train_test_split(
                        impostor_from_train_ids, test_size=0.1, random_state=SEED
                    )
                else:
                    train_impostor = pd.DataFrame()
                    extra_test_from_train = pd.DataFrame()
                
                train_data = pd.concat([train_genuine, train_impostor])
                test_data = pd.concat([test_genuine, test_impostor, extra_test_from_train])
                
                dataset_info = {
                    'User_ID': user_id,
                    'Game_Type': game_type,
                    'Train_Size': len(train_data),
                    'Train_Genuine': len(train_genuine),
                    'Train_Impostor': len(train_impostor),
                    'Test_Size': len(test_data),
                    'Test_Genuine': len(test_genuine),
                    'Test_Impostor_New_IDs': len(test_impostor),
                    'Test_Impostor_Train_IDs': len(extra_test_from_train)
                }
            
            if game_type == 'forklift_simulator':
                self.forklift_test_sets[user_id] = test_data
            else:
                self.beat_saber_test_sets[user_id] = test_data
                            
            X_train, y_train = self.preprocess_data(train_data)
            X_test, y_test = self.preprocess_data(test_data)
            
            genuine_count_before = sum(y_train == 1)
            impostor_count_before = sum(y_train == 0)
            dataset_info['Train_Genuine_Before_SMOTE'] = genuine_count_before
            dataset_info['Train_Impostor_Before_SMOTE'] = impostor_count_before
            
            if apply_smote:
                try:
                    smote = SMOTE(random_state=SEED)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
                    y_train = pd.Series(y_train_resampled)
                    
                    genuine_count_after = sum(y_train == 1)
                    impostor_count_after = sum(y_train == 0)
                    dataset_info['Train_Genuine_After_SMOTE'] = genuine_count_after
                    dataset_info['Train_Impostor_After_SMOTE'] = impostor_count_after
                    dataset_info['SMOTE_Applied'] = True
                    
                except Exception as e:
                    dataset_info['Train_Genuine_After_SMOTE'] = genuine_count_before
                    dataset_info['Train_Impostor_After_SMOTE'] = impostor_count_before
                    dataset_info['SMOTE_Applied'] = False
                    dataset_info['SMOTE_Error'] = str(e)
            else:
                dataset_info['Train_Genuine_After_SMOTE'] = genuine_count_before
                dataset_info['Train_Impostor_After_SMOTE'] = impostor_count_before
                dataset_info['SMOTE_Applied'] = False
                
            dataset_metrics.append(dataset_info)
            
            if not compare_models:
                models_to_train = {'RandomForest': models['RandomForest']}
            else:
                models_to_train = models
            
            best_score = 0
            best_model_name = None
            
            for model_name, model in models_to_train.items():
                if len(X_train) < 10:
                    continue
                    
                model.fit(X_train, y_train)
                
                model_dict[user_id][model_name] = model
                
                model_path = os.path.join(user_models_dir, f'{model_name}_{filename}.joblib')
                try:
                    joblib.dump(model, model_path)
                except Exception as e:
                    pass
                
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
                far = fp / (fp + tn) if (fp + tn) > 0 else 0  
                frr = fn / (fn + tp) if (fn + tp) > 0 else 0  
                
                auc_value = None
                if y_prob is not None:
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    auc_value = auc(fpr, tpr)
                
                self.performance_metrics = pd.concat([
                    self.performance_metrics,
                    pd.DataFrame({
                        'User_ID': [user_id],
                        'Game_Type': [game_type],
                        'Model_Type': [model_name],
                        'Accuracy': [accuracy],
                        'Precision': [precision],
                        'Recall': [recall],
                        'F1': [f1],
                        'AUC': [auc_value],
                        'FAR': [far],
                        'FRR': [frr]
                    })
                ], ignore_index=True)
                
                if f1 > best_score:
                    best_score = f1
                    best_model_name = model_name
            
            if best_model_name:
                self.best_models[game_type][user_id] = best_model_name
                
        if dataset_metrics:
            results_dir = os.path.join('../../results', 'user_authentification', 'training_evaluation')
            os.makedirs(results_dir, exist_ok=True)
            
            dataset_metrics_df = pd.DataFrame(dataset_metrics)
            dataset_metrics_path = os.path.join(results_dir, f'dataset_metrics_{game_type}_{filename}.csv')
            dataset_metrics_df.to_csv(dataset_metrics_path, index=False)

    def _save_metrics_to_csv(self, filname=""):
        """Save model performance metrics to CSV."""
        
        results_dir = os.path.join('../../results', 'user_authentification', 'training_evaluation')
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, f'model_metrics_{filname}.csv')
        
        self.performance_metrics.to_csv(file_path, index=False)
        
        best_models_data = []
        for game_type in self.best_models:
            for user_id, model_name in self.best_models[game_type].items():
                if model_name: 
                    user_metrics = self.performance_metrics[
                        (self.performance_metrics['User_ID'] == user_id) & 
                        (self.performance_metrics['Game_Type'] == game_type) & 
                        (self.performance_metrics['Model_Type'] == model_name)
                    ]
                    
                    if not user_metrics.empty:
                        best_models_data.append({
                            'User_ID': user_id,
                            'Game_Type': game_type,
                            'Best_Model': model_name,
                            'Accuracy': user_metrics['Accuracy'].values[0],
                            'Precision': user_metrics['Precision'].values[0],
                            'Recall': user_metrics['Recall'].values[0],
                            'F1': user_metrics['F1'].values[0],
                            'AUC': user_metrics['AUC'].values[0],
                            'FAR': user_metrics['FAR'].values[0],
                            'FRR': user_metrics['FRR'].values[0]
                        })
        
        if best_models_data:
            best_models_df = pd.DataFrame(best_models_data)
            best_models_file_path = os.path.join(results_dir, f'best_models_{filname}.csv')
            best_models_df.to_csv(best_models_file_path, index=False)

    def get_performance_summary(self):
        """Get performance summary."""
        return self.performance_metrics
    
    def visualize_performance(self, sample_user_forklift=None, sample_user_beat_saber=None, model_name=None, filename=""):
        """Visualize performance metrics and save as TeX."""

        results_dir = os.path.join('../../results', 'user_authentification', 'training_evaluation')
        os.makedirs(results_dir, exist_ok=True)
        
        if not filename:
            from datetime import datetime
            filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if model_name is not None:
            filtered_metrics = self.performance_metrics[self.performance_metrics['Model_Type'] == model_name]
            if filtered_metrics.empty:
                return
            metrics_to_use = filtered_metrics
            model_title = f" ({model_name} Model)"
        else:
            metrics_to_use = self.performance_metrics
            model_title = ""
        
        avg_metrics = metrics_to_use.groupby(['Game_Type', 'Model_Type']).mean().reset_index()
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
        
        plt.figure(figsize=(fig_width, fig_height))
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        if model_name is not None:
            model_metrics = avg_metrics[avg_metrics['Model_Type'] == model_name]
        else:
            model_metrics = avg_metrics[avg_metrics['Model_Type'] == 'RandomForest']
            if len(model_metrics) == 0:
                model_metrics = avg_metrics.groupby('Game_Type').first().reset_index()
        
        forklift_metrics = model_metrics[model_metrics['Game_Type'] == 'forklift_simulator'][metrics_to_plot].values[0] if 'forklift_simulator' in model_metrics['Game_Type'].values else np.zeros(len(metrics_to_plot))
        beat_saber_metrics = model_metrics[model_metrics['Game_Type'] == 'beat_saber'][metrics_to_plot].values[0] if 'beat_saber' in model_metrics['Game_Type'].values else np.zeros(len(metrics_to_plot))
        
        plt.bar(x - width/2, forklift_metrics, width, label='Forklift Simulator')
        plt.bar(x + width/2, beat_saber_metrics, width, label='Beat Saber')
        
        plt.ylabel('Score')
        plt.title(f'Average Authentication Performance Metrics')
        plt.xticks(x, metrics_to_plot)
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        tikzplotlib.save(os.path.join(results_dir, f'avg_performance_metrics_{filename}.tex'))
        plt.savefig(os.path.join(results_dir, f'avg_performance_metrics_{filename}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(fig_width, fig_height))
        
        for game_type, color in zip(['forklift_simulator', 'beat_saber'], ['dodgerblue', 'orange']):
            game_data = metrics_to_use[
                (metrics_to_use['Game_Type'] == game_type) & 
                (metrics_to_use['Model_Type'] == model_name if model_name else True)
            ]
            if not game_data.empty:
                plt.hist(game_data['Accuracy'], alpha=0.5, label=f'{game_type.capitalize()}', color=color, bins=10)
        
        plt.xlabel('Accuracy')
        plt.ylabel('Number of Users')
        plt.title(f'Distribution of Authentication Accuracy Across Users')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        tikzplotlib.save(os.path.join(results_dir, f'accuracy_distribution_{filename}.tex'))
        plt.savefig(os.path.join(results_dir, f'accuracy_distribution_{filename}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(fig_width, fig_height))
        
        for game_type, color, marker in zip(['forklift_simulator', 'beat_saber'], ['dodgerblue', 'orange'], ['o', 's']):
            game_data = metrics_to_use[
                (metrics_to_use['Game_Type'] == game_type) & 
                (metrics_to_use['Model_Type'] == model_name if model_name else True)
            ]
            if not game_data.empty:
                plt.scatter(game_data['FAR'], game_data['FRR'], alpha=0.7, label=f'{game_type.capitalize()}', 
                        color=color, marker=marker, s=80)
        
        plt.plot([0, 1], [1, 0], 'k--', alpha=0.3)
        plt.xlabel('False Acceptance Rate (FAR)')
        plt.ylabel('False Rejection Rate (FRR)')
        plt.title(f'FAR vs FRR Trade-off Across Users')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        tikzplotlib.save(os.path.join(results_dir, f'far_frr_tradeoff_{filename}.tex'))
        plt.savefig(os.path.join(results_dir, f'far_frr_tradeoff_{filename}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        rf_model_name = 'RandomForest'
        rf_metrics = self.performance_metrics[self.performance_metrics['Model_Type'] == rf_model_name]

        if not rf_metrics.empty:
            forklift_rf = rf_metrics[rf_metrics['Game_Type'] == 'forklift_simulator'].sort_values('User_ID')
            beatsaber_rf = rf_metrics[rf_metrics['Game_Type'] == 'beat_saber'].sort_values('User_ID')
            
            if len(forklift_rf) > 0:
                plt.figure(figsize=(fig_width, fig_height))
                user_ids = forklift_rf['User_ID'].astype(int).tolist()
                
                forklift_bars = plt.bar(range(len(forklift_rf)), forklift_rf['Accuracy'], 
                                    color='dodgerblue', label='Forklift Simulator Users', width=0.6)
                
                plt.xticks(range(len(forklift_rf)), user_ids)
                plt.xlabel('User ID')
                plt.ylabel('Accuracy')
                plt.title(f'Forklift Simulator - Model Accuracy Per User')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1.1)
                
                plt.tight_layout()
                tikzplotlib.save(os.path.join(results_dir, f'forklift_user_accuracy_{filename}.tex'))
                plt.savefig(os.path.join(results_dir, f'forklift_user_accuracy_{filename}.png'), dpi=300, bbox_inches='tight')
                plt.show()
            
            if len(beatsaber_rf) > 0:
                plt.figure(figsize=(fig_width, fig_height))
                user_ids = beatsaber_rf['User_ID'].astype(int).tolist()
                
                beatsaber_bars = plt.bar(range(len(beatsaber_rf)), beatsaber_rf['Accuracy'], 
                                    color='orangered', label='Beat Saber Users', width=0.6)
                
                plt.xticks(range(len(beatsaber_rf)), user_ids)
                plt.xlabel('User ID')
                plt.ylabel('Accuracy')
                plt.title(f'Beat Saber - Model Accuracy Per User')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1.1)
                
                plt.tight_layout()
                tikzplotlib.save(os.path.join(results_dir, f'beatsaber_user_accuracy_{filename}.tex'))
                plt.savefig(os.path.join(results_dir, f'beatsaber_user_accuracy_{filename}.png'), dpi=300, bbox_inches='tight')
                plt.show()
            
    def authenticate_user(self, user_id, gameplay_data, game_type, filename="normal", model_name="RandomForest"):
        """Authenticate a user based on gameplay data."""
        
        model_path = os.path.join('../../models', 'user_authentification', 'training_evaluation', 
                                game_type, f'user_{user_id}', 
                                f'{model_name}_{filename}.joblib')
        
        if not os.path.exists(model_path):
            return False, 0.0
        
        try:
            model = joblib.load(model_path)
        except Exception as e:
            return False, 0.0
        
        if hasattr(gameplay_data, 'to_frame') and not hasattr(gameplay_data, 'columns'):
            gameplay_data = gameplay_data.to_frame().T
        
        if hasattr(gameplay_data, 'columns'):
            if 'time_interval' in gameplay_data.columns:
                gameplay_data = gameplay_data.drop(columns=['time_interval'])
            
            if 'is_genuine' in gameplay_data.columns:
                gameplay_data = gameplay_data.drop(columns=['is_genuine'])
            
            if 'original_id' in gameplay_data.columns:
                gameplay_data = gameplay_data.drop(columns=['original_id'])
        
        if hasattr(gameplay_data, 'values'):
            predict_data = gameplay_data.values
        else:
            predict_data = gameplay_data
        
        if len(predict_data.shape) == 1:
            predict_data = predict_data.reshape(1, -1)
        
        predict_data = np.nan_to_num(predict_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        prediction = model.predict(predict_data)[0]
        confidence = model.predict_proba(predict_data)[0][1] if hasattr(model, "predict_proba") else 0.5
        
        is_genuine = bool(prediction)
        
        return is_genuine, confidence   
    
    
    def threshold_analysis(self, user_id=None, game_type='forklift_simulator', model_name="RandomForest", filename="normal"):
        """Analyze authentication performance at different thresholds."""
        
        results_dir = os.path.join('../../results', 'user_authentification', 'training_evaluation')
        os.makedirs(results_dir, exist_ok=True)
        
        test_sets = self.forklift_test_sets if game_type == 'forklift_simulator' else self.beat_saber_test_sets
        user_ids = self.forklift_user_ids if game_type == 'forklift_simulator' else self.beat_saber_user_ids
        
        if user_id is not None:
            if user_id not in test_sets:
                return
            
            model_path = os.path.join('../../models', 'user_authentification', 'training_evaluation', 
                                    game_type, f'user_{user_id}', 
                                    f'{model_name}_{filename}.joblib')
            
            if not os.path.exists(model_path):
                return
            
            try:
                model = joblib.load(model_path)
            except Exception as e:
                return
            
            test_data = test_sets[user_id]
            X_test, y_test = self.preprocess_data(test_data)
            
            if not hasattr(model, "predict_proba"):
                return
                
            y_prob = model.predict_proba(X_test)[:, 1]
            
            precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)
            fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
            
            thresholds = np.linspace(0, 1, 11)
            threshold_results = []
            
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
                far = fp / (fp + tn) if (fp + tn) > 0 else 0
                frr = fn / (fn + tp) if (fn + tp) > 0 else 0
                
                threshold_results.append({
                    'Threshold': threshold,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1': f1,
                    'FAR': far,
                    'FRR': frr
                })
            
            threshold_df = pd.DataFrame(threshold_results)
            
            threshold_csv_path = os.path.join(results_dir, f'threshold_analysis_{game_type}_user{user_id}_{model_name}_{filename}.csv')
            threshold_df.to_csv(threshold_csv_path, index=False)
            
            model_info = f" ({model_name} Model)"
            
            plt.figure(figsize=(fig_width, fig_height))
            for metric in ['Precision', 'Recall', 'F1', 'Accuracy']:
                plt.plot(threshold_df['Threshold'], threshold_df[metric], marker='o', label=metric)
            plt.xlabel('Threshold')
            plt.ylabel('Metric Value')
            plt.title(f'Performance Metrics vs. Threshold - User {user_id}, {game_type.capitalize()} Game{model_info}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            tikzplotlib.save(os.path.join(results_dir, f'threshold_metrics_{game_type}_user{user_id}_{model_name}_{filename}.tex'))
            plt.savefig(os.path.join(results_dir, f'threshold_metrics_{game_type}_user{user_id}_{model_name}_{filename}.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(fig_width, fig_height))
            plt.plot(threshold_df['Threshold'], threshold_df['FAR'], marker='o', label='FAR', color='orangered')
            plt.plot(threshold_df['Threshold'], threshold_df['FRR'], marker='o', label='FRR', color='dodgerblue')
            
            diffs = abs(threshold_df['FAR'] - threshold_df['FRR'])
            eer_idx = diffs.idxmin()
            eer_threshold = threshold_df.loc[eer_idx, 'Threshold']
            eer_rate = (threshold_df.loc[eer_idx, 'FAR'] + threshold_df.loc[eer_idx, 'FRR']) / 2
            
            plt.plot(eer_threshold, eer_rate, 'ko', markersize=10)
            plt.annotate(f'EER ~= {eer_rate:.3f}\nThreshold = {eer_threshold:.2f}', 
                        xy=(eer_threshold, eer_rate), xytext=(eer_threshold + 0.1, eer_rate + 0.1),
                        arrowprops=dict(arrowstyle='->'))
            
            plt.xlabel('Threshold')
            plt.ylabel('Rate')
            plt.title(f'FAR and FRR vs. Threshold - User {user_id}, {game_type.capitalize()} Game{model_info}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            tikzplotlib.save(os.path.join(results_dir, f'threshold_far_frr_{game_type}_user{user_id}_{model_name}_{filename}.tex'))
            plt.savefig(os.path.join(results_dir, f'threshold_far_frr_{game_type}_user{user_id}_{model_name}_{filename}.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(fig_width, fig_height))
            plt.plot(fpr, tpr, lw=2)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
            roc_auc = auc(fpr, tpr)
            plt.annotate(f'AUC = {roc_auc:.3f}', xy=(0.6, 0.2))
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - User {user_id}, {game_type.capitalize()} Game{model_info}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            tikzplotlib.save(os.path.join(results_dir, f'threshold_roc_{game_type}_user{user_id}_{model_name}_{filename}.tex'))
            plt.savefig(os.path.join(results_dir, f'threshold_roc_{game_type}_user{user_id}_{model_name}_{filename}.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
            return threshold_df

    def visualize_feature_importance(self, user_id=None, game_type='forklift_simulator', top_n=10, 
                        model_name="RandomForest", use_shap=False, filename="normal"):
        """Visualize feature importance for models."""
        
        results_dir = os.path.join('../../results', 'user_authentification', 'training_evaluation')
        os.makedirs(results_dir, exist_ok=True)
        
        test_sets = self.forklift_test_sets if game_type == 'forklift_simulator' else self.beat_saber_test_sets
        game_user_ids = self.forklift_user_ids if game_type == 'forklift_simulator' else self.beat_saber_user_ids
        
        if user_id is not None:
            model_path = os.path.join('../../models', 'user_authentification', 'training_evaluation', 
                                    game_type, f'user_{user_id}', 
                                    f'{model_name}_{filename}.joblib')
            
            if not os.path.exists(model_path):
                return
            
            try:
                model = joblib.load(model_path)
            except Exception as e:
                return
            
            if user_id not in test_sets:
                return
            
            test_data = test_sets[user_id]
            X_test, y_test = self.preprocess_data(test_data)
            feature_names = X_test.columns
            
            plt.figure(figsize=(fig_width, fig_height))
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_n = min(top_n, len(importances))
                indices = np.argsort(importances)[-top_n:]
                
                plt.barh(range(top_n), importances[indices])
                plt.yticks(range(top_n), [feature_names[i] for i in indices])
                plt.title(f'Feature Importance for User {user_id} ({game_type.capitalize()} Game, {model_name})')
                plt.xlabel('Importance')
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                importance_path = os.path.join(results_dir, f'feature_importance_{game_type}_user{user_id}_{model_name}_{filename}.csv')
                importance_df.to_csv(importance_path, index=False)
                
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
                top_n = min(top_n, len(importances))
                indices = np.argsort(importances)[-top_n:]
                
                plt.barh(range(top_n), importances[indices])
                plt.yticks(range(top_n), [feature_names[i] for i in indices])
                plt.title(f'Coefficient Magnitude for User {user_id} ({game_type.capitalize()} Game, {model_name})')
                plt.xlabel('Coefficient Magnitude')
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': importances
                }).sort_values('Coefficient', ascending=False)
                
                importance_path = os.path.join(results_dir, f'feature_coef_{game_type}_user{user_id}_{model_name}_{filename}.csv')
                importance_df.to_csv(importance_path, index=False)
                
            else:
                return
                
            plt.tight_layout()
            tikzplotlib.save(os.path.join(results_dir, f'feature_plot_{game_type}_user{user_id}_{model_name}_{filename}.tex'))
            plt.savefig(os.path.join(results_dir, f'feature_plot_{game_type}_user{user_id}_{model_name}_{filename}.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
            return importance_df
                
        else:
            if len(game_user_ids) == 0:
                return
            
            sample_user = game_user_ids[0]
            if sample_user not in test_sets:
                return
                
            sample_data = test_sets[sample_user]
            X_sample, _ = self.preprocess_data(sample_data)
            feature_names = X_sample.columns
            
            all_importances = {name: 0 for name in feature_names}
            count = 0
            
            for user_id in game_user_ids:
                model_path = os.path.join('../../models', 'user_authentification', 'training_evaluation', 
                                        game_type, f'user_{user_id}', 
                                        f'{model_name}_{filename}.joblib')
                
                if not os.path.exists(model_path):
                    continue
                
                try:
                    model = joblib.load(model_path)
                except Exception as e:
                    continue
                
                if hasattr(model, 'feature_importances_'):
                    for i, name in enumerate(feature_names):
                        if i < len(model.feature_importances_):
                            all_importances[name] += model.feature_importances_[i]
                    count += 1
            
            if count > 0:
                for name in all_importances:
                    all_importances[name] /= count
                
                plt.figure(figsize=(fig_width, fig_height))
                sorted_features = sorted(all_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
                features, importances = zip(*sorted_features)
                
                plt.barh(range(len(features)), importances)
                plt.yticks(range(len(features)), features)
                
                model_info = f" ({model_name} Model)"
                plt.title(f'Average Feature Importance Across All Users ({game_type.capitalize()} Game){model_info}')
                plt.xlabel('Average Importance')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                agg_importance_df = pd.DataFrame({
                    'Feature': all_importances.keys(),
                    'Average_Importance': all_importances.values()
                }).sort_values('Average_Importance', ascending=False)
                
                agg_importance_path = os.path.join(results_dir, f'feature_importance_agg_{game_type}_{model_name}_{filename}.csv')
                agg_importance_df.to_csv(agg_importance_path, index=False)
                
                tikzplotlib.save(os.path.join(results_dir, f'feature_plot_agg_{game_type}_{model_name}_{filename}.tex'))
                plt.savefig(os.path.join(results_dir, f'feature_plot_agg_{game_type}_{model_name}_{filename}.png'), dpi=300, bbox_inches='tight')
                plt.show()
                
                return agg_importance_df

    def authenticate_user_with_voting(self, user_id, gameplay_data_list, game_type, voting_threshold=0.5, 
                               filename="normal", model_name="RandomForest"):
        """Authenticate using voting across multiple samples."""
        
        if game_type == 'forklift_simulator':
            test_sets = self.forklift_test_sets if hasattr(self, 'forklift_test_sets') else {}
        else:
            test_sets = self.beat_saber_test_sets if hasattr(self, 'beat_saber_test_sets') else {}
        
        if user_id not in test_sets:
            return False, 0.0, {"genuine_votes": 0, "impostor_votes": 0, "genuine_percent": 0}
        
        sample_data = test_sets[user_id]
        train_features = sample_data.drop(columns=['is_genuine', 'original_id', 'time_interval']).columns
        
        processed_samples = []
        
        if hasattr(gameplay_data_list, 'iterrows'):
            samples = [row for _, row in gameplay_data_list.iterrows()]
        else:
            samples = gameplay_data_list if isinstance(gameplay_data_list, list) else [gameplay_data_list]
        
        for sample in samples:
            processed_sample = pd.Series(0, index=train_features)
            
            if hasattr(sample, 'index'):
                for feature in sample.index:
                    if feature in train_features:
                        processed_sample[feature] = sample[feature]
            elif hasattr(sample, 'columns'):
                for feature in sample.columns:
                    if feature in train_features:
                        processed_sample[feature] = sample[feature].iloc[0] if hasattr(sample[feature], 'iloc') else sample[feature]
            
            processed_samples.append(processed_sample)
        
        genuine_votes = 0
        impostor_votes = 0
        confidence_scores = []
        
        for sample in processed_samples:
            is_genuine, confidence = self.authenticate_user(
                user_id, sample, game_type, filename=filename, model_name=model_name
            )
            
            confidence_scores.append(confidence if confidence is not None else 0.5)
            
            if is_genuine:
                genuine_votes += 1
            else:
                impostor_votes += 1
        
        total_votes = genuine_votes + impostor_votes
        genuine_percent = genuine_votes / total_votes if total_votes > 0 else 0
        
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        final_is_genuine = genuine_percent >= voting_threshold
        
        voting_stats = {
            "genuine_votes": genuine_votes,
            "impostor_votes": impostor_votes,
            "total_votes": total_votes,
            "genuine_percent": genuine_percent,
            "individual_confidences": confidence_scores
        }
        
        return final_is_genuine, overall_confidence, voting_stats

    def analyze_voting_sample_effect_by_user(self, max_samples=None, voting_threshold=0.5, model_name="RandomForest", filename="initial"):
        results_dir = os.path.join('../../results', 'user_authentification', 'training_evaluation')
        os.makedirs(results_dir, exist_ok=True)
        
        output_prefix = f"voting_effect_{filename}_{model_name}"
        
        results = []
        user_results_by_game = {'forklift_simulator': {}, 'beat_saber': {}}
        
        for game_type in ['forklift_simulator', 'beat_saber']:
            
            if game_type == 'forklift_simulator':
                game_user_ids = self.forklift_user_ids
                game_test_sets = self.forklift_test_sets
            else:
                game_user_ids = self.beat_saber_user_ids
                game_test_sets = self.beat_saber_test_sets
            
            for user_id in game_user_ids:
                if user_id not in game_test_sets:
                    continue
                    
                model_path = os.path.join('../../models', 'user_authentification', 'training_evaluation', 
                                        game_type, f'user_{user_id}', 
                                        f'{model_name}_{filename}.joblib')
                
                if not os.path.exists(model_path):
                    continue
                
                test_data = game_test_sets[user_id]
                
                try:
                    loaded_model = joblib.load(model_path)
                except Exception as e:
                    continue
                
                genuine_test = test_data[test_data['is_genuine'] == 1].reset_index(drop=True)
                
                impostor_test_df = test_data[test_data['is_genuine'] == 0]
                impostor_ids = impostor_test_df['original_id'].unique()
                
                if len(genuine_test) < 3:
                    continue
                    
                if len(impostor_ids) == 0:
                    continue
                
                if game_type == 'forklift_simulator':
                    if user_id not in self.forklift_models:
                        self.forklift_models[user_id] = {}
                    self.forklift_models[user_id][model_name] = loaded_model
                else:
                    if user_id not in self.beat_saber_models:
                        self.beat_saber_models[user_id] = {}
                    self.beat_saber_models[user_id][model_name] = loaded_model
                
                self.best_models[game_type][user_id] = model_name
                
                user_results_by_game[game_type][user_id] = []
                
                user_max_samples = len(genuine_test)
                actual_sample_sizes = range(1, min(max_samples or 10, user_max_samples) + 1)
                
                for n_samples in actual_sample_sizes:
                    
                    group = genuine_test.iloc[:n_samples]
                    group_features = group.drop(columns=['is_genuine', 'original_id', 'time_interval'])
                    
                    genuine_votes = 0
                    for _, sample in group_features.iterrows():
                        is_genuine, confidence = self.authenticate_user(
                            user_id, sample, game_type, filename=filename, model_name=model_name
                        )
                        if is_genuine:
                            genuine_votes += 1
                    
                    genuine_percentage = genuine_votes / len(group_features)
                    genuine_accuracy = 1 if genuine_percentage >= voting_threshold else 0
                    
                    impostor_accuracy_by_id = {}
                    
                    for impostor_id in impostor_ids:
                        impostor_samples = impostor_test_df[impostor_test_df['original_id'] == impostor_id].reset_index(drop=True)
                        
                        if len(impostor_samples) < n_samples:
                            continue
                        
                        group = impostor_samples.iloc[:n_samples]
                        group_features = group.drop(columns=['is_genuine', 'original_id', 'time_interval'])
                        
                        genuine_votes = 0
                        for _, sample in group_features.iterrows():
                            is_genuine, confidence = self.authenticate_user(
                                user_id, sample, game_type, filename=filename, model_name=model_name
                            )
                            if is_genuine:
                                genuine_votes += 1
                        
                        genuine_percentage = genuine_votes / len(group_features)
                        impostor_accuracy_by_id[impostor_id] = 1 if genuine_percentage < voting_threshold else 0
                    
                    all_impostor_acc = list(impostor_accuracy_by_id.values())
                    avg_impostor_acc = sum(all_impostor_acc) / len(all_impostor_acc) if all_impostor_acc else 0
                    overall_acc = (genuine_accuracy + avg_impostor_acc) / 2

                    result = {
                        'User_ID': user_id,
                        'Game_Type': game_type,
                        'Model_Type': model_name,
                        'Num_Samples': n_samples,
                        'Genuine_Accuracy': genuine_accuracy,
                        'Impostor_Accuracy': avg_impostor_acc,
                        'Overall_Accuracy': overall_acc,
                        'Num_Impostor_IDs': len(impostor_accuracy_by_id)
                    }
                    
                    user_results_by_game[game_type][user_id].append(result)
                    results.append(result)
        
        results_df = pd.DataFrame(results)
        
        csv_path = os.path.join(results_dir, f'{output_prefix}.csv')
        results_df.to_csv(csv_path, index=False)
        
        for game_type in ['forklift_simulator', 'beat_saber']:
            if game_type not in user_results_by_game or not user_results_by_game[game_type]:
                continue
                
            plt.figure(figsize=(14, 8))
            
            for i, (user_id, user_results) in enumerate(user_results_by_game[game_type].items()):
                if not user_results:
                    continue
                    
                user_df = pd.DataFrame(user_results)
                plt.plot(user_df['Num_Samples'], user_df['Overall_Accuracy'], label=f'User {int(user_id)}')
            
            plt.title(f'Overall Accuracy by Voting Samples - {game_type.replace("_", " ").title()}', fontsize=14)
            plt.xlabel('Number of Samples Used for Voting', fontsize=12)
            plt.ylabel('Overall Accuracy', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='lower right', bbox_to_anchor=(1, 0), ncol=3)
            plt.ylim(0, 1.01)
            
            tikzplotlib.save(os.path.join(results_dir, f'{output_prefix}_{game_type}_overall.tex'))
            plt.savefig(os.path.join(results_dir, f'{output_prefix}_{game_type}_overall.png'), dpi=300, bbox_inches='tight')
            
            plt.tight_layout()
            plt.show()
        
        return results_df