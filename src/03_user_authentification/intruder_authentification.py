import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplot2tikz as tikzplotlib
from tqdm import tqdm
import glob
import joblib
from collections import defaultdict

class IntruderAuthEvaluator:
    
    def __init__(self, authenticator, intruder_data_path=None, filename="normal", model_name="RandomForest"):
        self.authenticator = authenticator
        self.intruder_data_path = intruder_data_path
        self.filename = filename
        self.model_name = model_name
        self.evaluation_results = pd.DataFrame(columns=[
            'Target_User_ID', 'Base_User_ID', 'Intruder_Type', 'Adaptation_Factor',
            'Authentication_Success', 'Confidence', 'Game_Type'
        ])
        
    def load_intruder_files(self, pattern=None, game_type=None):
        if not self.intruder_data_path:
            raise ValueError("Intruder data path not specified")
        
        if pattern is None:
            if game_type == 'forklift_simulator':
                pattern = "intruder_slow_base*_target*_*.csv"
            elif game_type == 'beat_saber':
                pattern = "intruder_fast_base*_target*_*.csv"
                
        intruder_files = {
            'physical': [],
            'movement': [],
            'combined': []
        }
        
        search_path = os.path.join(self.intruder_data_path, pattern)
        direct_files = glob.glob(search_path)
        
        subdir_files = []
        for intruder_type in intruder_files.keys():
            subdir_search_path = os.path.join(self.intruder_data_path, intruder_type, pattern)
            subdir_files.extend(glob.glob(subdir_search_path))
        
        all_files = direct_files + subdir_files
        
        if not all_files:
            raise ValueError(f"No files found matching pattern {pattern} in {self.intruder_data_path} or its subdirectories")
        
        valid_files = []
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            parent_dir = os.path.basename(os.path.dirname(file_path))
            
            is_slow = 'slow' in file_name
            is_fast = 'fast' in file_name
            
            if (game_type == 'forklift_simulator' and is_slow) or (game_type == 'beat_saber' and is_fast):
                try:
                    parts = file_name.split('_')
                    
                    base_id_part = [p for p in parts if p.startswith('base')][0]
                    target_id_part = [p for p in parts if p.startswith('target')][0]
                    base_user_id = int(base_id_part.replace('base', ''))
                    target_user_id = int(target_id_part.replace('target', ''))
                    
                    if game_type == 'forklift_simulator' and (30 <= base_user_id <= 59):
                        valid_files.append(file_path)
                    elif game_type == 'beat_saber' and (0 <= base_user_id <= 29): 
                        valid_files.append(file_path)

                except (IndexError, ValueError) as e:
                    continue
        
        for file_path in valid_files:
            file_name = os.path.basename(file_path)
            parent_dir = os.path.basename(os.path.dirname(file_path))
            
            if 'physical' in file_name or parent_dir == 'physical':
                intruder_files['physical'].append(file_path)
            elif 'movement' in file_name or parent_dir == 'movement':
                intruder_files['movement'].append(file_path)
            elif 'combined' in file_name or parent_dir == 'combined':
                intruder_files['combined'].append(file_path)
            else:
                parts = file_name.split('_')
                if len(parts) >= 4:
                    if parts[3] in ['physical', 'movement', 'combined']:
                        intruder_files[parts[3]].append(file_path)
                    elif parent_dir in intruder_files:
                        intruder_files[parent_dir].append(file_path)                                                                      
        return intruder_files
    
    def evaluate_intruders(self, intruder_files=None, target_users=None, game_type='forklift_simulator', without_height = False):
        if intruder_files is None:
            intruder_files = self.load_intruder_files(game_type=game_type)
        
        results = []
        
        for intruder_type, files in intruder_files.items():
            
            for file_path in tqdm(files, desc=f"{intruder_type} intruders"):
                try:
                    file_name = os.path.basename(file_path)
                    parts = file_name.split('_')
                    
                    base_id_part = [p for p in parts if p.startswith('base')][0]
                    target_id_part = [p for p in parts if p.startswith('target')][0]
                    
                    base_user_id = int(base_id_part.replace('base', ''))
                    target_user_id = int(target_id_part.replace('target', ''))
                    
                    if target_users is not None and target_user_id not in target_users:
                        continue
                    
                    adaptation_parts = [i for i, p in enumerate(parts) if p.startswith('0.') or p.startswith('1.')]
                    if adaptation_parts:
                        adaptation_factor = float(parts[adaptation_parts[0]])
                    else:
                        adaptation_factor = 0.5
                    
                    intruder_data = pd.read_csv(file_path)
                    meta_columns = ['time_25%', 'time_50%', 'time_75%', 'time_max', 'time_mean', 'time_min', 'time_std', 
                                    'Δt_mean','Δt_std','Δt_min','Δt_25%','Δt_50%','Δt_75%','Δt_max',
                                    'base_user_id','target_user_id','attribute_type','adaptation_factor']
            
                    columns_to_drop = [col for col in meta_columns if col in intruder_data.columns]
                    if columns_to_drop:
                        intruder_data = intruder_data.drop(columns=columns_to_drop)
                    
                    if without_height:
                        for col in intruder_data.columns.values:
                            if "PosY" in col and "Accel" not in col and "Velocity" not in col:
                                intruder_data[col] = intruder_data[col] / np.mean(intruder_data[col])
                                
                    if 'time_interval' in intruder_data.columns:
                        for interval, interval_data in intruder_data.groupby('time_interval'):
                            is_genuine, confidence = self.authenticator.authenticate_user(
                                target_user_id, interval_data, game_type=game_type,
                                filename=self.filename, model_name=self.model_name
                            )
                            
                            results.append({
                                'Target_User_ID': target_user_id,
                                'Base_User_ID': base_user_id,
                                'Intruder_Type': intruder_type,
                                'Adaptation_Factor': adaptation_factor,
                                'Authentication_Success': is_genuine,
                                'Confidence': confidence,
                                'Game_Type': game_type,
                                'Time_Interval': interval
                            })
                    else:
                        is_genuine, confidence = self.authenticator.authenticate_user(
                            target_user_id, intruder_data, game_type=game_type,
                            filename=self.filename, model_name=self.model_name
                        )
                        
                        results.append({
                            'Target_User_ID': target_user_id,
                            'Base_User_ID': base_user_id,
                            'Intruder_Type': intruder_type,
                            'Adaptation_Factor': adaptation_factor,
                            'Authentication_Success': is_genuine,
                            'Confidence': confidence,
                            'Game_Type': game_type,
                            'Time_Interval': None
                        })
                
                except Exception as e:
                    continue
        
        self.evaluation_results = pd.DataFrame(results)
        return self.evaluation_results
    
    def analyze_results(self, by_user=True, by_adaptation=True):
        if self.evaluation_results.empty:
            return None, None
            
        results_dir = os.path.join('../../results', 'user_authentification', 'intruders_evaluation')
        os.makedirs(results_dir, exist_ok=True)
        
        type_summary = self.evaluation_results.groupby('Intruder_Type')['Authentication_Success'].agg(
            ['mean', 'count']).rename(columns={'mean': 'Success_Rate', 'count': 'Sample_Count'})
        
        type_summary.to_csv(os.path.join(results_dir, f'type_summary_{self.model_name}_{self.filename}.csv'))
        
        plt.figure(figsize=(14, 6))
        ax = type_summary['Success_Rate'].plot(kind='bar')
        plt.title('Success Rate by Intruder Type', fontsize=14)
        plt.ylabel('Success Rate', fontsize=12)
        plt.xlabel('Intruder Type', fontsize=12)
        plt.ylim(0, max(type_summary['Success_Rate'].max() * 1.2, 0.1))
        plt.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(type_summary['Success_Rate']):
            ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
            
        for i, v in enumerate(type_summary['Sample_Count']):
            ax.text(i, 0.01, f"n={v}", ha='center', fontsize=9, color='dimgray')
            
        plt.tight_layout()
        try:
            tikzplotlib.save(os.path.join(results_dir, f'success_rate_by_type_{self.model_name}_{self.filename}.tex'), encoding='utf-8')
        except UnicodeEncodeError:
            pass
        plt.savefig(os.path.join(results_dir, f'success_rate_by_type_{self.model_name}_{self.filename}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        summary_by_user = None
        if by_user:
            summary_by_user = self.evaluation_results.groupby(
                ['Target_User_ID', 'Intruder_Type'])['Authentication_Success'].agg(
                ['mean', 'count']).rename(columns={'mean': 'Success_Rate', 'count': 'Sample_Count'})
            
            summary_by_user.to_csv(os.path.join(results_dir, f'user_summary_{self.model_name}_{self.filename}.csv'))
            
            user_vulnerability = self.evaluation_results.groupby('Target_User_ID')['Authentication_Success'].mean()
            most_vulnerable = user_vulnerability.sort_values(ascending=False)
            
            most_vulnerable.to_csv(os.path.join(results_dir, f'most_vulnerable_{self.model_name}_{self.filename}.csv'))
            
            plt.figure(figsize=(14, 6))
            ax = most_vulnerable.plot(kind='bar', color='orangered')
            plt.title('Most Vulnerable Users (Highest Intruder Success Rate)', fontsize=14)
            plt.ylabel('Success Rate', fontsize=12)
            plt.xlabel('Target User ID', fontsize=12)
            plt.ylim(0, min(most_vulnerable.max() * 1.2, 1.0))
            plt.grid(axis='y', alpha=0.3)
            
            for i, v in enumerate(most_vulnerable):
                ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)
                
            plt.tight_layout()
            try:
                tikzplotlib.save(os.path.join(results_dir, f'most_vulnerable_{self.model_name}_{self.filename}.tex'), encoding='utf-8')
            except UnicodeEncodeError:
                pass
            plt.savefig(os.path.join(results_dir, f'most_vulnerable_{self.model_name}_{self.filename}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        summary_by_adaptation = None
        if by_adaptation and 'Adaptation_Factor' in self.evaluation_results.columns:
            summary_by_adaptation = self.evaluation_results.groupby(
                ['Intruder_Type', 'Adaptation_Factor'])['Authentication_Success'].agg(
                ['mean', 'count']).rename(columns={'mean': 'Success_Rate', 'count': 'Sample_Count'})
            
            summary_by_adaptation.to_csv(os.path.join(results_dir, f'adaptation_summary_{self.model_name}_{self.filename}.csv'))
               
        return summary_by_user, summary_by_adaptation
    
    def compare_intruder_target_features(self, target_user_id, base_user_id, intruder_type, game_type='forklift_simulator', top_n=15):
        results_dir = os.path.join('../../results', 'user_authentification', 'intruders_evaluation')
        os.makedirs(results_dir, exist_ok=True)
        
        target_model_path = os.path.join('../../models', 'user_authentification', 'training_evaluation', 
                                        game_type, f'user_{target_user_id}', 
                                        f'{self.model_name}_{self.filename}.joblib')
        
        base_model_path = os.path.join('../../models', 'user_authentification', 'training_evaluation', 
                                      game_type, f'user_{base_user_id}', 
                                      f'{self.model_name}_{self.filename}.joblib')
        
        if not os.path.exists(target_model_path) or not os.path.exists(base_model_path):
            return None
        
        try:
            target_model = joblib.load(target_model_path)
            base_model = joblib.load(base_model_path)
        except Exception as e:
            return None
        
        if not hasattr(target_model, 'feature_importances_') or not hasattr(base_model, 'feature_importances_'):
            return None
        
        test_sets = self.authenticator.forklift_test_sets if game_type == 'forklift_simulator' else self.authenticator.beat_saber_test_sets
        
        if target_user_id not in test_sets:
            return None
        
        test_data = test_sets[target_user_id]
        X_test, _ = self.authenticator.preprocess_data(test_data)
        feature_names = X_test.columns
        
        target_importances = target_model.feature_importances_
        base_importances = base_model.feature_importances_
        
        target_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': target_importances
        }).sort_values('Importance', ascending=False)
        
        base_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': base_importances
        }).sort_values('Importance', ascending=False)
        
        top_target = target_df.head(top_n)
        top_base = base_df.head(top_n)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1.barh(range(len(top_target)), top_target['Importance'].values, color='dodgerblue')
        ax1.set_yticks(range(len(top_target)))
        ax1.set_yticklabels(top_target['Feature'].values)
        ax1.set_xlabel('Importance')
        ax1.set_title(f'Target User {target_user_id} - Top {top_n} Features')
        ax1.invert_yaxis()
        
        ax2.barh(range(len(top_base)), top_base['Importance'].values, color='orangered')
        ax2.set_yticks(range(len(top_base)))
        ax2.set_yticklabels(top_base['Feature'].values)
        ax2.set_xlabel('Importance')
        ax2.set_title(f'Intruder (Base User {base_user_id}) - Top {top_n} Features')
        ax2.invert_yaxis()
        
        plt.suptitle(f'Feature Importance Comparison - {intruder_type.capitalize()} Intruder\n{game_type.replace("_", " ").title()}', fontsize=14)
        plt.tight_layout()
        
        plot_filename = f'feature_comparison_target{target_user_id}_base{base_user_id}_{intruder_type}_{self.model_name}_{self.filename}'
        try:
            tikzplotlib.save(os.path.join(results_dir, f'{plot_filename}.tex'), encoding='utf-8')
        except UnicodeEncodeError:
            pass
        plt.savefig(os.path.join(results_dir, f'{plot_filename}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        common_features = set(top_target['Feature'].values) & set(top_base['Feature'].values)
        
        if common_features:
            plt.figure(figsize=(14, 8))
            
            common_df = pd.DataFrame()
            for feature in common_features:
                target_imp = target_df[target_df['Feature'] == feature]['Importance'].values[0]
                base_imp = base_df[base_df['Feature'] == feature]['Importance'].values[0]
                common_df = pd.concat([common_df, pd.DataFrame({
                    'Feature': [feature],
                    'Target_Importance': [target_imp],
                    'Base_Importance': [base_imp]
                })], ignore_index=True)
            
            common_df = common_df.sort_values('Target_Importance', ascending=False)
            
            x = np.arange(len(common_df))
            width = 0.35
            
            plt.bar(x - width/2, common_df['Target_Importance'], width, label=f'Target User {target_user_id}', color='dodgerblue')
            plt.bar(x + width/2, common_df['Base_Importance'], width, label=f'Base User {base_user_id}', color='orangered')
            
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.title(f'Common Top Features - {intruder_type.capitalize()} Intruder Attack')
            plt.xticks(x, common_df['Feature'], rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            common_plot_filename = f'common_features_target{target_user_id}_base{base_user_id}_{intruder_type}_{self.model_name}_{self.filename}'
            try:
                tikzplotlib.save(os.path.join(results_dir, f'{common_plot_filename}.tex'), encoding='utf-8')
            except UnicodeEncodeError:
                pass
            plt.savefig(os.path.join(results_dir, f'{common_plot_filename}.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
            common_df.to_csv(os.path.join(results_dir, f'{common_plot_filename}.csv'), index=False)
        
        return {
            'target_features': top_target,
            'base_features': top_base,
            'common_features': common_features,
            'num_common': len(common_features)
        }
    
    def compare_actual_feature_values(self, target_user_id, base_user_id, intruder_type, game_type='forklift_simulator', 
                                     intruder_data_sample=None, top_n=10):
        results_dir = os.path.join('../../results', 'user_authentification', 'intruders_evaluation')
        os.makedirs(results_dir, exist_ok=True)
        
        target_model_path = os.path.join('../../models', 'user_authentification', 'training_evaluation', 
                                        game_type, f'user_{target_user_id}', 
                                        f'{self.model_name}_{self.filename}.joblib')
        
        if not os.path.exists(target_model_path):
            return None
        
        try:
            target_model = joblib.load(target_model_path)
        except Exception as e:
            return None
        
        if not hasattr(target_model, 'feature_importances_'):
            return None
        
        test_sets = self.authenticator.forklift_test_sets if game_type == 'forklift_simulator' else self.authenticator.beat_saber_test_sets
        
        if target_user_id not in test_sets:
            return None
        
        test_data = test_sets[target_user_id]
        X_test, _ = self.authenticator.preprocess_data(test_data)
        feature_names = X_test.columns
        
        target_importances = target_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': target_importances
        }).sort_values('Importance', ascending=False)
        
        top_features = feature_importance_df.head(top_n)['Feature'].tolist()
        
        if game_type == 'forklift_simulator':
            target_original_data = self.authenticator.forklift_datasets[target_user_id]
            base_original_data = self.authenticator.forklift_datasets.get(base_user_id)
        else:
            target_original_data = self.authenticator.beat_saber_datasets[target_user_id]
            base_original_data = self.authenticator.beat_saber_datasets.get(base_user_id)
        
        if base_original_data is None:
            return None
        
        target_genuine = target_original_data[target_original_data['is_genuine'] == 1]
        base_genuine = base_original_data[base_original_data['is_genuine'] == 1]
        
        comparison_results = {}
        
        for feature in top_features:
            if feature in target_genuine.columns and feature in base_genuine.columns:
                target_values = target_genuine[feature].dropna()
                base_values = base_genuine[feature].dropna()
                
                comparison_results[feature] = {
                    'target_mean': target_values.mean(),
                    'target_std': target_values.std(),
                    'target_median': target_values.median(),
                    'base_mean': base_values.mean(),
                    'base_std': base_values.std(),
                    'base_median': base_values.median(),
                    'target_values': target_values.values,
                    'base_values': base_values.values,
                    'importance': feature_importance_df[feature_importance_df['Feature'] == feature]['Importance'].iloc[0]
                }
                
                if intruder_data_sample is not None and feature in intruder_data_sample.columns:
                    intruder_values = intruder_data_sample[feature].dropna()
                    comparison_results[feature]['intruder_mean'] = intruder_values.mean()
                    comparison_results[feature]['intruder_std'] = intruder_values.std()
                    comparison_results[feature]['intruder_median'] = intruder_values.median()
                    comparison_results[feature]['intruder_values'] = intruder_values.values
        
        n_features = min(15, len(comparison_results))
        n_cols = 2
        n_rows = int(np.ceil(n_features / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        feature_comparison_data = []
        
        for i, (feature, data) in enumerate(list(comparison_results.items())[:n_features]):
            ax = axes[i]
            
            plot_data = []
            labels = []
            
            plot_data.append(data['target_values'])
            labels.append(f'Target {target_user_id}')
            
            plot_data.append(data['base_values'])
            labels.append(f'Base {base_user_id}')
            
            if 'intruder_values' in data:
                plot_data.append(data['intruder_values'])
                labels.append('Intruder')
            
            parts = ax.violinplot(plot_data, positions=range(len(plot_data)), showmeans=True, showmedians=True)
            
            colors = ['dodgerblue', 'orangered', 'green']
            for j, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[j % len(colors)])
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Value')
            ax.set_title(f'{feature}\n(Importance: {data["importance"]:.3f})')
            ax.grid(True, alpha=0.3)
            
            for j, label in enumerate(labels):
                if j == 0:
                    mean_val = data['target_mean']
                elif j == 1:
                    mean_val = data['base_mean']
                else:
                    mean_val = data.get('intruder_mean', 0)
                
                ax.text(j, ax.get_ylim()[1] * 0.9, f'μ={mean_val:.2f}', 
                       ha='center', va='center', fontsize=9, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            feature_comparison_data.append({
                'Feature': feature,
                'Target_Mean': data['target_mean'],
                'Target_Std': data['target_std'],
                'Base_Mean': data['base_mean'],
                'Base_Std': data['base_std'],
                'Intruder_Mean': data.get('intruder_mean', np.nan),
                'Intruder_Std': data.get('intruder_std', np.nan),
                'Importance': data['importance'],
                'Mean_Diff_Target_Base': abs(data['target_mean'] - data['base_mean']),
                'Mean_Diff_Target_Intruder': abs(data['target_mean'] - data.get('intruder_mean', np.nan)) if 'intruder_mean' in data else np.nan
            })
        
        for j in range(n_features, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(f'Feature Value Comparison - {intruder_type.capitalize()} Intruder\nTarget {target_user_id} vs Base {base_user_id} ({game_type.replace("_", " ").title()})', 
                     fontsize=16, y=0.98)
        plt.tight_layout()
        
        plot_filename = f'feature_values_comparison_target{target_user_id}_base{base_user_id}_{intruder_type}_{self.model_name}_{self.filename}'
        try:
            tikzplotlib.save(os.path.join(results_dir, f'{plot_filename}.tex'), encoding='utf-8')
        except UnicodeEncodeError:
            pass
        plt.savefig(os.path.join(results_dir, f'{plot_filename}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        comparison_df = pd.DataFrame(feature_comparison_data)
        comparison_df.to_csv(os.path.join(results_dir, f'{plot_filename}.csv'), index=False)
        
        similarity_scores = {}
        for feature, data in comparison_results.items():
            target_mean = data['target_mean']
            base_mean = data['base_mean']
            
            target_std = data['target_std']
            base_std = data['base_std']
            
            mean_similarity = 1 - abs(target_mean - base_mean) / (abs(target_mean) + abs(base_mean) + 1e-8)
            std_similarity = 1 - abs(target_std - base_std) / (target_std + base_std + 1e-8)
            
            similarity_scores[feature] = {
                'mean_similarity': mean_similarity,
                'std_similarity': std_similarity,
                'combined_similarity': (mean_similarity + std_similarity) / 2,
                'importance': data['importance']
            }
        
        return {
            'comparison_results': comparison_results,
            'similarity_scores': similarity_scores,
            'feature_comparison_df': comparison_df
        }
        
    def generate_comprehensive_report(self, output_path=None):
        if self.evaluation_results.empty:
            return
        
        results_dir = os.path.join('../../results', 'user_authentification', 'intruders_evaluation')
        os.makedirs(results_dir, exist_ok=True)
        
        if output_path is None:
            output_path = os.path.join(results_dir, f'comprehensive_report_{self.model_name}_{self.filename}.txt')
        
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("INTRUDER AUTHENTICATION SYSTEM EVALUATION REPORT")
        report_lines.append("=" * 80)
        
        total_intruders = len(self.evaluation_results)
        success_rate = self.evaluation_results['Authentication_Success'].mean()
        
        report_lines.append(f"\nOverall Performance:")
        report_lines.append(f"Total intruder samples evaluated: {total_intruders}")
        report_lines.append(f"Overall intruder success rate: {success_rate:.2%}")
        
        type_results = self.evaluation_results.groupby('Intruder_Type').agg({
            'Authentication_Success': ['mean', 'count'],
            'Confidence': 'mean'
        })
        
        report_lines.append("\nPerformance by Intruder Type:")
        for intruder_type, row in type_results.iterrows():
            report_lines.append(f"- {intruder_type.capitalize()}: {row[('Authentication_Success', 'mean')]:.2%} success rate " +
                            f"(Avg. Confidence: {row[('Confidence', 'mean')]:.2f}, Samples: {row[('Authentication_Success', 'count')]})")
        
        if 'Adaptation_Factor' in self.evaluation_results.columns:
            report_lines.append("\nEffect of Adaptation Factor:")
            for intruder_type in self.evaluation_results['Intruder_Type'].unique():
                type_data = self.evaluation_results[self.evaluation_results['Intruder_Type'] == intruder_type]
                
                min_adapt = type_data['Adaptation_Factor'].min()
                max_adapt = type_data['Adaptation_Factor'].max()
                
                min_success = type_data[type_data['Adaptation_Factor'] == min_adapt]['Authentication_Success'].mean()
                max_success = type_data[type_data['Adaptation_Factor'] == max_adapt]['Authentication_Success'].mean()
                
                report_lines.append(f"- {intruder_type.capitalize()} intruders:")
                report_lines.append(f"  * Low adaptation ({min_adapt:.2f}): {min_success:.2%} success rate")
                report_lines.append(f"  * High adaptation ({max_adapt:.2f}): {max_success:.2%} success rate")
        
        user_vulnerability = self.evaluation_results.groupby('Target_User_ID')['Authentication_Success'].mean().sort_values(ascending=False)
        
        report_lines.append("\nMost Vulnerable Users (Top 5):")
        for i, (user_id, success_rate) in enumerate(user_vulnerability.head(5).items()):
            report_lines.append(f"{i+1}. User {user_id}: {success_rate:.2%} intruder success rate")
        
        report_lines.append("\nMost Resistant Users (Top 5):")
        for i, (user_id, success_rate) in enumerate(user_vulnerability.tail(5).items()):
            report_lines.append(f"{i+1}. User {user_id}: {success_rate:.2%} intruder success rate")
        
        report_text = "\n".join(report_lines)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def evaluate_intruders_with_voting(self, intruder_files=None, target_users=None, game_type='forklift_simulator', 
                            voting_threshold=0.5, min_votes=3, max_votes=10, group_by='time_interval', without_height = False):
        if intruder_files is None:
            intruder_files = self.load_intruder_files(game_type=game_type)
        
        results = []
        
        for intruder_type, files in intruder_files.items():
            
            grouped_data = {}
            
            for file_path in tqdm(files, desc=f"{intruder_type} intruders"):
                try:
                    file_name = os.path.basename(file_path)
                    parts = file_name.split('_')
                    
                    base_id_part = [p for p in parts if p.startswith('base')][0]
                    target_id_part = [p for p in parts if p.startswith('target')][0]
                    
                    base_user_id = int(base_id_part.replace('base', ''))
                    target_user_id = int(target_id_part.replace('target', ''))
                    
                    if target_users is not None and target_user_id not in target_users:
                        continue
                    
                    if game_type == 'forklift_simulator':
                        if not (30 <= target_user_id <= 60):
                            continue
                    else:
                        if not (0 <= target_user_id <= 30):
                            continue
                    
                    adaptation_parts = [i for i, p in enumerate(parts) if p.startswith('0.') or p.startswith('1.')]
                    if adaptation_parts:
                        adaptation_factor = float(parts[adaptation_parts[0]])
                    else:
                        adaptation_factor = 0.5
                    
                    intruder_data = pd.read_csv(file_path)
                    meta_columns = ['time_25%', 'time_50%', 'time_75%', 'time_max', 'time_mean', 'time_min', 'time_std', 
                                    'Δt_mean','Δt_std','Δt_min','Δt_25%','Δt_50%','Δt_75%','Δt_max',
                                    'base_user_id','target_user_id','attribute_type','adaptation_factor']
                    
                    columns_to_drop = [col for col in meta_columns if col in intruder_data.columns]
                    if columns_to_drop:
                        intruder_data = intruder_data.drop(columns=columns_to_drop)
                    
                    if without_height:
                        for col in intruder_data.columns.values:
                            if "PosY" in col and "Accel" not in col and "Velocity" not in col:
                                intruder_data[col] = intruder_data[col] / np.mean(intruder_data[col])

                    group_key = (target_user_id, base_user_id, intruder_type, adaptation_factor)
                    
                    if group_key not in grouped_data:
                        grouped_data[group_key] = []
                    
                    if group_by == 'time_interval' and 'time_interval' in intruder_data.columns:
                        for interval, interval_data in intruder_data.groupby('time_interval'):
                            interval_data_no_time = interval_data.drop(columns=['time_interval'])
                            grouped_data[group_key].append(interval_data_no_time)
                    else:
                        if 'time_interval' in intruder_data.columns:
                            intruder_data = intruder_data.drop(columns=['time_interval'])
                        grouped_data[group_key].append(intruder_data)
                    
                except Exception as e:
                    continue
            
            for group_key, samples in grouped_data.items():
                target_user_id, base_user_id, curr_intruder_type, adaptation_factor = group_key
                
                if len(samples) < min_votes:
                    continue
                
                for n_samples in range(min_votes, min(max_votes+1, len(samples)+1)):
                    n_groups = len(samples) // n_samples
                    if n_groups == 0:
                        continue
                        
                    n_groups = min(n_groups, 5)
                    
                    for i in range(n_groups):
                        start_idx = i * n_samples
                        group = samples[start_idx:start_idx + n_samples]
                        
                        if len(group) < n_samples:
                            continue
                            
                        genuine_votes = 0
                        confidence_scores = []
                        
                        for sample in group:
                            is_genuine, confidence = self.authenticator.authenticate_user(
                                target_user_id, sample, game_type=game_type,
                                filename=self.filename, model_name=self.model_name
                            )
                            
                            if is_genuine:
                                genuine_votes += 1
                            
                            confidence_scores.append(confidence)
                        
                        total_votes = len(group)
                        genuine_percent = genuine_votes / total_votes if total_votes > 0 else 0
                        
                        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                        
                        final_is_genuine = genuine_percent >= voting_threshold
                        
                        results.append({
                            'Target_User_ID': target_user_id,
                            'Base_User_ID': base_user_id,
                            'Intruder_Type': curr_intruder_type,
                            'Adaptation_Factor': adaptation_factor,
                            'Authentication_Success': final_is_genuine,
                            'Confidence': avg_confidence,
                            'Game_Type': game_type,
                            'Num_Samples': n_samples,
                            'Genuine_Votes': genuine_votes,
                            'Total_Votes': total_votes,
                            'Genuine_Percent': genuine_percent
                        })
        
        voting_evaluation_results = pd.DataFrame(results)
        
        results_dir = os.path.join('../../results', 'user_authentification', 'intruders_evaluation')
        os.makedirs(results_dir, exist_ok=True)
        voting_evaluation_results.to_csv(os.path.join(results_dir, 
                                                f'voting_results_{game_type}_{self.model_name}_{self.filename}.csv'), 
                                    index=False)
        
        return voting_evaluation_results

    def analyze_voting_results(self, voting_results):
        if voting_results.empty:
            return None, None
            
        results_dir = os.path.join('../../results', 'user_authentification', 'intruders_evaluation')
        os.makedirs(results_dir, exist_ok=True)

        type_summary = voting_results.groupby('Intruder_Type')['Authentication_Success'].agg(
            ['mean', 'count']).rename(columns={'mean': 'Success_Rate', 'count': 'Group_Count'})
        
        type_summary.to_csv(os.path.join(results_dir, f'voting_type_summary_{self.model_name}_{self.filename}.csv'))
        
        plt.figure(figsize=(14, 6))
        ax = type_summary['Success_Rate'].plot(kind='bar')
        plt.title('Voting-Based Success Rate by Intruder Type', fontsize=14)
        plt.ylabel('Success Rate', fontsize=12)
        plt.xlabel('Intruder Type', fontsize=12)
        plt.ylim(0, max(type_summary['Success_Rate'].max() * 1.2, 0.1))
        plt.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(type_summary['Success_Rate']):
            ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
            
        for i, v in enumerate(type_summary['Group_Count']):
            ax.text(i, 0.01, f"n={v}", ha='center', fontsize=9, color='dimgray')
            
        plt.tight_layout()
        try:
            tikzplotlib.save(os.path.join(results_dir, f'voting_success_rate_by_type_{self.model_name}_{self.filename}.tex'), encoding='utf-8')
        except UnicodeEncodeError:
            pass
        plt.savefig(os.path.join(results_dir, f'voting_success_rate_by_type_{self.model_name}_{self.filename}.png'), 
                dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(14, 6))
        thresholds = np.linspace(0, 1, 11)
        success_rates = []
        
        for threshold in thresholds:
            success_rate = (voting_results['Genuine_Percent'] >= threshold).mean()
            success_rates.append(success_rate)
        
        threshold_df = pd.DataFrame({'Threshold': thresholds, 'Success_Rate': success_rates})
        threshold_df.to_csv(os.path.join(results_dir, 
                                    f'voting_threshold_effect_{self.model_name}_{self.filename}.csv'),
                        index=False)
        
        plt.plot(thresholds, success_rates, marker='o', linestyle='-', linewidth=2, color='purple')
        plt.title('Intruder Success Rate by Voting Threshold', fontsize=14)
        plt.ylabel('Success Rate', fontsize=12)
        plt.xlabel('Voting Threshold', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        try:
            tikzplotlib.save(os.path.join(results_dir, f'voting_threshold_effect_{self.model_name}_{self.filename}.tex'), encoding='utf-8')
        except UnicodeEncodeError:
            pass
        plt.savefig(os.path.join(results_dir, f'voting_threshold_effect_{self.model_name}_{self.filename}.png'), 
                dpi=300, bbox_inches='tight')
        plt.show()

def evaluate_authentication_system(authenticator, intruder_data_path, game_type='forklift_simulator',
                                target_users=None, output_dir=None, filename="normal", model_name="RandomForest"):
    
    results_dir = os.path.join('../../results', 'user_authentification', 'intruders_evaluation')
    if output_dir:
        results_dir = output_dir
        
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    evaluator = IntruderAuthEvaluator(
        authenticator, 
        intruder_data_path,
        filename=filename,
        model_name=model_name
    )
    
    intruder_files = evaluator.load_intruder_files(game_type=game_type)
    
    evaluation_results = evaluator.evaluate_intruders(
        intruder_files=intruder_files,
        target_users=target_users,
        game_type=game_type
    )
    
    evaluator.analyze_results(by_user=True, by_adaptation=True)
    
    evaluation_results.to_csv(os.path.join(results_dir, f'intruder_evaluation_results_{model_name}_{filename}.csv'), index=False)
    
    evaluator.generate_comprehensive_report(os.path.join(results_dir, f'intruder_evaluation_report_{model_name}_{filename}.txt'))
    
    if not evaluation_results.empty:
        user_vulnerability = evaluation_results.groupby('Target_User_ID')['Authentication_Success'].mean().sort_values(ascending=False)
        top_vulnerable = user_vulnerability.head(3)
        
        for target_user_id in top_vulnerable.index:
            target_results = evaluation_results[evaluation_results['Target_User_ID'] == target_user_id]
            if not target_results.empty:
                base_user_id = target_results.iloc[0]['Base_User_ID']
                intruder_type = target_results.iloc[0]['Intruder_Type']
                
                evaluator.compare_intruder_target_features(
                    target_user_id, base_user_id, intruder_type, game_type
                )
    
    return evaluator

def evaluate_authentication_system_with_voting(authenticator, intruder_data_path, game_type='forklift_simulator',
                                            target_users=None, output_dir=None, voting_threshold=0.5,
                                            min_votes=3, max_votes=10, compare_with_standard=True,
                                            filename="normal", model_name="RandomForest", without_height = False):
    
    results_dir = os.path.join('../../results', 'user_authentification', 'intruders_evaluation')
    if output_dir:
        results_dir = output_dir
        
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    standard_evaluator = None
    if compare_with_standard:
        standard_evaluator = IntruderAuthEvaluator(
            authenticator, 
            intruder_data_path,
            filename=filename,
            model_name=model_name
        )
        standard_files = standard_evaluator.load_intruder_files(game_type=game_type)
        evaluation_results = standard_evaluator.evaluate_intruders(
            intruder_files=standard_files,
            target_users=target_users,
            game_type=game_type,
            without_height = without_height
        )
        standard_evaluator.analyze_results()
        
        standard_evaluator.evaluation_results.to_csv(
            os.path.join(results_dir, f'standard_evaluation_results_{model_name}_{filename}.csv'), index=False)
        standard_evaluator.generate_comprehensive_report(
            os.path.join(results_dir, f'standard_evaluation_report_{model_name}_{filename}.txt'))
    
    voting_evaluator = IntruderAuthEvaluator(
        authenticator, 
        intruder_data_path,
        filename=filename,
        model_name=model_name
    )
    intruder_files = voting_evaluator.load_intruder_files(game_type=game_type)
    
    voting_results = voting_evaluator.evaluate_intruders_with_voting(
        intruder_files=intruder_files,
        target_users=target_users,
        game_type=game_type,
        voting_threshold=voting_threshold,
        min_votes=min_votes,
        max_votes=max_votes,
        without_height = without_height
    )
    
    voting_evaluator.analyze_voting_results(voting_results)
    
    if compare_with_standard and not evaluation_results.empty:
        user_vulnerability = evaluation_results.groupby('Target_User_ID')['Authentication_Success'].mean().sort_values(ascending=False)
        successful_targets = user_vulnerability[user_vulnerability > 0].index
        
        feature_analysis_results = {}
        
        for target_user_id in successful_targets:
            target_results = evaluation_results[evaluation_results['Target_User_ID'] == target_user_id]
            if not target_results.empty:
                base_user_id = target_results.iloc[0]['Base_User_ID']
                
                feature_comparison = standard_evaluator.compare_intruder_target_features(
                    target_user_id, base_user_id, 'general', game_type
                )
                
                value_comparison = standard_evaluator.compare_actual_feature_values(
                    target_user_id, base_user_id, 'general', game_type
                )
                
                feature_analysis_results[target_user_id] = {
                    'target_user_id': target_user_id,
                    'feature_importance_comparison': feature_comparison,
                    'feature_value_comparison': value_comparison,
                    'base_user_id': base_user_id,
                    'success_rate': user_vulnerability[target_user_id]
                }
        
        feature_analysis_summary = []
        for target_id, analysis in feature_analysis_results.items():
            if analysis['feature_value_comparison'] is not None:
                similarity_scores = analysis['feature_value_comparison']['similarity_scores']
                for feature, scores in similarity_scores.items():
                    feature_analysis_summary.append({
                        'Target_User_ID': analysis['target_user_id'],
                        'Base_User_ID': analysis['base_user_id'],
                        'Feature': feature,
                        'Mean_Similarity': scores['mean_similarity'],
                        'Std_Similarity': scores['std_similarity'],
                        'Combined_Similarity': scores['combined_similarity'],
                        'Feature_Importance': scores['importance'],
                        'Success_Rate': analysis['success_rate']
                    })
        
        if feature_analysis_summary:
            feature_summary_df = pd.DataFrame(feature_analysis_summary)
            feature_summary_df.to_csv(
                os.path.join(results_dir, f'feature_similarity_analysis_{model_name}_{filename}.csv'), 
                index=False
            )
            
            plt.figure(figsize=(16, 10))
            top_features_by_importance = feature_summary_df.nlargest(20, 'Feature_Importance')
            
            scatter = plt.scatter(top_features_by_importance['Mean_Similarity'], 
                                top_features_by_importance['Combined_Similarity'],
                                s=top_features_by_importance['Feature_Importance'] * 800,
                                alpha=0.6, 
                                c=top_features_by_importance['Success_Rate'],
                                cmap='Reds')
            
            plt.xlabel('Mean Value Similarity')
            plt.ylabel('Combined Similarity (Mean + Std)')
            plt.title('Feature Similarity vs Importance for All Successful Targets\n(Bubble size = Feature Importance, Color = Success Rate)')
            plt.colorbar(scatter, label='Intruder Success Rate')
            plt.grid(True, alpha=0.3)
            
            for idx, row in top_features_by_importance.head(10).iterrows():
                plt.annotate(f"{row['Feature'][:8]}...", 
                           (row['Mean_Similarity'], row['Combined_Similarity']),
                           xytext=(5, 5), textcoords='offset points', fontsize=7)
            
            plt.tight_layout()
            try:
                tikzplotlib.save(os.path.join(results_dir, f'feature_similarity_scatter_{model_name}_{filename}.tex'), encoding='utf-8')
            except UnicodeEncodeError:
                pass
            plt.savefig(os.path.join(results_dir, f'feature_similarity_scatter_{model_name}_{filename}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(14, 8))
            target_comparison = feature_summary_df.groupby('Target_User_ID')['Combined_Similarity'].agg(['mean', 'std', 'count']).reset_index()
            target_success = feature_summary_df.groupby('Target_User_ID')['Success_Rate'].first().reset_index()
            target_comparison = target_comparison.merge(target_success, on='Target_User_ID')
            target_comparison = target_comparison.sort_values('Success_Rate', ascending=False)
            
            bars = plt.bar(range(len(target_comparison)), target_comparison['mean'], 
                          yerr=target_comparison['std'], capsize=3, alpha=0.7,
                          color=plt.cm.Reds(target_comparison['Success_Rate']))
            
            plt.xticks(range(len(target_comparison)), [f"User {int(uid)}" for uid in target_comparison['Target_User_ID']], rotation=45)
            
            for i, (idx, row) in enumerate(target_comparison.iterrows()):
                plt.text(i, row['mean'] + row['std'] + 0.02, f"{row['Success_Rate']:.2f}", 
                        ha='center', fontsize=8)
            
            plt.ylabel('Average Combined Similarity')
            plt.xlabel('Target User (Ordered by Success Rate)')
            plt.title('Average Feature Similarity by Target User')
            plt.grid(True, alpha=0.3, axis='y')
            plt.ylim(0, 1)
            
            plt.tight_layout()
            try:
                tikzplotlib.save(os.path.join(results_dir, f'target_similarity_comparison_{model_name}_{filename}.tex'), encoding='utf-8')
            except UnicodeEncodeError:
                pass
            plt.savefig(os.path.join(results_dir, f'target_similarity_comparison_{model_name}_{filename}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    if compare_with_standard:
        return standard_evaluator, voting_evaluator
    else:
        return None, voting_evaluator