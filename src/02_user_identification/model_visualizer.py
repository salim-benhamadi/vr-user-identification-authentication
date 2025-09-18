import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import shap


class VisualizationUtils:
    """
    A class to provide visualization utilities for machine learning models.

    Attributes:
        None
    """

    @staticmethod
    def feature_importance(best_model, columnsPos, top_n=50):
        """
        Plots feature importance for a tree-based model.

        Args:
            best_model (object): Best performing model.
            columnsPos (list): List of column names representing features.
            top_n (int): Number of top features to plot. Default is 50.

        Returns:
            None
        """
        # Extract feature importance from the model
        feature_importance = best_model.feature_importances_
        
        # Create a DataFrame to store feature importance and names
        importance_df = pd.DataFrame({'Feature': columnsPos, 'Importance': feature_importance})
        
        # Sort the DataFrame by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        # Select top n features
        top_features = importance_df.head(top_n)
        
        # Plotting
        plt.figure(figsize=(30, 30))
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Top {} Feature Importance Plot'.format(top_n))
        plt.show()

    @staticmethod
    def confusion_matrices(cm1, cm2, title1='Confusion matrix - Slow Game', title2='Confusion matrix - Fast Game'):
            """
            Plots two confusion matrices side by side.

            Args:
                cm1 (array-like): Confusion matrix for the first scenario.
                cm2 (array-like): Confusion matrix for the second scenario.
                title1 (str): Title for the first confusion matrix plot. Default is 'Confusion matrix - Slow Game'.
                title2 (str): Title for the second confusion matrix plot. Default is 'Confusion matrix - Fast Game'.

            Returns:
                None
            """
            fig, ax = plt.subplots(1, 2, figsize=(30, 15))

            im1 = ax[0].imshow(cm1, interpolation='nearest', cmap=plt.cm.Blues)
            ax[0].set_title(title1)
            ax[0].set_ylabel('True label')
            ax[0].set_xlabel('Predicted label')
            fig.colorbar(im1, ax=ax[0])

            im2 = ax[1].imshow(cm2, interpolation='nearest', cmap=plt.cm.Blues)
            ax[1].set_title(title2)
            ax[1].set_ylabel('True label')
            ax[1].set_xlabel('Predicted label')
            fig.colorbar(im2, ax=ax[1])

            plt.tight_layout()  
            plt.show()

    @staticmethod
    def shap_plotter(shap_values, X_test, title, class_names, max_display=30):
        """
        Plots the summary plot using SHAP values.

        Args:
            shap_values (list): List of SHAP values. Each element should correspond to a different class.
            X_test (array-like): Test data.
            title (str): Title of the plot.
            class_names (list): List of class names.
            max_display (int, optional): Maximum number of features to display. Defaults to 30.

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=(30, 20))
        shap.summary_plot(shap_values, X_test, plot_type='bar', show=False, class_names=class_names,
                          title=title, max_display=max_display)
        fig = plt.gcf()
        fig.set_size_inches(30, 20)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_partial_dependence(model, X, feature_names, feature_index, title=None, figsize=(30, 30)):
        """
        Plot Partial Dependence Plot (PDP) for a specified feature using scikit-learn.

        Args:
            model: Trained model (e.g., RandomForestClassifier).
            X (array-like or DataFrame): Features data.
            feature_names (list): List of feature names.
            feature_index (int): Index of the feature for which PDP will be plotted.
            title (str, optional): Title for the plot. Defaults to None.
            figsize (tuple, optional): Size of the figure (width, height) in inches. Defaults to (10, 6).

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=figsize)
        if title:
            plt.title(title)
        plt.show()

    @staticmethod
    def plot_tsne(data, clusters=None, title='t-SNE Visualization', perplexity=30, n_components=2, random_state=None):
        """
        Plot t-SNE visualization of the provided data.

        Args:
            data (array-like): Data to visualize.
            clusters (array-like, optional): Cluster labels for the data points. If provided, points will be colored accordingly.
            title (str, optional): Title of the plot. Defaults to 't-SNE Visualization'.
            perplexity (float, optional): Perplexity parameter for t-SNE. Defaults to 30.
            n_components (int, optional): Number of dimensions of the embedded space. Defaults to 2.
            random_state (int, optional): Random state for t-SNE. Defaults to None.

        Returns:
            None
        """
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        embedded_data = tsne.fit_transform(data)
        plt.figure(figsize=(30, 30))

        if clusters is not None:
            unique_clusters = np.unique(clusters)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

            for cluster, color in zip(unique_clusters, colors):
                mask = (clusters == cluster)
                cluster_mean = np.mean(embedded_data[mask], axis=0)
                plt.scatter(embedded_data[mask, 0], embedded_data[mask, 1], color=color, s=150, label=f'Cluster {cluster}')
                plt.text(cluster_mean[0], cluster_mean[1], "Cluster "+str(cluster), fontsize=20)
            plt.legend()
        else:
            plt.scatter(embedded_data[:, 0], embedded_data[:, 1])

        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.show()
