"""
Random Forest model for forex trading prediction.
Uses scikit-learn's RandomForestClassifier/Regressor.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import logging

from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest model for forex trading prediction."""
    
    def __init__(self, model_type: str = "classification"):
        """
        Initialize Random Forest model.
        
        Args:
            model_type: Type of model ('classification' or 'regression')
        """
        super().__init__(name="RandomForest", model_type=model_type)
        
        # Default hyperparameters
        self.n_estimators = 100
        self.max_depth = None
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.max_features = 'sqrt'
        self.bootstrap = True
        self.random_state = 42
        self.n_jobs = -1
    
    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> Any:
        """
        Build Random Forest model.
        
        Args:
            input_shape: Shape of input data (ignored for RF)
            **kwargs: Model hyperparameters
            
        Returns:
            RandomForest model instance
        """
        # Extract hyperparameters from kwargs
        n_estimators = kwargs.get('n_estimators', self.n_estimators)
        max_depth = kwargs.get('max_depth', self.max_depth)
        min_samples_split = kwargs.get('min_samples_split', self.min_samples_split)
        min_samples_leaf = kwargs.get('min_samples_leaf', self.min_samples_leaf)
        max_features = kwargs.get('max_features', self.max_features)
        bootstrap = kwargs.get('bootstrap', self.bootstrap)
        random_state = kwargs.get('random_state', self.random_state)
        n_jobs = kwargs.get('n_jobs', self.n_jobs)
        
        if self.model_type == "classification":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=random_state,
                n_jobs=n_jobs
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=random_state,
                n_jobs=n_jobs
            )
        
        return self.model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (not used in RF training)
            y_val: Validation targets (not used in RF training)
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics
        """
        # Build model if not already built
        if self.model is None:
            self.build_model(X_train.shape, **kwargs)
        
        self.logger.info(f"Training Random Forest with {X_train.shape[0]} samples and {X_train.shape[1]} features")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions = self.model.predict(X_train)
        
        if self.model_type == "classification":
            train_metrics = self._classification_metrics(y_train, train_predictions)
        else:
            train_metrics = self._regression_metrics(y_train, train_predictions)
        
        # Calculate validation metrics if validation data is provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            if self.model_type == "classification":
                val_metrics = self._classification_metrics(y_val, val_predictions)
                val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
            else:
                val_metrics = self._regression_metrics(y_val, val_predictions)
                val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
        
        # Store training history
        self.training_history = {**train_metrics, **val_metrics}
        
        self.logger.info(f"Training completed. Train accuracy/R2: {train_metrics.get('accuracy', train_metrics.get('r2_score', 'N/A')):.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with the Random Forest model.
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict class probabilities for classification.
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Class probabilities
        """
        if self.model_type != "classification":
            raise NotImplementedError("predict_proba only available for classification models")
        
        return self.model.predict_proba(X)
    
    def hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        method: str = "grid",
        cv: int = 5,
        n_iter: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            method: Tuning method ('grid' or 'random')
            cv: Number of cross-validation folds
            n_iter: Number of iterations for random search
            **kwargs: Additional parameters
            
        Returns:
            Best parameters and scores
        """
        # Define parameter grid
        if self.model_type == "classification":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            scoring = 'accuracy'
        else:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            scoring = 'neg_mean_squared_error'
        
        # Override with custom parameter grid if provided
        param_grid.update(kwargs.get('param_grid', {}))
        
        # Build base model
        if self.model is None:
            self.build_model(X_train.shape)
        
        self.logger.info(f"Starting {method} search for hyperparameter tuning...")
        
        # Perform hyperparameter search
        if method == "grid":
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=1
            )
        elif method == "random":
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown tuning method: {method}")
        
        # Fit the search
        search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = search.best_estimator_
        self.is_trained = True
        
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
        
        self.logger.info(f"Hyperparameter tuning completed. Best score: {search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {search.best_params_}")
        
        return results
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores from the Random Forest.
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self.is_trained:
            return None
        
        importances = self.model.feature_importances_
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        else:
            feature_names = self.feature_names
        
        # Sort by importance
        importance_dict = dict(zip(feature_names, importances))
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None) -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            save_path: Path to save the plot (optional)
        """
        importance_dict = self.get_feature_importance()
        
        if importance_dict is None:
            self.logger.warning("Model not trained or no feature importance available")
            return
        
        import matplotlib.pyplot as plt
        
        # Get top N features
        top_features = list(importance_dict.items())[:top_n]
        features, importances = zip(*top_features)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - Random Forest')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def get_tree_info(self) -> Dict[str, Any]:
        """
        Get information about the trees in the forest.
        
        Returns:
            Dictionary with tree information
        """
        if not self.is_trained:
            return {}
        
        # Calculate tree depths
        tree_depths = [tree.get_depth() for tree in self.model.estimators_]
        
        # Calculate number of leaves
        tree_leaves = [tree.get_n_leaves() for tree in self.model.estimators_]
        
        return {
            'n_trees': len(self.model.estimators_),
            'avg_depth': np.mean(tree_depths),
            'max_depth': np.max(tree_depths),
            'min_depth': np.min(tree_depths),
            'avg_leaves': np.mean(tree_leaves),
            'max_leaves': np.max(tree_leaves),
            'min_leaves': np.min(tree_leaves)
        }
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model_type == "classification":
            # For classification, use prediction probabilities as uncertainty measure
            probabilities = self.model.predict_proba(X)
            predictions = self.model.predict(X)
            
            # Calculate uncertainty as entropy
            uncertainties = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
            
        else:
            # For regression, use predictions from individual trees
            tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            predictions = np.mean(tree_predictions, axis=0)
            uncertainties = np.std(tree_predictions, axis=0)
        
        return predictions, uncertainties
    
    def partial_dependence_plot(self, X: np.ndarray, feature_idx: int, save_path: Optional[str] = None) -> None:
        """
        Create partial dependence plot for a specific feature.
        
        Args:
            X: Input features
            feature_idx: Index of the feature to plot
            save_path: Path to save the plot (optional)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before creating partial dependence plot")
        
        from sklearn.inspection import PartialDependenceDisplay
        import matplotlib.pyplot as plt
        
        feature_name = self.feature_names[feature_idx] if self.feature_names else f'Feature {feature_idx}'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        display = PartialDependenceDisplay.from_estimator(
            self.model, X, [feature_idx], ax=ax, feature_names=self.feature_names
        )
        
        plt.title(f'Partial Dependence Plot - {feature_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Partial dependence plot saved to {save_path}")
        
        plt.show()