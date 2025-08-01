"""
Base model class for forex trading ML models.
Defines the interface that all ML models must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path

class BaseModel(ABC):
    """Abstract base class for all ML models."""
    
    def __init__(self, name: str, model_type: str = "classification"):
        """
        Initialize the base model.
        
        Args:
            name: Name of the model
            model_type: Type of model ('classification' or 'regression')
        """
        self.name = name
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.training_history = {}
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> Any:
        """
        Build the model architecture.
        
        Args:
            input_shape: Shape of input data
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters
            
        Returns:
            Training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        pass
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict class probabilities (for classification models).
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Class probabilities
        """
        if self.model_type != "classification":
            raise NotImplementedError("predict_proba only available for classification models")
        return self.predict(X, **kwargs)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)
        
        if self.model_type == "classification":
            return self._classification_metrics(y_test, predictions)
        else:
            return self._regression_metrics(y_test, predictions)
    
    def _classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC AUC for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                if hasattr(self, 'predict_proba') and self.model_type == "classification":
                    y_proba = self.predict_proba(y_true)
                    if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def _regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'name': self.name,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save using joblib for sklearn models, or custom save for deep learning models
        try:
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.name = model_data['name']
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.training_history = model_data.get('training_history', {})
            self.is_trained = model_data['is_trained']
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self.is_trained:
            return None
        
        # Try to get feature importance from the model
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_).flatten()
        else:
            return None
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        else:
            feature_names = self.feature_names
        
        return dict(zip(feature_names, importances))
    
    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Set feature names for the model.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names) if self.feature_names else None,
            'training_history': self.training_history
        }
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.name} ({self.model_type}) - Trained: {self.is_trained}"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"<{self.__class__.__name__}(name='{self.name}', type='{self.model_type}', trained={self.is_trained})>"