"""
LSTM neural network model for forex price prediction.
Uses TensorFlow/Keras for deep learning implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import logging

from .base_model import BaseModel

class LSTMModel(BaseModel):
    """LSTM neural network for forex trading prediction."""
    
    def __init__(self, sequence_length: int = 60, model_type: str = "classification"):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Length of input sequences
            model_type: Type of model ('classification' or 'regression')
        """
        super().__init__(name="LSTM", model_type=model_type)
        self.sequence_length = sequence_length
        self.scaler_X = None
        self.scaler_y = None
        
        # Model architecture parameters
        self.lstm_units = [64, 32, 16]
        self.dropout_rate = 0.2
        self.l2_reg = 0.001
        self.learning_rate = 0.001
        
        # Training parameters
        self.batch_size = 32
        self.epochs = 100
        self.patience = 10
        
    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> tf.keras.Model:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            **kwargs: Additional model parameters
            
        Returns:
            Compiled Keras model
        """
        # Extract parameters from kwargs
        lstm_units = kwargs.get('lstm_units', self.lstm_units)
        dropout_rate = kwargs.get('dropout_rate', self.dropout_rate)
        l2_reg = kwargs.get('l2_reg', self.l2_reg)
        learning_rate = kwargs.get('learning_rate', self.learning_rate)
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=lstm_units[0],
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Second LSTM layer
        if len(lstm_units) > 1:
            model.add(LSTM(
                units=lstm_units[1],
                return_sequences=len(lstm_units) > 2,
                kernel_regularizer=l2(l2_reg),
                recurrent_regularizer=l2(l2_reg)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Third LSTM layer (if specified)
        if len(lstm_units) > 2:
            model.add(LSTM(
                units=lstm_units[2],
                return_sequences=False,
                kernel_regularizer=l2(l2_reg),
                recurrent_regularizer=l2(l2_reg)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
        
        # Output layer
        if self.model_type == "classification":
            model.add(Dense(1, activation='sigmoid'))
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
        else:
            model.add(Dense(1, activation='linear'))
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae']
            )
        
        self.model = model
        return model
    
    def prepare_sequences(self, data: np.ndarray, target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare sequential data for LSTM training/prediction.
        
        Args:
            data: Input features array
            target: Target values array (optional)
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = [] if target is not None else None
        
        for i in range(self.sequence_length, len(data)):
            X_sequences.append(data[i-self.sequence_length:i])
            if target is not None:
                y_sequences.append(target[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences) if y_sequences else None
        
        return X_sequences, y_sequences
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters
            
        Returns:
            Training history
        """
        # Extract training parameters
        batch_size = kwargs.get('batch_size', self.batch_size)
        epochs = kwargs.get('epochs', self.epochs)
        patience = kwargs.get('patience', self.patience)
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        # Build model if not already built
        if self.model is None:
            input_shape = (self.sequence_length, X_train.shape[1])
            self.build_model(input_shape, **kwargs)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Add model checkpoint if specified
        checkpoint_path = kwargs.get('checkpoint_path')
        if checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_loss' if validation_data else 'loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        self.logger.info(f"Training LSTM model with {len(X_train_seq)} sequences")
        
        # Train the model
        history = self.model.fit(
            X_train_seq, y_train_seq,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        self.training_history = history.history
        
        # Log training results
        final_loss = history.history['loss'][-1]
        if validation_data:
            final_val_loss = history.history['val_loss'][-1]
            self.logger.info(f"Training completed. Final loss: {final_loss:.4f}, Val loss: {final_val_loss:.4f}")
        else:
            self.logger.info(f"Training completed. Final loss: {final_loss:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with the LSTM model.
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare sequences
        X_seq, _ = self.prepare_sequences(X)
        
        if len(X_seq) == 0:
            raise ValueError(f"Input data too short. Need at least {self.sequence_length} samples")
        
        # Make predictions
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Flatten predictions if necessary
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        
        return predictions
    
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
        
        probabilities = self.predict(X, **kwargs)
        
        # For binary classification, return probabilities for both classes
        if probabilities.ndim == 1:
            prob_class_1 = probabilities
            prob_class_0 = 1 - probabilities
            return np.column_stack([prob_class_0, prob_class_1])
        
        return probabilities
    
    def predict_next(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Predict next N time steps using recursive prediction.
        
        Args:
            X: Input features (last sequence_length samples)
            steps: Number of steps to predict ahead
            
        Returns:
            Predictions for next N steps
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(X) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} samples for prediction")
        
        # Use the last sequence_length samples
        current_sequence = X[-self.sequence_length:].copy()
        predictions = []
        
        for _ in range(steps):
            # Reshape for prediction
            X_seq = current_sequence.reshape(1, self.sequence_length, -1)
            
            # Make prediction
            pred = self.model.predict(X_seq, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence for next prediction (simplified approach)
            # In practice, you might want to update with actual feature values
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred  # Update the first feature (e.g., close price)
        
        return np.array(predictions)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the LSTM model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save the Keras model
        model_path = filepath.replace('.pkl', '.h5')
        self.model.save(model_path)
        
        # Save additional metadata
        model_data = {
            'name': self.name,
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'model_path': model_path
        }
        
        import joblib
        joblib.dump(model_data, filepath)
        self.logger.info(f"LSTM model saved to {filepath} and {model_path}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load the LSTM model.
        
        Args:
            filepath: Path to load the model from
        """
        import joblib
        
        # Load metadata
        model_data = joblib.load(filepath)
        
        self.name = model_data['name']
        self.model_type = model_data['model_type']
        self.sequence_length = model_data['sequence_length']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data.get('training_history', {})
        self.is_trained = model_data['is_trained']
        
        # Load the Keras model
        model_path = model_data['model_path']
        self.model = tf.keras.models.load_model(model_path)
        
        self.logger.info(f"LSTM model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            String representation of model architecture
        """
        if self.model is None:
            return "Model not built yet"
        
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.training_history:
            self.logger.warning("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('LSTM Training History')
        
        # Loss
        axes[0, 0].plot(self.training_history['loss'], label='Training Loss')
        if 'val_loss' in self.training_history:
            axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy (for classification)
        if self.model_type == "classification" and 'accuracy' in self.training_history:
            axes[0, 1].plot(self.training_history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in self.training_history:
                axes[0, 1].plot(self.training_history['val_accuracy'], label='Validation Accuracy')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
        
        # Learning rate
        if 'lr' in self.training_history:
            axes[1, 0].plot(self.training_history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
        
        # Additional metrics
        if self.model_type == "regression" and 'mae' in self.training_history:
            axes[1, 1].plot(self.training_history['mae'], label='Training MAE')
            if 'val_mae' in self.training_history:
                axes[1, 1].plot(self.training_history['val_mae'], label='Validation MAE')
            axes[1, 1].set_title('Mean Absolute Error')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()