import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import joblib
import os

class LSTMForecaster:
    """
    LSTM-based demand forecasting model for supply chain optimization.
    Supports multiple product categories and temporal patterns.
    """
    
    def __init__(self, sequence_length: int = 30, features: int = 5, 
                 lstm_units: List[int] = [128, 64], dropout_rate: float = 0.2):
        self.sequence_length = sequence_length
        self.features = features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_model(self) -> keras.Model:
        """
        Build LSTM architecture with attention mechanism
        """
        inputs = layers.Input(shape=(self.sequence_length, self.features))
        
        x = inputs
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = layers.LSTM(units, return_sequences=return_sequences, 
                          name=f'lstm_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Attention layer
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(self.lstm_units[-1])(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Output layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate / 2)(x)
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='LSTM_Forecaster')
        return model
    
    def compile_model(self, learning_rate: float = 0.001):
        """
        Compile model with optimizer and loss function
        """
        self.model = self.build_model()
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mape']
        )
        print(self.model.summary())
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i+self.sequence_length])
            y.append(data[i+self.sequence_length, 0])  # Predict demand
        return np.array(X), np.array(y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32):
        """
        Train the LSTM model with early stopping
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance
        """
        predictions = self.predict(X_test)
        mse = np.mean((predictions.flatten() - y_test) ** 2)
        mae = np.mean(np.abs(predictions.flatten() - y_test))
        mape = np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'rmse': np.sqrt(mse)
        }
    
    def save_model(self, filepath: str):
        """
        Save model and scaler
        """
        self.model.save(filepath)
        joblib.dump(self.scaler, filepath.replace('.h5', '_scaler.pkl'))
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model and scaler
        """
        self.model = keras.models.load_model(filepath)
        self.scaler = joblib.load(filepath.replace('.h5', '_scaler.pkl'))
        print(f"Model loaded from {filepath}")
