#!/usr/bin/env python3
"""
Train and Evaluate LSTM Demand Forecasting Model
Generates synthetic data, trains model, and displays comprehensive results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.append('src')
from models.lstm_model import LSTMForecaster

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (15, 10)

def generate_synthetic_data(n_samples=1000, n_products=5):
    """
    Generate synthetic supply chain demand data
    """
    print("\n📊 Generating Synthetic Supply Chain Data...")
    
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    data = []
    
    for product_id in range(n_products):
        # Base demand with trend and seasonality
        trend = np.linspace(100, 300, n_samples)
        seasonality = 50 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
        weekly_pattern = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
        noise = np.random.normal(0, 15, n_samples)
        
        demand = trend + seasonality + weekly_pattern + noise
        demand = np.maximum(demand, 0)  # No negative demand
        
        product_data = pd.DataFrame({
            'date': dates,
            'product_id': f'PROD_{product_id:03d}',
            'demand': demand,
            'price': 50 + 20 * np.random.randn(n_samples).cumsum() / 10,
            'inventory': 500 + 100 * np.random.randn(n_samples).cumsum() / 10,
            'promotion': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        })
        data.append(product_data)
    
    df = pd.concat(data, ignore_index=True)
    print(f"✓ Generated {len(df)} records for {n_products} products")
    return df

def prepare_features(df):
    """
    Engineer features for the model
    """
    print("\n🔧 Engineering Features...")
    
    df = df.copy()
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear
    
    # Normalize features
    feature_cols = ['demand', 'price', 'inventory', 'promotion', 'day_of_week']
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print(f"✓ Created features: {feature_cols}")
    return df[feature_cols].values, scaler

def train_model(X_train, y_train, X_val, y_val):
    """
    Train LSTM model
    """
    print("\n🤖 Training LSTM Model...")
    print("=" * 50)
    
    # Initialize model
    model = LSTMForecaster(
        sequence_length=30,
        features=5,
        lstm_units=[128, 64],
        dropout_rate=0.2
    )
    
    # Compile
    model.compile_model(learning_rate=0.001)
    
    # Train
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    print("\n✓ Training Complete!")
    return model, history

def evaluate_and_visualize(model, X_test, y_test, history):
    """
    Evaluate model and create comprehensive visualizations
    """
    print("\n📈 Evaluating Model Performance...")
    print("=" * 50)
    
    # Get predictions
    predictions = model.predict(X_test).flatten()
    
    # Calculate metrics
    metrics = model.evaluate(X_test, y_test)
    
    print("\n📊 PERFORMANCE METRICS:")
    print(f"  • RMSE:  {metrics['rmse']:.4f}")
    print(f"  • MAE:   {metrics['mae']:.4f}")
    print(f"  • MAPE:  {metrics['mape']:.2f}%")
    print(f"  • MSE:   {metrics['mse']:.4f}")
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Training History
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Predictions vs Actual
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(y_test[:200], label='Actual Demand', linewidth=2, alpha=0.7)
    ax2.plot(predictions[:200], label='Predicted Demand', linewidth=2, alpha=0.7)
    ax2.set_title('Demand Forecasting: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Normalized Demand')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter Plot
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(y_test, predictions, alpha=0.5, s=10)
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax3.set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Actual Values')
    ax3.set_ylabel('Predicted Values')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error Distribution
    ax4 = plt.subplot(2, 3, 4)
    errors = y_test - predictions
    ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax4.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Error')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # 5. MAE over time
    ax5 = plt.subplot(2, 3, 5)
    if 'mae' in history.history:
        ax5.plot(history.history['mae'], label='Training MAE', linewidth=2)
    if 'val_mae' in history.history:
        ax5.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax5.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('MAE')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Metrics Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    metrics_text = f"""
    📊 MODEL PERFORMANCE SUMMARY
    
    Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}
    Mean Absolute Error (MAE): {metrics['mae']:.4f}
    Mean Absolute Percentage Error: {metrics['mape']:.2f}%
    Mean Squared Error (MSE): {metrics['mse']:.4f}
    
    🎯 Model Accuracy: {100 - metrics['mape']:.2f}%
    
    ✨ Training completed successfully!
    Total test samples: {len(y_test)}
    Sequence length: 30 days
    Features: 5 (demand, price, inventory, promotion, day_of_week)
    """
    ax6.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/model_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization to 'results/model_evaluation.png'")
    plt.show()
    
    return metrics

def main():
    """
    Main execution function
    """
    print("\n" + "="*60)
    print("🚀 AI-POWERED SUPPLY CHAIN DEMAND FORECASTING")
    print("="*60)
    
    # Generate data
    df = generate_synthetic_data(n_samples=1000, n_products=3)
    
    # Prepare features
    features, scaler = prepare_features(df)
    
    # Create sequences
    print("\n🔄 Creating Time Sequences...")
    model = LSTMForecaster(sequence_length=30, features=5)
    X, y = model.prepare_sequences(features)
    print(f"✓ Created {len(X)} sequences")
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Validation samples: {len(X_val)}")
    print(f"  • Test samples: {len(X_test)}")
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate and visualize
    metrics = evaluate_and_visualize(model, X_test, y_test, history)
    
    # Save model
    print("\n💾 Saving Model...")
    os.makedirs('models', exist_ok=True)
    model.save_model('models/lstm_forecaster.h5')
    print("✓ Model saved to 'models/lstm_forecaster.h5'")
    
    print("\n" + "="*60)
    print("✅ COMPLETE! Model trained and evaluated successfully.")
    print("="*60)
    print("\n📁 Output Files:")
    print("  • results/model_evaluation.png - Comprehensive visualizations")
    print("  • models/lstm_forecaster.h5 - Trained model")
    print("  • models/lstm_forecaster_scaler.pkl - Feature scaler")
    print("\n💡 Next Steps:")
    print("  1. Check results/model_evaluation.png for visual analysis")
    print("  2. Use the trained model for real-time predictions")
    print("  3. Deploy to AWS for production use")
    print("\n")

if __name__ == '__main__':
    main()
