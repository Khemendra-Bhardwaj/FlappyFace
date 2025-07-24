#!/usr/bin/env python3
"""
Flappy Bird Emotion Game - Model Training Script
Train a CNN model for facial emotion recognition using FER2013 dataset
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Import custom modules
from data_loader import load_fer2013_data
from model_architecture import create_emotion_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_emotion_model():
    """Train the emotion recognition model"""
    
    logger.info("Starting emotion model training...")
    
    # Load and preprocess data
    logger.info("Loading FER2013 dataset...")
    X_train, X_test, y_train, y_test, emotion_labels = load_fer2013_data()
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Emotion labels: {emotion_labels}")
    
    # Create model
    logger.info("Creating model architecture...")
    model = create_emotion_model(input_shape=(48, 48, 1), num_classes=7)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'models/emotion_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    logger.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    logger.info(f"Test loss: {test_loss:.4f}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    logger.info("Classification Report:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=emotion_labels))
    
    # Save training history
    save_training_history(history, test_accuracy, test_loss)
    
    # Plot training curves
    plot_training_curves(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test_classes, y_pred_classes, emotion_labels)
    
    logger.info("Training completed successfully!")
    return model

def save_training_history(history, test_accuracy, test_loss):
    """Save training history and metrics"""
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save training history
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'training_date': datetime.now().isoformat()
    }
    
    # Save to JSON
    import json
    with open('results/training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    logger.info("Training history saved to results/training_history.json")

def plot_training_curves(history):
    """Plot training and validation curves"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Training curves saved to results/training_curves.png")

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Confusion matrix saved to results/confusion_matrix.png")

def main():
    """Main training function"""
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train model
    model = train_emotion_model()
    
    # Save final model
    model.save('models/emotion_model_final.h5')
    logger.info("Final model saved to models/emotion_model_final.h5")
    
    # Save model for production (optimized)
    model.save('models/emotion_model.h5')
    logger.info("Production model saved to models/emotion_model.h5")

if __name__ == '__main__':
    main() 