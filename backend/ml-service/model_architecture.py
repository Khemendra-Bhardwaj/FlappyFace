#!/usr/bin/env python3
"""
Flappy Bird Emotion Game - Model Architecture
Define the CNN architecture for facial emotion recognition
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)

def create_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Create CNN model for emotion recognition
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of emotion classes
    
    Returns:
        Compiled Keras model
    """
    
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # First convolutional block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Second convolutional block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Third convolutional block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Fourth convolutional block
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    
    # First dense layer
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Second dense layer
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    logger.info(f"Created emotion recognition model with {model.count_params()} parameters")
    
    return model

def create_lightweight_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Create a lightweight CNN model for faster inference
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of emotion classes
    
    Returns:
        Compiled Keras model
    """
    
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # First convolutional block
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Second convolutional block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Third convolutional block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    
    # Dense layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    logger.info(f"Created lightweight emotion recognition model with {model.count_params()} parameters")
    
    return model

def create_mobilenet_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Create emotion recognition model based on MobileNet architecture
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of emotion classes
    
    Returns:
        Compiled Keras model
    """
    
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # Convert grayscale to RGB for MobileNet
    if input_shape[2] == 1:
        x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
        x = layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]))(x)
    else:
        x = inputs
    
    # Resize to 96x96 for better MobileNet performance
    x = layers.Resizing(96, 96)(x)
    
    # Load pre-trained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add base model
    x = base_model(x, training=False)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    logger.info(f"Created MobileNet-based emotion recognition model with {model.count_params()} parameters")
    
    return model

def get_model_summary(model):
    """Get a formatted summary of the model"""
    
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    return '\n'.join(summary_list)

def count_model_parameters(model):
    """Count trainable and non-trainable parameters"""
    
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    return {
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'total': total_params
    }

if __name__ == '__main__':
    # Test model creation
    model = create_emotion_model()
    print("Model Summary:")
    print(get_model_summary(model))
    
    params = count_model_parameters(model)
    print(f"\nParameters:")
    print(f"Trainable: {params['trainable']:,}")
    print(f"Non-trainable: {params['non_trainable']:,}")
    print(f"Total: {params['total']:,}") 