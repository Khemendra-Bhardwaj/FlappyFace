#!/usr/bin/env python3
"""
Flappy Bird Emotion Game - Data Loader
Load and preprocess FER2013 dataset for emotion recognition training
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import logging

logger = logging.getLogger(__name__)

def load_fer2013_data(test_size=0.2, random_state=42):
    """
    Load and preprocess FER2013 dataset
    
    Args:
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test, emotion_labels
    """
    
    # Emotion labels mapping
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Try to load from CSV file first
    csv_path = 'fer2013.csv'
    if os.path.exists(csv_path):
        logger.info(f"Loading FER2013 dataset from {csv_path}")
        return load_from_csv(csv_path, test_size, random_state, emotion_labels)
    
    # If CSV not found, create synthetic data for testing
    logger.warning("FER2013 CSV file not found. Creating synthetic data for testing.")
    return create_synthetic_data(test_size, random_state, emotion_labels)

def load_from_csv(csv_path, test_size, random_state, emotion_labels):
    """Load data from FER2013 CSV file"""
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Extract features and labels
    pixels = df['pixels'].values
    emotions = df['emotion'].values
    
    # Convert pixel strings to arrays
    X = []
    for pixel_string in pixels:
        pixel_array = np.array(pixel_string.split(), dtype=np.uint8)
        X.append(pixel_array.reshape(48, 48))
    
    X = np.array(X)
    
    # Convert emotions to categorical
    y = tf.keras.utils.to_categorical(emotions, num_classes=len(emotion_labels))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=emotions
    )
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Add channel dimension
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    logger.info(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test, emotion_labels

def create_synthetic_data(test_size, random_state, emotion_labels):
    """Create synthetic data for testing when real dataset is not available"""
    
    logger.info("Creating synthetic emotion recognition dataset...")
    
    # Generate synthetic data
    n_samples = 10000
    n_train = int(n_samples * (1 - test_size))
    n_test = n_samples - n_train
    
    # Create synthetic images (random patterns)
    X_train = np.random.rand(n_train, 48, 48, 1).astype('float32')
    X_test = np.random.rand(n_test, 48, 48, 1).astype('float32')
    
    # Create synthetic labels
    y_train = np.random.randint(0, len(emotion_labels), n_train)
    y_test = np.random.randint(0, len(emotion_labels), n_test)
    
    # Convert to categorical
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(emotion_labels))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(emotion_labels))
    
    logger.info(f"Created {n_train} synthetic training samples and {n_test} test samples")
    
    return X_train, X_test, y_train, y_test, emotion_labels

def download_fer2013_dataset():
    """Download FER2013 dataset (placeholder for future implementation)"""
    
    logger.info("FER2013 dataset download not implemented yet.")
    logger.info("Please download the dataset manually from:")
    logger.info("https://www.kaggle.com/datasets/msambare/fer2013")
    logger.info("And place the fer2013.csv file in the current directory.")
    
    return False

def augment_data(X, y, augmentation_factor=2):
    """
    Augment training data using TensorFlow data augmentation
    
    Args:
        X: Input images
        y: Labels
        augmentation_factor: How many times to augment the data
    
    Returns:
        Augmented X and y
    """
    
    # Create data augmentation pipeline
    data_augmentation = keras.Sequential([
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomFlip("horizontal"),
    ])
    
    # Augment data
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X)):
        # Original sample
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        
        # Augmented samples
        for _ in range(augmentation_factor - 1):
            augmented = data_augmentation(X[i:i+1], training=True)
            X_augmented.append(augmented[0])
            y_augmented.append(y[i])
    
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    
    logger.info(f"Data augmented: {len(X)} -> {len(X_augmented)} samples")
    
    return X_augmented, y_augmented

def preprocess_for_inference(image):
    """
    Preprocess a single image for inference
    
    Args:
        image: Input image (numpy array)
    
    Returns:
        Preprocessed image ready for model input
    """
    
    # Ensure image is grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB to grayscale
        image = np.mean(image, axis=2)
    
    # Resize to 48x48
    if image.shape != (48, 48):
        import cv2
        image = cv2.resize(image, (48, 48))
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Add batch and channel dimensions
    image = np.expand_dims(image, axis=[0, -1])
    
    return image

if __name__ == '__main__':
    # Test data loading
    X_train, X_test, y_train, y_test, labels = load_fer2013_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Labels: {labels}") 