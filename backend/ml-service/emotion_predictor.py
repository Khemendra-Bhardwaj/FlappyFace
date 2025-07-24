#!/usr/bin/env python3
"""
Flappy Bird Emotion Game - ML Service
Flask application for real-time emotion prediction using custom CNN model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import numpy as np
import cv2
from PIL import Image
import io
import logging
import tensorflow as tf
from tensorflow import keras
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
model = None
model_loaded = False
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_model():
    """Load the trained emotion recognition model"""
    global model, model_loaded
    
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'emotion_model.h5')
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            logger.info("Please place your trained model at: backend/ml-service/models/emotion_model.h5")
            logger.info("Creating a dummy model for testing purposes")
            model = create_dummy_model()
            model_loaded = True
            return
        
        logger.info(f"Loading your trained model from {model_path}")
        model = keras.models.load_model(model_path)
        model_loaded = True
        logger.info("Your trained emotion model loaded successfully!")
        
        # Test the model with a dummy prediction
        test_input = np.random.rand(1, 48, 48, 1).astype('float32')
        test_prediction = model.predict(test_input, verbose=0)
        logger.info(f"Model test successful. Output shape: {test_prediction.shape}")
        
    except Exception as e:
        logger.error(f"Error loading your model: {str(e)}")
        logger.info("Creating a dummy model for testing purposes")
        model = create_dummy_model()
        model_loaded = True

def create_dummy_model():
    """Create a dummy model for testing when the real model is not available"""
    model = keras.Sequential([
        keras.layers.Input(shape=(48, 48, 1)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image_data):
    """
    Preprocess image for emotion prediction
    Args:
        image_data: Base64 encoded image string
    Returns:
        Preprocessed image array
    """
    try:
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 48x48
        image = image.resize((48, 48))
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0
        
        # Add batch and channel dimensions
        image_array = np.expand_dims(image_array, axis=[0, -1])
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_emotion(image_array):
    """
    Predict emotion from preprocessed image
    Args:
        image_array: Preprocessed image array
    Returns:
        Tuple of (emotion_label, confidence_score)
    """
    try:
        if not model_loaded or model is None:
            raise Exception("Model not loaded")
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        emotion_index = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_index])
        
        emotion_label = emotion_labels[emotion_index]
        
        return emotion_label, confidence
        
    except Exception as e:
        logger.error(f"Error predicting emotion: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'flappy-bird-ml-service',
        'model_loaded': model_loaded,
        'timestamp': time.time()
    })

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion_endpoint():
    """Predict emotion from image data"""
    try:
        data = request.get_json()
        
        if not data or 'image_data' not in data:
            return jsonify({
                'success': False,
                'error': 'Image data required'
            }), 400
        
        image_data = data['image_data']
        
        # Preprocess image
        image_array = preprocess_image(image_data)
        
        # Predict emotion
        emotion, confidence = predict_emotion(image_array)
        
        logger.info(f"Emotion predicted: {emotion} (confidence: {confidence:.3f})")
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Error in emotion prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint that returns a dummy prediction"""
    try:
        # Create a dummy image (random noise)
        dummy_image = np.random.rand(48, 48) * 255
        dummy_image = dummy_image.astype(np.uint8)
        
        # Convert to base64
        pil_image = Image.fromarray(dummy_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Preprocess and predict
        image_array = preprocess_image(f"data:image/jpeg;base64,{image_base64}")
        emotion, confidence = predict_emotion(image_array)
        
        return jsonify({
            'success': True,
            'message': 'Test prediction completed',
            'test_emotion': emotion,
            'test_confidence': confidence,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        info = {
            'model_loaded': model_loaded,
            'emotion_labels': emotion_labels,
            'input_shape': None,
            'model_summary': None
        }
        
        if model_loaded and model is not None:
            info['input_shape'] = model.input_shape
            # Get model summary as string
            summary_list = []
            model.summary(print_fn=lambda x: summary_list.append(x))
            info['model_summary'] = '\n'.join(summary_list)
        
        return jsonify({
            'success': True,
            'model_info': info
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/emotions', methods=['GET'])
def get_emotions():
    """Get list of supported emotions"""
    return jsonify({
        'success': True,
        'emotions': emotion_labels,
        'count': len(emotion_labels)
    })

# Load model on startup
def initialize():
    """Initialize the model before first request"""
    load_model()

# Initialize model when module loads
initialize()

if __name__ == '__main__':
    # Load model immediately
    load_model()
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5003))
    
    logger.info(f"Starting Flappy Bird ML Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True) 