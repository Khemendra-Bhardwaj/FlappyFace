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
        # Look for the face_model.h5 (new trained model)
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'face_model.h5')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            logger.error("Please ensure face_model.h5 is in backend/ml-service/models/")
            return False
        
        logger.info(f"Loading face_model.h5 from {model_path}")
        
        # Load the model
        model = keras.models.load_model(model_path, compile=False)
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model_loaded = True
        logger.info("face_model.h5 loaded successfully!")
        
        # Test the model with a dummy prediction
        test_input = np.random.rand(1, 48, 48, 1).astype('float32')
        test_prediction = model.predict(test_input, verbose=0)
        logger.info(f"Model test successful. Output shape: {test_prediction.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False



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
        
        # Debug: Log all emotion predictions
        logger.info("All emotion predictions:")
        for i, (emotion, prob) in enumerate(zip(emotion_labels, predictions[0])):
            logger.info(f"  {emotion}: {prob:.4f}")
        
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
        
        # Debug: Log image array info
        logger.info(f"Image array shape: {image_array.shape}")
        logger.info(f"Image array min/max: {np.min(image_array):.3f}/{np.max(image_array):.3f}")
        logger.info(f"Image array mean: {np.mean(image_array):.3f}")
        
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

@app.route('/test_different_emotions', methods=['GET'])
def test_different_emotions():
    """Test endpoint that creates different types of images to test model bias"""
    try:
        results = []
        
        # Test 1: Random noise
        dummy_image = np.random.rand(48, 48) * 255
        dummy_image = dummy_image.astype(np.uint8)
        pil_image = Image.fromarray(dummy_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        image_array = preprocess_image(f"data:image/jpeg;base64,{image_base64}")
        emotion, confidence = predict_emotion(image_array)
        results.append({
            'test_type': 'random_noise',
            'emotion': emotion,
            'confidence': confidence
        })
        
        # Test 2: Bright image (should be happy-like)
        bright_image = np.ones((48, 48)) * 200
        bright_image = bright_image.astype(np.uint8)
        pil_image = Image.fromarray(bright_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        image_array = preprocess_image(f"data:image/jpeg;base64,{image_base64}")
        emotion, confidence = predict_emotion(image_array)
        results.append({
            'test_type': 'bright_image',
            'emotion': emotion,
            'confidence': confidence
        })
        
        # Test 3: Dark image (should be sad-like)
        dark_image = np.ones((48, 48)) * 50
        dark_image = dark_image.astype(np.uint8)
        pil_image = Image.fromarray(dark_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        image_array = preprocess_image(f"data:image/jpeg;base64,{image_base64}")
        emotion, confidence = predict_emotion(image_array)
        results.append({
            'test_type': 'dark_image',
            'emotion': emotion,
            'confidence': confidence
        })
        
        return jsonify({
            'success': True,
            'message': 'Different emotion tests completed',
            'results': results,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Error in test_different_emotions endpoint: {str(e)}")
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
    success = load_model()
    if not success:
        logger.error("Failed to load model. Service may not work properly.")

# Initialize model when module loads
initialize()

if __name__ == '__main__':
    # Load model immediately
    success = load_model()
    if not success:
        logger.error("Failed to load model. Exiting.")
        exit(1)
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5003))
    
    logger.info(f"Starting Flappy Bird ML Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True) 