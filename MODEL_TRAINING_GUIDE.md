# Emotion Recognition Model Training Guide

## Overview

This guide will help you train a proper emotion recognition model using the FER2013 dataset and integrate it with the Flappy Bird emotion game.

## Current Issue

The current model is predicting random values because it's either using a dummy model or a model with random weights. We need to train a proper model on the FER2013 dataset.

## Step 1: Prepare the Dataset

### Download FER2013 Dataset

1. Go to [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
2. Download the `fer2013.csv` file
3. Upload it to your Google Drive in the folder: `/content/drive/MyDrive/fer2013/`

### Dataset Structure

The FER2013 dataset contains:
- **35,887 images** of facial expressions
- **7 emotion classes**: angry, disgust, fear, happy, sad, surprise, neutral
- **48x48 pixel grayscale images**
- **CSV format** with columns: emotion, pixels, usage

## Step 2: Train the Model in Google Colab

### Option A: Use the Provided Script

1. Open Google Colab: https://colab.research.google.com/
2. Create a new notebook
3. Copy the contents of `train_emotion_model_colab.py` into a cell
4. Run the script

### Option B: Manual Training

```python
# Install required packages
!pip install tensorflow pandas matplotlib scikit-learn

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy the training script content here
# (Use the content from train_emotion_model_colab.py)
```

### Training Process

The script will:
1. **Load the FER2013 dataset** from your Google Drive
2. **Preprocess the data** (normalize, reshape, encode labels)
3. **Build a CNN model** with the following architecture:
   - 4 convolutional blocks with batch normalization and dropout
   - 2 dense layers with batch normalization and dropout
   - Output layer with softmax activation
4. **Train the model** for up to 100 epochs with early stopping
5. **Save the best model** based on validation accuracy
6. **Generate training curves** and evaluation metrics

### Expected Results

With proper training, you should achieve:
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 65-75%
- **Test Accuracy**: 65-75%

## Step 3: Download the Trained Model

After training completes:

1. The model will be saved to: `/content/drive/MyDrive/fer2013/models/emotion_model.h5`
2. Download the model file from Colab
3. Place it in your local project at: `backend/ml-service/models/emotion_model.h5`

## Step 4: Update the Flask Service

### Option A: Use the Simplified Service

Replace the current `emotion_predictor.py` with `emotion_predictor_simple.py`:

```bash
cd backend/ml-service
mv emotion_predictor.py emotion_predictor_backup.py
cp emotion_predictor_simple.py emotion_predictor.py
```

### Option B: Update the Current Service

If you want to keep the current service, update the `load_model()` function in `emotion_predictor.py`:

```python
def load_model():
    """Load the trained emotion recognition model"""
    global model, model_loaded
    
    try:
        # Look for the trained model
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'emotion_model.h5')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            logger.error("Please place your trained model at: backend/ml-service/models/emotion_model.h5")
            return False
        
        logger.info(f"Loading trained model from {model_path}")
        
        # Load the model
        model = keras.models.load_model(model_path, compile=False)
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model_loaded = True
        logger.info("Trained emotion model loaded successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False
```

## Step 5: Test the Model

### Test the Flask Service

1. Start the ML service:
```bash
cd backend/ml-service
python emotion_predictor.py
```

2. Test the health endpoint:
```bash
curl http://localhost:5003/health
```

3. Test the model info:
```bash
curl http://localhost:5003/model_info
```

4. Test with a dummy prediction:
```bash
curl http://localhost:5003/test
```

### Test with Real Images

You can test the model with real facial images by sending POST requests to `/predict_emotion` with base64-encoded image data.

## Step 6: Integration with the Game

The trained model will now provide accurate emotion predictions that can be used to control the Flappy Bird game:

- **Happy**: Normal game speed
- **Sad**: Slower game speed
- **Angry**: Faster game speed
- **Fear**: More obstacles
- **Surprise**: Random game behavior
- **Disgust**: Inverted controls
- **Neutral**: Standard game mode

## Troubleshooting

### Common Issues

1. **Model not found error**:
   - Ensure the model file is in the correct location
   - Check file permissions

2. **Low prediction accuracy**:
   - Retrain the model with more epochs
   - Try data augmentation
   - Adjust the model architecture

3. **Memory issues during training**:
   - Reduce batch size
   - Use Google Colab Pro for more memory
   - Train on a subset of data first

4. **Compatibility issues**:
   - Use the simplified emotion predictor
   - Ensure TensorFlow versions match

### Performance Optimization

1. **Model size**: The trained model should be around 20-50MB
2. **Inference time**: Should be <100ms per prediction
3. **Memory usage**: Should be <500MB for the ML service

## Expected File Structure

After training and setup:

```
backend/
├── ml-service/
│   ├── models/
│   │   └── emotion_model.h5          # Your trained model
│   ├── emotion_predictor.py          # Updated Flask service
│   └── requirements.txt
├── app.py
└── requirements.txt
```

## Next Steps

1. Train the model using the Colab script
2. Download and place the model in the correct location
3. Update the Flask service
4. Test the integration
5. Deploy and enjoy your emotion-controlled Flappy Bird game!

## Additional Resources

- [FER2013 Dataset Paper](https://arxiv.org/abs/1608.01041)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Google Colab Documentation](https://colab.research.google.com/) 