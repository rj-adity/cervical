import os
import zipfile
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)

# Define paths
MODEL_ZIP_PATH = 'cancer_classifier_model.zip'
MODEL_DIR = 'model'  # Directory where the model will be extracted

# Unzip the model if not already unzipped
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

# Load the model after unzipping
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'cancer_classifier.keras'))

# Categories (as per your model's prediction)
categories = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"]

def preprocess_image(image):
    """Preprocess the image to match the input shape expected by the model."""
    image = image.resize((224, 224))  # Resize to model's expected input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def home():
    """Render the homepage (HTML form)."""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """Classify an image uploaded via POST request."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Open and preprocess the image
        image = Image.open(file)
        image = preprocess_image(image)

        # Make predictions using the model
        predictions = model.predict(image)
        category = categories[np.argmax(predictions)]
        confidence = np.max(predictions) * 100  # Convert to percentage

        return render_template('index.html', category=category, confidence=confidence)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For production deployment, set debug=False and consider using a production-ready server like Gunicorn.
    app.run(debug=True, host='0.0.0.0', port=5000)  # This makes your app accessible externally for deployment
