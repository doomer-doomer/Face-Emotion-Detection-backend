# Your optimized Flask code
# Import necessary libraries
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
from flask_cors import CORS
import tensorflow as tf

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the trained model during application startup
trained_model = tf.keras.models.load_model('BasicRAF-DB794.h5')

# Define a function for image preprocessing
def preprocess_image(image):
    # Resize the image to (100, 100)
    resized_image = cv2.resize(image, (100, 100))
    # Normalize pixel values to the range [0, 1]
    resized_image = resized_image / 255.0
    return resized_image

# Define route for image processing
@app.route('/img', methods=['POST'])
def process_image():
    # Check if request contains file
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'message': 'No file part'}), 400

        # Get the uploaded file
        file = request.files['file']

        # Check if file is valid
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400

        # Read image from file
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Preprocess the image
        input_image = preprocess_image(img)

        # Make predictions using the trained model
        predictions = trained_model.predict(np.expand_dims(input_image, axis=0))

        # Get the predicted class label
        predicted_class = np.argmax(predictions)
        label = ['Surprised', 'Fear', 'Disgusted', 'Happy', 'Sad', 'Angry', 'Neutral']
        predicted_emotion = label[predicted_class]

        return jsonify({'message': 'Image processing successful', 'predicted_emotion': predicted_emotion}), 200
    elif request.method == 'GET':
        return jsonify({'message': 'This is a GET request for image processing'}), 200
    else:
        return jsonify({'message': 'Method not allowed'}), 405

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
