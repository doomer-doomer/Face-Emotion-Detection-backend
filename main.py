from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
from flask_cors import CORS
import tensorflow as tf
import functools

app = Flask(__name__)
CORS(app)

trained_model = tf.keras.models.load_model('BasicRAF-DB794.h5')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Preload an image and cache the preprocessing result at the start
preloaded_image_path = os.path.join(UPLOAD_FOLDER, 'smilee.jpg')

if not os.path.exists(preloaded_image_path):
    raise FileNotFoundError("Preloaded image not found!")

def preprocess_image(image_path):
    # Read the image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    img = cv2.imread(image_path)

    # Resize the image to (100, 100)
    img_resized = cv2.resize(img, (100, 100))

    faces = face_cascade.detectMultiScale(img_resized, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the detected face region
        face = img_resized[y:y+h, x:x+w]
        # Resize the face to 100x100 pixels
        resized_face = cv2.resize(face, (100, 100))

    # Normalize pixel values to the range [0, 1]
    resized_face = resized_face / 255.0

    return resized_face

# Preprocess the preloaded image and cache the result
preloaded_image_preprocessed = preprocess_image(preloaded_image_path)

# Decorate preprocess_image with caching
@functools.lru_cache(maxsize=128)
def cached_preprocess_image(image_path):
    return preprocess_image(image_path)



@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Hello, World!'})

@app.route('/img', methods=['GET', 'POST'])
def process_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'message': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400
        if file:
            # Save the uploaded file temporarily
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Preprocess the image using cached function
            input_image = cached_preprocess_image(file_path)

            # Remove the temporary file
            os.remove(file_path)

            # Make predictions
            predictions = trained_model.predict(np.expand_dims(input_image, axis=0))

            # Get the predicted class label
            predicted_class = np.argmax(predictions)

            label = ['Surprised', 'Fear', 'Disgusted', 'Happy', 'Sad', 'Angry', 'Neutral']
            predicted_emotion = label[predicted_class]

            return jsonify({'message': 'Image processing successful', 'predicted_emotion': predicted_emotion}), 200
        else:
            return jsonify({'message': 'Invalid file'}), 400
    else:
        # Preprocess the preloaded image
        input_image = preloaded_image_preprocessed

        # Make predictions for preloaded image
        predictions = trained_model.predict(np.expand_dims(input_image, axis=0))

        # Get the predicted class label
        predicted_class = np.argmax(predictions)

        label = ['Surprised', 'Fear', 'Disgusted', 'Happy', 'Sad', 'Angry', 'Neutral']
        predicted_emotion = label[predicted_class]

        return jsonify({'message': 'Preloaded image processing successful', 'predicted_emotion': predicted_emotion}), 200

if __name__ == '__main__':
    app.run(debug=True)
