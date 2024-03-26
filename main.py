from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
from flask_cors import CORS
import tensorflow as tf
import functools
from collections import Counter

app = Flask(__name__)
CORS(app)

trained_model = tf.keras.models.load_model('BasicRAF-DB794.h5')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    img_resized = cv2.resize(image, (100, 100))
    # Normalize pixel values to the range [0, 1]
    resized_face = img_resized / 255.0
    return resized_face

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Hello, World!'})

@app.route('/img', methods=['POST'])
def process_images():
    if 'file' not in request.files:
        return jsonify({'message': 'No files found'}), 400

    files = request.files.getlist('file')
    if len(files) == 0:
        return jsonify({'message': 'No files found'}), 400

    predicted_emotions = []
    for file in files:
        if file.filename == '':
            continue
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            predicted_emotions.append("No face detected")
            continue
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            resized_face = preprocess_image(face)
            predictions = trained_model.predict(np.expand_dims(resized_face, axis=0))
            predicted_class = np.argmax(predictions)
            label = ['Surprised', 'Fear', 'Disgusted', 'Happy', 'Sad', 'Angry', 'Neutral']
            predicted_emotion = label[predicted_class]
            predicted_emotions.append(predicted_emotion)

    if len(predicted_emotions) == 0:
        return jsonify({'message': 'No faces detected in any of the images'}), 400

    # Find the most frequent emotion prediction
    most_common_emotion = Counter(predicted_emotions).most_common(1)[0][0]
    
    return jsonify({'message': 'Image processing successful', 'predicted_emotion': most_common_emotion}), 200

if __name__ == "__main__":
    app.run(debug=True)
