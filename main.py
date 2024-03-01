from flask import Flask, request, jsonify
import os
from PIL import Image
import numpy as np
import cv2
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

trained_model = tf.keras.models.load_model('BasicRAF-DB794.h5') 

# Load your trained model here
# For example:
# from your_model import trained_model
# from preprocess import preprocess_image

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Hello, World!'})

@app.route('/img', methods=['GET','POST'])
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
        
            
            # Preprocess the image
            input_image = preprocess_image(file_path)
            
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
        return jsonify({'message': 'This is get request'}), 405

if __name__ == '__main__':
    app.run(debug=True)
