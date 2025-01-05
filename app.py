from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Charger le modèle
model = load_model('chest_xray_cnn_model.keras')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result="No file uploaded", error=True)
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result="No file selected", error=True)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Charger et prétraiter l'image
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Faire une prédiction
    prediction = model.predict(img_array)
    result = "Pneumonia" if prediction > 0.5 else "Normal"

    # Retourner le résultat
    return render_template('index.html', result=result, image_path=filepath, error=False)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
