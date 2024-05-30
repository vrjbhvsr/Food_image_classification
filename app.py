from flask import Flask, request, render_template, redirect, url_for
import os
import io
from PIL import Image as PILImage
import bentoml
import torch
from Food_Classification.constants import *
from main import start_training


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the BentoML model and runner
bento_model = bentoml.pytorch.get("food_classification_model:latest")
runner = bento_model.to_runner()
runner.init_local()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Prepare the image for the model
        image = PILImage.open(file.stream).convert("RGB")
        my_transforms = bento_model.custom_objects.get(TRAIN_TRANSFORM_KEY)  # Adjust key if needed
        image = my_transforms(image).unsqueeze(0)
        
        # Make prediction
        batch_ret = runner.run(image)
        pred = PREDICTION_LABEL[max(torch.argmax(batch_ret, dim=1).detach().cpu().tolist())]

        return render_template('results.html', label=pred, image_url=filename)

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    app.run(debug=True)
