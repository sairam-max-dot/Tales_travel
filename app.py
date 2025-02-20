from flask import Flask, request, jsonify, send_file
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import wikipedia
import numpy as np
import soundfile as sf
import os
from bark import generate_audio
from bark.generation import preload_models

app = Flask(__name__)

# Load ResNet50 Model
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust for number of classes
model.load_state_dict(torch.load("models/resnet50_goa.pth", map_location=torch.device('cpu')))
model.eval()

# Class names
class_names = ["Baga Beach", "Aguada Fort", "Basilica of Bom Jesus", "Chapora Fort",
               "Dudhsagar Waterfalls", "Arambol Beach", "Immaculate Conception Church",
               "Palolem Beach", "Shree Shantadurga Temple", "Sinquerim Beach"]

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load Bark AI voice models
preload_models()

# Supported languages
LANGUAGE_CODES = {
    "english": "en", "french": "fr", "spanish": "es", "german": "de",
    "japanese": "ja", "hindi": "hi", "tamil": "ta", "telugu": "te",
    "odia": "or", "malayalam": "ml"
}

def predict_location(images):
    """Predicts location from 3 images using majority voting."""
    predictions = []

    for image in images:
        image = Image.open(image)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            predicted_class = output.argmax().item()
            predictions.append(class_names[predicted_class])

    # Majority voting: Most common prediction
    location = max(set(predictions), key=predictions.count)
    return location

def get_wikipedia_info(location, language="english"):
    """Fetches a summary of the location from Wikipedia in the specified language."""
    lang_code = LANGUAGE_CODES.get(language.lower(), "en")
    wikipedia.set_lang(lang_code)

    try:
        summary = wikipedia.summary(location, sentences=5)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found. Try specifying: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found."

def text_to_speech(text):
    """Converts Wikipedia text to speech using Bark AI."""
    audio_array = generate_audio(text)
    audio_path = "static/output.wav"
    sf.write(audio_path, audio_array, samplerate=24000)  # Save audio
    return audio_path

@app.route("/predict", methods=["POST"])
def predict():
    if "file1" not in request.files or "file2" not in request.files or "file3" not in request.files:
        return jsonify({"error": "Please upload 3 images"}), 400

    images = [request.files["file1"], request.files["file2"], request.files["file3"]]
    location = predict_location(images)
    
    return jsonify({"Predicted Location": location})

@app.route("/tts", methods=["POST"])
def tts():
    data = request.json
    text = data.get("text", "")
    language = data.get("language", "english")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if language.lower() not in LANGUAGE_CODES:
        return jsonify({"error": "Unsupported language"}), 400

    audio_path = text_to_speech(text)
    
    return send_file(audio_path, mimetype="audio/wav", as_attachment=True)

@app.route("/full_process", methods=["POST"])
def full_process():
    """Handles the full pipeline: Image recognition → Wikipedia summary → Speech conversion."""
    if "file1" not in request.files or "file2" not in request.files or "file3" not in request.files:
        return jsonify({"error": "Please upload 3 images"}), 400

    images = [request.files["file1"], request.files["file2"], request.files["file3"]]
    location = predict_location(images)

    language = request.form.get("language", "english")
    info = get_wikipedia_info(location, language)

    audio_path = text_to_speech(info)

    return jsonify({
        "Predicted Location": location,
        "Wikipedia Info": info,
        "Audio URL": f"/static/output.wav"
    })

if __name__ == "__main__":
    app.run(debug=True)
