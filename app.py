from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, session
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo, Email
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import wikipedia
import numpy as np
import soundfile as sf
from bark import generate_audio
from bark.generation import preload_models

app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ----------------- USER MODEL -----------------

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ----------------- FORMS -----------------
class RegistrationForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField("Confirm Password", validators=[DataRequired(), EqualTo("password")])
    submit = SubmitField("Sign Up")

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

# ----------------- AUTH ROUTES -----------------

@app.route("/", methods=["GET", "POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode("utf-8")
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash("Account created! You can now log in.", "success")
        return redirect(url_for("login"))
    return render_template("index.html", form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Login failed! Check your email and password.", "danger")
    return render_template("login.html", form=form)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=current_user.username)

@app.route("/predict", methods=["POST"])
def predict():
    # Debugging: Print received files
    print("Received files:", request.files.keys())

    required_files = ["file1", "file2", "file3"]
    for file_key in required_files:
        if file_key not in request.files:
            return jsonify({"error": f"Missing file: {file_key}. Please upload 3 images."}), 400

    # Validate file types
    images = []
    allowed_extensions = {"png", "jpg", "jpeg"}
    for file_key in required_files:
        file = request.files[file_key]
        if file.filename == "":
            return jsonify({"error": f"File {file_key} is empty. Please upload a valid image."}), 400
        if file.filename.split(".")[-1].lower() not in allowed_extensions:
            return jsonify({"error": f"Invalid file type for {file_key}. Allowed formats: PNG, JPG, JPEG."}), 400
        images.append(file)

    # Run the prediction
    location = predict_location(images)
    return jsonify({"Predicted Location": location})


@app.route("/full_process", methods=["POST"])
def full_process():
    required_files = ["file1", "file2", "file3"]
    for file_key in required_files:
        if file_key not in request.files:
            return jsonify({"error": f"Missing file: {file_key}. Please upload 3 images."}), 400

    images = [request.files["file1"], request.files["file2"], request.files["file3"]]
    location = predict_location(images)

    language = request.form.get("language", "english")
    info = get_wikipedia_info(location, language)

    if "No Wikipedia page found" in info or "Multiple results found" in info:
        return jsonify({"Predicted Location": location, "Error": info}), 404

    audio_path = text_to_speech(info)
    return jsonify({
        "Predicted Location": location,
        "Wikipedia Info": info,
        "Audio URL": f"/static/output.wav"
    })


@app.route("/tts", methods=["POST"])
def tts():
    data = request.json
    text = data.get("text", "").strip()
    language = data.get("language", "english").lower()

    if not text:
        return jsonify({"error": "No text provided. Please enter some text."}), 400

    if language not in LANGUAGE_CODES:
        return jsonify({"error": f"Unsupported language: {language}. Choose from: {', '.join(LANGUAGE_CODES.keys())}"}), 400

    audio_path = text_to_speech(text)
    return send_file(audio_path, mimetype="audio/wav", as_attachment=True)


# ----------------- ML MODEL LOADING -----------------
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 10)  
model.load_state_dict(torch.load("models/resnet50_goa.pth", map_location=torch.device("cpu")))
model.eval()

class_names = ["Baga Beach", "Aguada Fort", "Basilica of Bom Jesus", "Chapora Fort",
               "Dudhsagar Waterfalls", "Arambol Beach", "Immaculate Conception Church",
               "Palolem Beach", "Shree Shantadurga Temple", "Sinquerim Beach"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

preload_models()

LANGUAGE_CODES = {
    "english": "en", "french": "fr", "spanish": "es", "german": "de",
    "japanese": "ja", "hindi": "hi", "tamil": "ta", "telugu": "te",
    "odia": "or", "malayalam": "ml"
}

# ----------------- IMAGE PROCESSING -----------------
def predict_location(images):
    predictions = []
    for image in images:
        image = Image.open(image)
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            predicted_class = output.argmax().item()
            predictions.append(class_names[predicted_class])
    return max(set(predictions), key=predictions.count)

def get_wikipedia_info(location, language="english"):
    lang_code = LANGUAGE_CODES.get(language.lower(), "en")
    wikipedia.set_lang(lang_code)
    try:
        return wikipedia.summary(location, sentences=5)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found. Try specifying: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found."

def text_to_speech(text):
    audio_array = generate_audio(text)
    audio_path = "static/output.wav"
    sf.write(audio_path, audio_array, samplerate=24000)
    return audio_path

# ----------------- PREDICTION ROUTES -----------------

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True)
