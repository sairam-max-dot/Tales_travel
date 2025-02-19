from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

app = Flask(__name__)

# Load Model
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 5)  # Adjust for class count
model.load_state_dict(torch.load("models/resnet50_goa.pth"))
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["file"]
    image = Image.open(image)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax().item()

    return jsonify({"Predicted Location": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
