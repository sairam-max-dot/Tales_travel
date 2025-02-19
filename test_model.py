from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

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

def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax().item()

    print(f"Predicted Location: {predicted_class}")

# Run Prediction
predict("test_image.jpg")
