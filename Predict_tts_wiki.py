import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import wikipedia
import pyttsx3

# Load Model
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust based on the number of classes
model.load_state_dict(torch.load("models/resnet50_goa.pth"))
model.eval()

# Define class names (Update with actual names)
class_names = ["Baga Beach", "Aguada Fort", "Basilica of Bom Jesus", "Chapora Fort", "Dudhsagar Waterfalls","Arambol Beach","Immaculate Conception Chruch","Palolem Beach","Shree Shantadurga Temple","Sinquerim Beach"]

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_location(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax().item()
        location = class_names[predicted_class]
    
    return location

def get_wikipedia_info(location):
    try:
        summary = wikipedia.summary(location, sentences=200)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found. Try specifying: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "No information found on Wikipedia."

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main(image_path):
    location = predict_location(image_path)
    print(f"Predicted Location: {location}")
    
    info = get_wikipedia_info(location)
    print(f"Wikipedia Info: {info}")
    
    text_to_speech(info)
    print("Speech Output Generated!")

# Run the pipeline with an example image
main("test_image.jpg")
