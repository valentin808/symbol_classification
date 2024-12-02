import torch
from PIL import Image
import os
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as F
from torchvision import transforms,datasets
import argparse
print(torch.cuda.is_available())
model = models.mobilenet_v2(weights=None) 
model.features[0][0]=nn.Conv2d(
    in_channels=1,  
    out_channels=32,
    kernel_size=3, 
    stride=2, 
    padding=1, 
    bias=False)
nn.init.kaiming_normal_(model.features[0][0].weight, mode='fan_out', nonlinearity='relu')
model.classifier[1] = nn.Linear(model.last_channel, 47)
nn.init.kaiming_normal_(model.classifier[1].weight, mode='fan_out', nonlinearity='relu')
model.load_state_dict(torch.load('MobileNet1.pt'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def preprocess_image(image_path):
    image = Image.open(image_path)  
    transform = transforms.Compose([
        transforms.Lambda(lambda img: F.rotate(img, angle=-90)),
        transforms.Grayscale(num_output_channels=1),  
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: F.hflip(img)),               
        transforms.ToTensor(),                       
        transforms.Normalize(mean=[0.5], std=[0.5]), 
    ])
    image = transform(image)
    image = image.unsqueeze(0) 
    return image

def predict(image_path, model):
    model.eval()  
    image = preprocess_image(image_path).to(device)  
    
    with torch.no_grad():  
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        
    return predicted.item() 


dict_new={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17:'H',
18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P',
26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X',
34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g',
42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'}


def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpeg', '.jpg')):
            image_path = os.path.join(directory_path, filename)
            predicted_class = predict(image_path, model)
            print(f'{ord(dict_new[predicted_class])}/{dict_new[predicted_class]}------{filename}')
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Інференс для моделі на зображенні")
    parser.add_argument('--input', type=str, required=True, help="Шлях до вхідного зображення")
    args = parser.parse_args()
    process_directory(args.input)


