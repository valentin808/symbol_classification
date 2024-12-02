import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision import datasets, transforms

def get_data(batch_size, data_root='data', num_workers=0):
    train_test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5], std=[0.5]),])
   
    train_loader = torch.utils.data.DataLoader(
        datasets.EMNIST(root="data", train=True,split="balanced", download=True, transform=train_test_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.EMNIST(root="data", train=False,split="balanced", download=True, transform=train_test_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return train_loader, test_loader

train_data, test_data=get_data(16)
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
criterion = torch.nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_data:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_data)}")


model.eval() 
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_data:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')

models = 'models'
if not os.path.exists(models):
    os.makedirs(models)

model_file_name = 'MobileNet1.pt'

model_path = os.path.join(models, model_file_name)

model.to('cpu')

torch.save(model.state_dict(), model_path)

