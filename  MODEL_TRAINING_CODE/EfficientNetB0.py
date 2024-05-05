#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install torch torchvision


# In[1]:


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import f1_score

# Define constants
BATCH_SIZE = 32
IMG_SIZE = 224  # EfficientNet in torchvision uses 224 as the default size
EPOCHS = 10
DATASET_PATH = "C:/Users/wanqi/Desktop/Train"

# Load EfficientNet
model = models.efficientnet_b0(pretrained=True)

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)


data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization parameters for ImageNet
])

full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=data_transforms)

train_size = int(0.75 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# In[2]:


# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
best_val_loss = float('inf')
device


# In[3]:


def train_model(model, criterion, optimizer, num_epochs=10):
    global best_val_loss
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader)}, Val Loss: {val_loss}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_eff.pth')
            print("Saved Best Model")

train_model(model, criterion, optimizer, EPOCHS)


# In[4]:


import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(pretrained=False)  # Assuming using EfficientNet B0
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 2)  # Modify for 2 classes

model.load_state_dict(torch.load('best_model_eff.pth'))  # Load your best model
model = model.to(device)
model.eval()
transformations = transforms.Compose([
    transforms.Resize((224, 224)),  # Match the input size that the model expects
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization used during training
])

data_path = "C:/Users/wanqi/Desktop/Test"
subtypes = ['sea', 'buildings', 'forest', 'glacier', 'mountain', 'street']
results = {}

for subtype in subtypes:
    path = f"{data_path}/{subtype}"
    dataset = datasets.ImageFolder(root=path, transform=transformations)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute F1 score and other statistics
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=dataset.classes)
    results[subtype] = {
        "F1 Score": f1,
        "Report": report
    }
    print(f"F1 Score for {subtype}: {f1}")
    print(report)

# Optionally, you can return or print results dictionary to see detailed results


# In[ ]:




