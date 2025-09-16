# ====== Modules ======
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
from torchvision.models import resnet34, ResNet34_Weights
from torchvision import transforms
import PIL
import numpy as np
import time
import copy



# ====== Configs ======
PATH = os.getcwd()
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = resnet34(weights= ResNet34_Weights.IMAGENET1K_V1)


# Classes
class CustomDataset(Dataset):
    
    def __init__(self, root_dir, transform= None):
        self.image_paths = glob(root_dir + r"\**\*.jpg")
        self.labels = [0 if "cat" in path else 1 for path in self.image_paths]
        self.transform = transform
    
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = PIL.Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    
    def __len__(self):
        return len(self.image_paths)


# ====== Functions ======
def counting_df_class():
    pth = PATH + "/dataset"
    num_classes = {
        "train_dog" : len(os.listdir(pth + "/training_set/dogs")),
        "train_cat" : len(os.listdir(pth + "/training_set/cats")),
        "test_dog" : len(os.listdir(pth + "/test_set/dogs")),
        "test_cat" : len(os.listdir(pth + "/test_set/cats"))
    }
    
    print(f"""
Training classes: 
    dogs: {num_classes['train_dog']}
    cats: {num_classes['train_cat']}
Testing classes:
    dogs: {num_classes['test_dog']}
    cats: {num_classes['test_cat']}""")


def show_examples():
    
    fig = plt.figure(figsize= (8, 6))
    
    img1 = cv2.imread(r"./dataset/single_prediction/cat_or_dog_1.jpg")
    img2 = cv2.imread(r"./dataset/single_prediction/cat_or_dog_2.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.axis("off")
    
    plt.show()


def create_dataloader(df, shuffle : bool):
    return DataLoader(df,
                      batch_size= 32,
                      shuffle= shuffle,
                      pin_memory= True
                      )


def train_model(model, train_loader, val_loader, criterion,
                optimizer, num_epochs, device= DEVICE, save_path= "best_resnet34.pth"):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        "train_loss" : [],
        "train_acc" : [],
        "val_loss" : [],
        "val_acc" : []
    }
    
    for epoch in range(num_epochs):
        since = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        
        for img, lbl in train_loader:
            image = img.to(DEVICE)
            label = lbl.to(DEVICE, dtype= torch.long)
            
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * image.size(0)
            running_correct += torch.sum(preds == label).item()
            total += image.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = running_correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for img, lbl in val_loader:
                image = img.to(DEVICE)
                label = lbl.to(DEVICE, dtype= torch.long)
                
                outputs = model(image)
                loss = criterion(outputs, label)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * image.size(0)
                val_correct += torch.sum(preds == label).item()
                val_total += image.size(0)
            
            val_epoch_loss = val_loss / val_total
            val_epoch_acc = val_correct / val_total
            history["val_loss"].append(val_epoch_loss)
            history["val_acc"].append(val_epoch_acc)
            
            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model with val_acc= {best_acc:.4f}")
                
            time_elapsed = time.time() - since
            print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.4f} | time: {time_elapsed:.1f}s \n")
            
    model.load_state_dict(best_model_wts)
    return model, history


def plot_history(hist):
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize= (8, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label= "train_loss")
    plt.plot(epochs, history["val_loss"], label= "val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label= ["train_acc"])
    plt.plot(epochs, history["val_acc"], label= ["val_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.show()
            

# ====== Runtime ======
counting_df_class()
# show_examples()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])
dataset_train = CustomDataset(root_dir= PATH + r"\dataset\training_set", transform= transform)
dataset_test = CustomDataset(root_dir= PATH + r"\dataset\test_set", transform= transform)
train_loader = create_dataloader(dataset_train, shuffle= True)
val_loader = create_dataloader(dataset_train, shuffle= False)

for param in MODEL.parameters():
    param.requires_grad = False

in_features = MODEL.fc.in_features
MODEL.fc = nn.Linear(in_features, 2)
model = MODEL.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr= 1e-3)
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr= 5e-3)

model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs= 14)
plot_history(history)

torch.save(model.state_dict(), "final_model.pth")