# ====== Modules ======
import torch
import torch.nn as nn
from torchvision.models import resnet34
from torchvision import transforms
import matplotlib.pyplot as plt
import PIL
from glob import glob


# ====== Configs ======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODEL_PATH = "final_model.pth"


# ====== Load Model ======
def load_model(model_path=MODEL_PATH, device=DEVICE):
    model = resnet34(weights= None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# ====== Transform ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


# ====== Predict Single Image ======
def predict_image(model, img_path, transform, device=DEVICE):
    image = PIL.Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)

    label = "cat" if pred.item() == 0 else "dog"
    return image, label


# ====== Show Images with Prediction ======
def show_predictions(model, img_paths, transform, cols=2):
    n_imgs = len(img_paths)
    rows = (n_imgs + cols - 1) // cols

    plt.figure(figsize=(12, 6))
    for i, path in enumerate(img_paths):
        img, label = predict_image(model, path, transform)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"Predicted: {label}", fontsize=12, color="blue")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ====== Example Run ======
if __name__ == "__main__":
    model = load_model()

    test_imgs = [path for path in glob("./dataset/single_prediction/*")]

    show_predictions(model, test_imgs, transform)