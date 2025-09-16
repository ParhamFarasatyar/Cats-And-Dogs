# 🐱🐶 Cats vs Dogs Classification using ResNet34

This project focuses on classifying images of cats and dogs using a **Convolutional Neural Network (CNN)** built on **ResNet34**.  
The implementation is based on **PyTorch** and includes complete training, evaluation, and visualization of the model’s performance.

---

## 📚 Libraries & Tools Used
- **PyTorch (torch, torch.nn)** → Building and training the deep learning model  
- **Torchvision (models, transforms)** → ResNet34 architecture and image preprocessing  
- **Matplotlib** → Plotting training loss & accuracy curves  
- **PIL (Python Imaging Library)** → Image handling and transformations  
- **Glob** → File handling for dataset images
- **Pandas**: For handling metadata and results
- **OpenCV (cv2)**: For image preprocessing and display.

---

## 🧠 Model Details
- **Architecture**: ResNet34  
- **Optimizer**: SGD (Stochastic Gradient Descent)  
- **Loss Function**: CrossEntropyLoss  
- **Dataset**: Cats and Dogs dataset  

> ⚠️ Note: In recent versions of `torchvision`, the argument `pretrained=True` is deprecated.  
> Instead, use:  
> ```python
> model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
> ```
> This ensures compatibility with the latest releases.

---

## 📊 Training Results
Below is the training progress in terms of **loss** and **accuracy**:



- Final Training Accuracy: ~97%  
- Final Validation Accuracy: ~98%  

---

📌 Future Improvements

* Experiment with other optimizers like Adam or RMSprop.

* Apply data augmentation techniques for better generalization.

* Test with deeper architectures (e.g., ResNet50, EfficientNet).

---

👨‍💻 Author

Developed by **Parham Farasatyar**
If you like this repo, don’t forget to ⭐ it!

---
