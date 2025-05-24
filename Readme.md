# 🧠 Real vs Fake Image Detector

A deep learning-based application that detects whether an image is real or AI-generated. This project uses a custom-trained convolutional neural network (CNN) built with PyTorch, designed to classify images with high accuracy.

## 🚀 Features
- Binary image classification: real vs AI-generated
- Custom-trained model saved as `real_vs_fake_model.pth`
- Simple Python script (`app.py`) to load the model and make predictions
- GitHub version control and open-source friendly

## 🛠️ Tech Stack
- Python 3.x
- PyTorch
- NumPy, PIL
- Git + GitHub

## 📁 Project Structure


Authenticator/
│
├── app.py # Script to load model and run predictions
├── real_vs_fake_model.pth # Pre-trained model file (42MB)
└── README.md # Project description and usage guide



## 🧪 How to Use

1. **Clone the repository:**
```bash
git clone https://github.com/nikhil99998/Authenticator.git
cd Authenticator

pip install torch torchvision pillow numpy

python app.py
