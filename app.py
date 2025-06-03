import streamlit as st
import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image,ImageDraw, ImageFont
import cv2
import numpy as np
from io import BytesIO
import gdown


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("C:/Users/Nikhil/real_vs_fake_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_gradcam(model, input_tensor, target_class):
    model.eval()

    # Hook the feature maps and gradients
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks on the last convolutional layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_layer = module

    forward_handle = last_conv_layer.register_forward_hook(forward_hook)
    backward_handle = last_conv_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    model.zero_grad()

    # Backward pass
    class_score = output[0, target_class]
    class_score.backward()

    # Get the gradients and activations
    gradients = gradients[0].detach().cpu().numpy()[0]
    activations = activations[0].detach().cpu().numpy()[0]

    # Global average pooling of gradients
    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations[i]

    # ReLU + normalize
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Convert to heatmap
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert input image tensor to numpy
    image_np = input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    image_np = (image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

    # BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Combine heatmap and image
    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    return overlay

st.title("Real vs Fake Image Detector")

def create_report_image(original_image, gradcam_image, label, confidence):

    from PIL import ImageDraw, ImageFont, ImageOps
    import datetime

    # Resize for consistency
    original = original_image.resize((300, 300))
    gradcam = gradcam_image.resize((300, 300))

    # Combine horizontally
    combined_width = original.width + gradcam.width + 40
    combined_height = 400
    report = Image.new("RGB", (combined_width, combined_height), color=(30, 30, 30))

    # Paste images
    report.paste(original, (20, 50))
    report.paste(gradcam, (original.width + 40, 50))

    draw = ImageDraw.Draw(report)

    # Fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        text_font = ImageFont.truetype("arial.ttf", 18)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    # Header text
    title_text = "🧠 Real vs Fake Image Prediction Report"
    draw.text((20, 10), title_text, font=title_font, fill=(255, 255, 255))

    # Prediction Card
    status_color = (0, 255, 0) if label == "Real" else (255, 0, 0)
    draw.rectangle([(20, 360), (combined_width - 20, 390)], fill=(50, 50, 50))
    draw.text((30, 365), f"Prediction: {label}    Confidence: {confidence:.2%}", font=text_font, fill=status_color)

    # Footer
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer = f"Model: ResNet18 | Timestamp: {now} | Size: {original_image.size[0]}x{original_image.size[1]}"
    draw.text((30, 395), footer, font=text_font, fill=(180, 180, 180))

    return report


model = load_model()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        label = "Fake" if pred.item() == 0 else "Real"
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][pred.item()].item()
        st.write(f"Prediction: **{label}** (Confidence: {confidence:.2%})")

    cam_image = generate_gradcam(model, input_tensor, target_class=pred.item())
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
    st.image(Image.fromarray(cam_image), caption="Grad-CAM Heatmap", use_column_width=True)

    # ---- Report generation ----
    report_image = create_report_image(
    original_image=image,
    gradcam_image=Image.fromarray(cam_image),
    label=label,
    confidence=confidence
)

    buffer = BytesIO()
    report_image.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="📥 Download Prediction Report",
        data=buffer,
        file_name="prediction_report.png",
        mime="image/png"
    )

@st.cache_resource
def load_model():
    # Google Drive model download
    url = "https://drive.google.com/uc?id=12HjPXAm5ojN_ksouAHFYutfag4C5vqeM"
    output = "real_vs_fake_model.pth"
    gdown.download(url, output, quiet=False)

    # Load model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(output, map_location=torch.device("cpu")))
    model.eval()
    return model
