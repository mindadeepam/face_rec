import gradio as gr
from torchvision import models, transforms
from PIL import Image
import requests
import torch
import numpy as np
from ds_utils.image_processing_utils import *

model = models.resnet18(pretrained=True)
model.eval()


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(img):
    try:
        img = preprocess(img)
        # Make a prediction using the model
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            _, predicted_class = output.max(1)
            predicted_label = str(predicted_class.item())

        return predicted_label, ",,"
    except Exception as e:
        return str(e)

iface = gr.Interface(
    fn=classify_image,
    inputs="image",
    outputs=[gr.Textbox(label="Is the person wearing a helmet?"), gr.Textbox(label="Who is this dude?")],
    title="Image Classification",
    description="Upload an image, and this app will classify it.",
    live=True,
)

if __name__ == "__main__":
    iface.launch(share=True)