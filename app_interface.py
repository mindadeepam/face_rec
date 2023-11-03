import gradio as gr
from torchvision import models, transforms
from PIL import Image
import requests
import torch
import numpy as np
from ds_utils.image_processing_utils import *
from helmet import detect_helmet
model = models.resnet18(pretrained=True)
model.eval()
from face2 import add_person, classify_face


def handle_action(input_image, action, name="dummy"):
    if action == "add this person":
        # Call the image classification function
        add_person(input_image, name)
        return None, None
    elif action == "predict":
        # Call the other function
        result2 = inference(input_image)
        return result2

def inference(img):
    try:
        helmet_prediction = detect_helmet(img)
        name = "NA"
        if not helmet_prediction=="MORE than one person in the photo":
            name = classify_face(img)
        return helmet_prediction, f"this dude is {name}"

    except Exception as e:
        return "error", "error"


iface = gr.Interface(
    fn=handle_action,
    inputs=["image", gr.Dropdown(["add this person", "predict"], label="Choose Action"), "text"],
    outputs=[gr.Textbox(label="Is the person wearing a helmet?"), gr.Textbox(label="Who is this dude?")],
    title="Image Classification",
    description="Upload an image, and this app will classify it.",
)




if __name__ == "__main__":
    iface.launch(share=True)