import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np 

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


def detect_helmet(img):
    l = ["Human face wearing a helmet", "Human face NOT wearing a helmet", "MORE than one person in the photo"]
    inputs = processor(text=l, images=img, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)
    # print(probs)
    return l[probs.detach().numpy().argmax()]