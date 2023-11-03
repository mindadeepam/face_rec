import torch
from transformers import ViTFeatureExtractor, ViTModel
# from deepface import DeepFace
from PIL import Image
import numpy as np 
import joblib
import os

# Load the feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained("jayanta/vit-base-patch16-224-in21k-face-recognition")
model = ViTModel.from_pretrained("jayanta/vit-base-patch16-224-in21k-face-recognition")

img_db_dir = './image_db'
os.makedirs(img_db_dir, exist_ok=True)
emb_path = "./embeddings.pkl"
if os.path.exists(emb_path):
    dict_of_embeddings = joblib.load(emb_path)    
else:    
    dict_of_embeddings = {}


def get_embedding(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    # Pass the preprocessed image through the model to obtain embeddings
    with torch.no_grad():
        outputs = model(pixel_values)
        embedding = outputs.last_hidden_state.mean(dim=1)  # This takes the mean of the embeddings from the last layer
        return embedding.detach().numpy()
    
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2.T)  # Transpose the second vector
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return (dot_product / (norm_vec1 * norm_vec2)).item()  # Extract the scalar value from the result

    
def find_face(img1, img2):
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)
    e1,e2 = get_embedding(img1), get_embedding(img2)
    if cosine_similarity(e1, e2) > 0.75:
        return 'SAME FACE!'
    return 'DIFF FACE'
    
def classify_face(img1):
    img1 = Image.fromarray(img1)
    # img_path = os.path.join('tmp.png')
    # Image.fromarray(img1).save(img_path)

    # dfs = DeepFace.find(img_path = img_path, db_path = img_db_dir)
    # return dfs

    emb_query = get_embedding(img1)
    similarites = {}
    for id_, emb in dict_of_embeddings.items():
        similarites[id_] = cosine_similarity(emb_query, emb)

    sorted_similarities = dict(sorted(similarites.items(), key=lambda item: item[1], reverse=True))
    print(sorted_similarities)
    if list(sorted_similarities.values())[0]>0.50:
        return list(sorted_similarities.keys())[0]
    else:
        return "none match"

def add_person(img, name):
    global list_of_embeddings
    img_path = os.path.join(img_db_dir, f'{name}.png')
    # Image.fromarray(img).save(img_path)
    # emb = DeepFace.represent(img_path)
    emb = get_embedding(img)
    i = len(dict_of_embeddings)
    dict_of_embeddings[name] = emb

    joblib.dump(dict_of_embeddings, emb_path)
    print(f"person stored as {name}")
    return 
    


