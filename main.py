import os
import vertexai
import numpy as np
import streamlit as st
import json
from vertexai.vision_models import MultiModalEmbeddingModel, Image
from vertexai.generative_models import GenerativeModel, Part
from sklearn.metrics.pairwise import cosine_similarity
from google.oauth2 import service_account

# --- 1. Setup Credentials & Initialize ---
PROJECT_ID = "lost-found-483214-j7"
LOCATION = "us-central1"

def initialize_vertex():
    """Handles authentication for both Local and Streamlit Cloud"""
    try:
        if "gcp_service_account" in st.secrets:
            # OPTION A: Streamlit Cloud
            secret_info = st.secrets["gcp_service_account"]
            
            # --- FIX STARTS HERE ---
            # Convert AttrDict to a standard Python Dictionary
            creds_dict = {key: value for key, value in secret_info.items()}
            # --- FIX ENDS HERE ---
            
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
        else:
            # OPTION B: Local Development
            current_dir = os.path.dirname(os.path.abspath(__file__))
            key_path = os.path.join(current_dir, "key.json")
            if os.path.exists(key_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
                vertexai.init(project=PROJECT_ID, location=LOCATION)
    except Exception as e:
        st.error(f"Initialization Error: {e}")
# Run the initialization immediately
initialize_vertex()

# --- 2. Core Functions ---

def get_image_embedding(image_path):
    # Using 'multimodalembedding@001' is often more stable for specific versions
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    image = Image.load_from_file(image_path)
    embeddings = model.get_embeddings(image=image)
    return embeddings.image_embedding

def calculate_similarity(img_path1, img_path2):
    try:
        vec1 = np.array(get_image_embedding(img_path1)).reshape(1, -1)
        vec2 = np.array(get_image_embedding(img_path2)).reshape(1, -1)
        score = cosine_similarity(vec1, vec2)[0][0]
        # Ensure the score stays within 0-100%
        return round(max(0, min(100, score * 100)), 2)
    except Exception as e:
        return f"Embedding Error: {str(e)}"

def get_ai_explanation(img_path1, img_path2):
    try:
        # Note: Gemini 2.5 Flash is the model requested for 2026 stability
        model = GenerativeModel("gemini-2.5-flash")
        
        with open(img_path1, "rb") as f1, open(img_path2, "rb") as f2:
            img1_data = f1.read()
            img2_data = f2.read()
        
        image1 = Part.from_data(data=img1_data, mime_type="image/jpeg")
        image2 = Part.from_data(data=img2_data, mime_type="image/jpeg")
        
        prompt = (
            "You are a forensic lost-and-found assistant. Look at these two items. "
            "Provide exactly 2 bullet points explaining why they are likely the same object. "
            "Focus on unique identifiers like scratches, specific colors, or branding."
        )
        
        response = model.generate_content([prompt, image1, image2])
        return response.text
    except Exception as e:
        return f"AI Logic Error: {str(e)}"