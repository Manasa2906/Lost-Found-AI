import os
import vertexai
import numpy as np
import streamlit as st
import json
from vertexai.vision_models import MultiModalEmbeddingModel, Image
from vertexai.generative_models import GenerativeModel, Part
from sklearn.metrics.pairwise import cosine_similarity
from google.oauth2 import service_account

# --- 1. Global Configuration ---
# Keep these as strings
PROJECT_ID = "lost-found-483214-j7"
LOCATION = "us-central1"

def initialize_vertex():
    """
    Handles authentication for both Local and Streamlit Cloud.
    Converts Streamlit AttrDict to a standard dict to avoid JSON errors.
    """
    try:
        if "gcp_service_account" in st.secrets:
            # OPTION A: Streamlit Cloud (Reading from Secrets)
            secret_info = st.secrets["gcp_service_account"]
            
            # Create a clean dictionary from the secret info
            creds_dict = {
                "type": secret_info["type"],
                "project_id": secret_info["project_id"],
                "private_key_id": secret_info["private_key_id"],
                "private_key": secret_info["private_key"].replace("\\n", "\n"),
                "client_email": secret_info["client_email"],
                "client_id": secret_info["client_id"],
                "auth_uri": secret_info["auth_uri"],
                "token_uri": secret_info["token_uri"],
                "auth_provider_x509_cert_url": secret_info["auth_provider_x509_cert_url"],
                "client_x509_cert_url": secret_info["client_x509_cert_url"],
                "universe_domain": secret_info.get("universe_domain", "googleapis.com")
            }
            
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
        else:
            # OPTION B: Local Development (Reading from key.json)
            key_path = "key.json"
            if os.path.exists(key_path):
                credentials = service_account.Credentials.from_service_account_file(key_path)
                vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
            else:
                st.warning("No credentials found. Please check key.json or Streamlit Secrets.")
                
    except Exception as e:
        # This will catch the 'AttrDict' error if it still persists
        st.error(f"Cloud Connection Error: {str(e)}")

# Run initialization immediately when the app starts
initialize_vertex()

# --- 2. Logic Functions ---

def get_image_embedding(image_path):
    """Generates visual vectors using Vertex AI"""
    # Re-verify init to ensure the session remains active
    initialize_vertex()
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    image = Image.load_from_file(image_path)
    embeddings = model.get_embeddings(image=image)
    return embeddings.image_embedding

def calculate_similarity(img_path1, img_path2):
    """Calculates the % match between two images"""
    try:
        vec1 = np.array(get_image_embedding(img_path1)).reshape(1, -1)
        vec2 = np.array(get_image_embedding(img_path2)).reshape(1, -1)
        score = cosine_similarity(vec1, vec2)[0][0]
        # Normalize and round the score
        return round(max(0, min(100, score * 100)), 2)
    except Exception as e:
        return f"Embedding Error: {str(e)}"

def get_ai_explanation(img_path1, img_path2):
    """Uses Gemini 1.5 Flash to explain why items match"""
    try:
        initialize_vertex()
        model = GenerativeModel("gemini-1.5-flash")
        
        with open(img_path1, "rb") as f1, open(img_path2, "rb") as f2:
            image1 = Part.from_data(data=f1.read(), mime_type="image/jpeg")
            image2 = Part.from_data(data=f2.read(), mime_type="image/jpeg")
        
        prompt = (
            "You are a Lost and Found assistant. Compare these two images. "
            "List 2 specific bullet points explaining why they are likely the same object. "
            "Mention color, brand, or unique wear and tear."
        )
        
        response = model.generate_content([prompt, image1, image2])
        return response.text
    except Exception as e:
        return f"AI Logic Error: {str(e)}"