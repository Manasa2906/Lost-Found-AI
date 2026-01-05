import os
import vertexai
import numpy as np
import streamlit as st
from vertexai.vision_models import MultiModalEmbeddingModel, Image
from vertexai.generative_models import GenerativeModel, Part
from sklearn.metrics.pairwise import cosine_similarity
from google.oauth2 import service_account

# Configuration
PROJECT_ID = "lost-found-483214-j7"
LOCATION = "us-central1"

def initialize_vertex():
    """Authenticates using Streamlit Secrets. No manual typing of keys in code."""
    try:
        if "gcp_service_account" in st.secrets:
            s = st.secrets["gcp_service_account"]
            
            # This manual mapping is mandatory to fix the 'AttrDict' serialization error
            creds_dict = {
                "type": s["type"],
                "project_id": s["project_id"],
                "private_key_id": s["private_key_id"],
                "private_key": s["private_key"].replace("\\n", "\n"), 
                "client_email": s["client_email"],
                "client_id": s["client_id"],
                "auth_uri": s["auth_uri"],
                "token_uri": s["token_uri"],
                "auth_provider_x509_cert_url": s["auth_provider_x509_cert_url"],
                "client_x509_cert_url": s["client_x509_cert_url"],
                "universe_domain": s.get("universe_domain", "googleapis.com")
            }
            
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
        else:
            st.error("Secrets not found! Paste your TOML into the Streamlit Cloud Settings.")
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")

# Initialize immediately
initialize_vertex()

def get_image_embedding(image_path):
    initialize_vertex()
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    image = Image.load_from_file(image_path)
    embeddings = model.get_embeddings(image=image)
    return embeddings.image_embedding

def calculate_similarity(img_path1, img_path2):
    try:
        vec1 = np.array(get_image_embedding(img_path1)).reshape(1, -1)
        vec2 = np.array(get_image_embedding(img_path2)).reshape(1, -1)
        score = cosine_similarity(vec1, vec2)[0][0]
        return round(max(0, min(100, score * 100)), 2)
    except Exception as e:
        return f"Embedding Error: {str(e)}"

def get_ai_explanation(img_path1, img_path2):
    """UPDATED: Using Gemini 2.5 Flash for 2026 stability"""
    try:
        # Using the updated 2026 stable model name
        model = GenerativeModel("gemini-2.5-flash")
        
        img1_data = open(img_path1, "rb").read()
        img2_data = open(img_path2, "rb").read()
        
        image1 = Part.from_data(data=img1_data, mime_type="image/jpeg")
        image2 = Part.from_data(data=img2_data, mime_type="image/jpeg")
        
        prompt = "Look at these two items. Explain in 2 bullet points why they are the same object. Mention specific details like shape, color, or markings."
        
        response = model.generate_content([prompt, image1, image2])
        return response.text
    except Exception as e:
        # This will now show you the EXACT error in Streamlit
        return f"AI Logic Error: {str(e)}"