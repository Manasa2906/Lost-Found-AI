import os
import vertexai
import numpy as np
import streamlit as st
from vertexai.vision_models import MultiModalEmbeddingModel, Image
from vertexai.generative_models import GenerativeModel, Part
from sklearn.metrics.pairwise import cosine_similarity
from google.oauth2 import service_account

# --- 1. Global Configuration ---
PROJECT_ID = "lost-found-483214-j7"
LOCATION = "us-central1"

def initialize_vertex():
    """
    Handles authentication for both Local (key.json) and Streamlit Cloud (Secrets).
    This manually maps secrets to a dict to fix the 'AttrDict is not JSON serializable' error.
    """
    try:
        # Check if running on Streamlit Cloud (Secrets)
        if "gcp_service_account" in st.secrets:
            s = st.secrets["gcp_service_account"]
            
            # Manually reconstructing the dictionary ensures compatibility with Google SDK
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
            
        # Fallback for local VS Code testing
        elif os.path.exists("key.json"):
            credentials = service_account.Credentials.from_service_account_file("key.json")
            vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
            
    except Exception as e:
        st.error(f"Cloud Connection Error: {str(e)}")

# Initialize Vertex AI immediately on script load
initialize_vertex()

# --- 2. Core Logic Functions ---

def get_image_embedding(image_path):
    """Generates visual vectors using Vertex AI Multimodal Embeddings."""
    # Re-initialize to ensure the Project context is active in the session
    initialize_vertex()
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    image = Image.load_from_file(image_path)
    embeddings = model.get_embeddings(image=image)
    return embeddings.image_embedding

def calculate_similarity(img_path1, img_path2):
    """Calculates cosine similarity percentage between two images."""
    try:
        vec1 = np.array(get_image_embedding(img_path1)).reshape(1, -1)
        vec2 = np.array(get_image_embedding(img_path2)).reshape(1, -1)
        score = cosine_similarity(vec1, vec2)[0][0]
        
        # Scale score to 0-100 range and round to 2 decimal places
        return round(max(0, min(100, score * 100)), 2)
    except Exception as e:
        return f"Embedding Error: {str(e)}"

def get_ai_explanation(img_path1, img_path2):
    """Uses Gemini 2.5 Flash to provide a logical explanation for the match."""
    try:
        initialize_vertex()
        model = GenerativeModel("gemini-2.5-flash")
        
        with open(img_path1, "rb") as f1, open(img_path2, "rb") as f2:
            image1 = Part.from_data(data=f1.read(), mime_type="image/jpeg")
            image2 = Part.from_data(data=f2.read(), mime_type="image/jpeg")
        
        prompt = (
            "Analyze these two images for a Lost and Found system. "
            "Explain in exactly 2 bullet points why these items are likely a match. "
            "Focus on visual details like color, shape, and unique markings."
        )
        
        response = model.generate_content([prompt, image1, image2])
        return response.text
    except Exception as e:
        return f"AI Logic Error: {str(e)}"