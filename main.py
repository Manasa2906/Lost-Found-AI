import os
import vertexai
import numpy as np
from vertexai.vision_models import MultiModalEmbeddingModel, Image
from vertexai.generative_models import GenerativeModel, Part
from sklearn.metrics.pairwise import cosine_similarity

# 1. Setup Credentials
current_dir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(current_dir, "key.json")
if os.path.exists(key_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

# 2. Initialize Vertex AI
PROJECT_ID = "lost-found-483214-j7" 
vertexai.init(project=PROJECT_ID, location="us-central1")

def get_image_embedding(image_path):
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    image = Image.load_from_file(image_path)
    embeddings = model.get_embeddings(image=image)
    return embeddings.image_embedding

def calculate_similarity(img_path1, img_path2):
    try:
        vec1 = np.array(get_image_embedding(img_path1)).reshape(1, -1)
        vec2 = np.array(get_image_embedding(img_path2)).reshape(1, -1)
        score = cosine_similarity(vec1, vec2)[0][0]
        return round(score * 100, 2)
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