import streamlit as st
import os
from main import calculate_similarity, get_ai_explanation
from PIL import Image
# 1. Page Configuration
st.set_page_config(page_title="AI Lost & Found", page_icon="🔍", layout="wide")

# Custom CSS for a clean, modern look
st.markdown("""
    <style>
    [data-testid="stImage"] img {
        height: 300px;
        object-fit: cover; /* This crops the image to fill the box without stretching it */
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 AI Match: Smart Lost & Found")
st.info("Compare items using Google Vertex AI Multimodal Embeddings.")

# Create uploads folder if missing
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# 2. Sidebar for System Status
with st.sidebar:
    st.header("System Status")
    if os.path.exists("key.json"):
        st.success("Google Cloud Key: FOUND")
    else:
        st.error("Google Cloud Key: MISSING")
    st.markdown("---")
    st.write("**Core Technology:**")
    st.write("- Multimodal Vectors")
    st.write("- Gemini 2.5 Flash")

# 3. Image Upload Section with Visual Previews
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Lost Item")
    lost_file = st.file_uploader("Upload Image", type=['jpg','jpeg','png'], key="lost")
    if lost_file:
        img = Image.open(lost_file)
        # Force the image to be 350px wide and 300px tall
        resized_img = img.resize((350, 300)) 
        st.image(resized_img, caption="Target: Lost Object")

with col2:
    st.subheader("📥 Found Item")
    found_file = st.file_uploader("Upload Image", type=['jpg','jpeg','png'], key="found")
    if found_file:
        img = Image.open(found_file)
        resized_img = img.resize((350, 300))
        st.image(resized_img, caption="Target: Found Object")
st.markdown("---")

# 4. Analysis Logic
if lost_file and found_file:
    if st.button("🔥 Run AI Match Analysis", use_container_width=True):
        # Save files temporarily for the AI model to read
        path1 = os.path.join("uploads", "lost_temp.jpg")
        path2 = os.path.join("uploads", "found_temp.jpg")
        with open(path1, "wb") as f: f.write(lost_file.getbuffer())
        with open(path2, "wb") as f: f.write(found_file.getbuffer())

        with st.spinner("🔄 Analyzing visual features..."):
            score = calculate_similarity(path1, path2)
            
            if isinstance(score, (float, int)):
                # Results Display
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.metric(label="Visual Match Confidence", value=f"{score}%")
                
                with res_col2:
                    # Logic for Colored Success Boxes
                    if score >= 85:
                        st.success("### ✅ High Probability\nThese items look identical! The AI has high confidence this is a match.")
                    elif 65 <= score < 85:
                        st.warning("### ⚠️ Possible Match\nThere are strong similarities. Please verify specific details manually.")
                    else:
                        st.error("### ❌ Not a Match\nThese objects appear to be different based on visual feature mapping.")
                
                # 5. AI Explanation Section (Restored)
                st.markdown("---")
                st.subheader("🤖 AI Explanation (Gemini Analysis)")
                with st.info(""):
                    reason = get_ai_explanation(path1, path2)
                    st.write(reason)
            else:
                st.error(f"Analysis Failed: {score}")
else:
    st.warning("Please upload both images to start the analysis.")