import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="AI vs Real Classifier", page_icon="🤖")

st.title("🖼️ AI vs Real Image Classifier")
st.write("Upload an image to check if it was **AI-generated** or a **Real photo**.")

uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_bytes = uploaded.getvalue()

    # Roboflow API details (replace with your actual key)
    api_url = "https://serverless.roboflow.com"
    workflow = "ziad-f3ycp/custom-workflow"
    api_key = "88SAMULX00OA2WNAnJG6"   # <-- your real API key

    endpoint = f"{api_url}/{workflow}?api_key={api_key}"

   with st.spinner("🔎 Classifying..."):
    try:
        resp = requests.post(
            endpoint,
            files={"image": ("uploaded.jpg", img_bytes, "image/jpeg")}
        )
        resp.raise_for_status()
        result = resp.json()

        # Parse nested JSON like in your notebook
        pred_dict = result[0]["predictions"]["predictions"][0]
        label = pred_dict["class"]
        conf = pred_dict["confidence"]

        st.success(f"✅ Prediction: **{label}** ({conf:.2f})")
    except Exception as e:
        st.error("❌ Error running inference")
        st.write(str(e))
        st.json(result if 'result' in locals() else {})