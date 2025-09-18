import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile

st.title("🖼️ AI vs Real Image Classifier")
st.write("Upload an image to check if it's **AI-generated** or a **Real photo**.")

uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ✅ Roboflow Client
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=st.secrets["ROBOFLOW_API_KEY"]
    )

    with st.spinner("🔎 Classifying..."):
        try:
            # Save uploaded file to a temporary path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded.getvalue())
                tmp_path = tmp_file.name

            # Run inference workflow
            result = client.run_workflow(
                workspace_name="ziad-f3ycp",
                workflow_id="custom-workflow",
                images={"image": tmp_path},  # ✅ file path works
                use_cache=True
            )

            pred_dict = result[0]["predictions"]["predictions"][0]
            label = pred_dict["class"]
            conf = pred_dict["confidence"]

            st.success(f"✅ Prediction: **{label}** ({conf:.2%} confidence)")

        except Exception as e:
            st.error("❌ Error running inference")
            st.write(str(e))
            if 'result' in locals():
                st.json(result)