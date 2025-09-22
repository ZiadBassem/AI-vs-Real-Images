#  AI vs Real Image Classifier

##  Project Overview
This project builds a **binary classifier** to distinguish between **AI-generated images** and **real-world photos**.  
The model was trained using **Roboflow’s Vision Transformer (ViT)** and fine-tuned on a custom dataset.

---

##  Dataset
- **AI-Generated Images (~500)** → Generated via Stable Diffusion / Craiyon with varied prompts.  
- **Real Images (~470)** → Collected with the Pexels API.  

### Preprocessing
- Resize → 224x224
- Normalize pixel intensities
- Split → Train 70%, Validation 20%, Test 10%

### Augmentation
- Horizontal flip
- Small rotations (±15°)
- Brightness & contrast adjustments (±10%)
- Gaussian blur (~2–3px)

---

##  Model & Training
- Architecture: **Vision Transformer (ViT)**
- Pretrained on: **ImageNet**
- Training Platform: **Roboflow GPU cluster**
- Training Time: ~34 minutes / 12 epochs
- Loss: Cross Entropy
- Optimizer: Adam

---

## Results
- **Test Accuracy**: ~99.0%  

The training loss decreased smoothly, while the validation accuracy stabilized at ~99%, showing strong generalization.

---

## Deployment & Usage

### Option 1: Streamlit Demo App
[streamlit run streamlit_app.py](https://ziad-image-classifier.streamlit.app/)

### Option 2: Python Inference via Roboflow API
```python
from inference_sdk import InferenceHTTPClient
import json

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="YOUR_API_KEY"
)

result = client.run_workflow(
    workspace_name="ziad-f3ycp",
    workflow_id="custom-workflow",
    images={"image": "test_image.jpg"},
    use_cache=True
)

print(json.dumps(result, indent=2))
