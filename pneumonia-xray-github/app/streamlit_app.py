
import streamlit as st
import torch
from PIL import Image
from io import BytesIO
from src.model import get_model
from src.inference_utils import get_preprocess, load_label_map

st.title("Chest X-Ray Pneumonia Classifier")
st.caption("For research/education only — not for clinical use.")

@st.cache_resource
def load_checkpoint(checkpoint_path, label_map_path):
    label_map, inv_map = load_label_map(label_map_path)
    model = get_model(num_classes=len(label_map), pretrained=False)
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model, label_map, inv_map

ckpt = st.text_input("Checkpoint path", "artifacts/best_model.pt")
lblp = st.text_input("Label map path", "artifacts/label_map.json")
img = st.file_uploader("Upload a chest X-ray", type=["png","jpg","jpeg"])

if img and ckpt and lblp:
    model, label_map, inv_map = load_checkpoint(ckpt, lblp)
    image = Image.open(BytesIO(img.read())).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    preprocess = get_preprocess(224)
    x = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].tolist()

    pred_idx = int(torch.tensor(prob).argmax().item())
    st.subheader(f"Prediction: **{inv_map[pred_idx]}**")
    st.write({inv_map[i]: float(p) for i, p in enumerate(prob)})
