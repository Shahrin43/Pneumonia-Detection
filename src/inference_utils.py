
import json, torch
from PIL import Image
from torchvision import transforms

def load_label_map(path='artifacts/label_map.json'):
    with open(path, 'r') as f:
        label_map = json.load(f)
    inv_map = {v:k for k, v in label_map.items()}
    return label_map, inv_map

def get_preprocess(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def predict_image(model, image_path, device='cpu', img_size=224):
    preprocess = get_preprocess(img_size)
    image = Image.open(image_path).convert('RGB')
    x = preprocess(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs
