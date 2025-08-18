
import argparse, os, json
import torch
from model import get_model
from inference_utils import load_label_map, predict_image

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    label_map, inv_map = load_label_map(args.label_map)
    model = get_model(num_classes=len(label_map), pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    paths = []
    if os.path.isdir(args.input_path):
        for root, _, files in os.walk(args.input_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(root, f))
    else:
        paths = [args.input_path]

    results = []
    for p in paths:
        probs = predict_image(model, p, device=device, img_size=args.img_size)
        pred_idx = int(probs.argmax())
        results.append({
            'path': p,
            'pred_class': inv_map[pred_idx],
            'probabilities': {inv_map[i]: float(prob) for i, prob in enumerate(probs)}
        })

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--label_map", type=str, default="artifacts/label_map.json")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
