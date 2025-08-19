
import argparse, os, json, random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import build_dataloaders
from model import get_model, freeze_backbone

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_confusion_matrix(cm, classes, out_path):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def evaluate(model, loader, device, criterion):
    model.eval()
    all_logits, all_y = [], []
    running_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            all_logits.append(logits.detach().cpu())
            all_y.append(y.detach().cpu())
    import torch.nn.functional as F
    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0).numpy()
    y_prob = F.softmax(logits, dim=1).numpy()
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob[:,1], average='macro')
    except Exception:
        auc = float('nan')
    loss = running_loss / len(loader.dataset)
    return {'loss': loss, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc,
            'y_true': y_true.tolist(), 'y_pred': y_pred.tolist()}

def train(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    train_loader, val_loader, test_loader, classes = build_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers
    )

    model = get_model(num_classes=len(classes), pretrained=args.pretrained)
    if args.freeze_backbone:
        model = freeze_backbone(model, True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=max(1, args.epochs))

    artifacts = Path('artifacts'); artifacts.mkdir(exist_ok=True)
    with open(artifacts / 'label_map.json', 'w') as f:
        json.dump({cls:i for i, cls in enumerate(classes)}, f, indent=2)

    best_f1, best_path = -1.0, artifacts / 'best_model.pt'

    if args.evaluate_only:
        assert args.checkpoint, "--checkpoint is required for --evaluate_only"
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        test_stats = evaluate(model, test_loader, device, criterion)
        with open(artifacts / 'metrics.json', 'w') as f:
            json.dump({'test': test_stats}, f, indent=2)
        cm = confusion_matrix(test_stats['y_true'], test_stats['y_pred'])
        save_confusion_matrix(cm, classes, artifacts / 'confusion_matrix.png')
        print(json.dumps(test_stats, indent=2))
        return

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running_loss = 0.0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            running_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item())
        scheduler.step()

        val_stats = evaluate(model, val_loader, device, criterion)
        train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_f1={val_stats['f1']:.4f} val_acc={val_stats['acc']:.4f} val_auc={val_stats['auc']:.4f}")

        if val_stats['f1'] > best_f1:
            best_f1 = val_stats['f1']
            torch.save(model.state_dict(), best_path)

    # Final test with best model
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_stats = evaluate(model, test_loader, device, criterion)
    with open(artifacts / 'metrics.json', 'w') as f:
        json.dump({'test': test_stats}, f, indent=2)

    cm = confusion_matrix(test_stats['y_true'], test_stats['y_pred'])
    save_confusion_matrix(cm, classes, artifacts / 'confusion_matrix.png')
    with open(artifacts / 'run_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("Training complete. Best model saved to:", str(best_path))
    print(json.dumps(test_stats, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()
    train(args)
