
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import os

def build_transforms(img_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms

def build_dataloaders(data_dir, batch_size=32, img_size=224, num_workers=2):
    train_tfms, eval_tfms = build_transforms(img_size)
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_dir, transform=eval_tfms)
    test_ds  = datasets.ImageFolder(test_dir, transform=eval_tfms)

    # Weighted sampling for imbalance
    class_counts = [0]*len(train_ds.classes)
    for _, y in train_ds.samples:
        class_counts[y] += 1
    class_weights = [0]*len(class_counts)
    total = sum(class_counts)
    for i, c in enumerate(class_counts):
        class_weights[i] = total / (len(class_counts) * c)

    sample_weights = [class_weights[y] for _, y in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes
