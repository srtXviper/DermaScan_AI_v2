import argparse, json, random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


def main():
    parser = argparse.ArgumentParser(description="Quick, balanced demo training for DermaScan ResNet18.")
    parser.add_argument("--data", default="training/data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-per-class", type=int, default=10,
                       help="Images per class for quick demo; set 0 to use all images")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.data)
    if not root.exists():
        raise SystemExit(f"Dataset folder not found: {root}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    train_all = datasets.ImageFolder(root, transform=train_transform)
    eval_all = datasets.ImageFolder(root, transform=eval_transform)
    if len(train_all.classes) < 2:
        raise SystemExit("Need at least 2 class folders with images.")

    per_class = {i: [] for i in range(len(train_all.classes))}
    for idx, (_, label) in enumerate(train_all.samples):
        per_class[label].append(idx)

    train_ids, val_ids = [], []
    sampled_counts = {}
    for label, indices in per_class.items():
        random.shuffle(indices)
        if args.max_per_class > 0:
            indices = indices[:args.max_per_class]
        sampled_counts[train_all.classes[label]] = len(indices)
        if len(indices) < 2:
            raise SystemExit(f"Class '{train_all.classes[label]}' needs at least 2 images for this quick split.")
        val_count = max(1, int(len(indices) * 0.2))
        val_ids.extend(indices[:val_count])
        train_ids.extend(indices[val_count:])

    print("Quick sampled class counts:", sampled_counts, flush=True)
    print(f"Images: train={len(train_ids)}, validation={len(val_ids)}", flush=True)

    train_loader = DataLoader(Subset(train_all, train_ids), batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(eval_all, val_ids), batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}. Loading pretrained ResNet18…", flush=True)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_all.classes))
    # Freeze feature extractor to make the demo training much faster.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("fc.")
    model.to(device)

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val = -1.0
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        loss_sum = correct = seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        model.eval()
        val_correct = val_seen = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                val_correct += (model(images).argmax(1) == labels).sum().item()
                val_seen += labels.size(0)
        val_acc = val_correct / max(val_seen, 1)
        train_acc = correct / max(seen, 1)
        train_loss = loss_sum / max(seen, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} | loss={train_loss:.4f} | train_acc={train_acc:.3f} | val_acc={val_acc:.3f}", flush=True)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.cpu().state_dict(), out_dir / "dermascan_resnet18.pth")
            (out_dir / "class_names.json").write_text(
                json.dumps(train_all.classes, indent=2), encoding="utf-8")
            model.to(device)

    print(f"Best validation accuracy: {best_val:.3f}", flush=True)
    print("Saved: models/dermascan_resnet18.pth and models/class_names.json", flush=True)
    print("Demo training only: not clinically validated or suitable for medical decisions.", flush=True)


if __name__ == "__main__":
    main()
