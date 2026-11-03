import argparse, json, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="training/data")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    root = Path(args.data)
    if not root.exists():
        raise SystemExit(f"Dataset folder not found: {root}")
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    ])
    ds = datasets.ImageFolder(root, transform=tfm)
    if len(ds.classes) < 2 or len(ds) < 20:
        raise SystemExit("Need at least 2 classes and 20 images for this basic training script.")
    counts = {c:0 for c in ds.classes}
    for _, y in ds.samples: counts[ds.classes[y]] += 1
    print("Class counts:", counts)
    if min(counts.values()) < 5:
        raise SystemExit("Each class needs at least 5 images for this demo. More is strongly recommended.")
    n_val = max(1, int(.2 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))
    # Disable augmentation on validation data.
    val_ds.dataset = datasets.ImageFolder(root, transform=transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(ds.classes))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    for epoch in range(args.epochs):
        model.train(); total=correct=seen=0; loss_sum=0.0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(); out=model(x); loss=loss_fn(out,y); loss.backward(); opt.step()
            loss_sum += loss.item()*y.size(0); correct += (out.argmax(1)==y).sum().item(); seen += y.size(0)
        model.eval(); vc=vs=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device); out=model(x)
                vc += (out.argmax(1)==y).sum().item(); vs += y.size(0)
        val_acc=vc/max(vs,1)
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={loss_sum/max(seen,1):.4f} train_acc={correct/max(seen,1):.3f} val_acc={val_acc:.3f}")
        if val_acc > best:
            best=val_acc
            outdir=Path("models"); outdir.mkdir(exist_ok=True)
            torch.save(model.cpu().state_dict(), outdir/"dermascan_resnet18.pth")
            (outdir/"class_names.json").write_text(json.dumps(ds.classes, indent=2), encoding="utf-8")
            model.to(device)
    print(f"Best validation accuracy: {best:.3f}")
    print("WARNING: image-level random split is not suitable for clinical claims; use patient-level/external validation.")

if __name__ == "__main__": main()
