import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data", default="training/test_data")
    p.add_argument("--weights", default="models/dermascan_resnet18.pth")
    args=p.parse_args()
    data=Path(args.data); weights=Path(args.weights)
    if not data.exists() or not weights.exists():
        raise SystemExit("Provide an untouched test dataset and trained weights first.")
    tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    ds=datasets.ImageFolder(data,transform=tfm)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=models.resnet18(weights=None); model.fc=nn.Linear(model.fc.in_features,len(ds.classes))
    model.load_state_dict(torch.load(weights,map_location=device)); model.to(device).eval()
    loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)
    y_true=[]; y_pred=[]
    with torch.no_grad():
        for x,y in loader:
            pred=model(x.to(device)).argmax(1).cpu().tolist()
            y_pred.extend(pred); y_true.extend(y.tolist())
    print("Accuracy:",accuracy_score(y_true,y_pred))
    print(classification_report(y_true,y_pred,target_names=ds.classes,zero_division=0))
    print("Confusion matrix:\n",confusion_matrix(y_true,y_pred))
    print("Metrics describe only this test set and do not establish clinical safety.")
if __name__=="__main__": main()
