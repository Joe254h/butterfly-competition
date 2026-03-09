
import os, numpy as np, pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from baseline.model import build_model

IMAGE_SIZE=224
BATCH_SIZE=32
EPOCHS=8
LR=3e-4

TRAIN_DIR="data/train"
META_FILE="data/Training_set.csv"

class ButterflyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df=df
        self.img_dir=img_dir
        self.transform=transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        path=os.path.join(self.img_dir,row["filename"])
        img=Image.open(path).convert("RGB")
        if self.transform: img=self.transform(img)
        return img,row["label_enc"]

def main():

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    meta=pd.read_csv(META_FILE)

    enc=LabelEncoder()
    meta["label_enc"]=enc.fit_transform(meta["label"])

    np.save("label_classes.npy",enc.classes_)
    NUM_CLASSES=len(enc.classes_)

    train,val=train_test_split(meta,train_size=0.8,stratify=meta["label_enc"],random_state=42)

    train_tf=transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    val_tf=transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    train_ds=ButterflyDataset(train,TRAIN_DIR,train_tf)
    val_ds=ButterflyDataset(val,TRAIN_DIR,val_tf)

    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE)

    model=build_model(NUM_CLASSES).to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)

    best=0

    for epoch in range(EPOCHS):

        model.train()
        train_correct=0
        train_total=0

        for x,y in train_loader:

            x=x.to(device)
            y=y.to(device)

            optimizer.zero_grad()

            out=model(x)
            loss=criterion(out,y)

            loss.backward()
            optimizer.step()

            pred=out.argmax(1)

            train_correct+=(pred==y).sum().item()
            train_total+=y.size(0)

        train_acc=train_correct/train_total

        model.eval()
        val_correct=0
        val_total=0

        with torch.no_grad():
            for x,y in val_loader:

                x=x.to(device)
                y=y.to(device)

                out=model(x)
                pred=out.argmax(1)

                val_correct+=(pred==y).sum().item()
                val_total+=y.size(0)

        val_acc=val_correct/val_total

        print(f"Epoch {epoch+1}/{EPOCHS} Train:{train_acc:.4f} Val:{val_acc:.4f}")

        if val_acc>best:
            best=val_acc
            torch.save(model.state_dict(),"model_weights.pth")

if __name__=="__main__":
    main()
