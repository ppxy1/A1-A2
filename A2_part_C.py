import os, argparse, json
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class BrainCNN(nn.Module):
    def __init__(self, num_classes: int, img_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 20, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2, 2)
        feat_hw = img_size // 8
        self.fc   = nn.Linear(32 * feat_hw * feat_hw, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        return self.fc(x)

def compute_class_weights(dataset):
    num_classes = len(dataset.classes)
    counts = Counter(dataset.targets)
    freq = torch.tensor([counts.get(i, 0) for i in range(num_classes)], dtype=torch.float)
    freq = torch.clamp(freq, min=1.0)
    w = 1.0 / freq
    w = w * (num_classes / torch.clamp(w.sum(), min=1e-12))
    return w

def compute_sample_weights(dataset):
    num_classes = len(dataset.classes)
    counts = Counter(dataset.targets)
    class_w = torch.tensor([1.0 / counts.get(i, 1) for i in range(num_classes)], dtype=torch.float)
    targets = torch.as_tensor(dataset.targets, dtype=torch.long)
    return class_w[targets]

def get_loaders(train_dir, test_dir, img_size, bs, workers, use_sampler=True):
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    tf_test  = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=tf_train)
    test_ds  = datasets.ImageFolder(test_dir,  transform=tf_test)
    if use_sampler:
        sw = compute_sample_weights(train_ds)
        sampler = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)
        train_ld = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=workers, pin_memory=True)
    else:
        train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=bs*2, shuffle=False, num_workers=workers, pin_memory=True)
    return train_ld, test_ld, len(train_ds.classes), train_ds

def train_one_epoch(model, loader, optimizer, loss_fct, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fct(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, loss_fct, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fct(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir",  required=True)
    ap.add_argument("--img_size",  type=int, default=150)
    ap.add_argument("--epochs",    type=int, default=10)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--bs",        type=int, default=32)
    ap.add_argument("--workers",   type=int, default=4)
    ap.add_argument("--save_path", type=str, default="best_cnn.pt")
    ap.add_argument("--sampler",   type=int, default=1)
    ap.add_argument("--use_class_weight", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ld, test_ld, n_classes, train_ds = get_loaders(
        args.train_dir, args.test_dir, args.img_size, args.bs, args.workers, use_sampler=bool(args.sampler)
    )
    model = BrainCNN(num_classes=n_classes, img_size=args.img_size).to(device)
    if args.use_class_weight:
        class_w = compute_class_weights(train_ds).to(device)
        loss_fct = nn.CrossEntropyLoss(weight=class_w)
    else:
        loss_fct = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = -1.0
    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(1, args.epochs+1), desc="Epochs"):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, optimizer, loss_fct, device)
        te_loss, te_acc = evaluate(model, test_ld, loss_fct, device)
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["test_loss"].append(te_loss);  hist["test_acc"].append(te_acc)
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "test_acc": te_acc, "args": vars(args)}, args.save_path)
        print(f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.3f} | test {te_loss:.4f}/{te_acc:.3f} | best {best_acc:.3f}")

    with open("train_log.json", "w") as f:
        json.dump(hist, f, indent=2)

    epochs = range(1, args.epochs+1)
    plt.figure(figsize=(8,4))
    plt.plot(epochs, hist["train_loss"], label="Train loss")
    plt.plot(epochs, hist["test_loss"],  label="Test loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss over epochs"); plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=160, bbox_inches="tight")
    print("Saved best model to", args.save_path)
    print("Saved loss plot to loss_curves.png and logs to train_log.json")

if __name__ == "__main__":
    main()

# python A2_c.py \
#   --train_dir /raid/baiyang/xiaoyu/assignment/ML_test/question2/Training \
#   --test_dir  /raid/baiyang/xiaoyu/assignment/ML_test/question2/Testing \
#   --img_size 150 --epochs 10 --bs 16 --sampler 1 --use_class_weight 1