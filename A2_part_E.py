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

class SelfAttention2D(nn.Module):
    def __init__(self, in_dim, reduction: int = 4):
        super().__init__()
        self.reduction = max(int(reduction), 1)
        mid = max(in_dim // 8, 1)
        self.q = nn.Conv2d(in_dim, mid, 1)
        self.k = nn.Conv2d(in_dim, mid, 1)
        self.v = nn.Conv2d(in_dim, in_dim, 1)
        self.pool = nn.Identity() if self.reduction == 1 else nn.AvgPool2d(self.reduction, self.reduction)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).view(b, -1, h*w).transpose(1, 2)
        x_r = self.pool(x)
        hr, wr = x_r.shape[-2:]
        k = self.k(x_r).view(b, -1, hr*wr)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.v(x_r).view(b, c, hr*wr)
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        return self.gamma * out + x
    
class AttnBrainCNN(nn.Module):
    def __init__(self, num_classes: int, img_size: int,
                 sr1: int = 8, sr2: int = 4, sr3: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(12)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.attn1 = SelfAttention2D(12, reduction=sr1)

        self.conv2 = nn.Conv2d(12, 20, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.attn2 = SelfAttention2D(20, reduction=sr2)

        self.conv3 = nn.Conv2d(20, 32, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.attn3 = SelfAttention2D(32, reduction=sr3)

        feat_hw = img_size // 8
        self.fc   = nn.Linear(32 * feat_hw * feat_hw, num_classes)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x); x = self.attn1(x)
        x = F.relu(self.conv2(x));          x = self.pool2(x); x = self.attn2(x)
        x = F.relu(self.bn3(self.conv3(x)));x = self.pool3(x); x = self.attn3(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def compute_class_weights(dataset):
    n = len(dataset.classes)
    cnt = Counter(dataset.targets)
    freq = torch.tensor([cnt.get(i, 0) for i in range(n)], dtype=torch.float)
    freq = torch.clamp(freq, min=1.0)
    w = 1.0 / freq
    w = w * (n / torch.clamp(w.sum(), min=1e-12))
    return w

def compute_sample_weights(dataset):
    n = len(dataset.classes)
    cnt = Counter(dataset.targets)
    class_w = torch.tensor([1.0 / cnt.get(i, 1) for i in range(n)], dtype=torch.float)
    idx = torch.as_tensor(dataset.targets, dtype=torch.long)
    return class_w[idx]

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

def epoch_loop(model, loader, optimizer, loss_fct, device, train=True):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    desc = "Train" if train else "Eval "
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = loss_fct(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

def fit_model(name, model, loaders, loss_fct, lr, epochs, save_path, device):
    train_ld, test_ld = loaders
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_ep = -1.0, -1
    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for ep in tqdm(range(1, epochs+1), desc=f"{name} | Epochs"):
        tr_loss, tr_acc = epoch_loop(model, train_ld, opt, loss_fct, device, train=True)
        te_loss, te_acc = epoch_loop(model, test_ld,  opt, loss_fct, device, train=False)
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["test_loss"].append(te_loss);  hist["test_acc"].append(te_acc)
        if te_acc > best:
            best, best_ep = te_acc, ep
            torch.save({"epoch": ep, "state_dict": model.state_dict(), "test_acc": te_acc}, save_path)
        print(f"{name} | Epoch {ep:02d} | train {tr_loss:.4f}/{tr_acc:.3f} | test {te_loss:.4f}/{te_acc:.3f} | best {best:.3f} @ {best_ep}")
    return hist, best, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir",  required=True)
    ap.add_argument("--img_size",  type=int, default=150)
    ap.add_argument("--epochs",    type=int, default=10)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--bs",        type=int, default=32)
    ap.add_argument("--workers",   type=int, default=4)
    ap.add_argument("--sampler",   type=int, default=1)
    ap.add_argument("--use_class_weight", type=int, default=1)
    ap.add_argument("--save_base", type=str, default="best_baseline.pt")
    ap.add_argument("--save_attn", type=str, default="best_attention.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ld, test_ld, ncls, train_ds = get_loaders(
        args.train_dir, args.test_dir, args.img_size, args.bs, args.workers, use_sampler=bool(args.sampler)
    )
    if args.use_class_weight:
        w = compute_class_weights(train_ds).to(device)
        loss_fct = nn.CrossEntropyLoss(weight=w)
    else:
        loss_fct = nn.CrossEntropyLoss()

    base = BrainCNN(ncls, args.img_size).to(device)
    attn = AttnBrainCNN(ncls, args.img_size).to(device)

    base_hist, base_best, base_best_ep = fit_model(
        "Baseline", base, (train_ld, test_ld), loss_fct, args.lr, args.epochs, args.save_base, device
    )
    attn_hist, attn_best, attn_best_ep = fit_model(
        "Attention", attn, (train_ld, test_ld), loss_fct, args.lr, args.epochs, args.save_attn, device
    )

    with open("compare_logs.json", "w") as f:
        json.dump({"baseline": base_hist, "attention": attn_hist,
                   "best": {"baseline_acc": base_best, "attention_acc": attn_best}}, f, indent=2)

    epochs = range(1, args.epochs+1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, base_hist["train_acc"], "-o", label="Base Train Acc")
    plt.plot(epochs, base_hist["test_acc"],  "-o", label="Base Test Acc")
    plt.plot(epochs, attn_hist["train_acc"], "-s", label="Attn Train Acc")
    plt.plot(epochs, attn_hist["test_acc"],  "-s", label="Attn Test Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy (Baseline vs Attention)")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, base_hist["train_loss"], "-o", label="Base Train Loss")
    plt.plot(epochs, base_hist["test_loss"],  "-o", label="Base Test Loss")
    plt.plot(epochs, attn_hist["train_loss"], "-s", label="Attn Train Loss")
    plt.plot(epochs, attn_hist["test_loss"],  "-s", label="Attn Test Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss (Baseline vs Attention)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("baseline_vs_attention_curves.png", dpi=160, bbox_inches="tight")

    print("\n=== Summary ===")
    print(f"Baseline best Test Acc = {base_best:.3f} at epoch {base_best_ep} (saved: {args.save_base})")
    print(f"Attention best Test Acc = {attn_best:.3f} at epoch {attn_best_ep} (saved: {args.save_attn})")
    print("Curves saved to baseline_vs_attention_curves.png; logs -> compare_logs.json")

if __name__ == "__main__":
    main()

# python A2_part_E.py \
#   --train_dir /raid/baiyang/xiaoyu/assignment/ML_test/question2/Training \
#   --test_dir  /raid/baiyang/xiaoyu/assignment/ML_test/question2/Testing \
#   --img_size 150 --epochs 10 --bs 4 --sampler 1 --use_class_weight 1