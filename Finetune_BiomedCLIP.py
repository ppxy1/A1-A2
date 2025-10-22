# Finetune_BiomedCLIP_linear.py
import os, argparse, json
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import open_clip

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
    cls_w = torch.tensor([1.0 / cnt.get(i, 1) for i in range(n)], dtype=torch.float)
    idx = torch.as_tensor(dataset.targets, dtype=torch.long)
    return cls_w[idx]

@torch.no_grad()
def infer_feat_dim(clip_model, loader, device):
    x, _ = next(iter(loader))
    x = x.to(device)
    f = clip_model.encode_image(x)
    return f.shape[-1]

def epoch_loop(clip_model, head, loader, loss_fct, optimizer, device, train=True):
    if train:
        head.train()
    else:
        head.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in tqdm(loader, leave=False, desc="Train" if train else "Eval"):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(x)
        logits = head(feats)
        loss = loss_fct(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir",  required=True)
    ap.add_argument("--epochs",    type=int, default=10)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--bs",        type=int, default=8)
    ap.add_argument("--workers",   type=int, default=4)
    ap.add_argument("--sampler",   type=int, default=1, help="1: WeightedRandomSampler for train")
    ap.add_argument("--use_class_weight", type=int, default=1, help="1: CE class weights")
    ap.add_argument("--save_path", type=str, default="biomedclip_linear.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, preprocess = open_clip.create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    clip_model.eval().requires_grad_(False).to(device)  # 冻结

    train_ds = datasets.ImageFolder(args.train_dir, transform=preprocess)
    test_ds  = datasets.ImageFolder(args.test_dir,  transform=preprocess)

    if args.sampler:
        sw = compute_sample_weights(train_ds)
        train_ld = DataLoader(train_ds, batch_size=args.bs,
                              sampler=WeightedRandomSampler(sw, num_samples=len(sw), replacement=True),
                              num_workers=args.workers, pin_memory=True)
    else:
        train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=max(1, args.bs*2), shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    feat_dim = infer_feat_dim(clip_model, train_ld, device)
    num_classes = len(train_ds.classes)
    head = nn.Linear(feat_dim, num_classes).to(device)

    if args.use_class_weight:
        class_w = compute_class_weights(train_ds).to(device)
        loss_fct = nn.CrossEntropyLoss(weight=class_w)
    else:
        loss_fct = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc, best_ep = -1.0, -1

    for ep in tqdm(range(1, args.epochs+1), desc="Epochs"):
        tr_loss, tr_acc = epoch_loop(clip_model, head, train_ld, loss_fct, optimizer, device, train=True)
        te_loss, te_acc = epoch_loop(clip_model, head, test_ld,  loss_fct, optimizer, device, train=False)

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss);   history["test_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc, best_ep = te_acc, ep
            torch.save({
                "epoch": ep,
                "state_dict": head.state_dict(),
                "test_acc": te_acc,
                "feat_dim": feat_dim,
                "num_classes": num_classes
            }, args.save_path)

        print(f"Epoch {ep:02d} | train {tr_loss:.4f}/{tr_acc:.3f} | "
              f"test {te_loss:.4f}/{te_acc:.3f} | best {best_acc:.3f} @ {best_ep}")

    with open("linear_probe_log.json", "w") as f:
        json.dump(history, f, indent=2)

    epochs = range(1, args.epochs+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["test_loss"],  label="Test loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss (Linear Probe)"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, history["train_acc"], label="Train acc")
    plt.plot(epochs, history["test_acc"],  label="Test acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy (Linear Probe)"); plt.legend()

    plt.tight_layout()
    plt.savefig("linear_probe_curves.png", dpi=160, bbox_inches="tight")
    print(f"Saved best head to {args.save_path} and curves to linear_probe_curves.png")

if __name__ == "__main__":
    main()


# python assignment/Finetune_BiomedCLIP.py \
#   --train_dir assignment/ML_test/question2/Training \
#   --test_dir  assignment/ML_test/question2/Testing \
#   --epochs 10 --bs 8 --lr 3e-4 \
#   --sampler 1 --use_class_weight 1 \
#   --save_path biomedclip_linear.pt
