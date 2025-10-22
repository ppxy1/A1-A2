import os
import argparse
from collections import Counter
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler
from torchvision import datasets, transforms

def make_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return train_tf, test_tf

def balanced_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = Counter(dataset.targets)
    class_w = torch.tensor([1.0 / counts[i] for i in range(len(counts))], dtype=torch.float)
    return class_w[torch.as_tensor(dataset.targets, dtype=torch.long)]

def print_counts(name: str, ds: datasets.ImageFolder):
    counts = Counter(ds.targets)
    print(f"\n[{name}] #classes = {len(ds.classes)}")
    print(f"[{name}] classes: {ds.classes}")
    for i, c in enumerate(ds.classes):
        print(f"[{name}] {c:>20s}: {counts.get(i, 0)} images")

def resolve_dirs(args):
    if args.train_dir and args.test_dir:
        return args.train_dir, args.test_dir
    def pick(root, *names):
        for n in names:
            p = os.path.join(root, n)
            if os.path.isdir(p):
                return p
        return None
    train_dir = args.train_dir or pick(args.data_root, "train", "Train", "Training")
    test_dir  = args.test_dir  or pick(args.data_root, "test", "Test", "val", "Val", "Validation", "Testing")
    if not train_dir or not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train dir not found. Tried under {args.data_root}")
    if not test_dir or not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test dir not found. Tried under {args.data_root}")
    return train_dir, test_dir

def get_loaders(args):
    train_tf, test_tf = make_transforms(args.img_size)
    train_dir, test_dir = resolve_dirs(args)

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    test_set  = datasets.ImageFolder(test_dir,  transform=test_tf)

    print_counts("train", train_set)
    print_counts("test",  test_set)

    w = balanced_weights(train_set)
    train_sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    test_sampler  = SequentialSampler(test_set)

    train_loader = DataLoader(train_set, sampler=train_sampler,
                              batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, sampler=test_sampler,
                              batch_size=args.eval_batch_size,   num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root dir; will look for train/test (or Training/Testing)")
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--test_dir",  type=str, default=None)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train_loader, test_loader = get_loaders(args)

    xb, yb = next(iter(train_loader))
    print(f"\n[quick check] train batch: images {tuple(xb.shape)}, labels {tuple(yb.shape)}")

# python dataloader.py \
#   --train_dir assignment/ML_test/question2/Training \
#   --test_dir  assignment/ML_test/question2/Testing \
#   --img_size 150
