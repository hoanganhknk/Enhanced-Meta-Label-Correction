# pretrain_ssl_resnet32_cifar.py

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from resnet import ResNet32


class SimCLR(nn.Module):
    def __init__(self, base_encoder, proj_dim=128):
        super().__init__()
        self.encoder = base_encoder
        # giả sử encoder có .fc là linear cuối cùng
        dim_mlp = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return z


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    z1, z2: [N, D] normalized embeddings
    """
    N, D = z1.shape
    z = torch.cat([z1, z2], dim=0)  # [2N, D]
    sim = torch.matmul(z, z.T)      # [2N, 2N]
    sim = sim / temperature

    # mask self-similarity
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim_exp = torch.exp(sim) * (~mask)

    # positives: (i, i+N) and (i+N, i)
    pos = torch.cat([
        torch.diag(sim, N),
        torch.diag(sim, -N)
    ], dim=0)

    denom = sim_exp.sum(dim=1)
    loss = - pos + torch.log(denom)
    return loss.mean()


def get_simclr_transform():
    return T.Compose([
        T.RandomResizedCrop(32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])


class TwoCropsTransform:
    """Trả về 2 augment khác nhau của cùng 1 ảnh."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


def build_dataloader(dataset, data_path, batch_size, num_workers=4):
    transform = TwoCropsTransform(get_simclr_transform())
    if dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform
        )
    else:
        train_set = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=transform
        )

    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def pretrain_ssl(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = build_dataloader(args.dataset, args.data_path, args.bs, args.num_workers)

    num_classes = 10 if args.dataset == 'cifar10' else 100
    base_encoder = ResNet32(num_classes=num_classes)   # CIFAR ResNet32 backbone
    model = SimCLR(base_encoder, proj_dim=args.proj_dim).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for (x1, x2), _ in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            z1 = model(x1)
            z2 = model(x2)

            loss = nt_xent_loss(z1, z2, temperature=args.temp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                print(f"[SSL] Epoch {epoch} Step {global_step} Loss {loss.item():.4f}")
            global_step += 1

        # mỗi epoch save 1 checkpoint
        os.makedirs(args.out_dir, exist_ok=True)
        ckpt_path = os.path.join(
            args.out_dir,
            f"simclr_resnet32_{args.dataset}_epoch{epoch}.pth"
        )

        # để EMLC load được: lưu với key 'model' và prefix 'encoder.module.'
        full_sd = model.state_dict()
        adapted = {}
        for k, v in full_sd.items():
            if k.startswith("encoder."):
                new_key = "encoder.module." + k[len("encoder."):]
                adapted[new_key] = v

        torch.save({"model": adapted}, ckpt_path)
        print(f"[SSL] Saved checkpoint to {ckpt_path}")

    print("Done SSL pretraining!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--out_dir", type=str, default="ssl_ckpts")
    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=400)  # có thể giảm 200 nếu thời gian hạn chế
    parser.add_argument("--lr", type=float, default=0.5)    # typical SimCLR LR (tùy GPU)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--proj_dim", type=int, default=128)
    args = parser.parse_args()
    pretrain_ssl(args)
