import numpy as np
import torch
import torchvision as tv
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from transformer_backbone import FlowTransformer
from cfm import CFM
from copy import deepcopy
import time
import math
import os

def cifar10_loader(args):
    transform = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_data = tv.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    loaded_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loaded_train

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_constant_schedule_with_warmup(optimizer, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.data, alpha=1 - decay)

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')

    loader = cifar10_loader(args)

    model = FlowTransformer(
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio
    ).to(device)

    ema_model = deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    no_decay_keys = {'bias', 'norm'}
    decay_params = [p for n, p in model.named_parameters() if not any(k in n for k in no_decay_keys)]
    no_decay_params = [p for n, p in model.named_parameters() if any(k in n for k in no_decay_keys)]
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': args.wd},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=args.lr)

    total_steps = len(loader) * args.epochs

    if args.lr_schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_steps)

    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        ema_model.load_state_dict(ckpt['ema_model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params / 1e6:.2f} M")

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if epoch == args.ema_warmup:
            ema_model.load_state_dict(model.state_dict())
        running_loss = 0.0
        t0 = time.time()
        for z, _ in loader:
            z = z.to(device)
            t = torch.rand(z.shape[0], device=device)

            loss = CFM.loss(model, t, z)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            optimizer.step()
            scheduler.step()
            if epoch >= args.ema_warmup:
                update_ema(ema_model, model)
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"    Loss: {avg_loss:.4f}")
        print(f"    Grad Norm: {grad_norm:.4f}")
        print(f"    Time: {epoch_time:.1f}s")

        if (epoch+1) % args.sample_interval == 0:
            model.eval()
            t_sample = time.time()
            with torch.no_grad():
                samples = CFM.sample(model, args.num_samples, args.num_steps, device)
                samples_ema = CFM.sample(ema_model, args.num_samples, args.num_steps, device)
            sample_time = time.time() - t_sample
            print(f"    Sample time: {sample_time:.1f}s ({args.num_steps} steps)")

            samples = samples.clamp(-1, 1)
            samples = (samples + 1) / 2
            grid = tv.utils.make_grid(samples, nrow=int(math.sqrt(args.num_samples)))
            tv.utils.save_image(grid, f"{args.save_dir}/samples_epoch_{epoch+1}.png")

            samples_ema = samples_ema.clamp(-1, 1)
            samples_ema = (samples_ema + 1) / 2
            grid_ema = tv.utils.make_grid(samples_ema, nrow=int(math.sqrt(args.num_samples)))
            tv.utils.save_image(grid_ema, f"{args.save_dir}/samples_ema_epoch_{epoch+1}.png")

            model.train()

        if (epoch+1) % args.save_interval == 0:
            checkpoint = {
                'model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }
            torch.save(checkpoint, f"{args.save_dir}/checkpoint_{epoch+1}.pt")
            print(f"    Saved checkpoint to {args.save_dir}/checkpoint_{epoch+1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--ema_warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine"])

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)

    parser.add_argument("--sample_interval", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=100)

    parser.add_argument("--save_dir", type=str, default='./cfm_runs')
    parser.add_argument("--save_interval", type=int, default=10)

    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    main(args)