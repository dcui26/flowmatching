import torch
import torchvision as tv
from torchvision.models import inception_v3
import numpy as np
import prdc
import argparse
from torch.utils.data import DataLoader
from transformer_backbone import FlowTransformer
from cfm import CFM
from scipy.linalg import sqrtm
import math

def extract_features(images, inception, device, batch_size=128):
    features = []
    inception.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device).float() / 255.0
            batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            batch = (batch - mean) / std
            feat = inception(batch)
            features.append(feat.cpu().numpy())
    print("    Feature extraction done.")
    return np.concatenate(features, axis=0)

def compute_fid(real_features, fake_features):
    mu_r, mu_f = real_features.mean(0), fake_features.mean(0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)
    
    diff = mu_r - mu_f
    covmean = sqrtm(sigma_r @ sigma_f)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)

def evaluate(args):
    device = torch.device('cuda')

    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    real_data = tv.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    real_loader = DataLoader(real_data, batch_size=args.batch_size, num_workers=args.num_workers)

    all_real = []
    for x, _ in real_loader:
        x = x.clamp(-1, 1)
        x = ((x + 1) / 2 * 255).byte()
        all_real.append(x)
        if len(torch.cat(all_real, dim=0)) >= args.num_samples:
            break

    all_real = torch.cat(all_real, dim=0)[:args.num_samples]

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    saved_args = checkpoint['args']

    ema_model = FlowTransformer(
        patch_size = saved_args.patch_size,
        hidden_size = saved_args.hidden_size,
        depth = saved_args.depth,
        num_heads = saved_args.num_heads,
        mlp_ratio = saved_args.mlp_ratio
    ).to(device)

    ema_model.load_state_dict(checkpoint['ema_model'])
    ema_model.eval()

    all_ema_samples = []

    total_batches = math.ceil(args.num_samples / args.batch_size)

    with torch.no_grad():
        for i in range(total_batches):
            if args.heun:
                samples = CFM.heun_sample(ema_model, args.batch_size, args.num_steps, device)
            else:
                samples = CFM.sample(ema_model, args.batch_size, args.num_steps, device)
            samples = samples.clamp(-1, 1)
            samples = ((samples + 1) / 2 * 255).byte()
            all_ema_samples.append(samples.cpu())
            if i % 10 == 0:
                print(f"    EMA: {i+1}/{total_batches}", end='\n')
    print()

    all_ema_samples = torch.cat(all_ema_samples, dim=0)[:args.num_samples]

    inception = inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()
    inception = inception.to(device).eval()

    real_features = extract_features(all_real, inception, device)
    ema_features = extract_features(all_ema_samples, inception, device)

    ema_fid_score = compute_fid(real_features, ema_features)
    print(f"EMA FID: {ema_fid_score:.2f}")

    metrics = prdc.compute_prdc(real_features, ema_features, nearest_k=5)
    print(f"EMA Precision: {metrics['precision']:.4f}")
    print(f"EMA Recall: {metrics['recall']:.4f}")

    if args.eval_raw:
        model = FlowTransformer(
            patch_size = saved_args.patch_size,
            hidden_size = saved_args.hidden_size,
            depth = saved_args.depth,
            num_heads = saved_args.num_heads,
            mlp_ratio = saved_args.mlp_ratio
        ).to(device)

        model.load_state_dict(checkpoint['model'])
        model.eval()

        all_samples = []
        with torch.no_grad():
            for i in range(total_batches):
                if args.heun:
                    samples = CFM.heun_sample(model, args.batch_size, args.num_steps, device)
                else:
                    samples = CFM.sample(model, args.batch_size, args.num_steps, device)
                samples = samples.clamp(-1, 1)
                samples = ((samples + 1) / 2 * 255).byte()
                all_samples.append(samples.cpu())
                if i % 10 == 0:
                    print(f"    Raw: {i+1}/{total_batches}", end='\n')
        print()

        all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]

        features = extract_features(all_samples, inception, device)
        fid_score = compute_fid(real_features, features)
        print(f"Raw FID: {fid_score:.2f}")

        metrics = prdc.compute_prdc(real_features, features, nearest_k=5)
        print(f"Raw Precision: {metrics['precision']:.4f}")
        print(f"Raw Recall: {metrics['recall']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_raw", action="store_true")
    parser.add_argument("--heun", action="store_true")

    args = parser.parse_args()
    evaluate(args)
