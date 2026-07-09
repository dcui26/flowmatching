import torch
from transformer_backbone import FlowTransformer
from cfm import CFM
from rectify import Rectify
import argparse
import math
import os

def main(args):
    device = torch.device('cuda')
    x0, x1 = Rectify.extract_coupling(args.ckpt_path, args.coupling_size, device)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({'x0': x0, 'x1': x1}, f"{args.output_dir}/couplings.pt")
    print(f"Saved {len(x0)} couplings to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--coupling_size", type=int, default=50000)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()
    main(args)
