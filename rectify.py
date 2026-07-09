import torch
from transformer_backbone import FlowTransformer
from cfm import CFM
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math

class CoupledDataset(Dataset):
    def __init__(self, x0, x1):
        self.x0 = x0
        self.x1 = x1

    def __len__(self):
        return len(self.x0)
    
    def __getitem__(self, idx):
        return self.x0[idx], self.x1[idx]

#for ema model only rn
class Rectify:
    @staticmethod
    def extract_coupling(ckpt_path, coupling_size, device):
        checkpoint = torch.load(ckpt_path, map_location=device)
        saved_args = checkpoint['args']

        parent_model = FlowTransformer(
            patch_size = saved_args.patch_size,
            hidden_size = saved_args.hidden_size,
            depth = saved_args.depth,
            num_heads = saved_args.num_heads,
            mlp_ratio = saved_args.mlp_ratio
        ).to(device)

        parent_model.load_state_dict(checkpoint['ema_model'])
        parent_model.eval()

        all_x0 = []
        all_x1 = []
        
        total_batches = math.ceil(coupling_size / saved_args.batch_size)
        with torch.no_grad():
            for i in range(total_batches):
                x0 = torch.randn(saved_args.batch_size, 3, 32, 32, device=device)
                x1 = CFM.sample_from_noise_with_heun(parent_model, x0, saved_args.num_steps, device)
                all_x0.append(x0.cpu())
                all_x1.append(x1.cpu())
                if i % 10 == 0:
                    print(f"    Coupling: {i+1}/{total_batches}")
        
        all_x0 = torch.cat(all_x0, dim=0)[:coupling_size]
        all_x1 = torch.cat(all_x1, dim=0)[:coupling_size]
        return all_x0, all_x1
    
    @staticmethod
    def make_data_loader(x0, x1, batch_size, num_workers):
        return DataLoader(CoupledDataset(x0, x1), batch_size=batch_size, num_workers=num_workers, shuffle=True)

    @staticmethod
    def sample_xt(x0, x1, t):
        t = t.reshape(-1, 1, 1, 1)
        xt = t*x1 + (1-t)*x0
        return xt

    @staticmethod
    def compute_cvf(x0, x1):
        return x1 - x0

    @staticmethod
    def loss(model, x0, x1, t):
        xt = Rectify.sample_xt(x0, x1, t)
        ut = Rectify.compute_cvf(x0, x1)
        predicted = model(xt, t)
        return torch.mean((predicted - ut) **2)

