import torch

class CFM:
    @staticmethod
    def sample_xt(t, z):
        """
        z: (N, C, H, W)
        t: (N, )
        returns: sampled batch of xt
        """
        eps = torch.randn_like(z)
        t = t.reshape(-1, 1, 1, 1)
        xt = t*z + (1-t)*eps
        return xt, eps
    
    @staticmethod
    def compute_cvf(eps, z):
        return z - eps

    @staticmethod
    def loss(model, t, z):
        xt, eps = CFM.sample_xt(t, z)
        ut = CFM.compute_cvf(eps, z)
        predicted = model(xt, t)
        return torch.mean((predicted - ut) ** 2)
    
    @staticmethod
    def sample(model, num_samples, num_steps, device):
        #default to cifar10
        x = torch.randn(num_samples, 3, 32, 32, device=device)
        for i in range(num_steps):
            t = torch.full((num_samples, ), i / num_steps, device=device)
            x += (1 / num_steps) * model(x, t)
        return x

    @staticmethod
    def heun_sample(model, num_samples, num_steps, device):
        x = torch.randn(num_samples, 3, 32, 32, device=device)
        dt = 1 / num_steps
        for i in range(num_steps):
            t = torch.full((num_samples, ), i / num_steps, device=device)
            v_t = model(x, t)
            x_pred = x + dt * v_t
            v_th = model(x_pred, t+dt)
            x = x_pred + (dt / 2) * (v_th - v_t)
        return x
    
    @staticmethod
    def sample_from_noise_with_heun(model, x0, num_steps, device):
        #default cifar10 also
        x = x0.clone()
        dt = 1 / num_steps
        for i in range(num_steps):
            t = torch.full((x.shape[0], ), i / num_steps, device=device)
            v_t = model(x, t)
            x_pred = x + dt * v_t
            v_th = model(x_pred, t+dt)
            x = x_pred + (dt / 2) * (v_th - v_t)
        return x
