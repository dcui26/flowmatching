import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                        Timestep Embedding (YOUR JOB)                          #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps t in [0, 1] into vector representations of size hidden_size.
    
    TODO: implement this.
    Hint: you need two things:
      1. A sinusoidal embedding of the scalar t (like positional encodings in transformers)
      2. A small MLP to project that embedding to hidden_size

    The output should be a tensor of shape (N, hidden_size).
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim):
        """
        TODO: implement sinusoidal embedding of scalar t.
        t: (N,) tensor of timesteps in [0, 1]
        dim: output dimension
        returns: (N, dim) tensor
        """
        half = dim // 2
        omega = 1.0 / (10000 ** (torch.arange(0, half).float() / half)).to(t.device)
        args = t[:, None].float() * 1000 * omega[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t):
        """
        t: (N,) tensor of timesteps in [0, 1]
        returns: (N, hidden_size) tensor
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(t.device))


#################################################################################
#                              Core Transformer                                  #
#################################################################################

class TransformerBlock(nn.Module):
    """
    Transformer block with adaLN conditioning on timestep.
    adaLN predicts shift and scale for LayerNorm from the time embedding,
    plus gating values for the residual connections.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        # predicts 6 values: shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        x: (N, T, hidden_size) — token sequence
        c: (N, hidden_size)    — time conditioning vector
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    Final layer: adaLN + linear projection from hidden_size to patch pixels.
    Projects each token back to patch_size * patch_size * out_channels values.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class FlowTransformer(nn.Module):
    """
    Transformer backbone for Conditional Flow Matching on CIFAR-10.
    Takes a noisy image x_t and timestep t, returns predicted velocity field.

    Forward signature: forward(x, t) -> velocity field of same shape as x
    """
    def __init__(
        self,
        input_size=32,      # CIFAR-10 image size
        patch_size=2,
        in_channels=3,      # RGB
        hidden_size=384,    # DiT-S width
        depth=12,           # number of transformer blocks
        num_heads=6,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels  # velocity field has same shape as input
        self.patch_size = patch_size

        # patchify input image into token sequence
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches

        # fixed 2D sin-cos positional embedding, not learned
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # time conditioning
        self.t_embedder = TimestepEmbedder(hidden_size)

        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # final projection back to pixel space
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # fixed sin-cos positional embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # patch embedding init
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # zero-init adaLN modulation (so blocks are identity at init)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # zero-init final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        returns: (N, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x, t):
        """
        x: (N, C, H, W) noisy image at time t
        t: (N,) timesteps in [0, 1]
        returns: (N, C, H, W) predicted velocity field
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, hidden_size)
        c = self.t_embedder(t)                    # (N, hidden_size)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)               # (N, T, patch_size**2 * C)
        x = self.unpatchify(x)                   # (N, C, H, W)
        return x


#################################################################################
#                        Positional Embedding Utilities                          #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)