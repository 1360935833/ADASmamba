"""
AdaS-Mamba: Adaptive Multi-Scale Skeleton Scanning for 3D Human Pose Estimation

Main model integrating three innovations:
1. Adaptive Scan Path Generator — data-driven joint scan order selection
2. Structure-Gated SSM — dynamic global/local information fusion
3. Adaptive Multi-Scale Fusion — learnable scale weights

Based on the SasMamba architecture (WACV 2026), extended with adaptive mechanisms.

Design principle: The adaptive scan path selection reorders the joint dimension
BEFORE the SSM scan, and restores the original order AFTER. This keeps the SSM
internals (BiSTSSM, CrossScan, etc.) unchanged while achieving adaptive scanning.

Architecture:
    2D Pose Input [B, T, J, 2]
        │
        ▼
    Spatial-Temporal Embedding
        │
        ▼
    AdaSM Block × N (spatial-temporal alternating)
    ┌─────────────────────────────────────────┐
    │  Adaptive Scan Path Generator           │  ← Innovation 1
    │  ├─ Candidate path library (K paths)    │
    │  ├─ Path selector MLP                   │
    │  └─ Joint reorder → SSM → restore       │
    │                                         │
    │  Structure-Gated SSM                    │  ← Innovation 2
    │  ├─ Structure encoder (static+dynamic)  │
    │  └─ Gate: g·SSM_out + (1-g)·local      │
    │                                         │
    │  Adaptive Multi-Scale Fusion            │  ← Innovation 3
    │  └─ Learnable scale weights (α₁,α₂,α₃) │
    └─────────────────────────────────────────┘
        │
        ▼
    Regression Head → 3D Pose [B, T, J, 3]
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

try:
    from model.modules.mambablocks import BiSTSSMBlock
    from model.modules.structure_gate import StructureGate, AdaptiveScaleFusion
    from model.modules.adaptive_scan import AdaptiveScanPathGenerator, build_candidate_paths
except ImportError:
    from modules.mambablocks import BiSTSSMBlock
    from modules.structure_gate import StructureGate, AdaptiveScaleFusion
    from modules.adaptive_scan import AdaptiveScanPathGenerator, build_candidate_paths


class AdaSMambaBlock(nn.Module):
    """Core AdaSMamba block with adaptive scan, structure gate, and adaptive fusion.

    Key design: Adaptive scan path selection reorders the joint dimension BEFORE
    SSM scanning, then restores the original order AFTER. This approach:
    - Keeps the SSM internals (BiSTSSM, CrossScan) unchanged
    - Works with any forward_type (v2_plus_poselimbs_v3, etc.)
    - Is compatible with the existing CUDA-optimized selective scan kernels
    - Adds minimal overhead (one MLP forward + one index operation)

    The reordering changes which joints are adjacent in the 1D scan sequence,
    effectively making the SSM "read" the skeleton in a different order.
    """

    def __init__(
            self,
            hidden_dim: int = 64,
            mlp_ratio: float = 2.0,
            drop_path: float = 0.0,
            norm_layer: nn.Module = nn.LayerNorm,
            ssm_conv: int = 7,
            forward_type: str = 'v2_plus_poselimbs_v3',
            num_joints: int = 17,
            # AdaSMamba-specific options
            use_adaptive_scan: bool = True,
            use_structure_gate: bool = True,
            use_adaptive_scale: bool = False,
            num_candidate_paths: int = 4,
            is_spatial: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        self.use_adaptive_scan = use_adaptive_scan
        self.use_structure_gate = use_structure_gate
        self.use_adaptive_scale = use_adaptive_scale
        self.is_spatial = is_spatial

        # Original BiSTSSMBlock (unchanged)
        self.ssm_block = BiSTSSMBlock(
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            ssm_conv=ssm_conv,
            forward_type=forward_type,
        )

        # Innovation 1: Adaptive Scan Path Generator
        # Only applied to spatial blocks (W = num_joints = 17)
        if use_adaptive_scan and is_spatial:
            self.path_generator = AdaptiveScanPathGenerator(
                dim=hidden_dim,
                num_joints=num_joints,
                num_paths=num_candidate_paths,
            )
        else:
            self.path_generator = None

        # Innovation 2: Structure Gate
        if use_structure_gate:
            self.structure_gate = StructureGate(
                dim=hidden_dim,
                num_joints=num_joints,
            )
        else:
            self.structure_gate = None

        # Innovation 3: Adaptive Scale Fusion
        if use_adaptive_scale:
            self.scale_fusion = AdaptiveScaleFusion(dim=hidden_dim)
        else:
            self.scale_fusion = None

    def _reorder_joints(self, x, scan_indices, inverse=False):
        """Reorder the joint dimension of feature tensor.

        Args:
            x: [B, F, N, C] feature tensor
            scan_indices: [N] or [B, N] scan order
            inverse: if True, restore original order (inverse permutation)

        Returns:
            x_reordered: [B, F, N, C]
        """
        if scan_indices.dim() == 1:
            if inverse:
                # Compute inverse permutation
                inv_indices = torch.argsort(scan_indices)
                return x[:, :, inv_indices, :]
            else:
                return x[:, :, scan_indices, :]
        else:
            # Per-sample scan order
            B, F, N, C = x.shape
            if inverse:
                # Compute inverse for each sample
                inv_indices = torch.argsort(scan_indices, dim=-1)  # [B, N]
            else:
                inv_indices = scan_indices
            # Use gather for per-sample reordering
            idx = inv_indices.unsqueeze(1).unsqueeze(-1).expand(B, F, N, C)
            return torch.gather(x, 2, idx)

    def forward(self, x):
        """Forward pass.

        Args:
            x: [B, F, N, C] feature tensor

        Returns:
            out: [B, F, N, C]
        """
        residual = x

        # --- Innovation 1: Adaptive Scan Path ---
        scan_indices = None
        path_weights = None
        if self.path_generator is not None:
            scan_indices, path_weights, _ = self.path_generator(x)

            # Reorder joints according to selected scan path
            x = self._reorder_joints(x, scan_indices, inverse=False)

        # --- SSM processing (inner BiSTSSMBlock) ---
        # The SSM scans the reordered joints in sequence
        x_ssm = self.ssm_block.op(self.ssm_block.norm(x))

        # --- Innovation 2: Structure Gate ---
        if self.structure_gate is not None:
            # Use the pre-reorder residual for gating
            x_gated = self.structure_gate(x_ssm, residual, scan_indices)
        else:
            x_gated = x_ssm + residual

        # --- MLP ---
        x_out = self.ssm_block.mlp(self.ssm_block.norm2(x_gated)) + x_gated

        return x_out


class AdaSMamba(nn.Module):
    """AdaS-Mamba: Adaptive Multi-Scale Skeleton Scanning for 3D HPE.

    Args:
        num_frame: number of input frames (default 243 for H36M)
        num_joints: number of joints (17 for H36M, 15 for others)
        in_chans: input channels (2 for 2D, 3 for 3D)
        embed_dim_ratio: embedding dimension (default 64)
        depth: number of spatial-temporal block pairs (default 5)
        mlp_ratio: MLP hidden dimension ratio
        ssm_conv: SSM convolution kernel size (default 7)
        drop_rate: dropout rate
        drop_path_rate: stochastic depth rate
        use_adaptive_scan: enable Innovation 1
        use_structure_gate: enable Innovation 2
        use_adaptive_scale: enable Innovation 3
        num_candidate_paths: K candidate scan paths
    """

    def __init__(
            self,
            num_frame: int = 243,
            num_joints: int = 17,
            in_chans: int = 2,
            embed_dim_ratio: int = 64,
            depth: int = 5,
            mlp_ratio: float = 2.0,
            ssm_conv: int = 7,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.2,
            norm_layer=None,
            # AdaSMamba options
            use_adaptive_scan: bool = True,
            use_structure_gate: bool = True,
            use_adaptive_scale: bool = False,
            num_candidate_paths: int = 4,
    ):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio
        out_dim = 3

        self.num_frame = num_frame
        self.num_joints = num_joints
        self.use_adaptive_scan = use_adaptive_scan
        self.use_structure_gate = use_structure_gate
        self.use_adaptive_scale = use_adaptive_scale

        # Spatial embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        # Temporal embedding
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        # Spatial blocks (adaptive scan applies here since W = num_joints = 17)
        self.STEblocks = nn.ModuleList([
            AdaSMambaBlock(
                hidden_dim=embed_dim_ratio,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                ssm_conv=ssm_conv,
                forward_type='v2_plus_poselimbs_v3',
                num_joints=num_joints,
                use_adaptive_scan=use_adaptive_scan,
                use_structure_gate=use_structure_gate,
                use_adaptive_scale=use_adaptive_scale,
                num_candidate_paths=num_candidate_paths,
                is_spatial=True,
            )
            for i in range(depth)
        ])

        # Temporal blocks (adaptive scan does NOT apply; W = num_frame, not joints)
        self.TTEblocks = nn.ModuleList([
            AdaSMambaBlock(
                hidden_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                ssm_conv=ssm_conv,
                forward_type='v2_plus_poselimbs_v3',
                num_joints=num_joints,
                use_adaptive_scan=False,  # No adaptive scan for temporal
                use_structure_gate=use_structure_gate,
                use_adaptive_scale=False,
                is_spatial=False,
            )
            for i in range(depth)
        ])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        # Regression head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.Spatial_pos_embed, std=0.02)
        trunc_normal_(self.Temporal_pos_embed, std=0.02)

    def STE_forward(self, x):
        """Spatial Transformer Encoder forward."""
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c  -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = rearrange(x, '(b f) n c  -> b f n c', f=f)
        blk = self.STEblocks[0]
        x = blk(x)
        x = self.Spatial_norm(x)
        return x

    def TTE_foward(self, x):
        """Temporal Transformer Encoder forward."""
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n cw -> (b n) f cw')
        x += self.Temporal_pos_embed[:, :f, :]
        x = self.pos_drop(x)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        blk = self.TTEblocks[0]
        x = blk(x)
        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        """Spatial-Temporal alternating forward."""
        for i in range(1, self.block_depth):
            x = self.STEblocks[i](x)
            x = self.Spatial_norm(x)
            x = self.TTEblocks[i](x)
            x = self.Temporal_norm(x)
        return x

    def forward(self, x):
        """Full forward pass.

        Args:
            x: [B, T, J, 2] 2D pose sequence

        Returns:
            out: [B, T, J, 3] 3D pose sequence
        """
        b, f, n, c = x.shape
        x = self.STE_forward(x)
        x = self.TTE_foward(x)
        x = self.ST_foward(x)
        x = self.head(x)
        x = x.view(b, f, n, -1)
        return x

    def get_path_selection_stats(self):
        """Get adaptive scan path selection statistics for analysis."""
        stats = {}
        for i, blk in enumerate(self.STEblocks):
            if blk.path_generator is not None:
                stats[f'STEblock_{i}'] = blk.path_generator.get_selection_stats()
        return stats

    def reset_path_stats(self):
        """Reset path selection statistics."""
        for blk in self.STEblocks:
            if blk.path_generator is not None:
                blk.path_generator.reset_stats()


def create_adasmamba(config=None):
    """Factory function to create AdaSMamba from config."""
    if config is None:
        return AdaSMamba(
            num_frame=243,
            num_joints=17,
            in_chans=2,
            embed_dim_ratio=64,
            depth=5,
            mlp_ratio=2.0,
            ssm_conv=7,
            use_adaptive_scan=True,
            use_structure_gate=True,
            use_adaptive_scale=False,
            num_candidate_paths=4,
        )

    kwargs = {
        'num_frame': getattr(config, 'n_frames', 243),
        'num_joints': getattr(config, 'num_joints', 17),
        'in_chans': 2,
        'embed_dim_ratio': getattr(config, 'dim_feat', 64),
        'depth': getattr(config, 'n_layers', 5),
        'mlp_ratio': getattr(config, 'mlp_ratio', 2.0),
        'ssm_conv': getattr(config, 'ssm_conv', 7),
        'use_adaptive_scan': getattr(config, 'use_adaptive_scan', True),
        'use_structure_gate': getattr(config, 'use_structure_gate', True),
        'use_adaptive_scale': getattr(config, 'use_adaptive_scale', False),
        'num_candidate_paths': getattr(config, 'num_candidate_paths', 4),
    }
    return AdaSMamba(**kwargs)


if __name__ == "__main__":
    # Test the model
    model = create_adasmamba()
    model = model.cuda()
    x = torch.randn(2, 243, 17, 2).cuda()
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    from thop import profile, clever_format
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")

    # Print path selection stats
    stats = model.get_path_selection_stats()
    print("\nPath selection stats:")
    for block_name, block_stats in stats.items():
        print(f"  {block_name}:")
        for path_name, prob in block_stats.items():
            print(f"    {path_name}: {prob:.4f}")