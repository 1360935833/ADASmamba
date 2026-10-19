"""
Structure-Gated SSM (Innovation 2 of AdaSMamba)
-----------------------------------------------
Dynamically regulates the fusion ratio between SSM's global state
propagation and local input based on joint structure attributes,
motion dynamics, and scan path context.

This module provides:
1. StructureGate: core gating mechanism
2. StructureEncoder: builds joint structure representations
3. HierarchicalScaleFusion: adaptive multi-scale weight fusion (Innovation 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureEncoder(nn.Module):
    """Builds dynamic structure encoding for each joint at each timestep.

    Encoding consists of three parts:
    - Static skeleton attributes (joint type, tree depth, limb group)
    - Dynamic motion features (velocity, acceleration, parent-relative angle)
    - Scan path context (relationship with previous joint in scan order)
    """

    def __init__(self, dim, num_joints=17, struct_dim=32):
        super().__init__()
        self.num_joints = num_joints
        self.struct_dim = struct_dim
        self.component_dim = struct_dim // 4

        # Static skeleton attribute embeddings
        self.joint_type_embed = nn.Parameter(torch.randn(1, 1, num_joints, self.component_dim))

        # Learnable limb group embedding (5 groups: torso, left arm, right arm, left leg, right leg)
        # Human3.6M joint groups:
        # 0: hip, 1: r_hip, 2: r_knee, 3: r_ankle, 4: l_hip, 5: l_knee, 6: l_ankle,
        # 7: spine, 8: thorax, 9: neck/nose, 10: head, 11: l_shoulder, 12: l_elbow,
        # 13: l_wrist, 14: r_shoulder, 15: r_elbow, 16: r_wrist
        self.register_buffer('limb_group', torch.tensor([
            0,  # hip -> torso
            4,  # r_hip -> right leg
            4,  # r_knee -> right leg
            4,  # r_ankle -> right leg
            3,  # l_hip -> left leg
            3,  # l_knee -> left leg
            3,  # l_ankle -> left leg
            0,  # spine -> torso
            0,  # thorax -> torso
            0,  # neck/nose -> torso
            0,  # head -> torso
            1,  # l_shoulder -> left arm
            1,  # l_elbow -> left arm
            1,  # l_wrist -> left arm
            2,  # r_shoulder -> right arm
            2,  # r_elbow -> right arm
            2,  # r_wrist -> right arm
        ], dtype=torch.long))
        self.limb_embed = nn.Embedding(5, self.component_dim)

        # Tree depth embedding (root=0, leaf=3)
        self.register_buffer('tree_depth', torch.tensor([
            0,  # hip
            1,  # r_hip
            2,  # r_knee
            3,  # r_ankle
            1,  # l_hip
            2,  # l_knee
            3,  # l_ankle
            1,  # spine
            2,  # thorax
            3,  # neck
            3,  # head
            2,  # l_shoulder
            3,  # l_elbow
            3,  # l_wrist
            2,  # r_shoulder
            3,  # r_elbow
            3,  # r_wrist
        ], dtype=torch.long))
        self.depth_embed = nn.Embedding(4, self.component_dim)

        # Dynamic motion projector
        self.motion_proj = nn.Linear(6, self.component_dim)  # vel(2) + acc(2) + parent_rel(2)

        # Scan context projector
        self.scan_ctx_proj = nn.Linear(3, self.component_dim)  # same_limb, tree_dist, is_parent_child

        # Fusion layer
        self.fusion = nn.Linear(3 * self.component_dim, struct_dim)

    def get_scan_context(self, scan_indices, num_joints):
        """Compute scan context: relationship between consecutive joints in scan order.

        Args:
            scan_indices: [W] or [B, W] the scan order
            num_joints: int

        Returns:
            ctx: [B, W, 3] in original joint order, with
                (same_limb, tree_dist_norm, is_parent_child)
        """
        if scan_indices.dim() == 1:
            scan_indices = scan_indices.unsqueeze(0)  # [1, W]
        B, W = scan_indices.shape

        # Previous joint in scan order
        prev_idx = torch.cat([scan_indices[:, :1], scan_indices[:, :-1]], dim=1)  # [B, W]

        # Same limb group?
        same_limb = (self.limb_group[scan_indices] == self.limb_group[prev_idx]).float()  # [B, W]

        # Tree distance
        tree_dist = (self.tree_depth[scan_indices] - self.tree_depth[prev_idx]).abs().float() / 3.0

        # Parent-child check (using H36M skeleton: 0→1, 0→4, 0→7, 7→8, 8→9, 9→10,
        # 8→11, 11→12, 12→13, 8→14, 14→15, 15→16, 1→2, 2→3, 4→5, 5→6)
        parent_map = {1: 0, 4: 0, 7: 0, 8: 7, 9: 8, 10: 9, 11: 8, 12: 11,
                      13: 12, 14: 8, 15: 14, 16: 15, 2: 1, 3: 2, 5: 4, 6: 5}
        is_parent = torch.zeros(B, W, device=scan_indices.device)
        for b in range(B):
            for w in range(W):
                cur = scan_indices[b, w].item()
                prev = prev_idx[b, w].item()
                if parent_map.get(cur) == prev or parent_map.get(prev) == cur:
                    is_parent[b, w] = 1.0

        ctx_in_scan_order = torch.stack([same_limb, tree_dist, is_parent], dim=-1)  # [B, W, 3]
        ctx = torch.zeros_like(ctx_in_scan_order)
        scatter_idx = scan_indices.unsqueeze(-1).expand_as(ctx_in_scan_order)
        ctx.scatter_(1, scatter_idx, ctx_in_scan_order)
        return ctx

    def forward(self, x, scan_indices=None):
        # x: [B, F, N, C] latent joint features
        batch_size, num_frames, num_joints, channels = x.shape

        # Static encoding
        limb_emb = self.limb_embed(self.limb_group[:num_joints])  # [N, D/4]
        depth_emb = self.depth_embed(self.tree_depth[:num_joints])  # [N, D/4]
        static_enc = (self.joint_type_embed[:, :, :num_joints, :] +
                      limb_emb.unsqueeze(0).unsqueeze(0) +
                      depth_emb.unsqueeze(0).unsqueeze(0))  # [1, 1, N, D/4]

        # Dynamic encoding: velocity, acceleration, parent-relative
        if num_frames > 1:
            vel = x[:, 1:] - x[:, :-1]  # [B, F-1, N, C]
            vel = F.pad(vel, (0, 0, 0, 0, 1, 0))  # [B, F, N, C]
            if num_frames > 2:
                acc = vel[:, 1:] - vel[:, :-1]  # [B, F-1, N, C]
                acc = F.pad(acc, (0, 0, 0, 0, 1, 0))  # [B, F, N, C]
            else:
                acc = torch.zeros_like(vel)
        else:
            vel = torch.zeros_like(x)
            acc = torch.zeros_like(x)

        # Parent-relative (simplified: use parent joint index 0 as reference)
        parent_rel = x - x[:, :, 0:1, :]  # relative to hip

        # Project the first two latent feature channels from each motion signal
        vel_xy = vel[..., :2] if channels >= 2 else torch.zeros(
            batch_size, num_frames, num_joints, 2, device=x.device, dtype=x.dtype)
        acc_xy = acc[..., :2] if channels >= 2 else torch.zeros(
            batch_size, num_frames, num_joints, 2, device=x.device, dtype=x.dtype)
        pr_xy = parent_rel[..., :2] if channels >= 2 else torch.zeros(
            batch_size, num_frames, num_joints, 2, device=x.device, dtype=x.dtype)

        motion_enc = self.motion_proj(torch.cat([vel_xy, acc_xy, pr_xy], dim=-1))  # [B, F, N, D/4]

        # Scan context encoding
        if scan_indices is not None:
            scan_ctx = self.get_scan_context(scan_indices, num_joints)  # [B, N, 3]
            scan_ctx = scan_ctx.unsqueeze(1).expand(batch_size, num_frames, num_joints, 3)
            scan_enc = self.scan_ctx_proj(scan_ctx)  # [B, F, N, D/4]
        else:
            scan_enc = torch.zeros(
                batch_size, num_frames, num_joints, self.component_dim,
                device=x.device, dtype=x.dtype)

        # Concatenate and fuse
        struct_enc = torch.cat([
            static_enc.expand(batch_size, num_frames, -1, -1), motion_enc, scan_enc
        ], dim=-1)
        struct_enc = self.fusion(struct_enc)  # [B, F, N, struct_dim]

        return struct_enc


class StructureGate(nn.Module):
    """Structure-gated SSM: dynamically regulates global vs local information fusion.

    g = σ(W_g · [x_local, s_struct])
    y = g ⊙ y_ssm + (1 - g) ⊙ x_local

    where s_struct encodes static joint attributes, dynamic motion, and scan context.
    """

    def __init__(self, dim, num_joints=17, struct_dim=32, reduction=4):
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints

        # Structure encoder
        self.struct_encoder = StructureEncoder(dim, num_joints, struct_dim)

        # Gate MLP
        self.gate = nn.Sequential(
            nn.Linear(dim + struct_dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x_ssm, x_local, scan_indices=None):
        """Apply structure gate.

        Args:
            x_ssm: [B, F, N, C] SSM output (global information)
            x_local: [B, F, N, C] local input (before SSM)
            scan_indices: optional scan order for context

        Returns:
            y: [B, F, N, C] gated output
        """
        B, F, N, C = x_ssm.shape

        # Build structure encoding
        s = self.struct_encoder(x_local, scan_indices)  # [B, F, N, struct_dim]

        # Compute gate
        gate_input = torch.cat([x_local, s], dim=-1)  # [B, F, N, C+struct_dim]
        g = self.gate(gate_input)  # [B, F, N, C]

        # Gated fusion
        return g * x_ssm + (1 - g) * x_local


class AdaptiveScaleFusion(nn.Module):
    """Adaptive multi-scale fusion (Innovation 3).

    Learns data-dependent weights for fusing 3 spatial scales
    (stride-1, stride-2, stride-3) instead of simple addition.
    """

    def __init__(self, dim, num_scales=3, reduction=4):
        super().__init__()
        self.num_scales = num_scales

        # Scale weight predictor
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, num_scales),
        )

    def forward(self, scale_features):
        """Compute adaptive weights and fuse scale features.

        Args:
            scale_features: list of [B, F, N, C] tensors for each scale

        Returns:
            fused: [B, F, N, C] weighted fusion
            weights: [B, num_scales] scale weights (for analysis)
        """
        # Compute weights from finest scale
        weights = self.weight_net(scale_features[0].permute(0, 3, 1, 2))  # [B, 3]
        weights = F.softmax(weights, dim=-1)  # [B, 3]

        # Weighted fusion
        fused = sum(w.unsqueeze(1).unsqueeze(1).unsqueeze(1) * f
                     for w, f in zip(weights.unbind(-1), scale_features))

        return fused, weights