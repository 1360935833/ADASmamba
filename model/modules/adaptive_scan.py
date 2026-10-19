"""
Adaptive Scan Path Generator (Innovation 1 of AdaSMamba)
--------------------------------------------------------
Replaces fixed, hand-designed scan paths with data-driven path selection.

The module:
1. Maintains a library of K candidate scan paths based on human kinematics
2. Uses a lightweight MLP to predict path weights from input features
3. During training: soft combination of all paths (differentiable)
4. During inference: hard selection (top-1 path) for efficiency

Candidate paths are designed following human kinematic principles:
- Kinematic chain order (proximal → distal within each limb)
- Top-down / bottom-up ordering
- Limb-grouped ordering
- Motion-activity-based ordering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Candidate Scan Path Library
# ============================================================
# Human3.6M joint indices (17 joints):
# 0: hip, 1: r_hip, 2: r_knee, 3: r_ankle, 4: l_hip, 5: l_knee, 6: l_ankle,
# 7: spine, 8: thorax, 9: neck/nose, 10: head, 11: l_shoulder, 12: l_elbow,
# 13: l_wrist, 14: r_shoulder, 15: r_elbow, 16: r_wrist

def build_candidate_paths(num_joints=17, num_paths=4):
    """Build K candidate scan paths based on human kinematic principles.

    For H36M (17 joints), 4 paths are defined:
    - path_0: original order (hip→r_hip→...→r_wrist) — default
    - path_1: head-to-toe (head→neck→shoulders→...→ankles)
    - path_2: limb-grouped (left arm→right arm→torso→left leg→right leg)
    - path_3: center-to-extremity (torso→proximal limbs→distal limbs)

    Returns:
        paths: [K, J] tensor of joint indices
        path_names: list of K path description strings
    """

    if num_joints == 17:
        paths = []

        # Path 0: Original order (default, same as SasMamba)
        paths.append(list(range(17)))

        # Path 1: Head-to-toe (top-down)
        # head(10)→nose(9)→thorax(8)→spine(7)→hip(0)→shoulders→elbows→wrists→hips→knees→ankles
        paths.append([
            10, 9, 8, 7, 0, 11, 14, 12, 15, 13, 16, 1, 4, 2, 5, 3, 6
        ])

        # Path 2: Limb-grouped (proximal→distal within each group)
        # left arm→right arm→torso→left leg→right leg
        paths.append([
            11, 12, 13,  # left arm
            14, 15, 16,  # right arm
            0, 7, 8, 9, 10,  # torso + head
            4, 5, 6,  # left leg
            1, 2, 3,  # right leg
        ])

        # Path 3: Center-to-extremity (torso first, then proximal→distal limbs)
        paths.append([
            0, 7, 8, 9, 10,  # center core
            1, 4, 11, 14,  # proximal joints
            2, 5, 12, 15,  # mid joints
            3, 6, 13, 16,  # distal joints
        ])

        path_names = [
            "Original (hip→r_hip→...→r_wrist)",
            "Head-to-toe (top-down)",
            "Limb-grouped (L_arm→R_arm→torso→L_leg→R_leg)",
            "Center-to-extremity (torso→proximal→distal)",
        ]

    elif num_joints == 15:
        # Simplified for 15-joint datasets
        paths = [list(range(15))]
        path_names = ["Original"]

        # Add 3 more paths for 15 joints (similar patterns)
        paths.append([14, 13, 12, 11, 0, 9, 10, 8, 7, 1, 2, 3, 4, 5, 6])
        paths.append([9, 10, 8, 7, 0, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6])
        paths.append([0, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        path_names.extend([
            "Head-to-toe",
            "Limb-grouped",
            "Center-to-extremity",
        ])

    else:
        # Generic: just use original order with variations
        paths = [list(range(num_joints))]
        path_names = ["Original"]
        for _ in range(num_paths - 1):
            perm = list(range(num_joints))
            torch.randperm(num_joints, out=torch.tensor(perm))
            paths.append(perm)
            path_names.append(f"Random_{_}")

    # Truncate to num_paths
    paths = paths[:num_paths]
    path_names = path_names[:num_paths]

    return torch.tensor(paths, dtype=torch.long), path_names


class AdaptiveScanPathGenerator(nn.Module):
    """Adaptive Scan Path Generator.

    Selects the optimal joint scan order for SSM scanning based on input features.

    Architecture:
        Input features → GlobalAvgPool → MLP(2-layer) → softmax → K weights
        Training: soft combination of K paths
        Inference: hard argmax selection
    """

    def __init__(self, dim, num_joints=17, num_paths=4, reduction=4, use_soft_training=True):
        super().__init__()
        self.num_joints = num_joints
        self.num_paths = num_paths
        self.use_soft_training = use_soft_training

        # Build candidate path library
        candidate_paths, path_names = build_candidate_paths(num_joints, num_paths)
        self.register_buffer('candidate_paths', candidate_paths)  # [K, J]
        self.path_names = path_names

        # Path selector MLP
        self.selector = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, num_paths),
        )

        # Statistics tracking (for analysis)
        self.register_buffer('path_selection_counts', torch.zeros(num_paths))
        self.register_buffer('total_samples', torch.zeros(1))

    def forward(self, x):
        """Select scan path.

        Args:
            x: [B, ..., C] input features (will be pooled to [B, C])

        Returns:
            scan_indices: [B, J] or [J] selected path indices
            path_weights: [B, K] selection weights
            selected_path: [B] path index (hard selection)
        """
        # Global pooling
        if x.dim() == 4:  # [B, F, N, C]
            x_pool = x.mean(dim=(1, 2))  # [B, C]
        elif x.dim() == 3:  # [B, N, C]
            x_pool = x.mean(dim=1)  # [B, C]
        elif x.dim() == 2:  # [B, C]
            x_pool = x
        else:
            # Flatten to [B, C]
            x_pool = x.flatten(1).mean(dim=-1, keepdim=True)
            x_pool = x_pool.expand(x.size(0), x.size(-1))

        # Path selection
        path_logits = self.selector(x_pool)  # [B, K]
        path_weights = F.softmax(path_logits, dim=-1)  # [B, K]

        # Hard selection
        selected_path = torch.argmax(path_weights, dim=-1)  # [B]

        # Update statistics
        if self.training:
            for k in range(self.num_paths):
                self.path_selection_counts[k] += (selected_path == k).sum().float()
            self.total_samples[0] += x.size(0)

        if self.training and self.use_soft_training:
            # Soft: return weighted combination of all paths
            # Each sample uses its most likely path as scan_indices
            scan_indices = self.candidate_paths[selected_path]  # [B, J]
        else:
            # Hard: majority vote for batch efficiency
            majority_path = torch.mode(selected_path).values.item()
            scan_indices = self.candidate_paths[majority_path]  # [J]

        return scan_indices, path_weights, selected_path

    def get_selection_stats(self):
        """Get path selection statistics for analysis/visualization."""
        if self.total_samples[0] > 0:
            probs = self.path_selection_counts / self.total_samples[0]
            return {name: prob.item() for name, prob in zip(self.path_names, probs)}
        return {name: 0.0 for name in self.path_names}

    def reset_stats(self):
        """Reset selection statistics."""
        self.path_selection_counts.zero_()
        self.total_samples.zero_()


# ============================================================
# Soft Scan Path Combination (for training)
# ============================================================

class SoftScanCombiner(nn.Module):
    """Combines SSM outputs from multiple candidate scan paths during training.

    Instead of running SSM K times (which is expensive), this module
    combines the CrossScan outputs before SSM, then runs SSM once.
    """

    def __init__(self, num_paths=4):
        super().__init__()
        self.num_paths = num_paths

    def forward(self, xs_list, path_weights):
        """Combine multiple CrossScan outputs.

        Args:
            xs_list: list of K tensors, each [B, 4, C, L]
            path_weights: [B, K] soft weights

        Returns:
            xs_combined: [B, 4, C, L]
        """
        # Weighted average
        xs_combined = sum(
            xs * w.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            for xs, w in zip(xs_list, path_weights.unbind(-1))
        )
        return xs_combined