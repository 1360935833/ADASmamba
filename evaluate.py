"""
AdaSMamba Evaluation Script
---------------------------
Evaluates the AdaSMamba model on Human3.6M and MPI-INF-3DHP benchmarks.
Supports ablation studies by toggling individual innovations.

Usage:
    # Standard evaluation
    python evaluate.py --config configs/h36m/adasmamba_h36m_243.yaml --checkpoint checkpoint/36m/AdaSMamba_243/

    # Ablation: disable adaptive scan
    python evaluate.py --config configs/h36m/adasmamba_h36m_243.yaml --checkpoint ... --no-adaptive-scan

    # Ablation: disable structure gate
    python evaluate.py --config configs/h36m/adasmamba_h36m_243.yaml --checkpoint ... --no-structure-gate

    # Path selection analysis
    python evaluate.py --config configs/h36m/adasmamba_h36m_243.yaml --checkpoint ... --analyze-paths
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from loss.pose3d import mpjpe as calculate_mpjpe
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from data.reader.h36m import DataReaderH36M
from data.reader.motion_dataset import MotionDataset3D
from utils.tools import set_random_seed, get_config, print_args
from utils.data import flip_data


def parse_args():
    parser = argparse.ArgumentParser(description='AdaSMamba Evaluation')
    parser.add_argument("--config", type=str, default="configs/h36m/adasmamba_h36m_243.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint")
    parser.add_argument("--checkpoint-file", type=str, default="best_epoch.bin",
                        help="Checkpoint file name")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--num-cpus", default=16, type=int, help="Number of CPU cores")
    parser.add_argument("--batch-size", default=256, type=int, help="Batch size for evaluation")

    # Ablation flags
    parser.add_argument("--no-adaptive-scan", action="store_true",
                        help="Disable adaptive scan path (Innovation 1 ablation)")
    parser.add_argument("--no-structure-gate", action="store_true",
                        help="Disable structure gate (Innovation 2 ablation)")
    parser.add_argument("--no-adaptive-scale", action="store_true",
                        help="Disable adaptive scale fusion (Innovation 3 ablation)")

    # Analysis flags
    parser.add_argument("--analyze-paths", action="store_true",
                        help="Analyze path selection distribution per action")
    parser.add_argument("--analyze-gates", action="store_true",
                        help="Analyze structure gate values")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Save results to file")

    # Dataset
    parser.add_argument("--dataset", type=str, default="h36m",
                        choices=["h36m", "mpi"],
                        help="Dataset for evaluation")

    opts = parser.parse_args()
    return opts


def evaluate_h36m(args, model, test_loader, device):
    """Evaluate on Human3.6M test set."""
    model.eval()

    # Per-action metrics
    action_names = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
                    'Photo', 'Posing', 'Purchases', 'Sitting', 'SittingDown',
                    'Smoking', 'Waiting', 'WalkDog', 'Walking', 'WalkTogether']
    action_mpjpe = {name: [] for name in action_names}
    all_mpjpe = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs_2d = batch['inputs_2d'].to(device)
            targets_3d = batch['targets_3d'].to(device)
            action = batch['action']

            # Forward pass
            outputs_3d = model(inputs_2d)

            # Compute MPJPE
            mpjpe = calculate_mpjpe(outputs_3d, targets_3d)
            all_mpjpe.append(mpjpe.cpu())

            # Per-action
            for i, act in enumerate(action):
                if act in action_mpjpe:
                    action_mpjpe[act].append(mpjpe[i].cpu())

    # Aggregate results
    results = {
        'MPJPE (P1)': torch.cat(all_mpjpe).mean().item() if all_mpjpe else 0,
    }
    for name in action_names:
        if action_mpjpe[name]:
            results[f'  {name}'] = torch.cat(action_mpjpe[name]).mean().item()

    return results


def analyze_path_selection(model, test_loader, device):
    """Analyze which scan paths are selected for different action types."""
    model.eval()
    model.reset_path_stats()

    action_path_counts = {}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Analyzing paths"):
            inputs_2d = batch['inputs_2d'].to(device)
            action = batch['action']

            _ = model(inputs_2d)

            # Record per-action path selections
            stats = model.get_path_selection_stats()
            for act in action:
                if act not in action_path_counts:
                    action_path_counts[act] = []
                action_path_counts[act].append(stats)

    # Aggregate
    print("\n=== Path Selection Analysis ===")
    for act in sorted(action_path_counts.keys()):
        print(f"\n{act}:")
        # Average across all blocks and samples
        all_stats = action_path_counts[act]
        for block_name in all_stats[0].keys():
            path_probs = {}
            for s in all_stats:
                for path_name, prob in s[block_name].items():
                    path_probs.setdefault(path_name, []).append(prob)
            print(f"  {block_name}:")
            for path_name, probs in path_probs.items():
                print(f"    {path_name}: {np.mean(probs):.4f}")

    return action_path_counts


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config = get_config(args.config)
    print_args(config)

    # Override with ablation flags
    if args.no_adaptive_scan:
        config.use_adaptive_scan = False
    if args.no_structure_gate:
        config.use_structure_gate = False
    if args.no_adaptive_scale:
        config.use_adaptive_scale = False

    # Create model
    try:
        from model.AdaSMamba import create_adasmamba
    except ImportError:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from model.AdaSMamba import create_adasmamba

    model = create_adasmamba(config).to(device)
    print(f"Model created: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Print enabled innovations
    print("\nEnabled innovations:")
    print(f"  Innovation 1 (Adaptive Scan): {config.use_adaptive_scan}")
    print(f"  Innovation 2 (Structure Gate): {config.use_structure_gate}")
    print(f"  Innovation 3 (Adaptive Scale): {config.use_adaptive_scale}")

    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint, args.checkpoint_file)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Running with random weights (results will be meaningless)")

    # Load dataset
    if args.dataset == "h36m":
        test_reader = DataReaderH36M(
            n_frames=config.n_frames,
            sample_stride=1,
            is_train=False,
            data_root=config.data_root,
            data_root_2d=config.data_root_2d,
            dt_file=config.dt_file,
            use_proj_as_2d=config.use_proj_as_2d,
        )
        test_dataset = MotionDataset3D(test_reader, None, config.n_frames, is_train=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_cpus,
            pin_memory=True,
        )

        # Evaluate
        results = evaluate_h36m(args, model, test_loader, device)
        print("\n=== Results ===")
        for key, value in results.items():
            print(f"{key}: {value:.2f} mm")

        # Path analysis
        if args.analyze_paths:
            analyze_path_selection(model, test_loader, device)

        # Save results
        if args.save_results:
            with open(args.save_results, 'w') as f:
                for key, value in results.items():
                    f.write(f"{key}: {value:.2f}\n")
            print(f"Results saved to {args.save_results}")

    return results


if __name__ == "__main__":
    main()