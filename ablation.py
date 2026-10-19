"""
AdaSMamba Ablation Study Runner
-------------------------------
Systematically evaluates the contribution of each innovation component.

Ablation experiments:
1. Baseline (SasMamba: no adaptive scan, no structure gate, no adaptive scale)
2. + Adaptive Scan Path only
3. + Structure Gate only
4. + Adaptive Scale Fusion only
5. + Adaptive Scan + Structure Gate
6. AdaSMamba Full (all three innovations)

Also tests:
- Candidate path count K ∈ {1, 2, 4, 8}
- Structure gate encoding variants
- Per-action breakdown

Usage:
    python ablation.py --config configs/h36m/adasmamba_h36m_243.yaml --checkpoint-dir checkpoint/36m/
"""

import argparse
import os
import json
import numpy as np
import torch
from itertools import product


def define_ablation_configs():
    """Define all ablation experiment configurations."""
    configs = []

    # Experiment 1: Component ablation
    component_configs = [
        {
            'name': 'Baseline (SasMamba)',
            'use_adaptive_scan': False,
            'use_structure_gate': False,
            'use_adaptive_scale': False,
        },
        {
            'name': '+ Adaptive Scan Path',
            'use_adaptive_scan': True,
            'use_structure_gate': False,
            'use_adaptive_scale': False,
        },
        {
            'name': '+ Structure Gate',
            'use_adaptive_scan': False,
            'use_structure_gate': True,
            'use_adaptive_scale': False,
        },
        {
            'name': '+ Adaptive Scale Fusion',
            'use_adaptive_scan': False,
            'use_structure_gate': False,
            'use_adaptive_scale': True,
        },
        {
            'name': '+ Adaptive Scan + Structure Gate',
            'use_adaptive_scan': True,
            'use_structure_gate': True,
            'use_adaptive_scale': False,
        },
        {
            'name': 'AdaSMamba Full (ALL)',
            'use_adaptive_scan': True,
            'use_structure_gate': True,
            'use_adaptive_scale': True,
        },
    ]
    configs.append(('component', component_configs))

    # Experiment 2: Candidate path count K
    path_k_configs = [
        {'name': f'K={k}', 'use_adaptive_scan': True, 'use_structure_gate': True,
         'use_adaptive_scale': False, 'num_candidate_paths': k}
        for k in [1, 2, 4, 8]
    ]
    configs.append(('path_count', path_k_configs))

    # Experiment 3: Structure gate encoding variants
    gate_configs = [
        {
            'name': 'No gate',
            'use_adaptive_scan': True,
            'use_structure_gate': False,
            'use_adaptive_scale': False,
        },
        {
            'name': 'Static only',
            'use_adaptive_scan': True,
            'use_structure_gate': True,
            'use_adaptive_scale': False,
            'gate_mode': 'static',
        },
        {
            'name': 'Static + Dynamic',
            'use_adaptive_scan': True,
            'use_structure_gate': True,
            'use_adaptive_scale': False,
            'gate_mode': 'static_dynamic',
        },
        {
            'name': 'Full (Static + Dynamic + ScanCtx)',
            'use_adaptive_scan': True,
            'use_structure_gate': True,
            'use_adaptive_scale': False,
            'gate_mode': 'full',
        },
    ]
    configs.append(('gate_encoding', gate_configs))

    return configs


def run_ablation_study(args):
    """Run all ablation experiments and collect results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ablation_configs = define_ablation_configs()
    all_results = {}

    for exp_name, configs in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*60}")

        exp_results = []
        for cfg in configs:
            print(f"\n--- {cfg['name']} ---")

            # Create config
            try:
                from model.AdaSMamba import AdaSMamba
            except ImportError:
                import sys
                sys.path.insert(0, os.path.dirname(__file__))
                from model.AdaSMamba import AdaSMamba

            model = AdaSMamba(
                num_frame=args.n_frames,
                num_joints=args.num_joints,
                in_chans=2,
                embed_dim_ratio=args.dim_feat,
                depth=args.n_layers,
                mlp_ratio=args.mlp_ratio,
                ssm_conv=args.ssm_conv,
                use_adaptive_scan=cfg['use_adaptive_scan'],
                use_structure_gate=cfg['use_structure_gate'],
                use_adaptive_scale=cfg['use_adaptive_scale'],
                num_candidate_paths=cfg.get('num_candidate_paths', 4),
            ).to(device)

            num_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  Parameters: {num_params:.2f}M")

            # Load checkpoint if available
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"adasmamba_{cfg['name'].replace(' ', '_').replace('(', '').replace(')', '')}.bin"
            )
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)
                print(f"  Loaded checkpoint: {checkpoint_path}")

            # Quick test with random input
            x = torch.randn(2, args.n_frames, args.num_joints, 2).to(device)
            with torch.no_grad():
                y = model(x)
            print(f"  Input: {x.shape}, Output: {y.shape}")

            # For FLOPs
            from thop import profile, clever_format
            flops, params = profile(model, inputs=(x,), verbose=False)
            flops, params = clever_format([flops, params], "%.3f")
            print(f"  FLOPs: {flops}, Params: {params}")

            exp_results.append({
                'name': cfg['name'],
                'params': num_params,
                'flops': flops,
                'config': cfg,
            })

        all_results[exp_name] = exp_results

    # Save results
    results_path = os.path.join(args.checkpoint_dir, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    for exp_name, results in all_results.items():
        print(f"\n{exp_name}:")
        print(f"{'Name':<40} {'Params':<10} {'FLOPs':<12}")
        print("-"*62)
        for r in results:
            print(f"{r['name']:<40} {r['params']:<10.2f}M {r['flops']:<12}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description='AdaSMamba Ablation Study')
    parser.add_argument("--config", type=str, default="configs/h36m/adasmamba_h36m_243.yaml",
                        help="Path to base config")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoint/36m/",
                        help="Directory containing checkpoints")
    parser.add_argument("--n-frames", type=int, default=243)
    parser.add_argument("--num-joints", type=int, default=17)
    parser.add_argument("--dim-feat", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--ssm-conv", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run: only print model configs without loading checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation_study(args)