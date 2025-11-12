#!/usr/bin/env python3
"""
Generate custom config file for finetuning SlowFast model

This script helps create a configuration file tailored for your finetuning task.
It asks for key parameters and generates a YAML config file.
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_config_template(
    output_path: str,
    base_config: str = "resnet50_4x16f_baseline",
    num_classes: int = 80,
    backbone: str = "resnet50",
    use_ia_structure: bool = False,
    ia_structure: str = "denseserial",
    dataset_name: str = "custom",
    frame_num: int = 64,
    frame_sample_rate: int = 1,
    tau: int = 16,
    alpha: int = 8,
    num_gpus: int = 8,
    batch_size: int = 16,
    base_lr: float = 0.0004,
    max_iter: int = 90000,
    output_dir: str = None
):
    """
    Generate a finetuning config file
    
    Args:
        output_path: Path to save the config file
        base_config: Base configuration template
        num_classes: Number of action classes
        backbone: Backbone architecture (resnet50 or resnet101)
        use_ia_structure: Whether to use Interaction Aggregation
        ia_structure: IA structure type (parallel, serial, denseserial)
        dataset_name: Name of your dataset
        frame_num: Number of frames to load
        frame_sample_rate: Frame sampling rate
        tau: Temporal sampling stride
        alpha: Fast/Slow pathway ratio
        num_gpus: Number of GPUs to use
        batch_size: Videos per batch
        base_lr: Base learning rate
        max_iter: Maximum training iterations
        output_dir: Output directory for checkpoints and logs
    """
    
    # Determine backbone settings
    if backbone == "resnet50":
        backbone_name = "Slowfast-Resnet50"
        pretrained_model = "SlowFast-ResNet50-4x16.pth"
    elif backbone == "resnet101":
        backbone_name = "Slowfast-Resnet101"
        pretrained_model = "SlowFast-ResNet101-8x8.pth"
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    # Determine IA structure settings
    ia_active = "True" if use_ia_structure else "False"
    
    if output_dir is None:
        output_dir = f"data/output/{dataset_name}_{backbone}_{frame_num}f"
        if use_ia_structure:
            output_dir += f"_{ia_structure}"
    
    # Calculate steps (typically at 55% and 78% of max_iter)
    step1 = int(max_iter * 0.55)
    step2 = int(max_iter * 0.78)
    
    # Calculate checkpoint and eval periods (every 10% of training)
    checkpoint_period = max(1000, int(max_iter * 0.1))
    eval_period = checkpoint_period
    
    # Warmup iterations (typically 2-5% of max_iter)
    warmup_iters = int(max_iter * 0.02)
    
    # Generate config content
    config_content = f"""# Custom Finetuning Configuration
# Generated for: {dataset_name}
# Backbone: {backbone}
# Number of Classes: {num_classes}
# IA Structure: {'Yes - ' + ia_structure if use_ia_structure else 'No'}

MODEL:
  WEIGHT: "data/models/pretrained_models/{pretrained_model}"
  BACKBONE:
    CONV_BODY: "{backbone_name}"
    FROZEN_BN: True  # Keep BatchNorm frozen for stability
    SLOWFAST:
      BETA: 0.125
      LATERAL: "tconv"
      SLOW:
        ACTIVE: True
        CONV3_NONLOCAL: False
        CONV4_NONLOCAL: False
      FAST:
        ACTIVE: True
        CONV3_NONLOCAL: False
        CONV4_NONLOCAL: False
  NONLOCAL:
    USE_ZERO_INIT_CONV: False
    BN_INIT_GAMMA: 0.0
    FROZEN_BN: True
  ROI_ACTION_HEAD:
    FEATURE_EXTRACTOR: "2MLPFeatureExtractor"
    POOLER_TYPE: "align3d"
    MEAN_BEFORE_POOLER: True
    POOLER_RESOLUTION: 7
    POOLER_SCALE: 0.0625
    POOLER_SAMPLING_RATIO: 0
    NUM_CLASSES: {num_classes}  # IMPORTANT: Set this to your number of action classes
    PROPOSAL_PER_CLIP: 10
    DROPOUT_RATE: 0.2
  IA_STRUCTURE:
    ACTIVE: {ia_active}
"""
    
    # Add IA structure configuration if enabled
    if use_ia_structure:
        if ia_structure == "parallel":
            config_content += """    STRUCTURE: "parallel"
    MAX_PER_SEC: 5
    MAX_PERSON: 25
    LENGTH_RATIO: 0.5
"""
        elif ia_structure == "serial":
            config_content += """    STRUCTURE: "serial"
    MAX_PER_SEC: 5
    MAX_PERSON: 25
    LENGTH_RATIO: 0.5
"""
        elif ia_structure == "denseserial":
            config_content += """    STRUCTURE: "denseserial"
    MAX_PER_SEC: 5
    MAX_PERSON: 25
    FUSION: "concat"
    LAYER_NUM: 2
"""
    
    config_content += f"""
INPUT:
  FRAME_NUM: {frame_num}  # Total frames to load
  FRAME_SAMPLE_RATE: {frame_sample_rate}  # Sample every N frames
  TAU: {tau}  # Temporal sampling stride (for Slow pathway)
  ALPHA: {alpha}  # Fast/Slow pathway ratio (Fast samples at TAU/ALPHA)
  SLOW_JITTER: True  # Temporal augmentation
  COLOR_JITTER: True  # Color augmentation

DATASETS:
  # IMPORTANT: Update these to match your dataset names registered in paths_catalog.py
  TRAIN: ("{dataset_name}_train",)
  TEST: ("{dataset_name}_val",)

DATALOADER:
  NUM_WORKERS: 4  # Increase for faster data loading (if CPU allows)
  SIZE_DIVISIBILITY: 16

SOLVER:
  BASE_LR: {base_lr}  # Base learning rate (scaled for {num_gpus} GPUs)
  WARMUP_FACTOR: 0.25
  WARMUP_ITERS: {warmup_iters}
  BIAS_LR_FACTOR: 2
  WEIGHT_DECAY: 1e-7
  STEPS: ({step1}, {step2})  # Learning rate decay at these iterations
  MAX_ITER: {max_iter}  # Total training iterations
  CHECKPOINT_PERIOD: {checkpoint_period}  # Save checkpoint every N iterations
  EVAL_PERIOD: {eval_period}  # Evaluate every N iterations
  VIDEOS_PER_BATCH: {batch_size}  # Batch size

TEST:
  BOX_THRESH: 0.8  # Person detection threshold
  ACTION_THRESH: 0.0  # Action prediction threshold (0 = no filtering)
  VIDEOS_PER_BATCH: {batch_size}

OUTPUT_DIR: "{output_dir}"
"""
    
    # Write config file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    logger.info(f"Config file generated: {output_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Print next steps
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"\n1. Review and edit the config file: {output_path}")
    print(f"\n2. Make sure your dataset is registered in alphaction/config/paths_catalog.py:")
    print(f"   Dataset names: {dataset_name}_train, {dataset_name}_val")
    print(f"\n3. Download pretrained model: {pretrained_model}")
    print(f"   Save to: data/models/pretrained_models/{pretrained_model}")
    print(f"\n4. Start training:")
    if num_classes != 80:
        print(f"   python train_net.py --config-file {output_path} --transfer --no-head --use-tfboard")
    else:
        print(f"   python train_net.py --config-file {output_path} --transfer --use-tfboard")
    print(f"\n5. Monitor training:")
    print(f"   tensorboard --logdir={output_dir}")
    print("\n" + "="*60 + "\n")


def interactive_mode():
    """Interactive mode to gather parameters from user"""
    
    print("\n" + "="*60)
    print("SlowFast Model Finetuning Config Generator")
    print("="*60 + "\n")
    
    # Output path
    output_path = input("Config file name (e.g., my_config.yaml): ").strip()
    if not output_path.endswith('.yaml'):
        output_path += '.yaml'
    if not output_path.startswith('config_files/'):
        output_path = f'config_files/{output_path}'
    
    # Dataset name
    dataset_name = input("\nDataset name (e.g., custom_dataset): ").strip()
    if not dataset_name:
        dataset_name = "custom"
    
    # Number of classes
    num_classes = int(input("\nNumber of action classes (default: 80): ").strip() or "80")
    
    # Backbone
    print("\nBackbone options:")
    print("  1. ResNet50 (faster, lighter)")
    print("  2. ResNet101 (slower, more accurate)")
    backbone_choice = input("Choose backbone (1 or 2, default: 1): ").strip() or "1"
    backbone = "resnet50" if backbone_choice == "1" else "resnet101"
    
    # IA structure
    use_ia = input("\nUse Interaction Aggregation? (y/n, default: n): ").strip().lower()
    use_ia_structure = use_ia == 'y'
    
    ia_structure = "denseserial"
    if use_ia_structure:
        print("\nIA Structure options:")
        print("  1. Parallel")
        print("  2. Serial")
        print("  3. Dense Serial (recommended)")
        ia_choice = input("Choose IA structure (1/2/3, default: 3): ").strip() or "3"
        ia_structure = ["parallel", "serial", "denseserial"][int(ia_choice)-1]
    
    # Training parameters
    print("\n--- Training Parameters ---")
    num_gpus = int(input("Number of GPUs (default: 8): ").strip() or "8")
    
    # Calculate default batch size and LR based on GPU count
    default_batch = max(2, 16 * num_gpus // 8)
    default_lr = 0.0004 * num_gpus / 8
    
    batch_size = int(input(f"Batch size (videos per batch, default: {default_batch}): ").strip() or str(default_batch))
    base_lr = float(input(f"Base learning rate (default: {default_lr:.6f}): ").strip() or str(default_lr))
    max_iter = int(input("Max iterations (default: 90000): ").strip() or "90000")
    
    # Video parameters
    print("\n--- Video Parameters ---")
    frame_num = int(input("Number of frames to load (default: 64): ").strip() or "64")
    tau = int(input("Temporal stride TAU (default: 16): ").strip() or "16")
    alpha = int(input("Fast/Slow ratio ALPHA (default: 8): ").strip() or "8")
    
    # Generate config
    create_config_template(
        output_path=output_path,
        num_classes=num_classes,
        backbone=backbone,
        use_ia_structure=use_ia_structure,
        ia_structure=ia_structure,
        dataset_name=dataset_name,
        frame_num=frame_num,
        tau=tau,
        alpha=alpha,
        num_gpus=num_gpus,
        batch_size=batch_size,
        base_lr=base_lr,
        max_iter=max_iter
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate custom config file for finetuning SlowFast model"
    )
    
    # Add arguments
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output config file path"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="custom",
        help="Name of your dataset (default: custom)"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=80,
        help="Number of action classes (default: 80)"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet50", "resnet101"],
        default="resnet50",
        help="Backbone architecture (default: resnet50)"
    )
    parser.add_argument(
        "--use-ia",
        action="store_true",
        help="Use Interaction Aggregation structure"
    )
    parser.add_argument(
        "--ia-structure",
        type=str,
        choices=["parallel", "serial", "denseserial"],
        default="denseserial",
        help="IA structure type (default: denseserial)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs (default: 8)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Videos per batch (default: auto-calculated)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Base learning rate (default: auto-calculated)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=90000,
        help="Maximum training iterations (default: 90000)"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode (ask for all parameters)"
    )
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive or args.output is None:
        interactive_mode()
        return
    
    # Auto-calculate batch size and LR if not provided
    if args.batch_size is None:
        args.batch_size = max(2, 16 * args.num_gpus // 8)
    
    if args.lr is None:
        args.lr = 0.0004 * args.num_gpus / 8
    
    # Generate config
    create_config_template(
        output_path=args.output,
        num_classes=args.num_classes,
        backbone=args.backbone,
        use_ia_structure=args.use_ia,
        ia_structure=args.ia_structure,
        dataset_name=args.dataset_name,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        base_lr=args.lr,
        max_iter=args.max_iter
    )


if __name__ == "__main__":
    main()

