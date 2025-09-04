#!/usr/bin/env python3
"""
Self-contained script to show data batch order bug when resuming from partial checkpoints from 
the second epoch.

Runs the test twice: once with target_ckpt_num=2 (within epoch 0) and once with 
target_ckpt_num=7 (within epoch 1).

Usage:
    python reproduce_wrong_resumed_epoch.py --trainer-class Trainer
    python reproduce_wrong_resumed_epoch.py --trainer-class TrainerFixed
"""

# Suppress TensorFlow and CUDA logs
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide CUDA devices to avoid CUDA logs

# Suppress codecarbon logs
os.environ["CODECARBON_LOG_LEVEL"] = "ERROR"

import argparse
import random
import shutil
import time
import warnings
import logging as python_logging
from pathlib import Path
from typing import Any, Dict, Tuple

warnings.filterwarnings("ignore")
python_logging.getLogger().setLevel(python_logging.ERROR)
python_logging.getLogger("codecarbon").setLevel(python_logging.ERROR)
python_logging.getLogger("transformers").setLevel(python_logging.ERROR)
python_logging.getLogger("torch").setLevel(python_logging.ERROR)

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import TrainingArguments
from transformers.utils import logging
import transformers

# Import fixed trainer from local module
from src.transformers.trainer_fixed import TrainerFixed


def seed_everything(seed: int) -> None:
    """Set the random seeds for random number generators in Pytorch, numpy and native Python."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    transformers.set_seed(seed)


class DummyDataset(Dataset):
    def __init__(self, size: int = 32) -> None:
        self.size = size
        self.data = torch.randn((size, 10))
        self.data[:, 0] = torch.arange(0, size)  # encode the data order for debugging
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"input_ids": self.data[idx], "labels": self.labels[idx]}


class DummyModel(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 10, bias=False)
        # data_order logs the order of data points seen by the model
        self.data_order: torch.Tensor
        self.register_buffer("data_order", torch.empty(0, dtype=torch.long))

    def load_state_dict(self, state_dict, strict=True):
        # Handle data_order buffer size mismatch during checkpoint loading
        if "data_order" in state_dict:
            saved_data_order = state_dict["data_order"]
            if hasattr(self, "data_order") and self.data_order.shape != saved_data_order.shape:
                # Resize the buffer to match the saved state
                self.data_order = saved_data_order.clone()

        return super().load_state_dict(state_dict, strict=strict)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor | None]:
        logits = self.fc(input_ids)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        time.sleep(0.01)  # Simulate some computation time
        data_indices = input_ids[:, 0].int()
        self.data_order = torch.cat([self.data_order, data_indices.detach().clone()])

        return {"loss": loss, "logits": logits}


def create_dummy_dataset_and_model(size: int = 32) -> Tuple[Dataset, nn.Module]:
    """Creates a dummy dataset and a corresponding neural network model."""
    return DummyDataset(size=size), DummyModel(size=size)


def load_data_order_from_safetensors(
    path: Path,
) -> torch.Tensor:
    """Load the 'data_order' tensor from a Hugging Face model saved in safetensors format."""
    try:
        state_dict = load_file(path / "model.safetensors")
        if "data_order" not in state_dict:
            raise KeyError("'data_order' tensor not found in the safetensors file.")

        return state_dict["data_order"]

    except Exception as e:
        raise RuntimeError(f"Error loading safetensors file: {e}") from e


def run_synthetic_experiment(
    seed: int,
    data_size: int,
    batch_size: int,
    grad_acc: int,
    save_steps: int,
    num_epochs: int,
    exp_dir: Path,
    trainer_class: type = TrainerFixed,
) -> None:
    """Run a synthetic training experiment."""
    # Set random seeds for reproducibility
    seed_everything(seed)

    exp_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, model = create_dummy_dataset_and_model(size=data_size)

    args = TrainingArguments(
        seed=seed,
        output_dir=str(exp_dir),
        learning_rate=0.1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        save_strategy="steps",
        save_steps=save_steps,
        num_train_epochs=num_epochs,
        optim="sgd",
        disable_tqdm=True,
        dataloader_num_workers=0,  # ensures that main process loads the data
        report_to=[],  # Disable all reporting including codecarbon
    )

    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )

    last_checkpoint = None
    print(f"Getting last checkpoint in {exp_dir}")

    last_checkpoint = get_last_checkpoint(str(exp_dir))
    print(f"Last checkpoint found in {exp_dir}: {last_checkpoint}")

    # Resume training from last available checkpoint
    trainer.train(resume_from_checkpoint=last_checkpoint)


def test_checkpoint_deletion_resume(
    target_ckpt_num: int, trainer_class: type = TrainerFixed
) -> None:
    """Test that the dataloader order is reproducible when resuming from partial checkpoints."""

    ## Experiment setup
    num_epochs = 3
    data_size = 10
    batch_size = 2
    grad_acc = 1
    save_steps = 1
    last_ckpt_num = int(data_size * num_epochs / (batch_size * grad_acc))

    print(f"\n{'='*80}")
    print(
        f"Testing resuming from middle of epoch {round((target_ckpt_num-1) * batch_size / data_size)} (target_ckpt_num = {target_ckpt_num}) using {trainer_class.__name__}"
    )
    print(f"{'='*80}")

    n_trials = 1
    seeds = list(range(1))

    for seed in seeds:
        for trial in range(n_trials):
            print(f"\nRunning program with seed: [{seed}], trial: [{trial+1}/{n_trials}]")
            exp_dir_baseline = Path(
                f"./tmp/exp_baseline_target{target_ckpt_num}_seed-{seed}_trial-{trial}"
            )
            exp_dir_checkpoint_deletion = Path(
                f"./tmp/exp_checkpoint_deletion_target{target_ckpt_num}_seed-{seed}_trial-{trial}"
            )

            # Cleanup directories at the beginning of the test
            print("\n[Scenario 1] Running baseline training to completion...")
            if exp_dir_baseline.exists():
                shutil.rmtree(exp_dir_baseline)

            run_synthetic_experiment(
                seed=seed,
                data_size=data_size,
                batch_size=batch_size,
                grad_acc=grad_acc,
                save_steps=save_steps,
                num_epochs=num_epochs,
                exp_dir=exp_dir_baseline,
                trainer_class=trainer_class,
            )

            baseline_data_order = load_data_order_from_safetensors(
                exp_dir_baseline / f"checkpoint-{last_ckpt_num}"
            )

            print("\n[Scenario 2] Running training with checkpoint deletion and resume...")
            if exp_dir_checkpoint_deletion.exists():
                shutil.rmtree(exp_dir_checkpoint_deletion)

            # First, run training to completion
            run_synthetic_experiment(
                seed=seed,
                data_size=data_size,
                batch_size=batch_size,
                grad_acc=grad_acc,
                save_steps=save_steps,
                num_epochs=num_epochs,
                exp_dir=exp_dir_checkpoint_deletion,
                trainer_class=trainer_class,
            )

            # Delete checkpoints from target_ckpt_num onwards
            print(f"Deleting checkpoints from checkpoint-{target_ckpt_num} onwards...")
            for ckpt_num in range(target_ckpt_num, last_ckpt_num + 1):
                ckpt_path = exp_dir_checkpoint_deletion / f"checkpoint-{ckpt_num}"
                if ckpt_path.exists():
                    # print(f"Deleting {ckpt_path}")
                    shutil.rmtree(ckpt_path)

            # Resume training from the remaining checkpoints
            print("Resuming training from remaining checkpoints...")
            run_synthetic_experiment(
                seed=seed,
                data_size=data_size,
                batch_size=batch_size,
                grad_acc=grad_acc,
                save_steps=save_steps,
                num_epochs=num_epochs,
                exp_dir=exp_dir_checkpoint_deletion,
                trainer_class=trainer_class,
            )

            # Load the final results after resume
            resumed_data_order = load_data_order_from_safetensors(
                exp_dir_checkpoint_deletion / f"checkpoint-{last_ckpt_num}"
            )

            # Compare results - using prints instead of assertions
            print("\n--- Comparing data orders ---")
            if torch.equal(baseline_data_order, resumed_data_order):
                print("PASSED: data order is identical after checkpoint deletion and resume.")
            else:
                print("FAILED: differences in data order after checkpoint deletion and resume.")
            print("Baseline data order:")
            print(baseline_data_order)
            print("Resumed data order:")
            print(resumed_data_order)


def main():
    """Main function to run the test with different target checkpoint numbers."""
    parser = argparse.ArgumentParser(
        description="Test dataloader order reproducibility when resuming from partial checkpoints"
    )
    parser.add_argument(
        "--trainer-class",
        choices=["Trainer", "TrainerFixed"],
        default="Trainer",
        help="Trainer class to use for the experiment (default: Trainer)",
    )

    args = parser.parse_args()

    # Get the trainer class based on the argument
    if args.trainer_class == "Trainer":
        trainer_class = Trainer
    else:
        trainer_class = TrainerFixed

    print(f"Starting checkpoint deletion resume tests using {trainer_class.__name__}...")

    # Test with target_ckpt_num = 2 -> in the middle of epoch 0
    test_checkpoint_deletion_resume(target_ckpt_num=2, trainer_class=trainer_class)

    # Test with target_ckpt_num = 7 -> in the middle of epoch 1
    test_checkpoint_deletion_resume(target_ckpt_num=7, trainer_class=trainer_class)

    # Clean up tmp directory
    tmp_dir = Path("./tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
