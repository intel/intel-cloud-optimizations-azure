import os
import math
import time
import torch
import fsspec
import numpy as np

from typing import Any, Dict
from collections import OrderedDict
from accelerate import Accelerator
from accelerate.logging import get_logger
from dataclasses import dataclass, asdict
import intel_extension_for_pytorch as ipex

logger = get_logger(__name__, log_level="INFO")

def log_info(msg, main_only=False, in_order=False):
    logger.info(msg, main_process_only=main_only, in_order=in_order)


@dataclass
class Snapshot:
    model_state: "OrderedDict[str, torch.Tensor]"
    optimizer_state: Dict[str, Any]
    iter_num: int
    best_val_loss: float

class Trainer:
    def __init__(self, trainer_config, data_dir, model, optimizer, block_size):
        self.trainer_config = trainer_config
        self.model = model
        self.optimizer = optimizer
        self.block_size = block_size
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.trainer_config.gradient_accumulation_steps,
            cpu=self.trainer_config.device == "cpu",
        )

        self.master_process = self.accelerator.is_main_process

        # data stuff
        self.train_data = np.memmap(
            os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
        )
        self.val_data = np.memmap(
            os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
        )

        log_info(
            f"[RANK {self.accelerator.process_index}] Total training samples (tokens/block_size) : {int(self.train_data.shape[0]/self.block_size)}",
        )
        log_info(
            f"[RANK {self.accelerator.process_index}] Total validation samples (tokens/block_size) : {int(self.val_data.shape[0]/self.block_size)}",
        )
        log_info(
            f"[RANK {self.accelerator.process_index}] One epoch (total_training_samples/batch_size): {int(self.train_data.shape[0]/self.block_size/self.trainer_config.batch_size)} iterations",
        )

        # initialize train states
        self.iter_num = 0
        self.best_val_loss = 1e9

        # load snapshot if available. only necessary on the first node.
        if self.trainer_config.snapshot_path is None:
            self.trainer_config.snapshot_path = "snapshot.pt"
        self._load_snapshot()

        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[
            self.trainer_config.mixed_precision
        ]

        self.autocast_ctx_manager = torch.cpu.amp.autocast(
            cache_enabled=True, dtype=dtype
        )

        self.model.train()
        self.model, self.optimizer = ipex.optimize(
            self.model,
            optimizer=self.optimizer,
            dtype=dtype,
            inplace=True,
            level="O1",
        )

        # don't use `accelerator.prepare` when model is wrapped in `ipex.optimize`
        # self.model, self.optimizer = self.accelerator.prepare(
        #     self.model, self.optimizer
        # )

    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.trainer_config.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")
        except FileNotFoundError:
            log_info(
                f"[RANK {self.accelerator.process_index}] Snapshot not found. Training model from pretrained gpt2 weights",
            )
            return

        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.iter_num = snapshot.iter_num
        self.best_val_loss = snapshot.best_val_loss
        log_info(
            f"[RANK {self.accelerator.process_index}] Resuming training from snapshot: Completed {self.iter_num} iterations | Best Val loss {self.best_val_loss}",
        )

    def _save_snapshot(self, iter_num):
        # capture snapshot
        model = self.accelerator.unwrap_model(self.model)
        snapshot = Snapshot(
            model_state=model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            iter_num=iter_num,
            best_val_loss=self.best_val_loss,
        )
        # save snapshot
        snapshot = asdict(snapshot)
        torch.save(snapshot, self.trainer_config.snapshot_path)
        torch.save(snapshot["model_state"], self.trainer_config.model_path)
        log_info(f"[RANK {self.accelerator.process_index}] Snapshot saved at {iter_num} iteration", main_only=True)

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(
            len(data) - self.block_size, (self.trainer_config.batch_size,)
        )
        x = torch.stack(
            [
                torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.trainer_config.eval_iters)
            for k in range(self.trainer_config.eval_iters):
                X, Y = self.get_batch(split)
                with self.autocast_ctx_manager:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.trainer_config.warmup_iters:
            return self.trainer_config.max_lr * it / self.trainer_config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.trainer_config.lr_decay_iters:
            return self.trainer_config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.trainer_config.warmup_iters) / (
            self.trainer_config.lr_decay_iters - self.trainer_config.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.trainer_config.min_lr + coeff * (
            self.trainer_config.max_lr - self.trainer_config.min_lr
        )

    def train(self):
        X, Y = self.get_batch("train")  # fetch the very first batch
        t0 = time.time()
        iter_num = self.iter_num

        while True:
            # determine and set the learning rate for this iteration
            lr = (
                self.get_lr(iter_num)
                if self.trainer_config.decay_lr
                else self.trainer_config.max_lr
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if (
                iter_num % self.trainer_config.eval_interval == 0
                and self.master_process
            ):
                losses = self.estimate_loss()

                if losses["val"] < self.best_val_loss:
                    if iter_num > 0:
                        self.best_val_loss = losses["val"]
                        self._save_snapshot(iter_num)

            if self.trainer_config.eval_only:
                break

            # gradient accumulation
            X, Y = self.get_batch("train")
            with self.accelerator.accumulate(self.model):
                with self.autocast_ctx_manager:
                    _, loss = self.model(X, Y)
                self.accelerator.backward(loss)
                loss = loss.detach()

                # gradient clipping
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.trainer_config.grad_clip
                )

                # optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.trainer_config.log_interval == 0:
                log_info(
                    f"[RANK {self.accelerator.process_index}] iter {iter_num}: train loss {loss:.4f}, time {dt:.2f}s",
                )

            # termination conditions
            if iter_num > self.trainer_config.max_iters:
                log_info(
                    f"[RANK {self.accelerator.process_index}] Total Samples used for training: {(iter_num-1)*self.trainer_config.batch_size}",
                )
                break

            iter_num += 1