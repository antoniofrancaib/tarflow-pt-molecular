from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

# Only use simple bidirectional NLL - removed all legacy acceptance losses

from .simple_losses import simple_bidirectional_nll
from src.accelmd.utils.config import load_config, setup_device, get_temperature_pairs, create_run_config, get_energy_threshold

__all__ = ["PTSwapTrainer"]


class PTSwapTrainer:
    """General trainer for coordinate transformations (PT swaps or distribution mapping)."""

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        temp_pair: Tuple[int, int] = None,
        device: str = "cpu",
        mode: str = "pt",  # "pt" for PT swaps, "distribution" for simple distribution mapping
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.temp_pair = temp_pair
        self.device = torch.device(device)
        self.mode = mode  # "pt" or "distribution"

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["training"]["learning_rate"]
        )
        self.clip_grad_norm = config["training"].get("clip_grad_norm", 1.0)
        
        # Mode-specific initialization
        if self.mode == "pt":
            self.energy_threshold = get_energy_threshold(config)
        else:
            self.energy_threshold = None  # Not used for distribution mode

        # ------------------------------------------------------------------
        # Loss-component weights
        # ------------------------------------------------------------------
        self.recon_weight = config["training"].get("recon_weight", 1.0)
        self.recon_weight_end = config["training"].get("recon_weight_end", 0.0)

        # Loss component weights from config (simplified naming)
        t_cfg = config["training"]
        
        # NLL weight schedule
        self.nll_weight_start = t_cfg.get("nll_start", t_cfg.get("nll_weight_start", t_cfg.get("nll_weight", 1.0)))
        self.nll_weight_end = t_cfg.get("nll_end", t_cfg.get("nll_weight_end", self.nll_weight_start))
        
        # No acceptance loss - removed legacy code
        self.nll_warmup_epochs = t_cfg.get("nll_warmup_epochs", 0)

        # Number of epochs over which we interpolate weights
        self.warmup_epochs = t_cfg.get("warmup_epochs", 0)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config["training"].get("lr_factor", 0.5),
            patience=config["training"].get("lr_patience", 3),
            threshold=5e-3,  # 0.5 % relative improvement
            verbose=True,
            min_lr=config["training"].get("min_lr", 5e-5),
        )
        self.num_epochs = config["training"]["num_epochs"]
        self.early_patience = config["training"].get("early_stopping_patience", 10)

        # Optional: track and plot swap acceptance per epoch
        t_cfg = config["training"]
        self.track_epoch_acceptance = t_cfg.get("track_epoch_acceptance", False)
        self.acceptance_eval_max_batches = t_cfg.get("acceptance_eval_max_batches", 2)

        # Output directory for this pair
        self.output_dir = Path(config["output"]["pair_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)

        self.best_metric = float("inf")
        self.epochs_no_improve = 0

    # ------------------------------------------------------------------
    def train(self) -> Dict:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        start_time = time.time()
        train_hist = []
        val_hist = []
        clip_train_hist = []  # fraction of batches where loss was clipped
        clip_val_hist = []
        acceptance_hist = [] if self.track_epoch_acceptance else None
        for epoch in range(1, self.num_epochs + 1):
            self._current_epoch = epoch
            # Compute current NLL weight
            nll_weight = self._current_nll_weight(epoch)

            train_loss, train_nll_f, train_nll_r, train_clip_frac = self._run_epoch(
                self.train_loader, training=True, nll_weight=nll_weight
            )
            val_loss, val_nll_f, val_nll_r, val_clip_frac = self._run_epoch(
                self.val_loader, training=False, nll_weight=nll_weight
            )

            print(
                f"Epoch {epoch:03d} | loss {train_loss:.3e} (nll_f {train_nll_f:.3e} | nll_r {train_nll_r:.3e}) "
                f"| val_loss {val_loss:.3e} (nll_f {val_nll_f:.3e} | nll_r {val_nll_r:.3e}) "
                f"| weights (nll={nll_weight:.3f})"
            )

            train_hist.append(train_loss)
            val_hist.append(val_loss)
            clip_train_hist.append(train_clip_frac)
            clip_val_hist.append(val_clip_frac)

            # ----------------------------------------------------------
            # Early-stopping / LR-scheduler metric: always use the *total*
            # validation loss.  Previously we switched to the acceptance-loss
            # after warm-up, which caused best checkpoints to ignore better
            # overall fits.  Simpler and more transparent to monitor one
            # scalar throughout training.

            early_metric = val_loss

            # Save checkpoint whenever metric strictly improves
            if early_metric < self.best_metric:
                self.best_metric = early_metric
                self._save_checkpoint(epoch)
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            # LR scheduler monitors the same metric as early stopping
            self.lr_scheduler.step(early_metric)

            # ----------------------------------------------------------
            # NEW: log|det J| statistics (mean ± std) – computed **once**
            # per epoch on a single validation batch to avoid the costly
            # extra forward/inverse passes inside every mini-batch.
            # ----------------------------------------------------------
            with torch.no_grad():
                det_batch = next(iter(self.val_loader))
                det_batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in det_batch.items()}

                # Check if this is a flow that needs molecular data

                _, ld_f = self.model.forward(det_batch["source_coords"], return_log_det=True)
                _, ld_inv = self.model.forward(det_batch["target_coords"], reverse=True, return_log_det=True)

                ld_vals = torch.cat([ld_f.flatten(), ld_inv.flatten()])
                ld_mean = ld_vals.mean().item()
                ld_std = ld_vals.std().item()
                print(f"    [epoch {epoch:03d}] log|det J| mean={ld_mean:.3f} ± {ld_std:.3f}")

            # ----------------------------------------------------------
            # Optional: estimate swap acceptance for the current model
            # on a few validation batches and record per-epoch history.
            # ----------------------------------------------------------
            if self.track_epoch_acceptance:
                from src.accelmd.evaluation.swap_acceptance import flow_acceptance
                try:
                    flow_acc = flow_acceptance(
                        loader=self.val_loader,
                        model=self.model,
                        base_low=self.model.base_low,
                        base_high=self.model.base_high,
                        device=str(self.device),
                        max_batches=self.acceptance_eval_max_batches,
                    )
                except Exception as _e:
                    # Be robust: in case evaluation fails (e.g., missing CPU targets),
                    # record NaN and continue training without interruption.
                    flow_acc = float('nan')
                acceptance_hist.append(flow_acc)

        hours = (time.time() - start_time) / 3600.0

        # ----------------------------------------------------------
        # Save training metrics (plots disabled per user request)
        # Only save model checkpoints

        # Save training metrics (acceptance data only, no plots)
        if self.track_epoch_acceptance and acceptance_hist is not None:
            metrics_dir = self.output_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            import json
            with open(metrics_dir / "acceptance_curve.json", "w") as fh:
                json.dump({
                    "epochs": list(range(1, len(acceptance_hist) + 1)),
                    "flow_acceptance": acceptance_hist,
                }, fh, indent=2)

        return {
            "best_metric": self.best_metric,
            "epochs_trained": epoch,
            "total_time_hours": hours,
        }

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, training: bool, nll_weight: float = 1.0) -> tuple[float, float, float]:
        self.model.train(mode=training)
        total_loss = 0.0
        nll_f_sum = 0.0
        nll_r_sum = 0.0
        n_batches = 0
        clipped_batches = 0
        n_samples = 0
        
        for batch in loader:
            # Move tensors to device (only coords present for now)
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Only bidirectional NLL loss
            with torch.set_grad_enabled(training):
                # Use simple bidirectional NLL for all modes
                loss_dict = simple_bidirectional_nll(
                    self.model,
                    source_coords=batch["source_coords"],
                    target_coords=batch["target_coords"],
                    source_temp=batch["source_temp"],
                    target_temp=batch["target_temp"],
                )
                nll_total = loss_dict["loss"]
                nll_forward = loss_dict["nll_forward"]
                nll_reverse = loss_dict["nll_backward"]
            
            # Apply NLL weight (always 1.0 for now)
            total = nll_weight * nll_total

            if training:
                self.optimizer.zero_grad()
                total.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()

            B = batch["source_coords"].shape[0]
            n_samples += B

            batch_loss_val = total.item()

            # Accumulate *per-sample* sums so we can average correctly.
            total_loss += batch_loss_val * B
            nll_f_sum += nll_forward.item() * B
            nll_r_sum += nll_reverse.item() * B

            # Detect sentinel clipping constant (≈1e4) rather than any large loss
            sentinel = 1e4
            tol = 1e-3
            if abs(batch_loss_val - sentinel) < tol or abs(nll_total.item() - sentinel) < tol:
                clipped_batches += 1

            n_batches += 1
        
        mean_total = total_loss / max(1, n_samples)
        mean_nll_f = nll_f_sum / max(1, n_samples)
        mean_nll_r = nll_r_sum / max(1, n_samples)

        clip_fraction = clipped_batches / max(1, n_batches)

        return mean_total, mean_nll_f, mean_nll_r, clip_fraction

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int):
        path = self.output_dir / "models" / f"best_model_epoch{epoch}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint: {path}") 

    # ------------------------------------------------------------------
    # Helper – geometric warm-up of NLL weight
    def _current_nll_weight(self, epoch: int) -> float:
        if self.warmup_epochs == 0:
            return self.nll_weight_end  # fixed / legacy behaviour

        t = min(epoch, self.warmup_epochs)
        
        # Handle case where starting weight is 0 - use linear interpolation
        if self.nll_weight_start == 0.0:
            beta = self.nll_weight_end * (t / self.warmup_epochs)
        else:
            # Use geometric interpolation when both start and end are non-zero
            ratio = self.nll_weight_end / max(1e-12, self.nll_weight_start)
            beta = self.nll_weight_start * (ratio ** (t / self.warmup_epochs))
        return beta
    
    # Removed acceptance weight scheduling - no longer used 

    # ------------------------------------------------------------------
    @property
    def _current_epoch(self):
        """Private helper set by train() loop just before _run_epoch call."""
        return getattr(self, "__cur_epoch", 0)

    @_current_epoch.setter
    def _current_epoch(self, value):
        setattr(self, "__cur_epoch", value)

    def _interp(self, start: float, end: float, epoch: int):
        if self.warmup_epochs == 0:
            return end
        t = min(epoch, self.warmup_epochs)
        return start + (end - start) * (t / self.warmup_epochs) 