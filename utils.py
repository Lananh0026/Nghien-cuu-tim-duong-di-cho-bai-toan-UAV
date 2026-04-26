from __future__ import annotations

import os
import random
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch


# =====================================================
# RANDOM SEED
# =====================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================
# CREATE FOLDER
# =====================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =====================================================
# MOVING AVERAGE
# =====================================================
def moving_average(values: Iterable[float], window: int = 20) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)

    if len(arr) == 0:
        return arr

    if len(arr) < window:
        return arr.copy()

    kernel = np.ones(window) / window
    out = np.convolve(arr, kernel, mode="valid")

    prefix = np.full(window - 1, out[0])

    return np.concatenate([prefix, out])


# =====================================================
# SINGLE ALGORITHM TRAINING CURVES
# =====================================================
def plot_training_curves(
    scores: List[float],
    success_rates: List[float],
    collision_rates: List[float],
    out_dir: str
) -> None:

    ensure_dir(out_dir)

    # ---------------- FIG 6 ----------------
    plt.figure(figsize=(8, 4.5))

    plt.plot(scores, alpha=0.35, label="Raw")
    plt.plot(
        moving_average(scores, 30),
        linewidth=2,
        label="Smoothed"
    )

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Đường cong huấn luyện của thuật toán")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "training_score.png"),
        dpi=220
    )
    plt.close()

    # ---------------- FIG 7 ----------------
    plt.figure(figsize=(8, 4.5))

    x = np.arange(1, len(success_rates) + 1) * 10

    plt.plot(
        x,
        success_rates,
        linewidth=2,
        label="Success Rate"
    )

    plt.xlabel("Trained Episodes")
    plt.ylabel("SR")
    plt.title("Success rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "success_rate.png"),
        dpi=220
    )
    plt.close()

    # ---------------- FIG 8 ----------------
    plt.figure(figsize=(8, 4.5))

    x = np.arange(1, len(collision_rates) + 1) * 10

    plt.plot(
        x,
        collision_rates,
        linewidth=2,
        label="Collision Rate"
    )

    plt.xlabel("Trained Episodes")
    plt.ylabel("CR")
    plt.title("Collision rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "collision_rate.png"),
        dpi=220
    )
    plt.close()


# =====================================================
# COMPARE 3 ALGORITHMS
# =====================================================
def plot_comparison(
    sac_sr: List[float],
    td3_sr: List[float],
    ddpg_sr: List[float],
    sac_cr: List[float],
    td3_cr: List[float],
    ddpg_cr: List[float],
    out_dir: str
) -> None:

    ensure_dir(out_dir)

    # 🔥 FIX: bảo vệ trường hợp 3 list có length khác nhau
    # (xảy ra nếu train bị interrupt hoặc episodes không chia hết cho 10)
    min_sr = min(len(sac_sr), len(td3_sr), len(ddpg_sr))
    min_cr = min(len(sac_cr), len(td3_cr), len(ddpg_cr))

    sac_sr  = sac_sr[:min_sr]
    td3_sr  = td3_sr[:min_sr]
    ddpg_sr = ddpg_sr[:min_sr]

    sac_cr  = sac_cr[:min_cr]
    td3_cr  = td3_cr[:min_cr]
    ddpg_cr = ddpg_cr[:min_cr]

    # =================================================
    # FIG 7 SUCCESS RATE
    # =================================================
    plt.figure(figsize=(8, 4.5))

    x_sr = np.arange(1, min_sr + 1) * 10

    plt.plot(x_sr, sac_sr,  linewidth=2, label="SAC")
    plt.plot(x_sr, ddpg_sr, linewidth=2, label="DDPG")
    plt.plot(x_sr, td3_sr,  linewidth=2, label="TD3")

    plt.xlabel("Trained Episodes")
    plt.ylabel("Success Rate")
    plt.title(
        "Tỷ lệ thành công theo số tập huấn luyện của các thuật toán SAC, DDPG và TD3"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "fig7_compare_success_rate.png"),
        dpi=220
    )
    plt.close()

    # =================================================
    # FIG 8 COLLISION RATE
    # =================================================
    plt.figure(figsize=(8, 4.5))

    x_cr = np.arange(1, min_cr + 1) * 10

    plt.plot(x_cr, sac_cr,  linewidth=2, label="SAC")
    plt.plot(x_cr, ddpg_cr, linewidth=2, label="DDPG")
    plt.plot(x_cr, td3_cr,  linewidth=2, label="TD3")

    plt.xlabel("Trained Episodes")
    plt.ylabel("Collision Rate")
    plt.title(
        "Tỷ lệ va chạm theo số tập huấn luyện của các thuật toán SAC, DDPG và TD3"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "fig8_compare_collision_rate.png"),
        dpi=220
    )
    plt.close()