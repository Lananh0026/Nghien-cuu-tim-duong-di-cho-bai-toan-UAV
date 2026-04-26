import json, os, sys
import numpy as np
import matplotlib.pyplot as plt

INDIR  = "E:\khoaluan\stable_baselines3\uav2\mt_sac\outputs"
OUTDIR = "E:\khoaluan\stable_baselines3\uav2\mt_sac\outputs"
os.makedirs(OUTDIR, exist_ok=True)

def load(name):
    with open(f"{INDIR}/{name}") as f:
        d = json.load(f)
    return d["success_rates"], d["collision_rates"]

def smooth(values, window=10):
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    out = np.convolve(arr, kernel, mode="valid")
    prefix = np.full(window - 1, out[0])
    return np.concatenate([prefix, out])

sac_sr,  sac_cr  = load("sac_results.json")
td3_sr,  td3_cr  = load("td3_results.json")
ddpg_sr, ddpg_cr = load("ddpg_results.json")

def plot(sac, td3, ddpg, ylabel, title, filename):
    n   = min(len(sac), len(td3), len(ddpg))
    x   = np.arange(1, n + 1) * 10

    plt.figure(figsize=(10, 5))
    plt.plot(x, smooth(sac[:n]),  linewidth=2, label="SAC")
    plt.plot(x, smooth(ddpg[:n]), linewidth=2, label="DDPG")
    plt.plot(x, smooth(td3[:n]),  linewidth=2, label="TD3")
    plt.xlabel("Trained Episodes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{filename}", dpi=220)
    plt.show()
    print(f"Saved: {filename}")

plot(sac_sr, td3_sr, ddpg_sr,
     "Success Rate",
     "Tỷ lệ thành công theo số tập huấn luyện của các thuật toán SAC, DDPG và TD3",
     "fig7_compare_success_rate.png")

plot(sac_cr, td3_cr, ddpg_cr,
     "Collision Rate",
     "Tỷ lệ va chạm theo số tập huấn luyện của các thuật toán SAC, DDPG và TD3",
     "fig8_compare_collision_rate.png")