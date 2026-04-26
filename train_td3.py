from __future__ import annotations

import argparse
import json
from typing import List

import numpy as np

from env_collision_avoidance import EnvConfig, MultiUAVCollisionEnv
from td3_agent import TD3Agent, TD3Config
from utils import ensure_dir, set_seed


# ================= EVALUATE =================
def evaluate(agent, env_cfg: EnvConfig, episodes: int = 10) -> tuple[float, float]:
    sr_list: List[float] = []
    cr_list: List[float] = []

    env = MultiUAVCollisionEnv(env_cfg)
    n_agents = env_cfg.n_agents

    for _ in range(episodes):
        states = env.reset()
        done = False
        collision_happened = False

        while not done:
            actions = np.vstack([
                agent.act(states[i], deterministic=True)
                for i in range(n_agents)
            ])
            states, rewards, done, info = env.step(actions)
            if info["collision_rate"] > 0:
                collision_happened = True

        sr_list.append(info["success_rate"])
        cr_list.append(1.0 if collision_happened else 0.0)

    return float(np.mean(sr_list)), float(np.mean(cr_list))


# ================= TRAIN =================
def train(agent, env_cfg: EnvConfig, episodes: int, name: str):
    env = MultiUAVCollisionEnv(env_cfg)

    scores = []
    success_rates = []
    collision_rates = []

    for ep in range(episodes):
        states = env.reset()
        done = False
        ep_score = 0.0

        while not done:
            actions = np.vstack([
                agent.act(states[i], deterministic=False)
                for i in range(env_cfg.n_agents)
            ])

            next_states, rewards, done, info = env.step(actions)
            dones_per_agent = info["done_agents"].astype(np.float32)

            for i in range(env_cfg.n_agents):
                agent.replay.add(
                    states[i], actions[i], float(rewards[i]),
                    next_states[i], float(dones_per_agent[i])
                )

            agent.update()
            states = next_states
            ep_score += float(np.sum(rewards))

        scores.append(ep_score)

        if (ep + 1) % 10 == 0:
            sr, cr = evaluate(agent, env_cfg, episodes=10)
            success_rates.append(sr)
            collision_rates.append(cr)
            print(f"[{name}] Ep {ep+1:4d}/{episodes} | Score={ep_score:8.2f} | SR={sr:.3f} | CR={cr:.3f}")
        else:
            print(f"[{name}] Ep {ep+1:4d}/{episodes} | Score={ep_score:8.2f}")

    return scores, success_rates, collision_rates


# ================= MAIN =================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--device",   type=str, default="cpu")
    parser.add_argument("--outdir",   type=str, default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)
    ensure_dir("checkpoints")

    env_cfg = EnvConfig()
    agent   = TD3Agent(TD3Config(device=args.device))

    print("\n========== TRAIN TD3 ==========")
    scores, sr, cr = train(agent, env_cfg, args.episodes, "TD3")

    # Lưu checkpoint
    agent.save("checkpoints/td3_actor.pt", "checkpoints/td3")

    # Lưu kết quả để plot_results.py đọc
    result = {"scores": scores, "success_rates": sr, "collision_rates": cr}
    out_path = f"{args.outdir}/td3_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f)

    print(f"\nDONE TD3! Saved → {out_path}")


if __name__ == "__main__":
    main()