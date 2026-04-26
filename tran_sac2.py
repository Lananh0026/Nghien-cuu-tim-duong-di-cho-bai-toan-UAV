from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np

from env_collision_avoidance import EnvConfig, MultiUAVCollisionEnv
from sac_agent import SACAgent, SACConfig
from utils import ensure_dir, plot_training_curves, set_seed


def evaluate(agent: SACAgent, n_agents: int, episodes: int = 10) -> tuple[float, float]:
    sr_list: List[float] = []
    cr_list: List[float] = []
    env = MultiUAVCollisionEnv(EnvConfig(n_agents=n_agents))
    for _ in range(episodes):
        states = env.reset()
        done = False
        agent_collided = np.zeros(n_agents, dtype=bool)
        last_info = None
        while not done:
            actions = np.vstack([
                agent.act(states[i], deterministic=True)
                for i in range(n_agents)
            ])
            states, rewards, done, info = env.step(actions)
            last_info = info
            agent_collided |= info["collision_mask"]

        sr_list.append(last_info["success_rate"])
        cr_list.append(float(np.mean(agent_collided)))

    return float(np.mean(sr_list)), float(np.mean(cr_list))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outdir", type=str, default="outputs")
    # FIX: update_ratio — số lần update SAC mỗi environment step
    # Với 5 agents, buffer tích lũy nhanh → cần update nhiều hơn để kịp học
    parser.add_argument("--update_ratio", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)
    ensure_dir("checkpoints")

    env_cfg = EnvConfig()
    env = MultiUAVCollisionEnv(env_cfg)
    agent = SACAgent(SACConfig(device=args.device))

    scores: List[float] = []
    success_rates: List[float] = []
    collision_rates: List[float] = []

    for ep in range(args.episodes):
        states = env.reset()
        done = False
        ep_score = 0.0

        # FIX: track done_agents từ bước trước để không add transition thừa
        prev_done = np.zeros(env_cfg.n_agents, dtype=bool)

        while not done:
            actions = np.vstack([
                agent.act(states[i], deterministic=False)
                for i in range(env_cfg.n_agents)
            ])
            next_states, rewards, done, info = env.step(actions)
            current_done = info["done_agents"]

            for i in range(env_cfg.n_agents):
                # FIX: chỉ add transition nếu agent chưa done ở bước trước
                # Tránh buffer bị nhiễm bởi các transition vô nghĩa sau khi agent đã kết thúc
                if not prev_done[i]:
                    # terminal = True chỉ đúng tại bước agent done lần đầu
                    terminal = float(current_done[i] and not prev_done[i])
                    agent.replay.add(
                        states[i], actions[i],
                        float(rewards[i]),
                        next_states[i],
                        terminal
                    )

            # FIX: update nhiều lần mỗi step để học kịp với tốc độ thu thập dữ liệu
            for _ in range(args.update_ratio):
                agent.update()

            prev_done = current_done.copy()
            states = next_states
            # FIX: dùng mean thay vì sum để score không bị thổi phồng bởi số steps
            ep_score += float(np.mean(rewards))

        scores.append(ep_score)

        if (ep + 1) % 10 == 0:
            sr, cr = evaluate(agent, env_cfg.n_agents, episodes=10)
            success_rates.append(sr)
            collision_rates.append(cr)
            print(
                f"Episode {ep+1:4d}/{args.episodes} | "
                f"score={ep_score:7.3f} | "
                f"SR={sr:.4f} | CR={cr:.4f} | "
                f"alpha={agent.alpha.item():.4f}"
            )
        else:
            print(f"Episode {ep+1:4d}/{args.episodes} | score={ep_score:7.3f}")

    agent.save("checkpoints/sac_actor.pt", "checkpoints/sac")
    plot_training_curves(scores, success_rates, collision_rates, args.outdir)

    # Lưu kết quả để plot_results.py đọc
    result = {"scores": scores, "success_rates": success_rates, "collision_rates": collision_rates}
    out_path = os.path.join(args.outdir, "sac_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"\nDONE SAC! Saved → {out_path}")


if __name__ == "__main__":
    main()
    