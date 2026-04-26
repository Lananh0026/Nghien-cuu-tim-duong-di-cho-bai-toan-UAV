from __future__ import annotations

import argparse
import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from env_collision_avoidance import EnvConfig, MultiUAVCollisionEnv
from sac_agent import SACAgent, SACConfig
from utils import ensure_dir, set_seed


def load_ga_result(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================
# XY -> LATLON (THÊM VÀO)
# =====================================================
def xy_to_latlon(x, y, ref_lat, ref_lon):
    R = 6371000.0  # Bán kính trái đất (m)
    dlat = y / R
    dlon = x / (R * np.cos(np.pi * ref_lat / 180.0))
    
    lat = ref_lat + dlat * 180 / np.pi
    lon = ref_lon + dlon * 180 / np.pi
    return lat, lon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/sac_actor.pt")
    parser.add_argument("--ga_result", type=str, default="outputs/ga_result.json")
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--goal_threshold", type=float, default=3.0)
    parser.add_argument("--spawn_noise", type=float, default=2.0)
    parser.add_argument("--dscale", type=float, default=300.0)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    ga_data = load_ga_result(args.ga_result)

    # ⚠️ QUAN TRỌNG: Cần lấy đúng cấu trúc dữ liệu cho map plotting
    # Kiểm tra xem ga_data có depot dạng dict hay list
    if isinstance(ga_data["depot"], dict):
        depot = np.asarray(ga_data["depot"]["xy"], dtype=np.float32)
        depot_lat = ga_data["depot"]["lat"]
        depot_lon = ga_data["depot"]["lon"]
    else:
        # Nếu depot là list [x, y] thì cần có lat/lon từ nơi khác
        depot = np.asarray(ga_data["depot"], dtype=np.float32)
        # Bạn cần có lat/lon trong ga_data, ví dụ:
        depot_lat = ga_data.get("depot_lat", 25.0173)  # Mặc định vĩ độ NTU
        depot_lon = ga_data.get("depot_lon", 121.5397) # Mặc định kinh độ NTU
        print("Warning: No lat/lon found in ga_data, using default NTU coordinates")
    
    customers = ga_data["customers"]
    routes: List[List[int]] = ga_data["routes"]
    map_width = float(ga_data["config"]["map_width"])

    customer_loc: Dict[int, np.ndarray] = {
        int(c["idx"]): np.asarray(c["location"], dtype=np.float32)
        for c in customers
    }

    n_customers = len(customers)
    n_uavs = max(len(routes), 1)

    env_cfg = EnvConfig(
        map_width=map_width,
        n_agents=n_uavs,
        dscale=args.dscale,
        max_steps=args.max_steps,
        goal_threshold=args.goal_threshold,
    )
    env = MultiUAVCollisionEnv(env_cfg)

    agent = SACAgent(SACConfig(device="cpu"))
    agent.load_actor(args.model)

    # Reset once to create arrays, then overwrite positions/goals
    states = env.reset()

    # Spawn all UAVs near the depot
    env.pos[:] = depot[None, :] + np.random.uniform(
        -args.spawn_noise, args.spawn_noise, size=(n_uavs, 2)
    ).astype(np.float32)
    env.depots[:] = env.pos.copy()
    env.vel[:] = 0.0
    env.heading[:] = 0.0
    env.done_agents[:] = False

    # Track route progress
    route_ptr = [0 for _ in range(n_uavs)]
    returned_to_depot = [False for _ in range(n_uavs)]

    def set_goals_from_routes() -> None:
        for i in range(n_uavs):
            route = routes[i]
            if route_ptr[i] < len(route):
                next_customer_id = route[route_ptr[i]]
                env.goals[i] = customer_loc[next_customer_id]
                returned_to_depot[i] = False
            else:
                env.goals[i] = depot.copy()

            env.initial_goal_dist[i] = max(
                float(np.linalg.norm(env.goals[i] - env.pos[i])), 1e-6
            )

    set_goals_from_routes()
    states = env._get_all_states()

    trajectories = [[env.pos[i].copy()] for i in range(n_uavs)]
    done = False
    step = 0

    while not done and step < args.max_steps:
        actions = np.vstack([agent.act(states[i], deterministic=True) for i in range(n_uavs)])
        states, rewards, env_done, info = env.step(actions)

        for i in range(n_uavs):
            trajectories[i].append(env.pos[i].copy())

            # Nếu UAV đã hoàn tất route và đã về depot rồi thì bỏ qua
            if returned_to_depot[i]:
                continue

            dist_to_goal = float(np.linalg.norm(env.pos[i] - env.goals[i]))
            if dist_to_goal <= env.cfg.goal_threshold:
                # Nếu vẫn còn khách trong route -> sang khách tiếp theo
                if route_ptr[i] < len(routes[i]):
                    route_ptr[i] += 1
                    if route_ptr[i] < len(routes[i]):
                        next_customer_id = routes[i][route_ptr[i]]
                        env.goals[i] = customer_loc[next_customer_id]
                    else:
                        # Hết route -> quay về depot
                        env.goals[i] = depot.copy()

                    env.initial_goal_dist[i] = max(
                        float(np.linalg.norm(env.goals[i] - env.pos[i])), 1e-6
                    )
                else:
                    # Đã về depot sau khi hoàn tất route
                    returned_to_depot[i] = True

        all_finished = all(returned_to_depot)
        done = bool(env_done or all_finished)
        step += 1

    # =====================================================
    # PLOT XY 
    # =====================================================
    plt.figure(figsize=(8, 8))
    plt.scatter([depot[0]], [depot[1]], c="red", s=130, marker="s", label="Depot")

    for c in customers:
        x, y = c["location"]
        plt.scatter([x], [y], c="blue", s=18)
        plt.text(x + 5, y + 5, f'{c["idx"]}', fontsize=7)

    colors = plt.cm.tab10(np.linspace(0, 1, n_uavs))
    for i in range(n_uavs):
        traj = np.asarray(trajectories[i])
        plt.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=1.6, label=f"UAV{i+1}")

    plt.title("Integrated parcel delivery simulation with GA routes and SAC avoidance")
    plt.xlabel("x-coordinate (m)")
    plt.ylabel("y-coordinate (m)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/fig16_integrated_simulation_routes.png", dpi=220)
    plt.close()

    # =====================================================
    # PLOT MAP 
    # =====================================================
    try:
        import geopandas as gpd
        import contextily as ctx
        from shapely.geometry import Point
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Vẽ depot lên bản đồ
        gpd.GeoSeries(
            [Point(depot_lon, depot_lat)],
            crs="EPSG:4326"
        ).to_crs(3857).plot(
            ax=ax,
            color="red",
            markersize=100,
            marker="s",
            label="Depot"
        )
        
        # Vẽ customers lên bản đồ
        for c in customers:
            # Kiểm tra mỗi customer có lat/lon không
            if "lat" in c and "lon" in c:
                gpd.GeoSeries(
                    [Point(c["lon"], c["lat"])],
                    crs="EPSG:4326"
                ).to_crs(3857).plot(
                    ax=ax,
                    color="blue",
                    markersize=30
                )
                # Thêm label cho customer
                x, y = gpd.GeoSeries([Point(c["lon"], c["lat"])], crs="EPSG:4326").to_crs(3857).geometry.x.iloc[0], \
                       gpd.GeoSeries([Point(c["lon"], c["lat"])], crs="EPSG:4326").to_crs(3857).geometry.y.iloc[0]
                ax.text(x + 5, y + 5, str(c["idx"]), fontsize=8)
            else:
                print(f"Warning: Customer {c['idx']} missing lat/lon, skipping map plot")
        
        # Vẽ trajectories lên bản đồ
        for i in range(n_uavs):
            traj_xy = np.array(trajectories[i])
            
            # Đảm bảo điểm cuối là depot nếu UAV đã về
            if returned_to_depot[i]:
                traj_xy[-1] = depot.copy()
            
            # Chuyển đổi từ XY sang lat/lon
            latlon = [
                xy_to_latlon(p[0], p[1], depot_lat, depot_lon)
                for p in traj_xy
            ]
            
            # Tạo các điểm shapely
            pts = [Point(lon, lat) for lat, lon in latlon]
            gdf = gpd.GeoSeries(pts, crs="EPSG:4326").to_crs(3857)
            
            ax.plot(
                gdf.geometry.x,
                gdf.geometry.y,
                linewidth=2,
                color=colors[i],
                label=f"UAV {i+1}"
            )
        
        # Thêm bản đồ nền
        ctx.add_basemap(
            ax,
            source=ctx.providers.OpenStreetMap.Mapnik
        )
        
        plt.title("UAV Delivery on Map")
        plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{args.outdir}/map_plot.png", dpi=220)
        plt.close()
        
        print(f"Map plot saved to {args.outdir}/map_plot.png")
        
    except ImportError as e:
        print(f"Warning: Could not create map plot - missing library: {e}")
        print("Install required packages: pip install geopandas contextily shapely")
    except Exception as e:
        print(f"Warning: Could not create map plot - error: {e}")

    print(
        f"Integrated simulation finished in {step} steps "
        f"with {n_uavs} UAVs and {n_customers} customers."
    )
    print("Used GA result file:", args.ga_result)
    print("Routes:", routes)


if __name__ == "__main__":
    main()