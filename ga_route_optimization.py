"""
ga_route_optimization.py
-------------------------
Tối ưu tuyến đường CVRP bằng Genetic Algorithm.

Thay đổi so với code cũ:
  1. load_input_data() đọc metadata (map_width, vehicle_capacity, max_parcels)
     từ Input_Data.json — không hardcode nữa
  2. split_routes() fix bug: điều kiện if/else sai khiến customer đầu tiên
     của mỗi route không bao giờ bị kiểm tra capacity
  3. Tính map_width thực tế từ dữ liệu (không dùng 1000 cứng)
  4. plot_solution() hiển thị tên địa điểm thực thay vì chỉ index
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field, asdict
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir, set_seed


# ================= DATA =================
@dataclass
class Customer:
    idx: int
    location: np.ndarray
    demand: float
    name: str = ""
    ctype: str = ""
    lat: float = 0.0     # NEW
    lon: float = 0.0     # NEW


@dataclass
class GAConfig:
    map_width: float       = 1000.0
    n_customers: int       = 20
    population_size: int   = 30
    weight_capacity: float = 1.0
    max_parcels: int       = 3
    mutation_rate: float   = 0.6
    generations: int       = 3000
    seed: int              = 42


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ================= LOAD DATA =================
def load_input_data(path: str):
    """
    Đọc Input_Data.json (output của process_osm_data.py).
    Trả về: depot, customers, map_width, weight_capacity, max_parcels
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Depot ---
    depot_raw = data["depart"]["xy_meters"]
    depot = np.array([depot_raw["x"], depot_raw["y"]], dtype=np.float32)
    depot_lat = data["depart"].get("lat", 0.0)   #  thêm
    depot_lon = data["depart"].get("lon", 0.0)   #  thêm
    # --- Customers ---
    customers = []
    i = 1
    while f"customer_{i}" in data:
        c = data[f"customer_{i}"]
        xy = c["xy_meters"]
        customers.append(Customer(
            idx=i,
            location=np.array([xy["x"], xy["y"]], dtype=np.float32),
            demand=float(c["demand"]),
            name=c.get("name", ""),
            ctype=c.get("type", ""),
            lat=c.get("lat", 0.0),     # NEW
            lon=c.get("lon", 0.0),     # NEW
        ))
        i += 1

    # --- Metadata (process_osm_data.py ghi vào) ---
    map_width        = float(data.get("map_width_meters", 1000.0))
    weight_capacity  = float(data.get("vehicle_capacity", 1.0))
    max_parcels      = int(data.get("max_parcels", 3))

    return depot, customers, map_width, weight_capacity, max_parcels, depot_lat, depot_lon

# ================= GA =================
class RouteOptimizerGA:
    def __init__(self, cfg: GAConfig, input_path: str):
        self.cfg = cfg
        set_seed(cfg.seed)

        depot, customers, map_width, weight_capacity, max_parcels, depot_lat, depot_lon = load_input_data(input_path)


        self.depot     = depot
        self.customers = customers
        self.depot_lat = depot_lat   # ✅ thêm
        self.depot_lon = depot_lon   # ✅ thêm

        # Ghi đè bằng giá trị thực từ file
        self.cfg.map_width        = map_width
        self.cfg.n_customers      = len(customers)
        self.cfg.weight_capacity  = weight_capacity
        self.cfg.max_parcels      = max_parcels

        print(f"[GA] {len(customers)} customers  |  map_width={map_width:.0f}m  "
              f"|  capacity={weight_capacity}  |  max_parcels={max_parcels}")
        print(f"[GA] Depot: {self.depot}")

    # ================= ROUTES =================
    def split_routes(self, chromosome: Sequence[int]) -> List[List[int]]:
        """
        Chia chromosome thành các route theo ràng buộc:
          - tổng demand <= weight_capacity
          - số lượng <= max_parcels

        BUG CŨ: code cũ dùng `if current and (...)` nên customer đầu tiên
        của mỗi route không bao giờ được kiểm tra → route đầu có thể vượt
        capacity ngay từ điểm 1 nếu demand > weight_capacity.

        FIX: luôn kiểm tra trước khi thêm.
        """
        demand_map = {c.idx: c.demand for c in self.customers}
        routes: List[List[int]] = []
        current: List[int] = []
        w_cur = 0.0
        cnt_cur = 0

        for cid in chromosome:
            d = demand_map[cid]
            # Kiểm tra TRƯỚC khi thêm (không phụ thuộc current có rỗng không)
            if w_cur + d > self.cfg.weight_capacity + 1e-9 or cnt_cur + 1 > self.cfg.max_parcels:
                if current:
                    routes.append(current)
                current = []
                w_cur = 0.0
                cnt_cur = 0
            current.append(cid)
            w_cur += d
            cnt_cur += 1

        if current:
            routes.append(current)

        return routes

    def route_energy(self, route: List[int]) -> float:
        """
        Hàm năng lượng theo bài báo:
          E = sum_i [ dist(prev, i) * (1 + payload_remaining) ] + dist(last, depot)
        """
        demand_map = {c.idx: c.demand for c in self.customers}
        loc_map = {0: self.depot, **{c.idx: c.location for c in self.customers}}

        payload = sum(demand_map[cid] for cid in route)
        prev    = 0
        energy  = 0.0

        for cid in route:
            energy  += distance(loc_map[prev], loc_map[cid]) * (1.0 + payload)
            payload -= demand_map[cid]
            prev     = cid

        energy += distance(loc_map[prev], loc_map[0])   # quay về depot (payload=0)
        return energy

    def total_energy(self, chrom: List[int]) -> float:
        return sum(self.route_energy(r) for r in self.split_routes(chrom))

    def fitness(self, chrom: List[int]) -> float:
        return 1.0 / max(self.total_energy(chrom), 1e-9)

    # ================= GA OPS =================
    def tournament_select(self, pop: List[List[int]]) -> List[int]:
        cand = random.sample(pop, 3)
        cand.sort(key=self.fitness, reverse=True)
        return cand[0][:]

    def crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        """Order Crossover (OX) — giữ nguyên theo bài báo."""
        n = len(p1)
        a, b = sorted(random.sample(range(n), 2))

        def make_child(pa: List[int], pb: List[int]) -> List[int]:
            child = [-1] * n
            child[a:b + 1] = pa[a:b + 1]
            fill = [x for x in pb if x not in child]
            j = 0
            for i in range(n):
                if child[i] == -1:
                    child[i] = fill[j]
                    j += 1
            return child

        return make_child(p1, p2), make_child(p2, p1)

    def mutate(self, chrom: List[int]) -> List[int]:
        """2-opt inversion mutation."""
        if random.random() > self.cfg.mutation_rate:
            return chrom
        a, b = sorted(random.sample(range(len(chrom)), 2))
        chrom[a:b + 1] = list(reversed(chrom[a:b + 1]))
        return chrom

    # ================= RUN =================
    def run(self) -> Tuple[List[int], List[float]]:
        n = self.cfg.n_customers
        pop = [
            random.sample(range(1, n + 1), n)
            for _ in range(self.cfg.population_size)
        ]
        curve: List[float] = []

        for gen in range(self.cfg.generations):
            pop.sort(key=self.fitness, reverse=True)
            curve.append(self.total_energy(pop[0]))

            if (gen + 1) % 500 == 0:
                print(f"  Gen {gen+1:4d}/{self.cfg.generations}  "
                      f"energy={curve[-1]:.1f}")

            new_pop = [pop[0][:]]   # elitism
            while len(new_pop) < self.cfg.population_size:
                c1, c2 = self.crossover(
                    self.tournament_select(pop),
                    self.tournament_select(pop),
                )
                new_pop.extend([self.mutate(c1), self.mutate(c2)])

            pop = new_pop[:self.cfg.population_size]

        pop.sort(key=self.fitness, reverse=True)
        return pop[0], curve

    # ================= OUTPUT =================
    def plot_solution(self, best: List[int], out_dir: str) -> None:
        ensure_dir(out_dir)
        routes   = self.split_routes(best)
        loc_map  = {0: self.depot, **{c.idx: c.location for c in self.customers}}
        name_map = {c.idx: c.name for c in self.customers}
        colors   = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                    "tab:purple", "tab:brown", "tab:pink", "tab:gray"]

        plt.figure(figsize=(9, 9))
        plt.scatter([self.depot[0]], [self.depot[1]],
                    c="red", s=160, marker="s", zorder=5, label="Depot")

        for c in self.customers:
            plt.scatter([c.location[0]], [c.location[1]], c="steelblue", s=30, zorder=4)
            short = c.name[:12] if c.name else str(c.idx)
            plt.text(c.location[0] + 4, c.location[1] + 4,
                     f"{c.idx}:{short}", fontsize=6.5, alpha=0.85)

        for ridx, route in enumerate(routes):
            seq = [0] + route + [0]
            xs = [loc_map[k][0] for k in seq]
            ys = [loc_map[k][1] for k in seq]
            label = f"Route {ridx+1} ({len(route)} stops)"
            plt.plot(xs, ys, color=colors[ridx % len(colors)],
                     linewidth=1.8, label=label)

        total_e = self.total_energy(best)
        plt.title(f"GA Route Optimization — {len(self.customers)} customers\n"
                  f"Total energy: {total_e:.1f}  |  {len(routes)} routes")
        plt.xlabel("x-coordinate (m)  [relative to depot]")
        plt.ylabel("y-coordinate (m)  [relative to depot]")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/fig11_ga_routes.png", dpi=220)
        plt.close()
        print(f"  → Saved: {out_dir}/fig11_ga_routes.png")

    def plot_curve(self, curve: List[float], out_dir: str) -> None:
        ensure_dir(out_dir)
        plt.figure(figsize=(7, 4.5))
        plt.plot(curve, linewidth=1.5)
        plt.xlabel("Generation")
        plt.ylabel("Total energy")
        plt.title("GA convergence curve")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/fig12_convergence.png", dpi=220)
        plt.close()
        print(f"  → Saved: {out_dir}/fig12_convergence.png")

    def save_result_json(self, best: List[int], out_dir: str) -> str:
        ensure_dir(out_dir)
        routes = self.split_routes(best)
        result = {
            "config": asdict(self.cfg),

            "depot": {
                    "xy": self.depot.tolist(),
                    "lat": self.depot_lat,
                    "lon": self.depot_lon
            },

            "customers": [
                {
                    "idx": c.idx,
                    "location": c.location.tolist(),
                    "lat": c.lat,          # NEW
                    "lon": c.lon,          # NEW
                    "demand": c.demand,
                    "name": c.name
                }
                for c in self.customers
            ],

            "best_chromosome": best,
            "total_energy": self.total_energy(best),
            "n_routes": len(routes),

            "routes": [r for r in routes]
        }
        path = f"{out_dir}/ga_result.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  → Saved: {path}")
        return path


# ================= MAIN =================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data",    default="outputs/Input_Data.json")
    parser.add_argument("--generations",   type=int,   default=3000)
    parser.add_argument("--population",    type=int,   default=30)
    parser.add_argument("--mutation_rate", type=float, default=0.6)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--outdir",        default="outputs")
    args = parser.parse_args()

    cfg = GAConfig(
        population_size=args.population,
        mutation_rate=args.mutation_rate,
        generations=args.generations,
        seed=args.seed,
    )

    ga = RouteOptimizerGA(cfg, args.input_data)
    best, curve = ga.run()

    ga.plot_solution(best, args.outdir)
    ga.plot_curve(curve, args.outdir)
    ga.save_result_json(best, args.outdir)

    print(f"\nBest energy : {ga.total_energy(best):.2f}")
    print(f"Routes      : {ga.split_routes(best)}")


if __name__ == "__main__":
    main()
