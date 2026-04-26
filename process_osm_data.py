"""
process_osm_data.py
-------------------
Chuyển đổi dữ liệu OSM (data.json) sang Input_Data.json dùng cho GA.

Vấn đề của code cũ:
  - Không lọc đường phố (highway=secondary/primary/residential/service...)
    → 125/185 elements là đường phố, KHÔNG phải điểm giao hàng
  - Không lọc bus_stop, bridge
  - Thiếu metadata (map_width_meters, vehicle_capacity, max_parcels)
    → ga_route_optimization.py không đọc được config đúng

Sau khi lọc đúng: ~37 điểm giao hàng thực sự (shop/amenity có tên)
"""

import json
import math
import os
import random
import argparse
import numpy as np

# ================= DEPOT =================
DEPOT_OSM_ID = 9420432154
DEPOT_LAT    = 20.9973974
DEPOT_LON    = 105.8267585

# Các loại highway là đường phố/hạ tầng, KHÔNG phải điểm giao hàng
HIGHWAY_ROAD_TYPES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "service", "proposed",
    "living_street", "pedestrian", "track", "road",
}

DEFAULT_INPUT  = "data.json"
DEFAULT_OUTPUT = os.path.join("outputs", "Input_Data.json")


# ================= DISTANCE =================
def haversine(p1, p2):
    R = 6371000
    phi1 = math.radians(p1["lat"])
    phi2 = math.radians(p2["lat"])
    dphi = math.radians(p2["lat"] - p1["lat"])
    dlambda = math.radians(p2["lon"] - p1["lon"])
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def latlon_to_xy_meters(lat, lon):
    """Chuyển (lat, lon) sang tọa độ phẳng (x, y) tính bằng mét, gốc = depot."""
    ref = {"lat": DEPOT_LAT, "lon": DEPOT_LON}
    dx = haversine(ref, {"lat": DEPOT_LAT, "lon": lon})
    dy = haversine(ref, {"lat": lat, "lon": DEPOT_LON})
    if lon < DEPOT_LON:
        dx = -dx
    if lat < DEPOT_LAT:
        dy = -dy
    return dx, dy


# ================= FILTER: CHỈ LẤY ĐIỂM GIAO HÀNG THỰC SỰ =================
def is_delivery_point(el: dict) -> bool:
    """
    Trả về True nếu element này là điểm giao hàng hợp lệ.

    Quy tắc lọc (theo thứ tự):
    1. Phải có tên (name tag) — không có tên thì UAV không biết giao cho ai
    2. Loại bỏ đường phố (highway in HIGHWAY_ROAD_TYPES)
    3. Loại bỏ bus_stop
    4. Loại bỏ man_made=bridge, barrier, natural
    5. Phải có ít nhất một trong: amenity, shop, building + amenity/shop
    """
    tags = el.get("tags", {})

    # 1. Phải có tên
    if not tags.get("name"):
        return False

    # 2. Loại bỏ đường phố hạ tầng
    hw = tags.get("highway", "")
    if hw in HIGHWAY_ROAD_TYPES:
        return False
    if hw == "bus_stop":
        return False

    # 3. Loại bỏ cầu / hạ tầng khác
    if tags.get("man_made") in ("bridge", "pier"):
        return False
    if tags.get("barrier"):
        return False
    if tags.get("natural"):
        return False

    # 4. Phải là địa điểm dịch vụ/thương mại
    has_amenity = bool(tags.get("amenity"))
    has_shop    = bool(tags.get("shop"))
    has_tourism = bool(tags.get("tourism"))
    has_office  = bool(tags.get("office"))
    has_leisure = bool(tags.get("leisure"))

    return has_amenity or has_shop or has_tourism or has_office or has_leisure


# ================= LOAD OSM =================
def load_customer_nodes(osm_file: str):
    with open(osm_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    customers = []
    skipped_depot = 0
    skipped_no_coord = 0
    skipped_filter = 0

    for el in data.get("elements", []):
        # Bỏ qua depot
        if int(el["id"]) == DEPOT_OSM_ID:
            skipped_depot += 1
            continue

        # Lấy tọa độ
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lon is None:
            skipped_no_coord += 1
            continue

        # Áp dụng filter
        if not is_delivery_point(el):
            skipped_filter += 1
            continue

        tags = el.get("tags", {})
        name = tags["name"]

        customers.append({
            "osm_id": el["id"],
            "lat": lat,
            "lon": lon,
            "name": name,
            "type": tags.get("amenity") or tags.get("shop") or tags.get("tourism") or "other",
        })

    print(f"  Bỏ qua depot:          {skipped_depot}")
    print(f"  Bỏ qua không tọa độ:   {skipped_no_coord}")
    print(f"  Bỏ qua không hợp lệ:   {skipped_filter}  (đường phố, cầu...)")
    print(f"  Điểm hợp lệ còn lại:   {len(customers)}")

    return customers


# ================= SPREAD SAMPLING =================
def select_spread_customers(customers, n_customers, min_dist=100, seed=42):
    """
    Chọn n_customers điểm phân bố đều, tránh tụ cụm.
    min_dist: khoảng cách tối thiểu giữa 2 điểm (mét).
    """
    random.seed(seed)

    def dist(c1, c2):
        x1, y1 = latlon_to_xy_meters(c1["lat"], c1["lon"])
        x2, y2 = latlon_to_xy_meters(c2["lat"], c2["lon"])
        return math.hypot(x1 - x2, y1 - y2)

    shuffled = customers[:]
    random.shuffle(shuffled)
    selected = []

    for c in shuffled:
        if len(selected) >= n_customers:
            break
        if all(dist(c, s) > min_dist for s in selected):
            selected.append(c)

    # Nếu vẫn chưa đủ (min_dist quá chặt), lấy thêm
    if len(selected) < n_customers:
        remaining = [c for c in shuffled if c not in selected]
        needed = n_customers - len(selected)
        selected.extend(remaining[:needed])
        print(f"  Cảnh báo: chỉ chọn được {len(selected)} điểm với min_dist={min_dist}m")

    return selected


# ================= BUILD DATA =================
def build_input_data(
    osm_file: str,
    output_file: str,
    n_customers: int = 0,
    min_dist: int = 100,
    vehicle_capacity: float = 1.0,
    max_parcels: int = 3,
    seed: int = 42,
):
    random.seed(seed)

    print(f"\n[process_osm_data] Đọc file: {osm_file}")
    all_customers = load_customer_nodes(osm_file)

    # Chọn số lượng
    if n_customers <= 0 or n_customers >= len(all_customers):
        selected = all_customers
        print(f"  → Lấy tất cả {len(selected)} điểm")
    else:
        selected = select_spread_customers(all_customers, n_customers, min_dist, seed)
        print(f"  → Đã chọn {len(selected)} điểm (min_dist={min_dist}m)")

    # Tính phạm vi bản đồ thực tế
    coords = [latlon_to_xy_meters(c["lat"], c["lon"]) for c in selected]
    max_dist = max(math.hypot(x, y) for x, y in coords) if coords else 500.0

    # map_width = 2 * (max_dist + buffer 10%)
    # Đảm bảo depot (0,0) luôn nằm trong tâm bản đồ
    map_width = round(2 * max_dist * 1.1, 1)
    print(f"  → map_width tính được: {map_width:.1f} m  (max_dist={max_dist:.1f} m)")

    # Build JSON
    vrp = {
        # Metadata cho GA đọc
        "map_width_meters": map_width,
        "vehicle_capacity": vehicle_capacity,
        "max_parcels": max_parcels,

        # Depot luôn là gốc tọa độ
        "depart": {
            "xy_meters": {"x": 0.0, "y": 0.0},
            "lat": DEPOT_LAT,   # ✅ thêm dòng này
            "lon": DEPOT_LON,   # ✅ thêm dòng này
            "name": "Bưu điện Khương Mai",
            "osm_id": DEPOT_OSM_ID,
        },
    }

    for i, c in enumerate(selected, 1):
        x, y = latlon_to_xy_meters(c["lat"], c["lon"])
        vrp[f"customer_{i}"] = {
            "xy_meters": {"x": round(x, 2), "y": round(y, 2)},
            "lat": c["lat"],   # ✅ thêm
            "lon": c["lon"],   # ✅ thêm
            "demand": round(random.uniform(0.05, 0.5), 2),
            "name": c["name"],
            "type": c["type"],
            "osm_id": c["osm_id"],
        }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vrp, f, indent=2, ensure_ascii=False)

    print(f"  → Đã lưu: {output_file}")
    print(f"  → Tổng khách hàng: {len(selected)}")
    return output_file


# ================= MAIN =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi OSM → Input_Data.json cho GA")
    parser.add_argument("--input",            default=DEFAULT_INPUT,   help="File OSM JSON")
    parser.add_argument("--output",           default=DEFAULT_OUTPUT,  help="File output")
    parser.add_argument("--n_customers",      type=int,   default=20,  help="Số điểm giao hàng (0=tất cả)")
    parser.add_argument("--min_dist",         type=int,   default=100, help="Khoảng cách tối thiểu giữa các điểm (m)")
    parser.add_argument("--vehicle_capacity", type=float, default=1.0, help="Sức chứa xe (tổng demand)")
    parser.add_argument("--max_parcels",      type=int,   default=3,   help="Số kiện hàng tối đa mỗi chuyến")
    parser.add_argument("--seed",             type=int,   default=42,  help="Random seed")
    args = parser.parse_args()

    build_input_data(
        osm_file=args.input,
        output_file=args.output,
        n_customers=args.n_customers,
        min_dist=args.min_dist,
        vehicle_capacity=args.vehicle_capacity,
        max_parcels=args.max_parcels,
        seed=args.seed,
    )
