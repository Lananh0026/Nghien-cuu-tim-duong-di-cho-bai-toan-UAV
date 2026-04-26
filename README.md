# Nghiên cứu tìm đường đi cho bài toán UAV (UAV Parcel Delivery System)

Đây là dự án nghiên cứu và triển khai hệ thống giao hàng bằng thiết bị bay không người lái (UAV), tập trung vào hai bài toán cốt lõi: **Tối ưu hóa lộ trình (Route Optimization)** và **Tránh va chạm đa tác nhân (Multi-agent Collision Avoidance)**.

## 📝 Giới thiệu đề tài
Khóa luận tập trung nghiên cứu giải pháp giao hàng tự động bằng UAV trong môi trường đô thị. Hệ thống kết hợp giữa thuật toán học sâu tăng cường (Deep Reinforcement Learning) để điều khiển UAV bay an toàn và thuật toán di truyền (Genetic Algorithm) để lập kế hoạch giao hàng tối ưu.

## 🚀 Thành phần cốt lõi của bài toán
Dựa trên kiến trúc được đề xuất trong nghiên cứu của Chun-Yuan Chi và cộng sự (2024), hệ thống bao gồm:
- **Tối ưu hóa lộ trình:** Giải quyết bài toán định tuyến phương tiện có giới hạn tải trọng (CVRP) bằng **Thuật toán Di truyền (GA)** với phép lai chéo OX.
- **Tránh va chạm đa UAV phi tập trung:** Mỗi UAV tự đưa ra quyết định bay dựa trên thông tin cục bộ.
- **Thuật toán SAC (Soft Actor-Critic):** Sử dụng cơ chế tự động điều chỉnh entropy giúp UAV khám phá môi trường tốt hơn.
- **Mô hình trạng thái & hành động:**
  - *Quan sát cục bộ 10 chiều (10D):* Bao gồm vị trí đích, khoảng cách và vận tốc của các UAV lân cận.
  - *Hành động 2D liên tục:* Điều khiển vận tốc và hướng bay.

## 📂 Danh mục tệp tin
- `process_osm_data.py`: Xử lý dữ liệu bản đồ từ OpenStreetMap.
- `env_collision_avoidance.py`: Môi trường giả lập 2D đa UAV.
- `sac_agent.py`: Triển khai thuật toán Soft Actor-Critic.
- `ga_route_optimization.py`: Tối ưu hóa lộ trình CVRP bằng GA.
- `simulate_delivery.py`: Mô phỏng quá trình giao hàng thực tế.
- `compare_algorithms.py`: So sánh hiệu suất giữa SAC, DDPG và TD3.
- `train_sac.py` / `train_sac2.py`: Huấn luyện mô hình SAC.
- `utils.py`: Các hàm hỗ trợ vẽ biểu đồ.
- `requirements.txt`: Danh sách thư viện cần thiết.

## 🛠 Hướng dẫn chạy
1. **Cài đặt:** `pip install -r requirements.txt`
2. **Xử lý dữ liệu:** `python process_osm_data.py --input data.json --n_customers 20`
3. **Huấn luyện:** `python train_sac.py --episodes 500`
4. **Mô phỏng:** `python simulate_delivery.py --model checkpoints/sac_actor.pt --ga_result outputs/ga_result.json`
5. **So sánh hiệu suất 3 thuật toán:** `python compare_algorithms.py
