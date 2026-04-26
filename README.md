Thành phần cốt lõi của bài toán:
 - Tránh va chạm đa UAV phi tập trung.
 - Thuật toán Soft Actor-Critic (SAC) với khả năng tự động điều chỉnh entropy.
 - Quan sát cục bộ 10 chiều (10D observation).
 - Hành động 2D liên tục (tốc độ, hướng bay).
 - Thiết kế phần thưởng (reward).
 - Tối ưu hóa lộ trình bằng cách sử dụng mục tiêu CVRP sửa đổi và Thuật toán Di truyền (GA) với phép lai chéo OX.

## Files
- process_osm_data.py: Xử lý dữ liệu bản đồ OpenStreetMap
- env_collision_avoidance.py: Môi trường 2D đa UAV dành cho SAC
- sac_agent.py: SAC với bộ phê bình đôi và tự động điều chỉnh nhiệt độ
- replay_buffer.py: Bộ đệm trải nghiệm (replay buffer)
- train_sac.py: Huấn luyện chính sách SAC dùng chung cho tất cả các tác nhân UAV
- train_sac2.py: File thực thi huấn luyện mô hình SAC thêm lưu file json
- train_ddpg.py: File thực thi huấn luyện mô hình DDPG
- train_td3.py: File thực thi huấn luyện mô hình TD3
- ga_route_optimization.py: Tối ưu hóa lộ trình CVRP sửa đổi bằng thuật toán di truyền (GA)
- simulate_delivery.py: Tích hợp lộ trình từ GA với chính sách tránh va chạm SAC đã huấn luyện
- compare_algorithms.py: So sánh hiệu suất giữa các thuật toán
- utils.py: Các hàm hỗ trợ vẽ biểu đồ và thiết lập seed
- requirements.txt: Các thư viện Python cần thiết
## Quick start
```bash
pip install -r requirements.txt
pip install contextily
python process_osm_data.py --input data.json --n_customers 20
python train_sac.py --episodes 500
python ga_route_optimization.py --input_data outputs/Input_Data.json 
python simulate_delivery.py --model checkpoints/sac_actor.pt --ga_result outputs/ga_result.json
python compare_algorithms.py 
```
