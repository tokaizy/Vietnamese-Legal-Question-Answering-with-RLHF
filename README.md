Luật Bảo vệ quyền lợi người tiêu dùng 2023

**Quy trình training** 
1. Tạo 1 folder mới chứa các folder của project
2. đi đến mục scripts chạy training tuần tự như sau:
train_SFT.py -> train_rw.py -> train_ppo 
3. chạy tiếp auto_feedback_gen.py -> convert_feedback_to_preference.py -> train_rw.py -> train_ppo 
để lặp lại theo quy trình của RLHF
4. Chạy file webserver.py để bắt đâù hỏi đáp

thư mục data - chứa dữ lieu dùng training
thư mục xlm_data - chứa dữ lieu thông tin văn bản luật 
thư mục models - chứa dữ lieu mô hình sau khi train
