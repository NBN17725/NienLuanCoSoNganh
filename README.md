Hướng dẫn Cài đặt và Chạy Dự án
Để mô hình hoạt động tối ưu và đạt trải nghiệm tốt nhất, vui lòng thực hiện theo các bước hướng dẫn sau:
📋 Yêu cầu hệ thống
Python: Phiên bản 3.10 trở lên.

Cài đặt các thư viện phụ thuộc
  Sau khi tải/clone thư mục dự án về máy, mở terminal tại thư mục gốc và chạy các lệnh sau để cài đặt thư viện:
    pip install chromadb
    pip install torch
    pip install gliclass
    pip install sentence-transformers
    pip install transformers
    
Lưu ý quan trọng trước khi khởi chạy
  Kiểm tra đường dẫn: Hãy đảm bảo các đường dẫn kết nối trong file cấu hình/code đã được trỏ đúng thư mục trước khi chạy.

ChromaDB Chunking: Nếu ChromaDB chưa hiển thị đủ 5267 chunks, vui lòng giữ nguyên và đợi hệ thống tự động tiến hành chia nhỏ (chunk) lại dữ liệu từ đầu.

Mở dự án bằng Visual Studio Code.
Mở Terminal trong VS Code (Ctrl + ~ hoặc Cmd + ~) và chạy lệnh sau: python -m uvicorn main:app --reload --port 8000

Sau khi hệ thống khởi chạy thành công, terminal sẽ xuất hiện đường dẫn: [http://127.0.0.1:8000](http://127.0.0.1:8000).
Copy đường dẫn [http://127.0.0.1:8000](http://127.0.0.1:8000) và dán vào trình duyệt web để bắt đầu trải nghiệm.
