Để cải thiện file `GEMINI.md` theo hướng "Hướng dẫn mở rộng dự án", bạn có thể thêm một phần chi tiết như sau:

---

## Hướng dẫn mở rộng dự án

Phần này cung cấp các hướng dẫn về cách mở rộng và tùy chỉnh dự án `VisionAI Sumi Detect` để phù hợp với các yêu cầu mới.

### 1. Thêm một mô hình YOLO mới

Nếu bạn có một mô hình YOLO (`.pt` file) đã được huấn luyện mới và muốn tích hợp vào hệ thống:

*   **Bước 1: Đặt file mô hình.**
    Sao chép file `.pt` của mô hình mới vào thư mục `models/` (ví dụ: `models/MyNewModel.pt`).
*   **Bước 2: Cập nhật cấu hình sản phẩm.**
    Trong file CSV cấu hình sản phẩm của bạn (ví dụ: `config/products/ABC123x.csv`), trong cột `model_name`, hãy thay thế tên mô hình hiện có bằng tên mô hình mới của bạn (ví dụ: `MyNewModel` - không bao gồm phần mở rộng `.pt`). Hệ thống sẽ tự động tải mô hình này khi cần.

### 2. Tích hợp một Camera mới

Để thêm một camera mới vào hệ thống giám sát:

*   **Bước 1: Cập nhật `config/camera_config.csv`.**
    Mở file `config/camera_config.csv` và thêm một dòng mới cho camera của bạn. Đảm bảo cung cấp các thông tin sau:
    *   `camera_name`: Tên duy nhất cho camera (ví dụ: `CAM4`).
    *   `input_folder`: Đường dẫn đến thư mục nơi camera này sẽ lưu ảnh đầu vào. Hệ thống sẽ theo dõi thư mục này để phát hiện ảnh mới.
    *   `temp_folder`: Đường dẫn đến thư mục tạm thời để xử lý ảnh của camera này.
    *   `enabled`: Đặt là `true` để kích hoạt camera.
*   **Bước 2: Cập nhật cấu hình sản phẩm (nếu cần).**
    Nếu camera mới này sẽ được sử dụng trong các quy tắc ROI của một sản phẩm cụ thể, hãy đảm bảo rằng file CSV cấu hình sản phẩm đó (ví dụ: `config/products/MyProduct.csv`) có các quy tắc ROI liên quan đến `camera_name` mới của bạn.

### 3. Thêm các Quy tắc ROI cho Sản phẩm mới

Để định nghĩa các quy tắc kiểm tra cho một sản phẩm mới:

*   **Bước 1: Tạo file CSV sản phẩm mới.**
    Trong thư mục `config/products/`, tạo một file CSV mới với tên file là mã sản phẩm của bạn (ví dụ: `config/products/NEWPROD.csv`).
*   **Bước 2: Điền các quy tắc ROI.**
    Sử dụng định dạng của các file CSV sản phẩm hiện có (ví dụ: `ABC123x.csv`) làm mẫu. Điền các cột sau cho mỗi quy tắc ROI:
    *   `roi_id`: ID duy nhất cho quy tắc.
    *   `camera`: Tên camera mà quy tắc này áp dụng.
    *   `model_name`: Tên mô hình YOLO sẽ sử dụng.
    *   `class_id`: ID của lớp đối tượng cần phát hiện.
    *   `class_name`: Tên của lớp đối tượng.
    *   `detect_x_min`, `detect_y_min`, `detect_x_max`, `detect_y_max`: Tọa độ ROI phát hiện.
    *   `compare_x_min`, `compare_y_min`, `compare_x_max`, `compare_y_max`: Tọa độ ROI so sánh.
    *   `confidence`: Ngưỡng tin cậy cho việc phát hiện.
*   **Bước 3: Cấu hình sản phẩm trong `config.yaml` hoặc qua COM.**
    Đảm bảo rằng hệ thống được cấu hình để sử dụng mã sản phẩm mới này, thông qua cài đặt `product.code` trong `config/config.yaml` (chế độ `manual`) hoặc thông qua tín hiệu COM (chế độ `auto`).

### 4. Thay đổi logic so sánh

Nếu bạn cần một logic so sánh kết quả phát hiện khác với logic hiện tại trong `modules/comparator.py`:

*   **Bước 1: Chỉnh sửa `modules/comparator.py`.**
    Mở file `modules/comparator.py` và sửa đổi hàm `compare_detection` hoặc tạo một hàm so sánh mới.
*   **Bước 2: Cập nhật `main.py` (nếu cần).**
    Nếu bạn tạo một hàm so sánh mới, hoặc thay đổi giao diện của `compare_detection`, bạn sẽ cần cập nhật file `main.py` để gọi hàm so sánh mới hoặc điều chỉnh cách truyền tham số cho nó.

---
Bạn có thể chèn nội dung này vào file `GEMINI.md` của mình.