from modules.csv_loader import load_product_csv
from modules.camera_selector import get_used_cameras
from modules.image_watcher import ImageWatcher
from modules.roi_manager import prepare_roi_data
from modules.model_manager import get_model
from modules.detector import detect_object
from modules.comparator import compare_detection
from modules.result_manager import aggregate_results
from modules.result_visualizer import visualize_detection_result
import os
import time
from datetime import datetime
import cv2

PRODUCT_CODE = "ABC123x"
CSV_PATH = f"config/products/{PRODUCT_CODE}.csv"
OUTPUT_DIR = "output/results"
LOG_DIR = "output/logs"

# Tao thu muc output neu chua ton tai
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def log_message(message: str) -> None:
    """In va luu log"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    # Luu vao file log
    log_file = os.path.join(LOG_DIR, "processing.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")


def wait_for_all_cameras_images(watchers: dict, used_cameras: list) -> dict:
    """
    Chờ tất cả camera có ảnh mới dùng ImageWatcher
    - Chỉ trả về khi tất cả camera đều có ảnh
    
    Args:
        watchers (dict): {"CAM1": ImageWatcher, "CAM2": ImageWatcher, ...}
        used_cameras (list): Danh sách camera cần chờ
        
    Returns:
        dict: {"CAM1": image_array, "CAM2": image_array, ...}
    """
    images = {}
    log_message(f"[WAITING] Waiting for images from: {used_cameras}")
    print("(Waiting for camera images...)\n")
    
    while len(images) < len(used_cameras):
        # Polling tất cả camera
        for cam in used_cameras:
            if cam in images:
                # Đã có ảnh từ camera này
                continue
            
            # Get new images từ ImageWatcher
            new_images = watchers[cam].get_new_images()
            
            if new_images:
                # Lấy ảnh đầu tiên trong danh sách
                image_path = new_images[0]['temp_path']
                image = cv2.imread(image_path)
                if image is not None:
                    images[cam] = image
                    log_message(f"[GOT] Camera {cam} - image received {image.shape}")
                    print()  # Newline
        
        # Nếu chưa đủ tất cả, chờ 0.5s rồi kiểm tra lại
        if len(images) < len(used_cameras):
            time.sleep(0.5)
    
    log_message(f"[READY] All cameras ready. Processing batch...")
    return images

def main():
    """
    Chỉ xử lý khi tất cả camera có ảnh
    - Khởi tạo ImageWatcher cho mỗi camera
    - Chờ tất cả camera
    - Xử lý 1 batch
    - Print kết quả
    - Lặp lại chờ batch tiếp
    """
    try:
        # Load config
        log_message("="*70)
        log_message("SYSTEM STARTED - Waiting for camera images")
        log_message(f"Product code: {PRODUCT_CODE}")
        log_message("="*70)
        
        roi_rules = load_product_csv(CSV_PATH)
        used_cameras = get_used_cameras(roi_rules)
        
        # Khởi tạo ImageWatcher cho mỗi camera
        watchers = {}
        for cam in used_cameras:
            watch_folder = f"images/temp/{cam}"
            temp_folder = f"images/process_temp/{cam}"
            os.makedirs(watch_folder, exist_ok=True)
            os.makedirs(temp_folder, exist_ok=True)
            watchers[cam] = ImageWatcher(watch_folder, temp_folder, poll_interval=0.5)
            log_message(f"[INIT] ImageWatcher for {cam}: {watch_folder}")
        
        batch_num = 0
        
        # Vòng lặp: chờ ảnh → xử lý → print → lặp lại
        while True:
            batch_num += 1
            
            # Chờ tất cả camera có ảnh
            images = wait_for_all_cameras_images(watchers, used_cameras)
            
            roi_results = []
            batch_start_time = time.time()

            # Xử lý từng camera
            for cam in used_cameras:
                image = images[cam]
                log_message(f"[BATCH {batch_num}] Processing {cam} - {image.shape}")
                
                cam_rules = [r for r in roi_rules if r["camera"] == cam]

                for rule in cam_rules:
                    try:
                        roi_data = prepare_roi_data(image, rule)
                        model = get_model(rule["model_name"])

                        # Lay offset tu detect_roi
                        detect_roi = rule["detect_roi"]
                        roi_offset = (detect_roi[0], detect_roi[1])

                        detect_result = detect_object(
                            model,
                            roi_data["detect_image"],
                            rule["class_id"],
                            rule["confidence"],
                            roi_offset=roi_offset
                        )

                        passed, reason = compare_detection(
                            detect_result,
                            rule["compare_roi"]
                        )

                        # Ve hinh anh ket qua
                        roi_data["rule"] = rule
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = f"{OUTPUT_DIR}/{timestamp}_{rule['roi_id']}_{cam}.jpg"
                        visualize_detection_result(image, roi_data, detect_result, passed, output_path)

                        status_str = "OK" if passed else "NG"
                        log_message(f"  {rule['roi_id']}: {status_str} ({reason})")

                        roi_results.append({
                            "roi_id": rule["roi_id"],
                            "camera": cam,
                            "pass": passed,
                            "reason": reason
                        })
                    except Exception as e:
                        log_message(f"  [ERROR] {rule['roi_id']}: {str(e)}")
                        roi_results.append({
                            "roi_id": rule["roi_id"],
                            "camera": cam,
                            "pass": False,
                            "reason": f"ERROR: {str(e)}"
                        })

            # Tính kết quả batch
            final_status = aggregate_results(roi_results)
            batch_time = time.time() - batch_start_time
            
            # Print kết quả
            print("\n" + "="*70)
            print(f"BATCH #{batch_num} - RESULT: {final_status} | Time: {batch_time:.2f}s")
            print("="*70)
            for result in roi_results:
                status_icon = "[OK]" if result["pass"] else "[NG]"
                camera = result.get("camera", "?")
                roi_id = result.get("roi_id", "?")
                reason = result.get("reason", "")
                print(f"  {status_icon} {roi_id} ({camera}): {reason}")
            print("="*70)
            print(f"Visualizations saved to: {os.path.abspath(OUTPUT_DIR)}\n")
            
            log_message(f"[BATCH {batch_num}] COMPLETED - RESULT: {final_status}")
            log_message("")
    
    except FileNotFoundError as e:
        log_message(f"[FATAL] {e}")
        print(f"\n[ERROR] File not found: {e}\n")
    except KeyboardInterrupt:
        log_message("="*70)
        log_message(f"SYSTEM STOPPED (User interrupted after {batch_num} batches)")
        log_message("="*70)
        print("\n[INFO] System stopped by user\n")
    except Exception as e:
        log_message(f"[FATAL] {str(e)}")
        print(f"\n[ERROR] {str(e)}\n")


if __name__ == "__main__":
    main()
