from modules.csv_loader import load_product_csv
from modules.camera_selector import get_used_cameras
from modules.image_watcher import ImageWatcher
from modules.roi_manager import prepare_roi_data
from modules.model_manager import get_model
from modules.detector import detect_object
from modules.comparator import compare_detection
from modules.result_manager import aggregate_results
from modules.result_visualizer import visualize_detection_result
from modules.camera_config_loader import load_camera_config, get_camera_folder, print_camera_config_summary
from modules.result_gui import ResultGUI
from modules.com_output import COMOutput
import os
import time
from datetime import datetime
import cv2

PRODUCT_CODE = "ABC123x"
CSV_PATH = f"config/products/{PRODUCT_CODE}.csv"
OUTPUT_DIR = "output/results"
LOG_DIR = "output/logs"

# === COM Output Config ===
ENABLE_COM_OUTPUT = True     # Bat/tat chuc nang COM (de debug)
COM_PORT = "COM5"            # Cong COM ket noi voi PLC/Arduino
COM_BAUDRATE = 9600          # Toc do baud
COM_RETRY = 3                # So lan thu lai neu that bai

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


def poll_cameras_once(watchers: dict, used_cameras: list, images: dict) -> dict:
    """
    Kiem tra 1 luot tat ca camera (NON-BLOCKING).
    - Khong cho, chi kiem tra co anh moi khong
    - Neu co → doc anh, them vao dict
    - Neu khong → bo qua, tra ve ngay
    
    Args:
        watchers: {"CAM1": ImageWatcher, ...}
        used_cameras: Danh sach camera can cho
        images: Dict anh da thu thap duoc (se duoc cap nhat)
        
    Returns:
        dict: images da duoc cap nhat
    """
    for cam in used_cameras:
        if cam in images:
            continue  # Da co anh tu camera nay
        
        new_images = watchers[cam].get_new_images()
        
        if new_images:
            image_path = new_images[0]['temp_path']
            image = cv2.imread(image_path)
            if image is not None:
                images[cam] = image
                log_message(f"[GOT] Camera {cam} - image received {image.shape}")
    
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
        
        # Load camera config
        log_message("[LOADING] Camera configuration...")
        camera_config = load_camera_config(
            csv_file="config/camera_config.csv",
            create_folders=True,
            verbose=False
        )
        print_camera_config_summary(camera_config)
        
        # Khởi tạo ImageWatcher cho mỗi camera
        watchers = {}
        for cam in used_cameras:
            # Lấy folder từ config
            input_folder = get_camera_folder(cam, camera_config, "input")
            temp_folder = get_camera_folder(cam, camera_config, "temp")
            
            # Kiểm tra camera có được cấu hình không
            if input_folder is None or temp_folder is None:
                log_message(f"[ERROR] Camera {cam} is not configured in camera_config.csv!")
                continue
            
            # Tạo folder nếu chưa có
            os.makedirs(input_folder, exist_ok=True)
            os.makedirs(temp_folder, exist_ok=True)
            
            # Khởi tạo watcher
            watchers[cam] = ImageWatcher(input_folder, temp_folder, poll_interval=0.5)
            log_message(f"[INIT] ImageWatcher for {cam}: {input_folder}")
        
        batch_num = 0
        images = {}  # Thu thap anh dan dan (non-blocking)
        
        # Khoi tao GUI
        gui = ResultGUI(window_name=f"VisionAI - {PRODUCT_CODE}")
        gui.start()
        
        # Khoi tao COM Output
        com_output = COMOutput(
            port=COM_PORT,
            baudrate=COM_BAUDRATE,
            enabled=ENABLE_COM_OUTPUT,
            retry_count=COM_RETRY
        )
        
        log_message(f"[WAITING] Waiting for images from: {used_cameras}")
        
        # === VONG LAP CHINH (Non-blocking) ===
        # Moi vong: poll anh → refresh GUI → xu ly neu du anh
        while True:
            
            # --- Buoc 1: Poll anh (KHONG block) ---
            images = poll_cameras_once(watchers, used_cameras, images)
            
            # --- Buoc 2: Refresh GUI + doc phim (BAT BUOC moi vong) ---
            action = gui.show(wait_time=30)
            if action == 'quit':
                log_message("[GUI] User pressed Quit")
                break
            
            # --- Buoc 3: Chua du anh → quay lai buoc 1 ---
            if len(images) < len(used_cameras):
                continue
            
            # --- Buoc 4: DU ANH → Xu ly batch ---
            # (Logic xu ly GIU NGUYEN 100%)
            batch_num += 1
            log_message(f"[READY] All cameras ready. Processing batch #{batch_num}...")
            
            roi_results = []
            gui_roi_items = []  # Thu thap data cho GUI
            batch_start_time = time.time()

            # Xu ly tung camera
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

                        # Ve hinh anh ket qua (VAN luu file nhu cu)
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
                        
                        # === THEM: Gom data cho GUI ===
                        gui_roi_items.append({
                            "roi_id": rule["roi_id"],
                            "camera": cam,
                            "crop_image": roi_data["detect_image"],
                            "passed": passed,
                            "reason": reason,
                            "detect_result": detect_result,
                            "detect_roi": rule["detect_roi"],
                            "compare_roi": rule["compare_roi"],
                        })
                        
                    except Exception as e:
                        log_message(f"  [ERROR] {rule['roi_id']}: {str(e)}")
                        roi_results.append({
                            "roi_id": rule["roi_id"],
                            "camera": cam,
                            "pass": False,
                            "reason": f"ERROR: {str(e)}"
                        })

            # Tinh ket qua batch
            final_status = aggregate_results(roi_results)
            batch_time = time.time() - batch_start_time
            
            # === THEM: Gui tin hieu COM ===
            com_output.send_result(final_status)
            
            # === THEM: Update GUI ===
            gui.update(gui_roi_items, final_status, {
                "batch_num": batch_num,
                "product_code": PRODUCT_CODE,
                "batch_time": batch_time,
            })
            
            # Print ket qua (GIU NGUYEN)
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
            
            # Reset cho batch tiep theo
            images = {}
            log_message(f"[WAITING] Waiting for images from: {used_cameras}")
    
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
    finally:
        # Dong GUI va COM an toan
        try:
            gui.close()
        except Exception:
            pass
        try:
            com_output.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
