"""
Module: result_visualizer
Chuc nang: Ve bbox + ROI len anh de hien thi ket qua
Khong phu thuoc: Chi import cv2 (opencv-python)
"""

import cv2
import os
from typing import Dict, Tuple, Any, Optional


def draw_roi(image: Any, roi: Tuple[int, int, int, int], 
             color: Tuple[int, int, int], thickness: int = 2, 
             label: str = "") -> None:
    """
    Ve hinh chu nhat ROI len anh
    
    Args:
        image: Anh (numpy array tu cv2.imread)
        roi: Toa do (x_min, y_min, x_max, y_max)
        color: Mau BGR (B, G, R), VD: (0, 255, 0) = green
        thickness: Do day duong ve
        label: Nhan van ban (VD: "detect_roi", "compare_roi")
    """
    x_min, y_min, x_max, y_max = roi
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        # Ve nhan tren canh trai-tren
        bg_x1 = x_min
        bg_y1 = y_min - text_size[1] - 5
        bg_x2 = x_min + text_size[0] + 5
        bg_y2 = y_min
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(image, label, (x_min + 2, y_min - 5), font, 
                   font_scale, (0, 0, 0), font_thickness)


def draw_bbox(image: Any, bbox: Tuple[int, int, int, int], 
              color: Tuple[int, int, int], thickness: int = 2, 
              confidence: float = 0.0) -> None:
    """
    Ve hinh chu nhat bbox tim duoc
    
    Args:
        image: Anh
        bbox: Toa do (x_min, y_min, x_max, y_max)
        color: Mau BGR
        thickness: Do day duong ve
        confidence: Diem tin cay (0.0-1.0), se hien thi tren anh
    """
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    
    # Ve diem tin cay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    conf_text = f"Conf: {confidence:.2f}"
    text_size = cv2.getTextSize(conf_text, font, font_scale, font_thickness)[0]
    
    bg_x1 = x_min
    bg_y1 = y_max + 5
    bg_x2 = x_min + text_size[0] + 5
    bg_y2 = y_max + 5 + text_size[1] + 5
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.putText(image, conf_text, (x_min + 2, y_max + 20), font, 
               font_scale, (0, 0, 0), font_thickness)


def draw_status(image: Any, status: str, roi_id: str = "") -> None:
    """
    Ve trang thai OK/NG len goc anh
    
    Args:
        image: Anh
        status: "OK" hoac "NG" hoac chi tiet khac
        roi_id: ID roi (tuong ung)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    
    # Chon mau theo status
    if "OK" in status:
        color_bg = (0, 255, 0)  # Green for OK
        color_text = (0, 0, 0)
    else:
        color_bg = (0, 0, 255)  # Red for NG
        color_text = (255, 255, 255)
    
    # Ve text status
    status_text = status if not roi_id else f"{roi_id}: {status}"
    text_size = cv2.getTextSize(status_text, font, font_scale, font_thickness)[0]
    
    x, y = 10, 40
    cv2.rectangle(image, (x, y - text_size[1] - 10), 
                 (x + text_size[0] + 10, y + 10), color_bg, -1)
    cv2.putText(image, status_text, (x + 5, y), font, font_scale, 
               color_text, font_thickness)


def draw_keypoints_and_angle(image: Any, detect_result: Dict[str, Any],
                              angle_info: Dict[str, Any],
                              color_kp: Tuple[int, int, int] = (0, 255, 255),
                              color_line: Tuple[int, int, int] = (255, 0, 255)) -> None:
    """
    Ve keypoints va goc len anh (toa do tuyet doi tren anh goc)
    
    Args:
        image: Anh (se ve truc tiep len anh nay)
        detect_result: Ket qua detect (co "keypoints")
        angle_info: Dict tu compare_angle():
            {"kp1": (x,y), "kp2": (x,y), "measured_angle": float, ...}
        color_kp: Mau cham keypoint (BGR) - Vang
        color_line: Mau duong noi + goc (BGR) - Tim
    """
    if not angle_info:
        return
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 1. Ve tat ca keypoints (cham nho + so thu tu)
    keypoints = detect_result.get("keypoints", [])
    if keypoints:
        for i, kp in enumerate(keypoints):
            kx, ky = int(kp[0]), int(kp[1])
            cv2.circle(image, (kx, ky), 4, color_kp, -1)
            cv2.putText(image, str(i), (kx + 6, ky - 6), font, 0.4, color_kp, 1)
    
    # 2. Ve 2 keypoints dung cho angle (cham to hon)
    kp1 = (int(angle_info["kp1"][0]), int(angle_info["kp1"][1]))
    kp2 = (int(angle_info["kp2"][0]), int(angle_info["kp2"][1]))
    cv2.circle(image, kp1, 8, color_line, -1)
    cv2.circle(image, kp2, 8, color_line, -1)
    
    # 3. Ve duong noi voi mui ten (kp1 -> kp2)
    cv2.arrowedLine(image, kp1, kp2, color_line, 2, tipLength=0.15)
    
    # 4. Ve thong tin goc o giua duong noi
    mid_x = (kp1[0] + kp2[0]) // 2
    mid_y = (kp1[1] + kp2[1]) // 2
    
    measured = angle_info["measured_angle"]
    expected = angle_info["expected_angle"]
    tolerance = angle_info["angle_tolerance"]
    
    angle_text = f"{measured:.1f} (exp:{expected:.0f}+/-{tolerance:.0f})"
    cv2.putText(image, angle_text, (mid_x - 50, mid_y - 15), font, 0.5, color_line, 2)


def visualize_detection_result(
    image: Any,
    roi_data: Dict[str, Any],
    detect_result: Dict[str, Any],
    passed: bool,
    output_path: str = None,
    angle_info: Dict[str, Any] = None
) -> Any:
    """
    Ve toan bo ket qua detection: detect_roi, compare_roi, bbox, status
    
    Args:
        image: Anh goc
        roi_data: Du lieu ROI chua:
            {
                "roi_id": str,
                "camera": str,
                "detect_image": anh (crop roi),
                "compare_roi": (x_min, y_min, x_max, y_max),
                "rule": dict
            }
        detect_result: Ket qua detection:
            {
                "found": bool,
                "bbox": (x_min, y_min, x_max, y_max) if found else None,
                "confidence": float if found else 0
            }
        passed: True neu OK, False neu NG
        output_path: Duong dan luu anh (neu None thi chi tra ve anh)
        
    Returns:
        Anh da ve (numpy array)
    """
    # Copy anh de khong chinh sua anh goc
    viz_image = image.copy()
    
    rule = roi_data["rule"]
    roi_id = roi_data["roi_id"]
    detect_roi = rule["detect_roi"]
    compare_roi = roi_data["compare_roi"]
    
    # Ve detect_roi (xanh)
    draw_roi(viz_image, detect_roi, (0, 255, 0), thickness=2, label="detect")
    
    # Ve compare_roi (do)
    draw_roi(viz_image, compare_roi, (0, 0, 255), thickness=2, label="compare")
    
    # Ve bbox neu tim thay
    if detect_result["found"]:
        bbox = detect_result["bbox"]
        confidence = detect_result["confidence"]
        draw_bbox(viz_image, bbox, (0, 255, 255), thickness=2, confidence=confidence)
    
    # Ve keypoints va goc (neu co)
    if angle_info:
        draw_keypoints_and_angle(viz_image, detect_result, angle_info)
    
    # Ve status OK/NG
    status_text = "OK" if passed else "NG"
    draw_status(viz_image, status_text, roi_id)
    
    # Luu file neu co duong dan
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        cv2.imwrite(output_path, viz_image)
        print(f"[OK] Saved visualization: {output_path}")
    
    return viz_image


# ============================================================================
# TEST
# ============================================================================

def create_sample_image(width: int = 640, height: int = 480) -> Any:
    """Tao anh mau de test"""
    # Tao anh trang
    image = cv2.cvtColor(
        cv2.create(height, width, cv2.CV_8UC3),
        cv2.COLOR_RGB2BGR
    )
    # Tao gradient
    for i in range(height):
        for j in range(width):
            image[i, j] = [50 + i % 50, 100 + j % 50, 150]
    return image

def main():
    """Test function"""
    import numpy as np
    
    print("[TEST] Result Visualizer\n")
    
    # Tao anh test
    print("[1] Creating test image...")
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Tao du lieu ROI
    roi_data = {
        "roi_id": "roi_001",
        "camera": "CAM1",
        "detect_image": image[100:400, 150:550],
        "compare_roi": (100, 150, 300, 250),
        "rule": {
            "detect_roi": (100, 100, 400, 400),
            "compare_roi": (120, 120, 380, 380),
            "class_id": 0,
            "class_name": "mark"
        }
    }
    
    # Test case 1: Tim thay object
    print("[2] Test case 1: Found object (PASS)")
    detect_result = {
        "found": True,
        "bbox": (150, 150, 250, 250),
        "confidence": 0.85
    }
    viz_image = visualize_detection_result(image, roi_data, detect_result, 
                                           passed=True, 
                                           output_path="output/test_ok.jpg")
    print("[OK] Visualization saved\n")
    
    # Test case 2: Khong tim thay
    print("[3] Test case 2: Object not found (FAIL)")
    detect_result = {
        "found": False,
        "bbox": None,
        "confidence": 0.0
    }
    viz_image = visualize_detection_result(image, roi_data, detect_result, 
                                           passed=False, 
                                           output_path="output/test_ng.jpg")
    print("[OK] Visualization saved\n")
    
    # Test case 3: Tim thay nhung o ngoai roi
    print("[4] Test case 3: Found but outside compare ROI (FAIL)")
    detect_result = {
        "found": True,
        "bbox": (350, 350, 450, 450),
        "confidence": 0.75
    }
    viz_image = visualize_detection_result(image, roi_data, detect_result, 
                                           passed=False, 
                                           output_path="output/test_out_of_roi.jpg")
    print("[OK] Visualization saved\n")
    
    print("[DONE] Tests completed!")


if __name__ == "__main__":
    main()
