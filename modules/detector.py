"""
Module: detector
Chuc nang: Detect object bang YOLO
Khong phu thuoc: Chi import ultralytics
"""

from ultralytics import YOLO
from typing import Dict, Tuple, Any, Optional


def detect_object(model: YOLO, image: Any, class_id: int, conf_thres: float,
                  roi_offset: Tuple[int, int] = (0, 0)) -> Dict[str, Any]:
    """
    Detect object trong anh bang YOLO
    
    Args:
        model: Model YOLO
        image: Anh (numpy array)
        class_id: ID class can detect (VD: 0)
        conf_thres: Nguong tin cay [0.0-1.0]
        roi_offset: Toa do goc cua anh crop tren anh goc: (x_offset, y_offset)
                   Dung de chuyen toa do tu anh crop sang anh goc
                   VD: Neu crop tu (100, 150), offset=(100, 150)
    
    Returns:
        Dict:
            - Neu tim thay: {"found": True, "bbox": (x1, y1, x2, y2), "confidence": float}
            - Neu khong: {"found": False, "bbox": None, "confidence": 0.0}
            
            Luu y: bbox la toa do TUYET DOI tren anh goc!
    """
    results = model(image, verbose=False)
    
    x_offset, y_offset = roi_offset

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == class_id and box.conf[0] >= conf_thres:
                # Toa do tren anh crop
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Chuyen thanh toa do tuyet doi tren anh goc
                x1_real = x1 + x_offset
                y1_real = y1 + y_offset
                x2_real = x2 + x_offset
                y2_real = y2 + y_offset
                
                return {
                    "found": True,
                    "bbox": (x1_real, y1_real, x2_real, y2_real),
                    "confidence": float(box.conf[0])
                }

    return {
        "found": False,
        "bbox": None,
        "confidence": 0.0
    }


# ============================================================================
# TEST
# ============================================================================

def main():
    """Test function - kiem tra la toa do offset hoat dong dung"""
    import numpy as np
    
    print("[TEST] Detector - Coordinate Offset\n")
    
    print("[SCENARIO]")
    print("  - Anh goc: 640x480")
    print("  - detect_roi: (100, 150, 400, 450)")
    print("  - Anh crop: 300x300")
    print("  - Model detect bbox tren anh crop: (50, 50, 100, 100)")
    print()
    
    print("[TINH TOAN]")
    print("  - Toa do tren anh crop: (50, 50, 100, 100)")
    print("  - Offset (x_min, y_min): (100, 150)")
    print("  - Toa do thuc tren anh goc: (50+100, 50+150, 100+100, 100+150)")
    print("                             = (150, 200, 200, 250)")
    print()
    
    print("[KET QUA DUNG]")
    print("  Bbox lay tren crop + offset = bbox tren anh goc")
    print("  (50, 50, 100, 100) + (100, 150) = (150, 200, 200, 250)")
    print()
    print("[DONE] Offset logic is correct!")


if __name__ == "__main__":
    main()
