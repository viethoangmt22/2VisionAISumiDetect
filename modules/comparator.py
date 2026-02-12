def is_center_inside_roi(bbox, roi):
    """
    Kiểm tra tâm của bounding box có nằm trong vùng ROI hay không.
    bbox: [x1, y1, x2, y2]
    roi:  [rx1, ry1, rx2, ry2]
    """
    bx1, by1, bx2, by2 = bbox
    rx1, ry1, rx2, ry2 = roi

    # Tính toán tọa độ tâm (center point)
    center_x = (bx1 + bx2) / 2
    center_y = (by1 + by2) / 2

    # Kiểm tra tâm có nằm trong giới hạn của ROI
    return rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2


def compare_detection(detect_result, compare_roi):
    """
    Hàm chính để so sánh kết quả detect với vùng ROI.
    """
    if not detect_result.get("found", False):
        return False, "NOT_FOUND"

    bbox = detect_result.get("bbox")
    if not bbox:
        return False, "INVALID_BBOX"

    if not is_center_inside_roi(bbox, compare_roi):
        return False, "OUT_OF_COMPARE_ROI"

    return True, "OK"


# --- PHẦN TEST NHANH ---
if __name__ == "__main__":
    # Vùng ROI ví dụ (Hình vuông từ 100 đến 500)
    roi_area = [100, 100, 500, 500]

    test_cases = [
        {
            "name": "Trường hợp 1: Tâm nằm trong ROI (Hợp lệ)",
            "detect": {"found": True, "bbox": [80, 80, 150, 150]}, # Tâm là (115, 115) -> Trong ROI
        },
        {
            "name": "Trường hợp 2: Toàn bộ box ở ngoài (Không hợp lệ)",
            "detect": {"found": True, "bbox": [0, 0, 50, 50]},    # Tâm là (25, 25) -> Ngoài ROI
        },
        {
            "name": "Trường hợp 3: Không tìm thấy đối tượng",
            "detect": {"found": False, "bbox": None},
        },
        {
            "name": "Trường hợp 4: Box lớn trùm ROI nhưng tâm ở ngoài",
            "detect": {"found": True, "bbox": [0, 0, 100, 100]},  # Tâm là (50, 50) -> Ngoài ROI
        }
    ]

    print(f"{'Test Case':<40} | {'Result':<10} | {'Status'}")
    print("-" * 70)
    
    for case in test_cases:
        res, msg = compare_detection(case["detect"], roi_area)
        print(f"{case['name']:<40} | {str(res):<10} | {msg}")