import csv
import os


def _parse_optional_int(value):
    """Parse optional int field - tra ve None neu trong/khong co"""
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _parse_optional_float(value):
    """Parse optional float field - tra ve None neu trong/khong co"""
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def load_product_csv(csv_path):
    """
    Đọc file CSV cấu hình mã hàng.
    
    Input:
        csv_path (str): đường dẫn file CSV
        
    Output:
        roi_rules (list[dict]): danh sách rule ROI
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

    roi_rules = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for index, row in enumerate(reader, start=1):
            try:
                rule = {
                    "roi_id": row["roi_id"].strip(),
                    "camera": row["camera"].strip(),
                    "model_name": row["model_name"].strip(),
                    "class_id": int(row["class_id"]),
                    "class_name": row["class_name"].strip(),

                    # ROI detect
                    "detect_roi": (
                        int(row["detect_x_min"]),
                        int(row["detect_y_min"]),
                        int(row["detect_x_max"]),
                        int(row["detect_y_max"]),
                    ),

                    # ROI compare
                    "compare_roi": (
                        int(row["compare_x_min"]),
                        int(row["compare_y_min"]),
                        int(row["compare_x_max"]),
                        int(row["compare_y_max"]),
                    ),

                    "confidence": float(row["confidence"]),

                    # Keypoint angle check (optional - de trong neu khong can)
                    "keypoint_idx_1": _parse_optional_int(row.get("keypoint_idx_1")),
                    "keypoint_idx_2": _parse_optional_int(row.get("keypoint_idx_2")),
                    "expected_angle": _parse_optional_float(row.get("expected_angle")),
                    "angle_tolerance": _parse_optional_float(row.get("angle_tolerance")),
                }

                roi_rules.append(rule)

            except Exception as e:
                print(f"[ERROR] Dòng {index} bị lỗi: {e}")
                print("Dữ liệu dòng:", row)

    return roi_rules


# ==========================================================
# TEST NHANH MODULE
# ==========================================================
def main():
    """
    Test nhanh file CSV
    """

    # Ví dụ đường dẫn CSV
    csv_path = "config/products/ABC123.csv"

    print("Đang đọc file:", csv_path)

    try:
        rules = load_product_csv(csv_path)

        print("\n===== KẾT QUẢ PARSE =====")
        print(f"Số lượng ROI đọc được: {len(rules)}\n")

        for r in rules:
            print(r)

    except Exception as e:
        print("LỖI:", e)


if __name__ == "__main__":
    main()
