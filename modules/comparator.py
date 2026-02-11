def is_bbox_inside_roi(bbox, roi):
    bx1, by1, bx2, by2 = bbox
    rx1, ry1, rx2, ry2 = roi

    return bx1 >= rx1 and by1 >= ry1 and bx2 <= rx2 and by2 <= ry2

def compare_detection(detect_result, compare_roi):
    if not detect_result["found"]:
        return False, "NOT_FOUND"

    if not is_bbox_inside_roi(detect_result["bbox"], compare_roi):
        return False, "OUT_OF_COMPARE_ROI"

    return True, "OK"
