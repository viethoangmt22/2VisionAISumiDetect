"""
Module: result_gui
Chuc nang: Hien thi ket qua detect len cua so GUI (OpenCV)
    - Crop tung ROI, ve bbox + compare_roi + OK/NG
    - Xep thanh luoi (grid)
    - Header: batch info, product code, final result
    - Footer: thong ke OK/NG qua cac batch
    - Non-blocking: gui.show() goi cv2.waitKey() de GUI khong bi do

Khong phu thuoc module khac trong project.
Chi import: cv2, numpy
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Optional


class ResultGUI:
    """
    GUI hien thi ket qua detect cua tung batch.
    
    Cach dung:
        gui = ResultGUI()
        gui.start()
        
        # Trong vong lap chinh:
        while True:
            action = gui.show(wait_time=30)   # Giu GUI song
            if action == 'quit':
                break
            
            # Khi co ket qua batch:
            gui.update(roi_items, final_status, batch_info)
        
        gui.close()
    """

    def __init__(self, window_name: str = "VisionAI - ROI Result", 
                 max_history: int = 10):
        """
        Args:
            window_name: Ten cua so hien thi
            max_history: So batch luu lich su (hien thi o footer)
        """
        self.window_name = window_name
        self.max_history = max_history
        
        # Trang thai
        self.is_running = False
        self.paused = False
        
        # Du lieu hien thi hien tai
        self._display_image = None    # Anh dang hien thi
        self._roi_items = []          # Danh sach ROI items cua batch hien tai
        self._final_status = "N/A"    # OK / NG
        self._batch_info = {}         # {"batch_num": 1, "product_code": "ABC123", ...}
        
        # Thong ke
        self.total_batches = 0
        self.total_ok = 0
        self.total_ng = 0
        
        # Mau sac (BGR)
        self.COLOR_OK = (0, 200, 0)       # Xanh la
        self.COLOR_NG = (0, 0, 220)       # Do
        self.COLOR_BBOX = (0, 255, 255)   # Vang
        self.COLOR_COMPARE = (255, 200, 0) # Cyan nhat
        self.COLOR_HEADER = (50, 50, 50)  # Xam dam
        self.COLOR_FOOTER = (40, 40, 40)  # Xam dam
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
    
    # ==================================================================
    # PUBLIC METHODS
    # ==================================================================
    
    def start(self) -> None:
        """Tao cua so OpenCV"""
        self.is_running = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        print(f"[GUI] Window started: {self.window_name}")
        print(f"[GUI] Controls: Q = Quit | SPACE = Pause/Resume")
    
    def update(self, roi_items: List[Dict], final_status: str, 
               batch_info: Dict) -> None:
        """
        Cap nhat GUI voi ket qua batch moi.
        
        Args:
            roi_items: Danh sach ROI, moi item la dict:
                {
                    "roi_id": "ROI_01",
                    "camera": "CAM1",
                    "crop_image": numpy array (anh crop tu detect_roi),
                    "passed": True/False,
                    "reason": "OK" / "NOT_FOUND" / "OUT_OF_COMPARE_ROI",
                    "detect_result": {"found": bool, "bbox": tuple, "confidence": float},
                    "detect_roi": (x1, y1, x2, y2),
                    "compare_roi": (x1, y1, x2, y2),
                }
            final_status: "OK" hoac "NG"
            batch_info: {"batch_num": int, "product_code": str, "batch_time": float}
        """
        self._roi_items = roi_items
        self._final_status = final_status
        self._batch_info = batch_info
        
        # Cap nhat thong ke
        self.total_batches += 1
        if final_status == "OK":
            self.total_ok += 1
        else:
            self.total_ng += 1
        
        # Tao anh hien thi
        self._display_image = self._create_display()
    
    def show(self, wait_time: int = 30) -> Optional[str]:
        """
        Hien thi GUI va doc phim nhan. GOI MOI VONG LAP.
        
        Args:
            wait_time: Thoi gian cho phim nhan (ms). 
                       30ms ~ 33 FPS, du muot.
        
        Returns:
            'quit'   - Nguoi dung nhan Q hoac ESC
            'pause'  - Nhan SPACE (dang chay → pause)
            'resume' - Nhan SPACE (dang pause → chay lai)
            None     - Khong co gi
        """
        if not self.is_running:
            return None
        
        # Tao anh hien thi
        if self._display_image is not None:
            display = self._display_image
        else:
            display = self._create_waiting_screen()
        
        # Hien thi
        cv2.imshow(self.window_name, display)
        
        # Doc phim nhan
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q') or key == ord('Q') or key == 27:  # Q hoac ESC
            return 'quit'
        elif key == ord(' '):  # SPACE
            self.paused = not self.paused
            state = "PAUSED" if self.paused else "RESUMED"
            print(f"[GUI] {state}")
            return 'pause' if self.paused else 'resume'
        
        return None
    
    def close(self) -> None:
        """Dong cua so"""
        self.is_running = False
        cv2.destroyAllWindows()
        print(f"[GUI] Window closed")
    
    # ==================================================================
    # PRIVATE: Tao anh hien thi
    # ==================================================================
    
    def _create_waiting_screen(self) -> np.ndarray:
        """Tao anh cho khi chua co du lieu"""
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Header
        cv2.rectangle(canvas, (0, 0), (1280, 60), self.COLOR_HEADER, -1)
        cv2.putText(canvas, "VisionAI Sumi Detect", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLOR_WHITE, 2)
        
        # Waiting message
        cv2.putText(canvas, "Waiting for images...", 
                    (420, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
        cv2.putText(canvas, "Q = Quit | SPACE = Pause", 
                    (440, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        return canvas
    
    def _create_display(self) -> np.ndarray:
        """Tao anh hien thi hoan chinh: header + grid ROI + footer"""
        
        if not self._roi_items:
            return self._create_waiting_screen()
        
        # --- Buoc 1: Ve tung ROI crop ---
        roi_images = []
        for item in self._roi_items:
            roi_img = self._draw_roi_crop(item)
            roi_images.append(roi_img)
        
        # --- Buoc 2: Xep luoi ---
        grid = self._arrange_grid(roi_images)
        grid_h, grid_w = grid.shape[:2]
        
        # --- Buoc 3: Ghep header + grid + footer ---
        header_h = 60
        footer_h = 50
        total_h = header_h + grid_h + footer_h
        total_w = max(grid_w, 800)  # Toi thieu 800px de header du cho
        
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        
        # Header
        self._draw_header(canvas, total_w, header_h)
        
        # Grid (dat giua neu grid nho hon canvas)
        x_offset = (total_w - grid_w) // 2
        canvas[header_h:header_h + grid_h, x_offset:x_offset + grid_w] = grid
        
        # Footer
        self._draw_footer(canvas, total_w, total_h, footer_h)
        
        # Pause indicator
        if self.paused:
            cv2.putText(canvas, "PAUSED", 
                        (total_w - 160, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 2)
        
        return canvas
    
    def _draw_header(self, canvas: np.ndarray, width: int, height: int) -> None:
        """Ve header: batch info + final result"""
        cv2.rectangle(canvas, (0, 0), (width, height), self.COLOR_HEADER, -1)
        
        # Final result (ben trai)
        status = self._final_status
        color = self.COLOR_OK if status == "OK" else self.COLOR_NG
        
        # Background cho status
        cv2.rectangle(canvas, (10, 8), (140, 52), color, -1)
        cv2.putText(canvas, status, 
                    (30, 43), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.COLOR_BLACK, 3)
        
        # Batch info (ben phai)
        batch_num = self._batch_info.get("batch_num", "?")
        product = self._batch_info.get("product_code", "?")
        batch_time = self._batch_info.get("batch_time", 0)
        
        info_text = f"Batch #{batch_num} | Product: {product} | Time: {batch_time:.2f}s"
        cv2.putText(canvas, info_text, 
                    (160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_WHITE, 1)
    
    def _draw_footer(self, canvas: np.ndarray, width: int, 
                     total_h: int, footer_h: int) -> None:
        """Ve footer: thong ke"""
        y_start = total_h - footer_h
        cv2.rectangle(canvas, (0, y_start), (width, total_h), self.COLOR_FOOTER, -1)
        
        stats = (f"Total Batches: {self.total_batches} | "
                 f"OK: {self.total_ok} | NG: {self.total_ng}")
        cv2.putText(canvas, stats, 
                    (20, y_start + 32), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, self.COLOR_WHITE, 1)
        
        # Controls hint (ben phai)
        cv2.putText(canvas, "Q=Quit | SPACE=Pause", 
                    (width - 280, y_start + 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (120, 120, 120), 1)
    
    # ==================================================================
    # PRIVATE: Ve tung ROI crop
    # ==================================================================
    
    def _draw_roi_crop(self, item: Dict) -> np.ndarray:
        """
        Ve ket qua len anh crop cua 1 ROI.
        
        item:
            roi_id, camera, crop_image, passed, reason,
            detect_result, detect_roi, compare_roi
        """
        crop = item.get("crop_image")
        if crop is None or crop.size == 0:
            # Tao anh den neu khong co crop
            result_img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.putText(result_img, "NO IMAGE", (30, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            return result_img
        
        result_img = crop.copy()
        h, w = result_img.shape[:2]
        
        roi_id = item.get("roi_id", "?")
        camera = item.get("camera", "?")
        passed = item.get("passed", False)
        reason = item.get("reason", "")
        detect_result = item.get("detect_result", {})
        detect_roi = item.get("detect_roi", (0, 0, 0, 0))
        compare_roi = item.get("compare_roi", (0, 0, 0, 0))
        
        # Offset: detect_roi top-left la goc (0,0) cua anh crop
        offset_x = detect_roi[0]
        offset_y = detect_roi[1]
        
        # Mau theo ket qua
        color = self.COLOR_OK if passed else self.COLOR_NG
        
        # --- 1. Ve border (vien xanh/do) ---
        cv2.rectangle(result_img, (0, 0), (w - 1, h - 1), color, 3)
        
        # --- 2. Ve ROI ID + status (goc tren trai) ---
        status_text = "OK" if passed else "NG"
        label = f"{roi_id} ({camera}): {status_text}"
        
        # Background cho label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.6, 2)[0]
        cv2.rectangle(result_img, (0, 0), (text_size[0] + 10, text_size[1] + 14), color, -1)
        cv2.putText(result_img, label, (5, text_size[1] + 7), font, 0.6, self.COLOR_BLACK, 2)
        
        # --- 3. Ve compare ROI (net dut) ---
        cx1 = compare_roi[0] - offset_x
        cy1 = compare_roi[1] - offset_y
        cx2 = compare_roi[2] - offset_x
        cy2 = compare_roi[3] - offset_y
        self._draw_dashed_rect(result_img, (cx1, cy1), (cx2, cy2), self.COLOR_COMPARE, 2)
        
        # --- 4. Ve bbox neu tim thay ---
        if detect_result.get("found", False):
            bbox = detect_result["bbox"]
            confidence = detect_result.get("confidence", 0.0)
            
            # Chuyen toa do tuyet doi → toa do crop
            bx1 = bbox[0] - offset_x
            by1 = bbox[1] - offset_y
            bx2 = bbox[2] - offset_x
            by2 = bbox[3] - offset_y
            
            # Ve bbox
            cv2.rectangle(result_img, (bx1, by1), (bx2, by2), self.COLOR_BBOX, 2)
            
            # Ve confidence (duoi bbox)
            conf_text = f"Conf: {confidence:.2f}"
            cv2.putText(result_img, conf_text, (bx1, by2 + 18), 
                        font, 0.5, self.COLOR_BBOX, 1)
            
            # Ve center point
            cx_pt = (bx1 + bx2) // 2
            cy_pt = (by1 + by2) // 2
            cv2.circle(result_img, (cx_pt, cy_pt), 4, color, -1)
        
        # --- 5. Ve reason (goc duoi) ---
        if reason:
            cv2.putText(result_img, reason, (5, h - 8), 
                        font, 0.4, (200, 200, 200), 1)
        
        return result_img
    
    def _draw_dashed_rect(self, img: np.ndarray, 
                          pt1: Tuple[int, int], pt2: Tuple[int, int],
                          color: Tuple[int, int, int], thickness: int) -> None:
        """Ve hinh chu nhat net dut"""
        x1, y1 = pt1
        x2, y2 = pt2
        dash = 10
        gap = 10
        
        # Canh tren
        for i in range(x1, x2, dash + gap):
            cv2.line(img, (i, y1), (min(i + dash, x2), y1), color, thickness)
        # Canh duoi
        for i in range(x1, x2, dash + gap):
            cv2.line(img, (i, y2), (min(i + dash, x2), y2), color, thickness)
        # Canh trai
        for i in range(y1, y2, dash + gap):
            cv2.line(img, (x1, i), (x1, min(i + dash, y2)), color, thickness)
        # Canh phai
        for i in range(y1, y2, dash + gap):
            cv2.line(img, (x2, i), (x2, min(i + dash, y2)), color, thickness)
    
    # ==================================================================
    # PRIVATE: Xep luoi
    # ==================================================================
    
    def _arrange_grid(self, roi_images: List[np.ndarray]) -> np.ndarray:
        """
        Xep danh sach anh ROI thanh luoi (grid).
        
        VD: 3 ROI → 1 hang x 3 cot
            4 ROI → 2 hang x 2 cot
            6 ROI → 2 hang x 3 cot
        """
        n = len(roi_images)
        
        if n == 0:
            return np.zeros((200, 400, 3), dtype=np.uint8)
        
        # Tinh so hang x cot
        if n <= 3:
            rows, cols = 1, n
        elif n <= 4:
            rows, cols = 2, 2
        elif n <= 6:
            rows, cols = 2, 3
        elif n <= 9:
            rows, cols = 3, 3
        else:
            cols = 4
            rows = (n + cols - 1) // cols  # Lam tron len
        
        # Kich thuoc o lon nhat
        max_h = max(img.shape[0] for img in roi_images)
        max_w = max(img.shape[1] for img in roi_images)
        
        padding = 8
        
        canvas_h = rows * max_h + (rows - 1) * padding
        canvas_w = cols * max_w + (cols - 1) * padding
        
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        for idx, roi_img in enumerate(roi_images):
            row = idx // cols
            col = idx % cols
            
            y_start = row * (max_h + padding)
            x_start = col * (max_w + padding)
            
            h, w = roi_img.shape[:2]
            
            # Dat vao giua o
            y_center = y_start + (max_h - h) // 2
            x_center = x_start + (max_w - w) // 2
            
            # Dam bao khong vuot khoi canvas
            if (y_center + h <= canvas_h) and (x_center + w <= canvas_w):
                canvas[y_center:y_center + h, x_center:x_center + w] = roi_img
        
        return canvas


# ============================================================================
# TEST
# ============================================================================

def main():
    """Test nhanh GUI voi du lieu gia"""
    print("[TEST] ResultGUI\n")
    
    # Tao anh gia
    fake_crop_1 = np.ones((200, 300, 3), dtype=np.uint8) * 80
    cv2.rectangle(fake_crop_1, (50, 50), (150, 150), (0, 255, 0), 2)
    
    fake_crop_2 = np.ones((200, 300, 3), dtype=np.uint8) * 60
    cv2.rectangle(fake_crop_2, (30, 30), (180, 180), (0, 0, 255), 2)
    
    fake_crop_3 = np.ones((200, 300, 3), dtype=np.uint8) * 100
    
    # Tao ROI items
    roi_items = [
        {
            "roi_id": "ROI_01", "camera": "CAM1",
            "crop_image": fake_crop_1,
            "passed": True, "reason": "OK",
            "detect_result": {"found": True, "bbox": (150, 150, 250, 250), "confidence": 0.92},
            "detect_roi": (100, 100, 400, 300),
            "compare_roi": (120, 120, 380, 280),
        },
        {
            "roi_id": "ROI_02", "camera": "CAM2",
            "crop_image": fake_crop_2,
            "passed": False, "reason": "NOT_FOUND",
            "detect_result": {"found": False, "bbox": None, "confidence": 0.0},
            "detect_roi": (50, 50, 350, 250),
            "compare_roi": (70, 70, 330, 230),
        },
        {
            "roi_id": "ROI_03", "camera": "CAM3",
            "crop_image": fake_crop_3,
            "passed": True, "reason": "OK",
            "detect_result": {"found": True, "bbox": (160, 160, 260, 260), "confidence": 0.88},
            "detect_roi": (100, 100, 400, 300),
            "compare_roi": (110, 110, 390, 290),
        },
    ]
    
    # Test GUI
    gui = ResultGUI()
    gui.start()
    
    # Simulate batch 1
    gui.update(roi_items, "NG", {
        "batch_num": 1,
        "product_code": "ABC123x",
        "batch_time": 1.23,
    })
    
    print("[INFO] Nhan Q de thoat, SPACE de pause/resume")
    
    while True:
        action = gui.show(wait_time=30)
        if action == 'quit':
            break
    
    gui.close()
    print("[DONE] Test completed!")


if __name__ == "__main__":
    main()
