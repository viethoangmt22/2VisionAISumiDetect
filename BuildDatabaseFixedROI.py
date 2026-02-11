import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

class ROIAnnotationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ROI Annotation Tool - YOLO Custom Models")
        self.root.geometry("1400x900")
        
        # Data
        self.image = None
        self.original_image = None
        self.rois = []
        self.roi_counter = 1
        self.drawing = False
        self.start_point = None
        self.current_roi_rect = None
        
        # Dual ROI mode
        self.drawing_stage = 'detect'  # 'detect' or 'compare'
        self.temp_detect_roi = None  # Temporary storage for detect ROI
        
        self.yolo_classes = {}
        self.model_configs = {}
        self.available_models = []
        
        self.current_model = None
        self.current_class = None
        self.current_confidence = 0.7
        self.product_code = ""
        
        # Keypoint mode
        self.keypoint_mode = False
        self.keypoint_idx_1 = 0
        self.keypoint_idx_2 = 1
        self.expected_angle = 0.0
        self.angle_tolerance = 10.0
        
        # Display settings
        self.canvas_width = 900
        self.canvas_height = 700
        self.scale = 1.0
        
        # Zoom and pan settings
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.panning = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Thiết lập giao diện"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # PanedWindow để resize left/right panels
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls với scrollbar
        left_container = ttk.Frame(paned_window, width=200)
        
        # Canvas cho scrollable left panel
        left_canvas = tk.Canvas(left_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        self.scrollable_left_frame = ttk.Frame(left_canvas)
        
        self.scrollable_left_frame.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        
        left_canvas.create_window((0, 0), window=self.scrollable_left_frame, anchor="nw")
        left_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar và canvas
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse wheel
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Right panel - Canvas
        right_panel = ttk.Frame(paned_window)
        
        # Add panels to PanedWindow
        paned_window.add(left_container, weight=0)
        paned_window.add(right_panel, weight=1)
        
        # === LEFT PANEL CONTENT (inside scrollable_left_frame) ===
        left_panel = self.scrollable_left_frame
        
        # 1. Model Management
        model_frame = ttk.LabelFrame(left_panel, text="1. Model Management", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(model_frame, text="📁 Scan Models Folder", 
                  command=self.scan_models_folder).pack(fill=tk.X, pady=2)
        ttk.Button(model_frame, text="➕ Add Single Model (.pt)", 
                  command=self.add_single_model).pack(fill=tk.X, pady=2)
        
        # Model list
        ttk.Label(model_frame, text="Available Models:").pack(anchor=tk.W, pady=(5,0))
        self.model_listbox = tk.Listbox(model_frame, height=4)
        self.model_listbox.pack(fill=tk.X, pady=5)
        self.model_listbox.bind('<<ListboxSelect>>', self.on_model_select)
        
        # Mode Selection
        mode_frame = ttk.LabelFrame(left_panel, text="1b. Detection Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        self.mode_frame = mode_frame  # Save reference for on_mode_change
        
        self.mode_var = tk.StringVar(value="detection")
        ttk.Radiobutton(mode_frame, text="🔍 Detection Only", 
                       variable=self.mode_var, value="detection",
                       command=self.on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="📐 Detection + Keypoint Angle", 
                       variable=self.mode_var, value="keypoint",
                       command=self.on_mode_change).pack(anchor=tk.W)
        
        # ROI Drawing Stage Indicator
        self.roi_stage_frame = ttk.LabelFrame(left_panel, text="1c. Drawing Stage", padding=10)
        self.roi_stage_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stage_label = ttk.Label(self.roi_stage_frame, 
                                     text="Next: Draw DETECT ROI (larger area)",
                                     foreground="blue", font=('Arial', 10, 'bold'))
        self.stage_label.pack(pady=5)
        
        ttk.Label(self.roi_stage_frame, 
                 text="Step 1: Draw DETECT ROI (detection area)\nStep 2: Draw COMPARE ROI (check area)",
                 foreground="gray", font=('Arial', 8)).pack()
        
        # Keypoint settings (initially hidden)
        self.keypoint_settings_frame = ttk.LabelFrame(left_panel, text="Keypoint Settings", padding=10)
        
        ttk.Label(self.keypoint_settings_frame, text="Keypoint Index 1:").pack(anchor=tk.W)
        self.kp_idx_1_var = tk.IntVar(value=0)
        ttk.Spinbox(self.keypoint_settings_frame, from_=0, to=20, 
                   textvariable=self.kp_idx_1_var, width=10).pack(anchor=tk.W, pady=2)
        
        ttk.Label(self.keypoint_settings_frame, text="Keypoint Index 2:").pack(anchor=tk.W, pady=(5,0))
        self.kp_idx_2_var = tk.IntVar(value=1)
        ttk.Spinbox(self.keypoint_settings_frame, from_=0, to=20, 
                   textvariable=self.kp_idx_2_var, width=10).pack(anchor=tk.W, pady=2)
        
        ttk.Label(self.keypoint_settings_frame, text="Expected Angle (degrees):").pack(anchor=tk.W, pady=(5,0))
        self.expected_angle_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(self.keypoint_settings_frame, from_=0, to=360, increment=1,
                   textvariable=self.expected_angle_var, width=10).pack(anchor=tk.W, pady=2)
        
        ttk.Label(self.keypoint_settings_frame, text="Angle Tolerance (degrees):").pack(anchor=tk.W, pady=(5,0))
        self.angle_tolerance_var = tk.DoubleVar(value=10.0)
        ttk.Spinbox(self.keypoint_settings_frame, from_=0, to=180, increment=1,
                   textvariable=self.angle_tolerance_var, width=10).pack(anchor=tk.W, pady=2)
        
        # Add trace callbacks to update instance variables
        self.kp_idx_1_var.trace('w', self.update_keypoint_settings)
        self.kp_idx_2_var.trace('w', self.update_keypoint_settings)
        self.expected_angle_var.trace('w', self.update_keypoint_settings)
        self.angle_tolerance_var.trace('w', self.update_keypoint_settings)
        
        # 2. Image Management
        image_frame = ttk.LabelFrame(left_panel, text="2. Load Image", padding=10)
        image_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(image_frame, text="🖼️ Load Image", 
                  command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(image_frame, text="📄 Load CSV & View ROIs", 
                  command=self.load_csv, style='Info.TButton').pack(fill=tk.X, pady=2)
        
        self.image_label = ttk.Label(image_frame, text="No image loaded", 
                                     foreground="gray")
        self.image_label.pack(pady=5)
        
        # 2b. Zoom Controls
        zoom_frame = ttk.LabelFrame(left_panel, text="2b. Zoom Controls", padding=10)
        zoom_frame.pack(fill=tk.X, pady=(0, 10))
        
        zoom_btn_frame = ttk.Frame(zoom_frame)
        zoom_btn_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(zoom_btn_frame, text="🔍+", width=5,
                  command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_btn_frame, text="🔍-", width=5,
                  command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_btn_frame, text="Reset", width=8,
                  command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        
        self.zoom_label = ttk.Label(zoom_frame, text="Zoom: 100%", foreground="blue")
        self.zoom_label.pack(pady=2)
        
        ttk.Label(zoom_frame, text="💡 Mouse wheel: Zoom\n💡 Right-click drag: Pan", 
                 foreground="gray", font=('Arial', 8)).pack(pady=2)
        
        # 3. Class Selection
        class_frame = ttk.LabelFrame(left_panel, text="3. Select Class", padding=10)
        class_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(class_frame, text="Current Model:").pack(anchor=tk.W)
        self.current_model_label = ttk.Label(class_frame, text="None", 
                                             foreground="blue", font=('Arial', 10, 'bold'))
        self.current_model_label.pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(class_frame, text="Classes:").pack(anchor=tk.W)
        self.class_listbox = tk.Listbox(class_frame, height=6)
        self.class_listbox.pack(fill=tk.X, pady=5)
        self.class_listbox.bind('<<ListboxSelect>>', self.on_class_select)
        
        # 4. Confidence
        conf_frame = ttk.LabelFrame(left_panel, text="4. Confidence", padding=10)
        conf_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.conf_var = tk.DoubleVar(value=0.7)
        self.conf_scale = ttk.Scale(conf_frame, from_=0.0, to=1.0, 
                                    variable=self.conf_var, orient=tk.HORIZONTAL)
        
        self.conf_scale.pack(fill=tk.X)
        
        self.conf_label = ttk.Label(conf_frame, text="0.7")
        self.conf_label.pack()
        self.conf_var.trace('w', self.update_conf_label)
        
        # 5. ROI List
        roi_frame = ttk.LabelFrame(left_panel, text="5. ROI List", padding=10)
        roi_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # ROI listbox with scrollbar
        roi_scroll_frame = ttk.Frame(roi_frame)
        roi_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(roi_scroll_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.roi_listbox = tk.Listbox(roi_scroll_frame, yscrollcommand=scrollbar.set)
        self.roi_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.roi_listbox.yview)
        
        ttk.Button(roi_frame, text="🗑️ Delete Selected ROI", 
                  command=self.delete_selected_roi).pack(fill=tk.X, pady=(5, 0))
        
        # 6. Product Code
        product_frame = ttk.LabelFrame(left_panel, text="6. Product Code", padding=10)
        product_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(product_frame, text="Mã hàng:").pack(anchor=tk.W)
        self.product_code_var = tk.StringVar()
        self.product_code_entry = ttk.Entry(product_frame, textvariable=self.product_code_var)
        self.product_code_entry.pack(fill=tk.X, pady=5)
        ttk.Label(product_frame, text="File sẽ lưu: <mã_hàng>.csv", 
                 foreground="gray", font=('Arial', 8)).pack(anchor=tk.W)
        
        # 7. Export
        export_frame = ttk.LabelFrame(left_panel, text="7. Export", padding=10)
        export_frame.pack(fill=tk.X)
        
        ttk.Button(export_frame, text="💾 Save CSV", 
                  command=self.save_csv, style='Accent.TButton').pack(fill=tk.X)
        
        # === RIGHT PANEL - CANVAS ===
        
        canvas_frame = ttk.LabelFrame(right_panel, text="Draw ROI (Click & Drag)", padding=5)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas
        self.canvas = tk.Canvas(canvas_frame, bg='gray', 
                               width=self.canvas_width, height=self.canvas_height,
                               cursor='cross')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        
        # Zoom and pan events
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        self.canvas.bind('<ButtonPress-3>', self.on_pan_start)
        self.canvas.bind('<B3-Motion>', self.on_pan_move)
        self.canvas.bind('<ButtonRelease-3>', self.on_pan_end)
        
        # Instructions
        instructions = """
        INSTRUCTIONS:
        1. Scan or add YOLO models (.pt files)
        2. Load an image
        3. Select a model → classes will auto-load
        4. Select a class
        5. Click & drag on image to draw ROI
        6. Adjust confidence if needed
        7. Repeat for multiple ROIs
        8. Save to CSV
        
        VIEW MODE:
        - Click "Load CSV & View ROIs" to view existing annotations
        - Load the corresponding image to see ROIs
        """
        
        inst_label = ttk.Label(right_panel, text=instructions, 
                              justify=tk.LEFT, foreground='darkblue')
        inst_label.pack(pady=5)
    
    def update_conf_label(self, *args):
        """Update confidence label"""
        self.conf_label.config(text=f"{self.conf_var.get():.2f}")
        self.current_confidence = round(self.conf_var.get(), 2)
    
    def update_keypoint_settings(self, *args):
        """Update keypoint settings from UI variables"""
        self.keypoint_idx_1 = self.kp_idx_1_var.get()
        self.keypoint_idx_2 = self.kp_idx_2_var.get()
        self.expected_angle = self.expected_angle_var.get()
        self.angle_tolerance = self.angle_tolerance_var.get()
    
    def on_mode_change(self):
        """Handle mode change between detection and keypoint"""
        mode = self.mode_var.get()
        self.keypoint_mode = (mode == "keypoint")
        
        if self.keypoint_mode:
            # Show keypoint settings - pack after mode_frame
            self.keypoint_settings_frame.pack(fill=tk.X, pady=(0, 10), after=self.mode_frame)
            print("✓ Switched to Keypoint Mode - Angle checking enabled")
        else:
            # Hide keypoint settings
            self.keypoint_settings_frame.pack_forget()
            print("✓ Switched to Detection Only Mode")
    
    def scan_models_folder(self):
        """Scan folder for .pt models"""
        folder = filedialog.askdirectory(title="Select Models Folder")
        if not folder:
            return
        
        models_path = Path(folder)
        pt_files = list(models_path.rglob('*.pt'))
        
        if not pt_files:
            messagebox.showwarning("No Models", f"No .pt files found in {folder}")
            return
        
        for pt_file in pt_files:
            if pt_file.parent.name != models_path.name:
                model_name = f"{pt_file.parent.name}/{pt_file.stem}"
            else:
                model_name = pt_file.stem
            
            self.add_model(model_name, str(pt_file))
        
        messagebox.showinfo("Success", f"Loaded {len(pt_files)} model(s)")
    
    def add_single_model(self):
        """Add a single .pt model"""
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        model_name = Path(file_path).stem
        self.add_model(model_name, file_path)
    
    def add_model(self, model_name, model_path):
        """Add model and load classes"""
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            
            if hasattr(model, 'names'):
                classes = model.names
                if isinstance(classes, list):
                    classes = {i: name for i, name in enumerate(classes)}
                
                self.model_configs[model_name] = {
                    'model_path': model_path,
                    'classes': classes
                }
                
                if model_name not in self.available_models:
                    self.available_models.append(model_name)
                    self.model_listbox.insert(tk.END, f"{model_name} ({len(classes)} classes)")
                
                print(f"✓ Added: {model_name} - {len(classes)} classes")
            else:
                messagebox.showerror("Error", f"Model {model_name} has no class information")
                
        except ImportError:
            messagebox.showerror("Error", "Please install ultralytics:\npip install ultralytics")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def on_model_select(self, event):
        """Handle model selection"""
        selection = self.model_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        self.current_model = self.available_models[idx]
        self.yolo_classes = self.model_configs[self.current_model]['classes']
        
        # Update UI
        self.current_model_label.config(text=self.current_model)
        
        # Update class listbox
        self.class_listbox.delete(0, tk.END)
        for class_id, class_name in self.yolo_classes.items():
            self.class_listbox.insert(tk.END, f"{class_id}: {class_name}")
    
    def on_class_select(self, event):
        """Handle class selection"""
        selection = self.class_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        class_id = list(self.yolo_classes.keys())[idx]
        class_name = self.yolo_classes[class_id]
        
        self.current_class = {
            'id': class_id,
            'name': class_name
        }
        
        print(f"Selected class: {class_name} (ID: {class_id})")
    
    def load_image(self):
        """Load image to annotate"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.original_image = cv2.imread(file_path)
        if self.original_image is None:
            messagebox.showerror("Error", "Failed to load image")
            return
        
        self.image = self.original_image.copy()
        self.display_image()
        
        # Update label
        img_h, img_w = self.image.shape[:2]
        self.image_label.config(
            text=f"{Path(file_path).name}\n{img_w}x{img_h}",
            foreground="green"
        )
    
    def display_image(self):
        """Display image on canvas with zoom and pan"""
        if self.image is None:
            return
        
        img = self.image.copy()
        
        # Draw existing ROIs
        for roi in self.rois:
            # Draw DETECT ROI (cyan/blue)
            if 'detect_x_min' in roi:
                cv2.rectangle(img, 
                             (roi['detect_x_min'], roi['detect_y_min']), 
                             (roi['detect_x_max'], roi['detect_y_max']), 
                             (255, 200, 0), 2)  # Cyan
                cv2.putText(img, "DETECT",
                           (roi['detect_x_min'] + 5, roi['detect_y_min'] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
            
            # Draw COMPARE ROI (green)
            if 'compare_x_min' in roi:
                cv2.rectangle(img, 
                             (roi['compare_x_min'], roi['compare_y_min']), 
                             (roi['compare_x_max'], roi['compare_y_max']), 
                             (0, 255, 0), 3)  # Green
                
                label = f"{roi['class_name']} ({roi['confidence']})"
                cv2.rectangle(img,
                             (roi['compare_x_min'], roi['compare_y_min'] - 25),
                             (roi['compare_x_min'] + 200, roi['compare_y_min']),
                             (0, 255, 0), -1)
                cv2.putText(img, label,
                           (roi['compare_x_min'] + 5, roi['compare_y_min'] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:
                # Old format - single ROI
                cv2.rectangle(img, 
                             (roi['x_min'], roi['y_min']), 
                             (roi['x_max'], roi['y_max']), 
                             (0, 255, 0), 2)
                
                label = f"{roi['class_name']} ({roi['confidence']})"
                cv2.rectangle(img,
                             (roi['x_min'], roi['y_min'] - 25),
                             (roi['x_min'] + 200, roi['y_min']),
                             (0, 255, 0), -1)
                cv2.putText(img, label,
                           (roi['x_min'] + 5, roi['y_min'] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw temporary detect ROI if in compare stage
        if self.temp_detect_roi is not None and self.drawing_stage == 'compare':
            cv2.rectangle(img,
                         (self.temp_detect_roi['x_min'], self.temp_detect_roi['y_min']),
                         (self.temp_detect_roi['x_max'], self.temp_detect_roi['y_max']),
                         (255, 200, 0), 2)  # Cyan
            cv2.putText(img, "DETECT (temp)",
                       (self.temp_detect_roi['x_min'] + 5, self.temp_detect_roi['y_min'] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calculate base scale to fit canvas
        h, w = img_rgb.shape[:2]
        base_scale = min(self.canvas_width / w, self.canvas_height / h)
        
        # Apply zoom level
        self.scale = base_scale * self.zoom_level
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(img_resized)
        self.photo = ImageTk.PhotoImage(img_pil)
        
        # Display with offset for panning
        self.canvas.delete('all')
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.photo)
    
    def canvas_to_image_coords(self, x, y):
        """Convert canvas coordinates to original image coordinates accounting for zoom and pan"""
        img_x = (x - self.offset_x) / self.scale
        img_y = (y - self.offset_y) / self.scale
        return int(img_x), int(img_y)
    
    def on_mouse_down(self, event):
        """Mouse down event"""
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.current_class is None:
            messagebox.showwarning("Warning", "Please select a class first")
            return
        
        self.drawing = True
        self.start_point = self.canvas_to_image_coords(event.x, event.y)
    
    def on_mouse_move(self, event):
        """Mouse move event"""
        if not self.drawing:
            return
        
        # Remove previous rectangle
        if self.current_roi_rect:
            self.canvas.delete(self.current_roi_rect)
        
        # Choose color based on stage
        color = 'cyan' if self.drawing_stage == 'detect' else 'green'
        
        # Draw current rectangle (accounting for offset)
        x1 = int(self.start_point[0] * self.scale + self.offset_x)
        y1 = int(self.start_point[1] * self.scale + self.offset_y)
        self.current_roi_rect = self.canvas.create_rectangle(
            x1, y1, event.x, event.y,
            outline=color, width=3
        )
    
    def on_mouse_up(self, event):
        """Mouse up event - handles dual ROI drawing"""
        if not self.drawing:
            return
        
        self.drawing = False
        
        end_point = self.canvas_to_image_coords(event.x, event.y)
        
        # Calculate ROI
        x_min = min(self.start_point[0], end_point[0])
        y_min = min(self.start_point[1], end_point[1])
        x_max = max(self.start_point[0], end_point[0])
        y_max = max(self.start_point[1], end_point[1])
        
        # Validate ROI size
        if x_max - x_min < 10 or y_max - y_min < 10:
            messagebox.showwarning("Warning", "ROI too small")
            self.display_image()
            return
        
        # Handle based on drawing stage
        if self.drawing_stage == 'detect':
            # Save detect ROI temporarily
            self.temp_detect_roi = {
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max
            }
            
            # Move to compare stage
            self.drawing_stage = 'compare'
            self.stage_label.config(
                text="Next: Draw COMPARE ROI (check area)",
                foreground="green"
            )
            
            print(f"Detect ROI saved: ({x_min},{y_min}) -> ({x_max},{y_max})")
            print("Now draw COMPARE ROI...")
            
            # Redraw to show temp detect ROI
            self.display_image()
            
        elif self.drawing_stage == 'compare':
            # Create complete ROI with both detect and compare
            roi = {
                'roi_id': f'roi_{self.roi_counter:03d}',
                'model_name': self.current_model,
                'class_id': self.current_class['id'],
                'class_name': self.current_class['name'],
                'detect_x_min': self.temp_detect_roi['x_min'],
                'detect_y_min': self.temp_detect_roi['y_min'],
                'detect_x_max': self.temp_detect_roi['x_max'],
                'detect_y_max': self.temp_detect_roi['y_max'],
                'compare_x_min': x_min,
                'compare_y_min': y_min,
                'compare_x_max': x_max,
                'compare_y_max': y_max,
                'confidence': self.current_confidence
            }
            
            # Add keypoint data if in keypoint mode
            if self.keypoint_mode:
                roi['keypoint_idx_1'] = self.kp_idx_1_var.get()
                roi['keypoint_idx_2'] = self.kp_idx_2_var.get()
                roi['expected_angle'] = self.expected_angle_var.get()
                roi['angle_tolerance'] = self.angle_tolerance_var.get()
                print(f"DEBUG: Added keypoint data - KP[{roi['keypoint_idx_1']}-{roi['keypoint_idx_2']}], Angle: {roi['expected_angle']}°±{roi['angle_tolerance']}°")
            
            print(f"DEBUG: Created dual ROI with confidence: {roi['confidence']}")
            self.rois.append(roi)
            self.roi_counter += 1
            
            # Update ROI list
            roi_display = f"{roi['roi_id']}: {roi['class_name']} ({roi['confidence']})"
            if self.keypoint_mode and 'keypoint_idx_1' in roi:
                roi_display += f" [KP{roi['keypoint_idx_1']}-{roi['keypoint_idx_2']}, {roi['expected_angle']}°±{roi['angle_tolerance']}°]"
            
            self.roi_listbox.insert(tk.END, roi_display)
            
            # Reset to detect stage
            self.drawing_stage = 'detect'
            self.temp_detect_roi = None
            self.stage_label.config(
                text="Next: Draw DETECT ROI (larger area)",
                foreground="blue"
            )
            
            print("ROI complete! Ready for next ROI.")
            
            # Redraw
            self.display_image()
    
    def delete_selected_roi(self):
        """Delete selected ROI"""
        selection = self.roi_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an ROI to delete")
            return
        
        idx = selection[0]
        self.roi_listbox.delete(idx)
        del self.rois[idx]
        
        self.display_image()
    
    def load_csv(self):
        """Load ROIs from CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Check if new format (with detect/compare ROIs) or old format
            is_new_format = 'detect_x_min' in df.columns and 'compare_x_min' in df.columns
            
            # Validate columns based on format
            if is_new_format:
                required_cols = ['roi_id', 'model_name', 'class_id', 'class_name', 
                               'detect_x_min', 'detect_y_min', 'detect_x_max', 'detect_y_max',
                               'compare_x_min', 'compare_y_min', 'compare_x_max', 'compare_y_max',
                               'confidence']
            else:
                required_cols = ['roi_id', 'model_name', 'class_id', 'class_name', 
                               'x_min', 'y_min', 'x_max', 'y_max', 'confidence']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                messagebox.showerror("Error", 
                    f"CSV missing required columns:\n{', '.join(missing_cols)}")
                return
            
            # Clear current ROIs
            self.rois.clear()
            self.roi_listbox.delete(0, tk.END)
            
            # Load ROIs from CSV
            has_keypoints = 'keypoint_idx_1' in df.columns
            
            for _, row in df.iterrows():
                roi = {
                    'roi_id': row['roi_id'],
                    'model_name': row['model_name'],
                    'class_id': int(row['class_id']),
                    'class_name': row['class_name'],
                    'confidence': float(row['confidence'])
                }
                
                # Add ROI coordinates based on format
                if is_new_format:
                    roi['detect_x_min'] = int(row['detect_x_min'])
                    roi['detect_y_min'] = int(row['detect_y_min'])
                    roi['detect_x_max'] = int(row['detect_x_max'])
                    roi['detect_y_max'] = int(row['detect_y_max'])
                    roi['compare_x_min'] = int(row['compare_x_min'])
                    roi['compare_y_min'] = int(row['compare_y_min'])
                    roi['compare_x_max'] = int(row['compare_x_max'])
                    roi['compare_y_max'] = int(row['compare_y_max'])
                else:
                    # Old format - single ROI
                    roi['x_min'] = int(row['x_min'])
                    roi['y_min'] = int(row['y_min'])
                    roi['x_max'] = int(row['x_max'])
                    roi['y_max'] = int(row['y_max'])
                
                # Add keypoint data if present
                if has_keypoints:
                    roi['keypoint_idx_1'] = int(row['keypoint_idx_1'])
                    roi['keypoint_idx_2'] = int(row['keypoint_idx_2'])
                    roi['expected_angle'] = float(row['expected_angle'])
                    roi['angle_tolerance'] = float(row['angle_tolerance'])
                
                self.rois.append(roi)
                
                # Add to listbox
                roi_display = f"{roi['roi_id']}: {roi['class_name']} ({roi['confidence']})"
                if has_keypoints and 'keypoint_idx_1' in roi:
                    roi_display += f" [KP{roi['keypoint_idx_1']}-{roi['keypoint_idx_2']}, {roi['expected_angle']}°±{roi['angle_tolerance']}°]"
                
                self.roi_listbox.insert(tk.END, roi_display)
            
            # Update counter
            if self.rois:
                max_roi_num = max([int(roi['roi_id'].split('_')[1]) for roi in self.rois])
                self.roi_counter = max_roi_num + 1
            
            # Update mode based on loaded data
            if has_keypoints:
                self.mode_var.set("keypoint")
                self.on_mode_change()
            
            # Show info
            format_type = "NEW (Dual ROI)" if is_new_format else "OLD (Single ROI)"
            data_type = "with Keypoint Angle data" if has_keypoints else "Detection only"
            messagebox.showinfo("Success", 
                f"Loaded {len(self.rois)} ROI(s) from CSV\n"
                f"Format: {format_type}\n"
                f"Type: {data_type}\n\n"
                f"Note: Please load the corresponding image to view ROIs")
            
            # If image is already loaded, display ROIs
            if self.image is not None:
                self.display_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{str(e)}")
    
    def zoom_in(self):
        """Zoom in on image"""
        if self.image is None:
            return
        self.zoom_level *= 1.2
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_level * 100)}%")
        self.display_image()
    
    def zoom_out(self):
        """Zoom out on image"""
        if self.image is None:
            return
        self.zoom_level /= 1.2
        if self.zoom_level < 0.1:
            self.zoom_level = 0.1
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_level * 100)}%")
        self.display_image()
    
    def reset_zoom(self):
        """Reset zoom and pan to default"""
        if self.image is None:
            return
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.zoom_label.config(text="Zoom: 100%")
        self.display_image()
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        if self.image is None:
            return
        
        # Get mouse position before zoom
        mouse_x = event.x
        mouse_y = event.y
        
        # Convert to image coordinates
        img_x, img_y = self.canvas_to_image_coords(mouse_x, mouse_y)
        
        # Zoom in or out
        if event.delta > 0:
            zoom_factor = 1.1
        else:
            zoom_factor = 0.9
        
        old_zoom = self.zoom_level
        self.zoom_level *= zoom_factor
        
        # Limit zoom level
        if self.zoom_level < 0.1:
            self.zoom_level = 0.1
        elif self.zoom_level > 10.0:
            self.zoom_level = 10.0
        
        # Adjust offset to keep mouse position fixed
        h, w = self.image.shape[:2]
        base_scale = min(self.canvas_width / w, self.canvas_height / h)
        
        new_scale = base_scale * self.zoom_level
        old_scale = base_scale * old_zoom
        
        # Calculate new offset to keep the same image point under the mouse
        self.offset_x = mouse_x - (img_x * new_scale)
        self.offset_y = mouse_y - (img_y * new_scale)
        
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_level * 100)}%")
        self.display_image()
    
    def on_pan_start(self, event):
        """Start panning with right mouse button"""
        if self.image is None:
            return
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor='fleur')
    
    def on_pan_move(self, event):
        """Pan the image"""
        if not self.panning:
            return
        
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        
        self.offset_x += dx
        self.offset_y += dy
        
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        
        self.display_image()
    
    def on_pan_end(self, event):
        """End panning"""
        self.panning = False
        self.canvas.config(cursor='cross')
    
    def save_csv(self):
        """Save ROIs to CSV"""
        if not self.rois:
            messagebox.showwarning("Warning", "No ROIs to save")
            return
        
        # Lấy mã hàng từ input
        product_code = self.product_code_var.get().strip()
        
        if not product_code:
            messagebox.showwarning("Warning", "Vui lòng nhập mã hàng trước khi lưu!")
            return
        
        # Tạo tên file từ mã hàng
        default_filename = f"{product_code}.csv"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile=default_filename
        )
        
        if not file_path:
            return
        
        df = pd.DataFrame(self.rois)
        
        # Check if ROIs have dual format (detect + compare)
        has_dual_roi = any('detect_x_min' in roi for roi in self.rois)
        
        # Check if any ROI has keypoint data
        has_keypoints = any('keypoint_idx_1' in roi for roi in self.rois)
        
        # Select columns based on format
        if has_dual_roi:
            # New format with dual ROI
            base_cols = ['roi_id', 'model_name', 'class_id', 'class_name', 
                        'detect_x_min', 'detect_y_min', 'detect_x_max', 'detect_y_max',
                        'compare_x_min', 'compare_y_min', 'compare_x_max', 'compare_y_max',
                        'confidence']
        else:
            # Old format with single ROI
            base_cols = ['roi_id', 'model_name', 'class_id', 'class_name', 
                        'x_min', 'y_min', 'x_max', 'y_max', 'confidence']
        
        if has_keypoints:
            # Include keypoint columns
            keypoint_cols = ['keypoint_idx_1', 'keypoint_idx_2', 'expected_angle', 'angle_tolerance']
            # Ensure all rows have keypoint data (fill with defaults if missing)
            for roi in self.rois:
                if 'keypoint_idx_1' not in roi:
                    roi['keypoint_idx_1'] = 0
                    roi['keypoint_idx_2'] = 1
                    roi['expected_angle'] = 0.0
                    roi['angle_tolerance'] = 10.0
            df = pd.DataFrame(self.rois)  # Recreate df after filling
            df = df[base_cols + keypoint_cols]
        else:
            df = df[base_cols]
        
        df.to_csv(file_path, index=False)
        
        format_info = "Dual ROI (detect + compare)" if has_dual_roi else "Single ROI"
        kp_info = " with Keypoint data" if has_keypoints else ""
        messagebox.showinfo("Success", 
            f"Saved {len(self.rois)} ROI(s) to:\n{file_path}\n\n"
            f"Format: {format_info}{kp_info}")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    root = tk.Tk()
    app = ROIAnnotationGUI(root)
    root.mainloop()


# ============================================
# HƯỚNG DẪN SỬ DỤNG
# ============================================

"""
1. Cài đặt:
-----------
pip install opencv-python pandas numpy pillow ultralytics

2. Chạy:
--------
python BuildDatabaseFixedROI.py

3. Workflow - ANNOTATION MODE (Dual ROI):
-----------------------------------------
- Click "Scan Models Folder" → chọn thư mục chứa .pt files
- Chọn Mode:
  * 🔍 Detection Only: Chỉ kiểm tra object detection
  * 📐 Detection + Keypoint Angle: Kiểm tra detection + góc keypoint
- Click "Load Image" → chọn ảnh cần annotation
- Click chọn model từ danh sách
- Click chọn class
- (Nếu chọn Keypoint Mode) Điều chỉnh keypoint settings
- Vẽ ROI theo 2 bước:
  * BƯỚC 1: Click & drag để vẽ DETECT ROI (vùng lớn hơn, màu CYAN)
  * BƯỚC 2: Click & drag để vẽ COMPARE ROI (vùng kiểm tra, màu GREEN)
- Điều chỉnh confidence nếu cần
- Tiếp tục vẽ thêm ROI (lặp lại bước 1 & 2)
- Nhập mã hàng (Product Code)
- Click "Save CSV" để xuất kết quả

4. CSV Format - DUAL ROI (NEW):
--------------------------------
Detection Only:
roi_id,model_name,class_id,class_name,detect_x_min,detect_y_min,detect_x_max,detect_y_max,compare_x_min,compare_y_min,compare_x_max,compare_y_max,confidence

Detection + Keypoint:
roi_id,model_name,class_id,class_name,detect_x_min,detect_y_min,detect_x_max,detect_y_max,compare_x_min,compare_y_min,compare_x_max,compare_y_max,confidence,keypoint_idx_1,keypoint_idx_2,expected_angle,angle_tolerance

Example:
roi_002,MarkF,0,mark,1900,1650,2250,2100,1953,1727,2193,2040,0.52

Giải thích:
- DETECT ROI (1900,1650,2250,2100): Vùng để chạy model YOLO - có thể lớn hơn
- COMPARE ROI (1953,1727,2193,2040): Vùng để kiểm tra kết quả - vùng cụ thể cần có object

5. Workflow - VIEW MODE:
-------------------------
- Click "Load CSV & View ROIs" → chọn file CSV đã lưu
  (Tool sẽ tự động nhận diện format: Dual ROI hoặc Single ROI)
- Click "Load Image" → chọn ảnh tương ứng
- Tất cả ROI sẽ được hiển thị:
  * DETECT ROI: màu CYAN
  * COMPARE ROI: màu GREEN
- Có thể xóa hoặc thêm ROI mới
- Lưu lại CSV nếu có thay đổi

6. Features:
------------
✅ Dual ROI Mode: Vẽ 2 vùng ROI riêng biệt cho mỗi annotation
✅ Hỗ trợ cả format mới (dual ROI) và format cũ (single ROI)
✅ Tự động nhận diện format khi load CSV
✅ Hiển thị DETECT ROI (cyan) và COMPARE ROI (green)
✅ Hỗ trợ keypoint mode
✅ Zoom & Pan để vẽ chính xác
✅ Edit: thêm/xóa ROI
✅ Re-save CSV sau khi chỉnh sửa

- Tất cả ROI sẽ được hiển thị trên ảnh
- Có thể xóa hoặc thêm ROI mới
- Lưu lại CSV nếu có thay đổi

6. Features:
------------
✅ Hỗ trợ 2 loại sản phẩm: có keypoint và không có keypoint
✅ Load và view ROI từ CSV có sẵn
✅ Tự động nhận diện format CSV khi load
✅ Hiển thị bounding box với label
✅ Hiển thị thông tin keypoint (nếu có)
✅ Xem thông tin model, class, confidence
✅ Edit: thêm/xóa ROI
✅ Re-save CSV sau khi chỉnh sửa
✅ Chuyển đổi linh hoạt giữa 2 mode
"""