#!/usr/bin/env python3
"""
Thermal-RGB BBox Interactive Correction Tool
=============================================
GUI-based tool for manual review and correction of bounding box alignment
between thermal and RGB images.

Usage:
    python interactive_tool.py

Features:
- Load thermal/RGB image pairs with annotations
- Visualize current alignment with overlay
- Manually adjust bounding box positions
- Apply automatic correction suggestions
- Save corrections to file
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
import os
import sys
from pathlib import Path

# Import core modules
sys.path.insert(0, str(Path(__file__).parent))
from src.correction_core import BBoxCorrectionEngine
from src.label_parsers import YOLOLabelParser, MetadataParser, detect_label_format


class InteractiveCorrectionTool:
    """Interactive GUI tool for bbox correction."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Thermal-RGB BBox Correction Tool")
        self.root.geometry("1400x900")
        
        # State
        self.thermal_dir = None
        self.rgb_dir = None
        self.labels_dir = None
        self.samples = []
        self.current_idx = 0
        self.current_shift = {'dx': 0, 'dy': 0}
        self.corrections = {}
        
        # Engine
        self.engine = BBoxCorrectionEngine(expansion_factor=1.0)
        
        # Setup UI
        self._setup_ui()
        
        # Auto-load sample data on startup
        self._load_sample_data()
    
    def _load_sample_data(self):
        """Load sample data from default paths (no dialogs)."""
        # Default paths - edit these to point to your data
        base_dir = Path(__file__).parent
        self.thermal_dir = base_dir / "sample_data" / "thermal"
        self.rgb_dir = base_dir / "sample_data" / "rgb"
        self.labels_dir = base_dir / "sample_data" / "metadata"
        
        if not self.thermal_dir.exists():
            self.status_var.set("Sample data not found. Use 'Load Data' button.")
            return
        
        try:
            label_format = detect_label_format(self.labels_dir)
            self.status_var.set(f"Loading sample data ({label_format} format)...")
            
            if label_format == 'yolo':
                self._load_yolo_samples()
            else:
                self._load_metadata_samples()
            
            if self.samples:
                self.current_idx = 0
                self.root.after(100, self._update_display)  # Delay to let UI render
                self.status_var.set(f"Loaded {len(self.samples)} samples from sample_data/")
            else:
                self.status_var.set("No samples found in sample_data/")
        except Exception as e:
            self.status_var.set(f"Error loading sample data: {e}")
    
    def _setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Top control bar
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        ttk.Button(control_frame, text="Load Data", command=self._load_data).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Auto Correct", command=self._auto_correct).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Save Corrections", command=self._save_corrections).pack(side="left", padx=5)
        
        ttk.Separator(control_frame, orient="vertical").pack(side="left", fill="y", padx=10)
        
        ttk.Button(control_frame, text="◀ Prev", command=self._prev_sample).pack(side="left", padx=2)
        ttk.Button(control_frame, text="Next ▶", command=self._next_sample).pack(side="left", padx=2)
        
        self.sample_label = ttk.Label(control_frame, text="No data loaded")
        self.sample_label.pack(side="left", padx=20)
        
        # Image display area
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Left panel - Thermal
        left_frame = ttk.LabelFrame(image_frame, text="Thermal Image")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        self.thermal_canvas = tk.Canvas(left_frame, bg="gray20")
        self.thermal_canvas.pack(fill="both", expand=True)
        
        # Right panel - RGB with bbox
        right_frame = ttk.LabelFrame(image_frame, text="RGB Image with BBox")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        image_frame.columnconfigure(1, weight=1)
        
        self.rgb_canvas = tk.Canvas(right_frame, bg="gray20")
        self.rgb_canvas.pack(fill="both", expand=True)
        
        # Bottom control panel
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        # Shift controls
        shift_frame = ttk.LabelFrame(bottom_frame, text="Manual Shift Adjustment")
        shift_frame.pack(side="left", padx=10)
        
        ttk.Label(shift_frame, text="X:").grid(row=0, column=0, padx=5)
        self.dx_var = tk.IntVar(value=0)
        self.dx_spinbox = ttk.Spinbox(shift_frame, from_=-100, to=100, 
                                       textvariable=self.dx_var, width=8,
                                       command=self._update_display)
        self.dx_spinbox.grid(row=0, column=1, padx=5)
        self.dx_spinbox.bind('<Return>', lambda e: self._update_display())
        
        ttk.Label(shift_frame, text="Y:").grid(row=0, column=2, padx=5)
        self.dy_var = tk.IntVar(value=0)
        self.dy_spinbox = ttk.Spinbox(shift_frame, from_=-100, to=100,
                                       textvariable=self.dy_var, width=8,
                                       command=self._update_display)
        self.dy_spinbox.grid(row=0, column=3, padx=5)
        self.dy_spinbox.bind('<Return>', lambda e: self._update_display())
        
        ttk.Button(shift_frame, text="Reset", command=self._reset_shift).grid(row=0, column=4, padx=10)
        ttk.Button(shift_frame, text="Apply", command=self._apply_correction).grid(row=0, column=5, padx=5)
        
        # Info panel
        info_frame = ttk.LabelFrame(bottom_frame, text="Current Sample Info")
        info_frame.pack(side="left", padx=20, fill="x", expand=True)
        
        self.info_label = ttk.Label(info_frame, text="Load data to begin")
        self.info_label.pack(padx=10, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken")
        status_bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self._prev_sample())
        self.root.bind('<Right>', lambda e: self._next_sample())
        self.root.bind('<a>', lambda e: self._auto_correct())
        self.root.bind('<s>', lambda e: self._save_corrections())
    
    def _load_data(self):
        """Load thermal, RGB, and label directories."""
        # Ask for thermal directory
        thermal_dir = filedialog.askdirectory(title="Select Thermal Images Directory")
        if not thermal_dir:
            return
        
        # Ask for RGB directory
        rgb_dir = filedialog.askdirectory(title="Select RGB Images Directory")
        if not rgb_dir:
            return
        
        # Ask for labels directory
        labels_dir = filedialog.askdirectory(title="Select Labels Directory (YOLO .txt or JSON)")
        if not labels_dir:
            return
        
        self.thermal_dir = Path(thermal_dir)
        self.rgb_dir = Path(rgb_dir)
        self.labels_dir = Path(labels_dir)
        
        # Detect format and load samples
        try:
            label_format = detect_label_format(self.labels_dir)
            self.status_var.set(f"Detected format: {label_format}")
            
            if label_format == 'yolo':
                self._load_yolo_samples()
            else:
                self._load_metadata_samples()
            
            if self.samples:
                self.current_idx = 0
                self._update_display()
                self.status_var.set(f"Loaded {len(self.samples)} samples")
            else:
                messagebox.showwarning("Warning", "No matching samples found!")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
    
    def _load_yolo_samples(self):
        """Load samples from YOLO format."""
        self.samples = []
        
        for thermal_path in self.thermal_dir.glob("*.jpg"):
            label_path = self.labels_dir / (thermal_path.stem + ".txt")
            rgb_path = self.rgb_dir / thermal_path.name
            
            if not label_path.exists() or not rgb_path.exists():
                continue
            
            img = cv2.imread(str(thermal_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            annotations = YOLOLabelParser.parse_file(str(label_path), w, h)
            
            for i, ann in enumerate(annotations):
                self.samples.append({
                    'thermal_path': str(thermal_path),
                    'rgb_path': str(rgb_path),
                    'label_path': str(label_path),
                    'bbox': ann['bbox'],
                    'annotation': ann,
                    'frame_id': thermal_path.stem,
                    'ann_idx': i,
                    'img_width': w,
                    'img_height': h
                })
    
    def _load_metadata_samples(self):
        """Load samples from JSON metadata."""
        self.samples = []
        
        for meta_path in self.labels_dir.glob("*.json"):
            try:
                metadata = MetadataParser.parse_file(str(meta_path))
                samples = MetadataParser.extract_samples(
                    metadata, self.thermal_dir, self.rgb_dir
                )
                self.samples.extend(samples)
            except Exception as e:
                print(f"Warning: Could not parse {meta_path}: {e}")
    
    def _update_display(self):
        """Update the image displays."""
        if not self.samples:
            return
        
        sample = self.samples[self.current_idx]
        
        # Load images
        thermal_img = cv2.imread(sample['thermal_path'])
        rgb_img = cv2.imread(sample['rgb_path'])
        
        if thermal_img is None or rgb_img is None:
            self.status_var.set("Error loading images")
            return
        
        # Get current shift
        dx = self.dx_var.get()
        dy = self.dy_var.get()
        
        bbox = sample['bbox']
        
        # Draw on thermal
        thermal_display = thermal_img.copy()
        cv2.rectangle(thermal_display,
                     (bbox['x_min'], bbox['y_min']),
                     (bbox['x_max'], bbox['y_max']),
                     (0, 255, 0), 2)
        
        # Draw on RGB - original and shifted
        rgb_display = rgb_img.copy()
        # Original in red
        cv2.rectangle(rgb_display,
                     (bbox['x_min'], bbox['y_min']),
                     (bbox['x_max'], bbox['y_max']),
                     (0, 0, 255), 2)
        # Shifted in green
        cv2.rectangle(rgb_display,
                     (bbox['x_min'] + dx, bbox['y_min'] + dy),
                     (bbox['x_max'] + dx, bbox['y_max'] + dy),
                     (0, 255, 0), 2)
        
        # Convert and display
        self._display_image(thermal_display, self.thermal_canvas)
        self._display_image(rgb_display, self.rgb_canvas)
        
        # Update labels
        self.sample_label.config(text=f"Sample {self.current_idx + 1} / {len(self.samples)}")
        self.info_label.config(text=f"Frame: {sample['frame_id']} | "
                                   f"BBox: ({bbox['x_min']}, {bbox['y_min']}) - "
                                   f"({bbox['x_max']}, {bbox['y_max']}) | "
                                   f"Shift: ({dx}, {dy})")
    
    def _display_image(self, img, canvas):
        """Display an image on a canvas."""
        # Get canvas size
        canvas.update()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        
        if cw <= 1 or ch <= 1:
            return
        
        # Resize image to fit
        h, w = img.shape[:2]
        scale = min(cw / w, ch / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Display
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=img_tk, anchor="center")
        canvas.image = img_tk  # Keep reference
    
    def _prev_sample(self):
        """Go to previous sample."""
        if self.samples and self.current_idx > 0:
            self.current_idx -= 1
            self._reset_shift()
    
    def _next_sample(self):
        """Go to next sample."""
        if self.samples and self.current_idx < len(self.samples) - 1:
            self.current_idx += 1
            self._reset_shift()
    
    def _reset_shift(self):
        """Reset shift to zero."""
        self.dx_var.set(0)
        self.dy_var.set(0)
        self._update_display()
    
    def _auto_correct(self):
        """Compute automatic correction for current sample."""
        if not self.samples:
            return
        
        sample = self.samples[self.current_idx]
        
        try:
            thermal_img = cv2.imread(sample['thermal_path'])
            rgb_img = cv2.imread(sample['rgb_path'])
            
            data = self.engine.compute_correction(thermal_img, rgb_img, sample['bbox'])
            
            # Get best shift
            best_shift = None
            best_conf = 0
            
            for method_name, method_data in data['methods'].items():
                if method_data.get('shift') and method_data.get('confidence', 0) > best_conf:
                    best_shift = method_data['shift']
                    best_conf = method_data['confidence']
            
            if best_shift:
                self.dx_var.set(int(best_shift['dx']))
                self.dy_var.set(int(best_shift['dy']))
                self._update_display()
                self.status_var.set(f"Auto correction: ({best_shift['dx']}, {best_shift['dy']}) "
                                   f"confidence: {best_conf:.2f}")
            else:
                self.status_var.set("No confident correction found")
        
        except Exception as e:
            self.status_var.set(f"Auto correction failed: {e}")
    
    def _apply_correction(self):
        """Apply current correction to the sample."""
        if not self.samples:
            return
        
        sample = self.samples[self.current_idx]
        frame_id = sample['frame_id']
        ann_idx = sample.get('ann_idx', 0)
        
        key = f"{frame_id}_{ann_idx}"
        self.corrections[key] = {
            'frame_id': frame_id,
            'ann_idx': ann_idx,
            'shift': {'dx': self.dx_var.get(), 'dy': self.dy_var.get()},
            'original_bbox': sample['bbox'],
            'sample': sample
        }
        
        self.status_var.set(f"Correction applied for {frame_id}")
    
    def _save_corrections(self):
        """Save all corrections to file."""
        if not self.corrections:
            messagebox.showinfo("Info", "No corrections to save")
            return
        
        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save correction log
        log_path = output_path / 'corrections.json'
        log_data = {
            'corrections': [
                {
                    'frame_id': c['frame_id'],
                    'ann_idx': c['ann_idx'],
                    'shift': c['shift']
                }
                for c in self.corrections.values()
            ]
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.status_var.set(f"Saved {len(self.corrections)} corrections to {output_path}")
        messagebox.showinfo("Success", f"Saved {len(self.corrections)} corrections")


def main():
    root = tk.Tk()
    app = InteractiveCorrectionTool(root)
    root.mainloop()


if __name__ == '__main__':
    main()
