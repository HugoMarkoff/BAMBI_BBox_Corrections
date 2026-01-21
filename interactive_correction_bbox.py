"""
Thermal-to-RGB BBox Correction Tool v2
=======================================
Focused on CROPS and showing different processing methods applied.

Shows:
1. Thermal crop (ground truth template)
2. RGB search region with different processing methods
3. Match results for each method
4. Human validation interface

Update 2026-01-21:
-----------------
Added Shift+Click / Shift+Number key feature:
- When user validates a shift on ONE bbox using Shift+click or Shift+1-5,
  the determined shift (offset_x, offset_y) is applied to ALL bboxes in the frame.
- This dramatically reduces annotation time from bbox-by-bbox to frame-by-frame.
- User can skip frames until finding a good representative bbox to validate.

Update 2026-01-21 v2:
--------------------
Made apply-to-all the DEFAULT behavior:
- When you validate a shift on any bbox, it automatically applies to ALL bboxes in that frame.
- Progress shows total processed (e.g., 55/77 after validating a frame with 55 detections).
- In demo mode (sample_data), saves visualization images when all samples are processed.
- Visualizations saved to output/visualizations/manual_frame_XXXX.png showing:
  * Thermal image with bboxes
  * RGB image with original (uncorrected) bboxes
  * RGB image with corrected bboxes
"""

import cv2
import numpy as np
import json
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path
import random
from datetime import datetime

class BBoxCorrectionTool:
    def __init__(self, data_dir, use_sample_data=False):
        self.data_dir = Path(data_dir)
        self.use_sample_data = use_sample_data  # Track demo mode
        
        if use_sample_data:
            # Use sample_data subdirectory structure
            self.thermal_dir = self.data_dir / "sample_data" / "thermal"
            self.rgb_dir = self.data_dir / "sample_data" / "rgb"
            self.labels_dir = self.data_dir / "sample_data" / "labels"
            self.metadata_dir = self.data_dir / "sample_data" / "metadata"
        else:
            # Use full dataset structure
            self.thermal_dir = self.data_dir / "images"
            self.rgb_dir = self.data_dir / "rgb_images"
            self.labels_dir = self.data_dir / "labels"
            self.metadata_dir = self.data_dir / "metadata"
        
        self.output_dir = self.data_dir / "corrected_metadata"
        self.output_dir.mkdir(exist_ok=True)
        
        # Visualization output directory
        self.viz_output_dir = self.data_dir / "output" / "visualizations"
        self.viz_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store validated corrections
        self.corrections = []
        self.good_params = []
        self.bad_cases = []
        
        # Track processed sample indices for accurate progress
        self.processed_indices = set()
        
        # Matching methods
        self.matching_methods = [
            ('Grayscale', self._convert_to_grayscale),
            ('CLAHE Contrast', self._enhance_contrast),
            ('Canny Edges', self._edge_detection),
            ('Adaptive Thresh', self._adaptive_threshold),
            ('LAB Luminance', self._color_to_heatmap),
        ]
        
        self.current_idx = 0
        self.samples = []
        self.session_file = self.output_dir / "correction_session.json"
        self.load_session()
        
    def load_session(self):
        if self.session_file.exists():
            with open(self.session_file, 'r') as f:
                session = json.load(f)
                self.corrections = session.get('corrections', [])
                self.good_params = session.get('good_params', [])
                self.bad_cases = session.get('bad_cases', [])
                print(f"Loaded session: {len(self.corrections)} corrections, {len(self.good_params)} good params")
    
    def save_session(self):
        session = {
            'corrections': self.corrections,
            'good_params': self.good_params,
            'bad_cases': self.bad_cases,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.session_file, 'w') as f:
            json.dump(session, f, indent=2)
        print(f"Session saved: {len(self.corrections)} corrections")
    
    def collect_samples(self, split='test', max_samples=100, exclude_occluded=True, use_flat_structure=False):
        """
        Collect samples from metadata.
        
        Args:
            split: Dataset split ('test', 'train', 'val') - ignored if use_flat_structure=True
            max_samples: Maximum number of samples to collect
            exclude_occluded: Skip occluded annotations
            use_flat_structure: If True, look for metadata files directly in metadata_dir (for sample_data)
        """
        samples = []
        skipped_occluded = 0
        
        if use_flat_structure:
            # Flat structure: metadata files directly in metadata_dir
            print(f"Collecting samples from {self.metadata_dir}...")
            metadata_path = self.metadata_dir
            image_split_prefix = ""  # No split subdirectory
        else:
            # Standard structure: metadata/split/*.json
            print(f"Collecting samples from {split}...")
            metadata_path = self.metadata_dir / split
            image_split_prefix = split
        
        if not metadata_path.exists():
            print(f"Warning: Metadata path does not exist: {metadata_path}")
            return []
        
        for meta_file in metadata_path.glob("*_metadata.json"):
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            flight_key = metadata.get('flight_key', meta_file.stem.replace('_metadata', ''))
            
            for frame_id, frame_data in metadata.get('frames', {}).items():
                thermal_img_name = frame_data.get('thermal_image')
                rgb_img_name = frame_data.get('rgb_image')
                annotations = frame_data.get('annotations', [])
                
                # For sample_data, construct filename from flight_key and frame_id
                if not thermal_img_name:
                    thermal_img_name = f"{flight_key}_{frame_id}.jpg"
                if not rgb_img_name:
                    rgb_img_name = f"{flight_key}_{frame_id}.jpg"
                
                if not annotations:
                    continue
                
                if use_flat_structure:
                    thermal_path = self.thermal_dir / thermal_img_name
                    rgb_path = self.rgb_dir / rgb_img_name
                else:
                    thermal_path = self.thermal_dir / image_split_prefix / thermal_img_name
                    rgb_path = self.rgb_dir / image_split_prefix / rgb_img_name
                
                if thermal_path.exists() and rgb_path.exists():
                    for ann in annotations:
                        bbox = ann.get('bbox')
                        if not bbox:
                            continue
                        
                        # Filter out occluded annotations (visibility > 0 means occluded)
                        visibility = ann.get('visibility', 0)
                        if exclude_occluded and visibility > 0:
                            skipped_occluded += 1
                            continue
                        
                        samples.append({
                            'thermal_path': str(thermal_path),
                            'rgb_path': str(rgb_path),
                            'bbox': bbox,
                            'annotation': ann,
                            'frame_id': frame_id,
                            'flight_key': flight_key,
                            'split': split if not use_flat_structure else 'sample',
                            'visibility': visibility
                        })
                else:
                    if not thermal_path.exists():
                        print(f"  Warning: Thermal image not found: {thermal_path}")
                    if not rgb_path.exists():
                        print(f"  Warning: RGB image not found: {rgb_path}")
        
        random.shuffle(samples)
        self.samples = samples[:max_samples]
        
        # Build frame-to-samples index for apply-to-all feature
        self.frame_samples = {}  # {(flight_key, frame_id): [sample_indices]}
        for idx, s in enumerate(self.samples):
            key = (s['flight_key'], s['frame_id'])
            if key not in self.frame_samples:
                self.frame_samples[key] = []
            self.frame_samples[key].append(idx)
        
        # Debug: Show visibility distribution in selected samples
        vis_counts = {}
        for s in self.samples:
            v = s.get('visibility', 0)
            vis_counts[v] = vis_counts.get(v, 0) + 1
        print(f"Collected {len(self.samples)} samples (skipped {skipped_occluded} occluded)")
        print(f"Visibility distribution in selected samples: {vis_counts}")
        print(f"Unique frames: {len(self.frame_samples)}")
        
        return self.samples
    
    def _convert_to_grayscale(self, img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def _enhance_contrast(self, img):
        gray = self._convert_to_grayscale(img)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    def _edge_detection(self, img):
        gray = self._convert_to_grayscale(img)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, 50, 150)
    
    def _adaptive_threshold(self, img):
        gray = self._convert_to_grayscale(img)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
    
    def _color_to_heatmap(self, img):
        if len(img.shape) == 2:
            gray = img
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            gray = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    def expand_search_area(self, img, bbox, expansion_factor=1.0):
        h, w = img.shape[:2]
        x_min, y_min = bbox['x_min'], bbox['y_min']
        x_max, y_max = bbox['x_max'], bbox['y_max']
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        expand_x = int(bbox_w * expansion_factor)
        expand_y = int(bbox_h * expansion_factor)
        
        new_x_min = max(0, x_min - expand_x)
        new_y_min = max(0, y_min - expand_y)
        new_x_max = min(w, x_max + expand_x)
        new_y_max = min(h, y_max + expand_y)
        
        expanded_region = img[new_y_min:new_y_max, new_x_min:new_x_max]
        
        return expanded_region, {
            'x_min': new_x_min, 'y_min': new_y_min,
            'x_max': new_x_max, 'y_max': new_y_max,
            'original_offset_x': x_min - new_x_min,
            'original_offset_y': y_min - new_y_min
        }
    
    def find_best_match(self, thermal_crop, rgb_search_region, convert_func):
        thermal_processed = convert_func(thermal_crop)
        rgb_processed = convert_func(rgb_search_region)
        
        if thermal_processed.shape[0] > rgb_processed.shape[0] or \
           thermal_processed.shape[1] > rgb_processed.shape[1]:
            return None, 0, thermal_processed, rgb_processed, None
        
        # EXHAUSTIVE SEARCH: matchTemplate slides template across EVERY position
        # and computes correlation at each point. Returns a heatmap of scores.
        result = cv2.matchTemplate(rgb_processed, thermal_processed, cv2.TM_CCOEFF_NORMED)
        
        # Find the GLOBAL BEST match (highest correlation in entire heatmap)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Normalize heatmap for visualization (shows where good matches are)
        heatmap = ((result - result.min()) / (result.max() - result.min() + 1e-8) * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return max_loc, max_val, thermal_processed, rgb_processed, heatmap_colored
    
    def rank_results_smart(self, results, thermal_brightness):
        """
        Rank results using learned patterns from user's previous method choices.
        This weights by: 1) how often user chose each method, 2) similarity of brightness conditions
        """
        if not self.good_params:
            # No history yet, fall back to confidence ranking
            results.sort(key=lambda x: x['confidence'], reverse=True)
            return results
        
        # Calculate method preference scores based on user's choices
        method_counts = {}
        method_brightness_matches = {}  # Track brightness similarity per method
        
        for param in self.good_params:
            method = param['method']
            if method not in method_counts:
                method_counts[method] = 0
                method_brightness_matches[method] = []
            method_counts[method] += 1
            # Handle old session format without thermal_brightness
            if 'thermal_brightness' in param:
                method_brightness_matches[method].append(param['thermal_brightness'])
        
        total_choices = len(self.good_params)
        
        # Score each result: smart_score = confidence * preference_weight * brightness_match
        for result in results:
            method = result['method']
            confidence = result['confidence']
            
            # Base preference weight: how often user chooses this method (0 to 2)
            if method in method_counts:
                preference_weight = 1.0 + (method_counts[method] / total_choices)
            else:
                preference_weight = 0.5  # Penalize methods user never chose
            
            # Brightness similarity weight: does user use this method at similar brightness?
            brightness_weight = 1.0
            if method in method_brightness_matches and method_brightness_matches[method]:
                brightness_diffs = [abs(b - thermal_brightness) for b in method_brightness_matches[method]]
                min_diff = min(brightness_diffs)
                # If minimum difference is small, boost the weight
                # 0 diff = 1.3x boost, large diff = 0.8x 
                brightness_weight = 1.3 - min(0.5, min_diff / 200.0)
            
            # Combined smart score
            result['smart_score'] = confidence * preference_weight * brightness_weight
            result['preference_weight'] = preference_weight
            result['brightness_weight'] = brightness_weight
        
        # Sort by smart score instead of raw confidence
        results.sort(key=lambda x: x['smart_score'], reverse=True)
        
        return results

    def analyze_auto_certainty(self, results, sample, thermal_crop):
        """
        Analyze whether we can auto-accept this correction with high certainty.
        Returns: (certainty_score, reason, suggested_result)
        
        Patterns that increase certainty:
        1. Multiple methods agree on the same offset
        2. High confidence from user-preferred methods
        3. Offset within typical range from past corrections
        4. Flight not in bad_cases list frequently
        
        Patterns that decrease certainty:
        1. Methods disagree significantly
        2. Flight appears in bad_cases frequently
        3. Very low confidence across all methods
        4. Unusual brightness/contrast compared to good cases
        """
        if not results:
            return 0.0, "No results", None
        
        # Get image characteristics
        thermal_mean = float(np.mean(thermal_crop))
        flight_key = sample.get('flight_key', '')
        
        certainty = 0.5  # Start neutral
        reasons = []
        
        # 1. Check method agreement on offset
        offsets = [(r['offset_x'], r['offset_y']) for r in results]
        offset_counts = {}
        for off in offsets:
            key = (off[0], off[1])
            offset_counts[key] = offset_counts.get(key, 0) + 1
        
        max_agreement = max(offset_counts.values())
        most_common_offset = max(offset_counts.keys(), key=lambda k: offset_counts[k])
        
        if max_agreement >= 4:
            certainty += 0.25
            reasons.append(f"{max_agreement}/5 methods agree")
        elif max_agreement >= 3:
            certainty += 0.15
            reasons.append(f"{max_agreement}/5 methods agree")
        elif max_agreement == 1:
            certainty -= 0.15
            reasons.append("All methods disagree")
        
        # 2. Check if flight is problematic (appears in bad_cases)
        bad_flight_count = sum(1 for bc in self.bad_cases if bc.get('flight_key') == flight_key)
        if bad_flight_count >= 3:
            certainty -= 0.25
            reasons.append(f"Flight has {bad_flight_count} rejections")
        elif bad_flight_count >= 1:
            certainty -= 0.1
            reasons.append(f"Flight has {bad_flight_count} rejection(s)")
        
        # 3. Check if top method is user-preferred
        top_result = results[0]
        user_method_counts = {}
        for gp in self.good_params:
            m = gp['method']
            user_method_counts[m] = user_method_counts.get(m, 0) + 1
        
        total_user_choices = len(self.good_params)
        if total_user_choices > 0:
            top_method = top_result['method']
            if top_method in user_method_counts:
                method_pref = user_method_counts[top_method] / total_user_choices
                if method_pref > 0.3:
                    certainty += 0.15
                    reasons.append(f"User prefers {top_method} ({method_pref:.0%})")
        
        # 4. Check confidence level
        top_conf = top_result['confidence']
        if top_conf > 0.5:
            certainty += 0.1
            reasons.append(f"High conf {top_conf:.2f}")
        elif top_conf < 0.2:
            certainty -= 0.15
            reasons.append(f"Low conf {top_conf:.2f}")
        
        # 5. Check if offset is within typical range from past corrections
        if self.corrections:
            past_offsets_x = [c['offset_x'] for c in self.corrections if 'offset_x' in c]
            past_offsets_y = [c['offset_y'] for c in self.corrections if 'offset_y' in c]
            if past_offsets_x and past_offsets_y:
                mean_ox, std_ox = np.mean(past_offsets_x), np.std(past_offsets_x) + 1
                mean_oy, std_oy = np.mean(past_offsets_y), np.std(past_offsets_y) + 1
                
                curr_ox, curr_oy = top_result['offset_x'], top_result['offset_y']
                z_score = abs((curr_ox - mean_ox) / std_ox) + abs((curr_oy - mean_oy) / std_oy)
                
                if z_score < 1.5:
                    certainty += 0.1
                    reasons.append("Typical offset")
                elif z_score > 3:
                    certainty -= 0.1
                    reasons.append("Unusual offset")
        
        # 6. Check brightness similarity to good cases
        if self.good_params:
            good_brightnesses = [gp['thermal_brightness'] for gp in self.good_params if 'thermal_brightness' in gp]
            if good_brightnesses:
                min_diff = min(abs(thermal_mean - b) for b in good_brightnesses)
                if min_diff < 30:
                    certainty += 0.05
                    reasons.append("Similar brightness to past")
        
        # 7. If agreeing methods include the consensus offset with the top-ranked method
        if most_common_offset == (top_result['offset_x'], top_result['offset_y']):
            certainty += 0.1
            reasons.append("Top method matches consensus")
        else:
            certainty -= 0.1
            reasons.append("Top method differs from consensus")
        
        # Clamp certainty
        certainty = max(0.0, min(1.0, certainty))
        
        # Find best result that matches consensus (if different from top)
        suggested_result = top_result
        for r in results:
            if (r['offset_x'], r['offset_y']) == most_common_offset:
                suggested_result = r
                break
        
        reason_str = " | ".join(reasons)
        return certainty, reason_str, suggested_result

    def compute_correction(self, sample):
        thermal_img = cv2.imread(sample['thermal_path'])
        rgb_img = cv2.imread(sample['rgb_path'])
        
        if thermal_img is None or rgb_img is None:
            return None
        
        bbox = sample['bbox']
        
        # Extract thermal crop
        thermal_crop = thermal_img[bbox['y_min']:bbox['y_max'], 
                                   bbox['x_min']:bbox['x_max']]
        
        if thermal_crop.size == 0:
            return None
        
        # Get expanded search region from RGB (9x area)
        rgb_search, search_coords = self.expand_search_area(rgb_img, bbox, expansion_factor=1.0)
        
        # Get original RGB crop (at same position as thermal - may be wrong)
        rgb_original_crop = rgb_img[bbox['y_min']:bbox['y_max'], 
                                    bbox['x_min']:bbox['x_max']]
        
        results = []
        
        for method_name, convert_func in self.matching_methods:
            try:
                match_loc, confidence, thermal_processed, rgb_processed, heatmap = self.find_best_match(
                    thermal_crop, rgb_search, convert_func
                )
                
                if match_loc is None:
                    continue
                
                # Corrected bbox in original image coordinates
                corrected_bbox = {
                    'x_min': search_coords['x_min'] + match_loc[0],
                    'y_min': search_coords['y_min'] + match_loc[1],
                    'x_max': search_coords['x_min'] + match_loc[0] + bbox['width'],
                    'y_max': search_coords['y_min'] + match_loc[1] + bbox['height'],
                    'width': bbox['width'],
                    'height': bbox['height']
                }
                
                offset_x = corrected_bbox['x_min'] - bbox['x_min']
                offset_y = corrected_bbox['y_min'] - bbox['y_min']
                
                # Extract the corrected RGB crop
                rgb_corrected_crop = rgb_img[corrected_bbox['y_min']:corrected_bbox['y_max'],
                                             corrected_bbox['x_min']:corrected_bbox['x_max']]
                
                results.append({
                    'method': method_name,
                    'confidence': float(confidence),
                    'corrected_bbox': corrected_bbox,
                    'offset_x': offset_x,
                    'offset_y': offset_y,
                    'match_loc': match_loc,
                    'thermal_processed': thermal_processed,
                    'rgb_processed': rgb_processed,
                    'rgb_corrected_crop': rgb_corrected_crop,
                    'heatmap': heatmap  # Shows exhaustive search result
                })
                
            except Exception as e:
                print(f"Error with method {method_name}: {e}")
                continue
        
        if not results:
            return None
        
        # Compute image characteristics for smart ranking
        thermal_mean = float(np.mean(thermal_crop))
        
        # Smart ranking: weight by user's previous method preferences
        results = self.rank_results_smart(results, thermal_mean)
        
        # Analyze auto-certainty for smart suggestions
        certainty, certainty_reason, suggested_result = self.analyze_auto_certainty(
            results, sample, thermal_crop
        )
        
        return {
            'sample': sample,
            'thermal_img': thermal_img,
            'rgb_img': rgb_img,
            'thermal_crop': thermal_crop,
            'rgb_original_crop': rgb_original_crop,
            'rgb_search': rgb_search,
            'search_coords': search_coords,
            'results': results,
            'original_bbox': bbox,
            'certainty': certainty,
            'certainty_reason': certainty_reason,
            'suggested_result': suggested_result
        }


class CorrectionGUI:
    def __init__(self, tool):
        self.tool = tool
        self.root = tk.Tk()
        self.root.title("Thermal-RGB BBox Correction Tool v2 - CROP FOCUSED")
        self.root.geometry("1650x950")
        self.root.configure(bg='#2b2b2b')
        
        self.current_data = None
        self.current_result_idx = 0
        self.photo_refs = []  # Keep references to prevent garbage collection
        
        self.setup_ui()
        
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TLabelframe', background='#2b2b2b', foreground='white')
        style.configure('TLabelframe.Label', background='#2b2b2b', foreground='white')
        style.configure('Good.TButton', background='#4CAF50', foreground='white')
        style.configure('Bad.TButton', background='#f44336', foreground='white')
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top info bar
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_label = ttk.Label(info_frame, text="Progress: 0/0", font=('Arial', 12, 'bold'))
        self.progress_label.pack(side=tk.LEFT)
        
        self.stats_label = ttk.Label(info_frame, text="Good: 0 | Bad: 0", font=('Arial', 12))
        self.stats_label.pack(side=tk.RIGHT)
        
        self.info_label = ttk.Label(info_frame, text="", font=('Arial', 10))
        self.info_label.pack(side=tk.LEFT, padx=50)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # LEFT COLUMN: Source crops
        left_frame = ttk.LabelFrame(content_frame, text="SOURCE CROPS", padding=5)
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Thermal crop (ground truth)
        ttk.Label(left_frame, text="THERMAL CROP (Ground Truth)", font=('Arial', 10, 'bold')).pack()
        self.thermal_crop_canvas = tk.Canvas(left_frame, width=200, height=200, bg='#1a1a1a', 
                                              highlightthickness=2, highlightbackground='green')
        self.thermal_crop_canvas.pack(pady=5)
        
        # RGB crop at original position (may be wrong)
        ttk.Label(left_frame, text="RGB CROP (Original Position)", font=('Arial', 10, 'bold')).pack(pady=(10, 0))
        self.rgb_orig_crop_canvas = tk.Canvas(left_frame, width=200, height=200, bg='#1a1a1a',
                                               highlightthickness=2, highlightbackground='red')
        self.rgb_orig_crop_canvas.pack(pady=5)
        
        # MIDDLE COLUMN: Processing methods comparison
        middle_frame = ttk.LabelFrame(content_frame, text="PROCESSING METHODS (Thermal vs RGB Search Region)", padding=5)
        middle_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Create grid for 5 methods x 2 columns (thermal processed, rgb processed with match)
        self.method_frames = []
        self.method_canvases = []
        
        for i, (method_name, _) in enumerate(self.tool.matching_methods):
            row_frame = ttk.Frame(middle_frame)
            row_frame.pack(fill=tk.X, pady=5)
            
            # Method name and confidence - add padding to push images right
            label = ttk.Label(row_frame, text=f"{method_name}", font=('Arial', 9, 'bold'), width=12)
            label.pack(side=tk.LEFT, padx=(15, 5))
            
            conf_label = ttk.Label(row_frame, text="Conf: -", font=('Arial', 8), width=10)
            conf_label.pack(side=tk.LEFT)
            
            offset_label = ttk.Label(row_frame, text="Offset: -", font=('Arial', 8), width=14)
            offset_label.pack(side=tk.LEFT)
            
            # Thermal processed
            thermal_canvas = tk.Canvas(row_frame, width=100, height=100, bg='#1a1a1a',
                                       highlightthickness=1, highlightbackground='#444')
            thermal_canvas.pack(side=tk.LEFT, padx=5)
            
            # RGB CROP processed (same method applied to the suggested crop)
            rgb_crop_processed_canvas = tk.Canvas(row_frame, width=100, height=100, bg='#1a1a1a',
                                                   highlightthickness=1, highlightbackground='#666')
            rgb_crop_processed_canvas.pack(side=tk.LEFT, padx=5)
            
            # RGB suggested crop (the actual corrected crop - raw)
            rgb_crop_canvas = tk.Canvas(row_frame, width=100, height=100, bg='#1a1a1a',
                                        highlightthickness=2, highlightbackground='#4CAF50')
            rgb_crop_canvas.pack(side=tk.LEFT, padx=8)
            
            # Use This button (acts as Good/Accept) - applies to ALL bboxes in frame by default
            select_btn = tk.Button(row_frame, text=f"✓ Use ({i+1}) [All]", width=16,
                                   bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'))
            select_btn.bind('<Button-1>', lambda e, idx=i: self._on_select_click(e, idx))
            select_btn.pack(side=tk.LEFT, padx=10)
            
            self.method_frames.append({
                'label': label,
                'conf_label': conf_label,
                'offset_label': offset_label,
                'thermal_canvas': thermal_canvas,
                'rgb_crop_processed_canvas': rgb_crop_processed_canvas,
                'rgb_crop_canvas': rgb_crop_canvas,
                'select_btn': select_btn
            })
        
        # RIGHT COLUMN: Search region overview + heatmap
        right_frame = ttk.LabelFrame(content_frame, text="SEARCH RESULT", padding=5)
        right_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        
        ttk.Label(right_frame, text="Search Region (Red=Orig, Green=Best)", font=('Arial', 9)).pack()
        self.search_canvas = tk.Canvas(right_frame, width=280, height=200, bg='#1a1a1a')
        self.search_canvas.pack(pady=3)
        
        ttk.Label(right_frame, text="Correlation Heatmap (Exhaustive Search)", font=('Arial', 9)).pack(pady=(10,0))
        ttk.Label(right_frame, text="Red=Best match, Blue=Poor match", font=('Arial', 8)).pack()
        self.heatmap_canvas = tk.Canvas(right_frame, width=280, height=150, bg='#1a1a1a')
        self.heatmap_canvas.pack(pady=3)
        
        self.best_method_label = ttk.Label(right_frame, text="Best: -", font=('Arial', 11, 'bold'))
        self.best_method_label.pack(pady=5)
        
        # Stats about the search
        self.search_stats_label = ttk.Label(right_frame, text="", font=('Arial', 9))
        self.search_stats_label.pack(pady=5)
        
        # Pattern analysis button and display
        ttk.Button(right_frame, text="Show Patterns", command=self.show_patterns).pack(pady=5)
        self.pattern_label = ttk.Label(right_frame, text="", font=('Arial', 8), justify=tk.LEFT)
        self.pattern_label.pack(pady=5)
        
        # Configure grid weights
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=2)
        content_frame.columnconfigure(2, weight=1)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Reject and Skip buttons (Use This on each method row acts as Good/Accept)
        self.bad_btn = tk.Button(button_frame, text="✗ REJECT (R)", command=self.mark_bad,
                                 bg='#f44336', fg='white', font=('Arial', 14, 'bold'),
                                 width=15, height=2)
        self.bad_btn.pack(side=tk.LEFT, padx=20)
        
        self.skip_btn = tk.Button(button_frame, text="Skip (S)", command=self.skip,
                                  bg='#666666', fg='white', font=('Arial', 14, 'bold'),
                                  width=12, height=2)
        self.skip_btn.pack(side=tk.LEFT, padx=20)
        
        # Navigation
        nav_frame = ttk.Frame(button_frame)
        nav_frame.pack(side=tk.RIGHT, padx=20)
        
        ttk.Button(nav_frame, text="← Prev", command=self.prev_sample).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next →", command=self.next_sample).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Save & Quit", command=self.save_and_quit).pack(side=tk.LEFT, padx=20)
        
        # Key bindings
        self.root.bind('r', lambda e: self.mark_bad())
        self.root.bind('R', lambda e: self.mark_bad())
        self.root.bind('s', lambda e: self.skip())
        self.root.bind('S', lambda e: self.skip())
        self.root.bind('<Left>', lambda e: self.prev_sample())
        self.root.bind('<Right>', lambda e: self.next_sample())
        self.root.bind('<Escape>', lambda e: self.save_and_quit())
        
        # 1-5: Apply shift to ALL bboxes in frame (default behavior)
        self.root.bind('1', lambda e: self.use_method_apply_all(0))
        self.root.bind('2', lambda e: self.use_method_apply_all(1))
        self.root.bind('3', lambda e: self.use_method_apply_all(2))
        self.root.bind('4', lambda e: self.use_method_apply_all(3))
        self.root.bind('5', lambda e: self.use_method_apply_all(4))
        
        # Shift+number: Apply to single bbox only (override)
        self.root.bind('!', lambda e: self.use_method(0))  # Shift+1
        self.root.bind('@', lambda e: self.use_method(1))  # Shift+2
        self.root.bind('#', lambda e: self.use_method(2))  # Shift+3
        self.root.bind('$', lambda e: self.use_method(3))  # Shift+4
        self.root.bind('%', lambda e: self.use_method(4))  # Shift+5
        
    def cv2_to_tk(self, img, max_w=200, max_h=200):
        if img is None or img.size == 0:
            return None
        
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return None
            
        scale = min(max_w / w, max_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        if len(img_resized.shape) == 3:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
        pil_img = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(pil_img)
        self.photo_refs.append(photo)  # Keep reference
        return photo
    
    def draw_bbox_on_crop(self, search_region, orig_offset, match_loc, bbox_size, scale=1.0):
        """Draw original (red) and matched (green) boxes on search region"""
        img = search_region.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        bw, bh = bbox_size
        
        # Original position (red)
        cv2.rectangle(img, 
                     (orig_offset[0], orig_offset[1]),
                     (orig_offset[0] + bw, orig_offset[1] + bh),
                     (0, 0, 255), 2)
        
        # Matched position (green)
        if match_loc:
            cv2.rectangle(img,
                         (match_loc[0], match_loc[1]),
                         (match_loc[0] + bw, match_loc[1] + bh),
                         (0, 255, 0), 2)
        
        return img
    
    def _on_select_click(self, event, idx):
        """Handle click on select button - default applies to all, Shift for single bbox only"""
        if event.state & 0x1:  # Shift key is pressed - single bbox only
            self.use_method(idx)
        else:
            self.use_method_apply_all(idx)  # Default: apply to all
    
    def select_method(self, idx):
        """Select a specific method as the best one (just preview)"""
        if self.current_data and idx < len(self.current_data['results']):
            self.current_result_idx = idx
            self.update_display()
    
    def use_method(self, idx):
        """Use a specific method and accept as Good"""
        if self.current_data and idx < len(self.current_data['results']):
            self.current_result_idx = idx
            self.mark_good()
    
    def use_method_apply_all(self, idx):
        """
        Use a specific method and apply its shift to ALL bboxes in the current frame.
        This is triggered by Shift+click or Shift+number key.
        
        The offset (offset_x, offset_y) determined from the current bbox is applied
        to all other bboxes in the same frame, dramatically speeding up annotation.
        """
        if self.current_data is None or not self.current_data['results']:
            return
        
        if idx >= len(self.current_data['results']):
            return
        
        result = self.current_data['results'][idx]
        sample = self.current_data['sample']
        offset_x = result['offset_x']
        offset_y = result['offset_y']
        method_name = result['method']
        
        # Get all samples in this frame
        frame_key = (sample['flight_key'], sample['frame_id'])
        frame_sample_indices = self.tool.frame_samples.get(frame_key, [])
        
        if len(frame_sample_indices) <= 1:
            # Only one bbox in frame, just use normal method
            self.use_method(idx)
            return
        
        print(f"\n[APPLY TO ALL] Applying offset ({offset_x:+d}, {offset_y:+d}) to {len(frame_sample_indices)} bboxes in frame {frame_key}")
        
        # Mark current bbox as good first
        self.current_result_idx = idx
        self._record_correction(result, sample, self.current_data)
        
        # Apply same offset to all other bboxes in this frame
        applied_count = 1  # Already applied to current
        skipped_indices = set()
        
        for sample_idx in frame_sample_indices:
            if sample_idx == self.tool.current_idx:
                continue  # Skip current (already processed)
            
            other_sample = self.tool.samples[sample_idx]
            other_bbox = other_sample['bbox']
            
            # Compute corrected bbox using the same offset
            corrected_bbox = {
                'x_min': other_bbox['x_min'] + offset_x,
                'y_min': other_bbox['y_min'] + offset_y,
                'x_max': other_bbox['x_max'] + offset_x,
                'y_max': other_bbox['y_max'] + offset_y,
                'width': other_bbox['width'],
                'height': other_bbox['height']
            }
            
            # Create correction record
            correction = {
                'flight_key': other_sample['flight_key'],
                'frame_id': other_sample['frame_id'],
                'split': other_sample['split'],
                'original_bbox': other_sample['bbox'],
                'corrected_bbox': corrected_bbox,
                'method': method_name,
                'confidence': result['confidence'],  # Use same confidence as reference
                'offset_x': offset_x,
                'offset_y': offset_y,
                'annotation': other_sample['annotation'],
                'applied_from_reference': True,  # Mark as derived from another bbox
                'reference_bbox': sample['bbox']
            }
            
            self.tool.corrections.append(correction)
            applied_count += 1
            skipped_indices.add(sample_idx)
            
            # Track as processed
            self.tool.processed_indices.add(sample_idx)
        
        print(f"[APPLY TO ALL] Applied correction to {applied_count} bboxes total")
        print(f"[PROGRESS] {len(self.tool.processed_indices)}/{len(self.tool.samples)} samples processed")
        
        # Mark these samples as processed by skipping them
        # We'll jump past all samples in this frame
        self._skip_frame_samples(skipped_indices)
    
    def _record_correction(self, result, sample, data):
        """Record a single correction without advancing to next sample"""
        thermal_crop = data['thermal_crop']
        rgb_crop = data['rgb_original_crop']
        
        thermal_mean = float(np.mean(thermal_crop))
        thermal_std = float(np.std(thermal_crop))
        rgb_mean = float(np.mean(rgb_crop)) if rgb_crop.size > 0 else 0
        rgb_std = float(np.std(rgb_crop)) if rgb_crop.size > 0 else 0
        
        correction = {
            'flight_key': sample['flight_key'],
            'frame_id': sample['frame_id'],
            'split': sample['split'],
            'original_bbox': sample['bbox'],
            'corrected_bbox': result['corrected_bbox'],
            'method': result['method'],
            'confidence': result['confidence'],
            'offset_x': result['offset_x'],
            'offset_y': result['offset_y'],
            'annotation': sample['annotation'],
            'thermal_brightness': thermal_mean,
            'thermal_contrast': thermal_std,
            'rgb_brightness': rgb_mean,
            'rgb_contrast': rgb_std,
            'bbox_size': sample['bbox']['width'] * sample['bbox']['height']
        }
        
        self.tool.corrections.append(correction)
        self.tool.good_params.append({
            'method': result['method'],
            'confidence': result['confidence'],
            'thermal_brightness': thermal_mean,
            'rgb_brightness': rgb_mean
        })
        
        # Track this sample as processed
        self.tool.processed_indices.add(self.tool.current_idx)
    
    def _skip_frame_samples(self, skipped_indices):
        """
        Skip all samples that were processed via apply-to-all.
        Find the next unprocessed sample.
        """
        # Find next unprocessed sample
        self.tool.current_idx += 1
        
        while self.tool.current_idx < len(self.tool.samples):
            if self.tool.current_idx in self.tool.processed_indices:
                self.tool.current_idx += 1
            else:
                break
        
        # Check if all samples are processed
        if len(self.tool.processed_indices) >= len(self.tool.samples):
            self.show_completion()
        elif self.tool.current_idx >= len(self.tool.samples):
            # Reached end but maybe there are unprocessed samples earlier (shouldn't happen normally)
            self.show_completion()
        else:
            self.load_current_sample()
        
        # Update pattern display
        if len(self.tool.corrections) >= 3:
            self.update_quick_pattern()
    
    def analyze_patterns(self):
        """Analyze patterns in good vs bad corrections"""
        if len(self.tool.corrections) < 3:
            return "Need more data (3+ corrections)"
        
        # Count methods
        method_counts = {}
        method_confs = {}
        method_brightness = {}
        flight_methods = {}
        
        for c in self.tool.corrections:
            method = c['method']
            method_counts[method] = method_counts.get(method, 0) + 1
            if method not in method_confs:
                method_confs[method] = []
                method_brightness[method] = {'thermal': [], 'rgb': []}
            method_confs[method].append(c['confidence'])
            
            # Track brightness patterns if available
            if 'thermal_brightness' in c:
                method_brightness[method]['thermal'].append(c['thermal_brightness'])
                method_brightness[method]['rgb'].append(c['rgb_brightness'])
            
            flight = c['flight_key']
            if flight not in flight_methods:
                flight_methods[flight] = {}
            flight_methods[flight][method] = flight_methods[flight].get(method, 0) + 1
        
        # Build analysis text
        lines = ["=" * 40, "PATTERN ANALYSIS", "=" * 40, ""]
        
        lines.append("METHOD SUCCESS RATE:")
        total = len(self.tool.corrections)
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            avg_conf = sum(method_confs[method]) / len(method_confs[method])
            pct = 100 * count / total
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"  {method:15s} [{bar}] {count:3d} ({pct:4.1f}%)")
            lines.append(f"                   avg_conf={avg_conf:.3f}")
            
            # Brightness analysis
            if method_brightness[method]['thermal']:
                t_avg = sum(method_brightness[method]['thermal']) / len(method_brightness[method]['thermal'])
                r_avg = sum(method_brightness[method]['rgb']) / len(method_brightness[method]['rgb'])
                lines.append(f"                   thermal_bright={t_avg:.0f}, rgb_bright={r_avg:.0f}")
        
        lines.append("")
        lines.append("BY FLIGHT (may indicate scene conditions):")
        for flight, methods in sorted(flight_methods.items()):
            best = max(methods.items(), key=lambda x: x[1])
            all_methods = ", ".join([f"{m}:{c}" for m,c in sorted(methods.items(), key=lambda x:-x[1])])
            lines.append(f"  Flight {flight:4s}: {all_methods}")
        
        # Offset patterns
        offsets_x = [c['offset_x'] for c in self.tool.corrections]
        offsets_y = [c['offset_y'] for c in self.tool.corrections]
        lines.append("")
        lines.append("OFFSET PATTERNS (camera sync drift):")
        lines.append(f"  X offset: avg={sum(offsets_x)/len(offsets_x):+.1f}px, range=[{min(offsets_x):+d}, {max(offsets_x):+d}]")
        lines.append(f"  Y offset: avg={sum(offsets_y)/len(offsets_y):+.1f}px, range=[{min(offsets_y):+d}, {max(offsets_y):+d}]")
        
        # Insights
        lines.append("")
        lines.append("INSIGHTS:")
        
        # Best overall method
        best_method = max(method_counts.items(), key=lambda x: x[1])
        lines.append(f"  • Best overall: {best_method[0]} ({best_method[1]}/{total})")
        
        # Check if edge-based methods work better on certain brightness
        for method in ['Canny Edges', 'Adaptive Thresh']:
            if method in method_brightness and method_brightness[method]['rgb']:
                avg_rgb = sum(method_brightness[method]['rgb']) / len(method_brightness[method]['rgb'])
                if avg_rgb > 150:
                    lines.append(f"  • {method} tends to work on BRIGHT scenes (avg RGB={avg_rgb:.0f})")
                elif avg_rgb < 100:
                    lines.append(f"  • {method} tends to work on DARK scenes (avg RGB={avg_rgb:.0f})")
        
        for method in ['Grayscale', 'CLAHE Contrast']:
            if method in method_brightness and method_brightness[method]['rgb']:
                avg_rgb = sum(method_brightness[method]['rgb']) / len(method_brightness[method]['rgb'])
                lines.append(f"  • {method} avg scene brightness: {avg_rgb:.0f}")
        
        return "\n".join(lines)
    
    def update_display(self):
        if self.current_data is None:
            return
        
        self.photo_refs = []  # Clear old references
        data = self.current_data
        results = data['results']
        best_result = results[self.current_result_idx] if results else None
        
        # Update info with visibility, certainty and smart ranking info
        sample = data['sample']
        visibility = sample.get('visibility', 0)
        vis_status = "VISIBLE" if visibility == 0 else f"OCCLUDED({visibility})"
        
        # Show certainty score with color indication
        certainty = data.get('certainty', 0)
        certainty_reason = data.get('certainty_reason', '')
        
        if certainty >= 0.75:
            cert_display = f"✓ CERTAIN ({certainty:.0%})"
        elif certainty >= 0.5:
            cert_display = f"? MAYBE ({certainty:.0%})"
        else:
            cert_display = f"✗ UNSURE ({certainty:.0%})"
        
        # Count bboxes in current frame for apply-to-all hint
        frame_key = (sample['flight_key'], sample['frame_id'])
        frame_bbox_count = len(self.tool.frame_samples.get(frame_key, []))
        frame_hint = f" | Frame has {frame_bbox_count} bboxes" if frame_bbox_count > 1 else ""
        
        self.info_label.config(text=f"Flight: {sample['flight_key']} | Frame: {sample['frame_id']} | "
                                    f"BBox: {data['original_bbox']['width']}x{data['original_bbox']['height']} | "
                                    f"{cert_display}{frame_hint}")
        
        # Show certainty reasoning in a tooltip-like way (update stats label)
        self.stats_label.config(text=f"Good: {len(self.tool.corrections)} | Bad: {len(self.tool.bad_cases)} | {certainty_reason[:60]}...")
        
        # 1. Thermal crop
        photo = self.cv2_to_tk(data['thermal_crop'])
        self.thermal_crop_canvas.delete("all")
        if photo:
            self.thermal_crop_canvas.create_image(100, 100, image=photo)
        
        # 2. RGB original crop
        photo = self.cv2_to_tk(data['rgb_original_crop'])
        self.rgb_orig_crop_canvas.delete("all")
        if photo:
            self.rgb_orig_crop_canvas.create_image(100, 100, image=photo)
        
        # 3. Update each method row
        for i, method_frame in enumerate(self.method_frames):
            if i < len(results):
                r = results[i]
                
                # Update labels - always show confidence
                method_frame['conf_label'].config(text=f"Conf: {r['confidence']:.3f}")
                method_frame['offset_label'].config(text=f"Δ({r['offset_x']:+d}, {r['offset_y']:+d})")
                
                # Highlight selected method (green), show preference if high
                if i == self.current_result_idx:
                    method_frame['label'].config(foreground='#00ff00')
                elif 'preference_weight' in r and r['preference_weight'] > 1.3:
                    method_frame['label'].config(foreground='#ffff00')  # Yellow = user prefers
                else:
                    method_frame['label'].config(foreground='white')
                
                # Thermal processed
                photo = self.cv2_to_tk(r['thermal_processed'], 100, 100)
                method_frame['thermal_canvas'].delete("all")
                if photo:
                    method_frame['thermal_canvas'].create_image(50, 50, image=photo)
                
                # RGB crop PROCESSED (apply same method to the corrected crop)
                if 'rgb_corrected_crop' in r and r['rgb_corrected_crop'] is not None and r['rgb_corrected_crop'].size > 0:
                    # Get the method's convert function and apply to crop
                    method_func = self.tool.matching_methods[i][1]
                    rgb_crop_processed = method_func(r['rgb_corrected_crop'])
                    photo = self.cv2_to_tk(rgb_crop_processed, 100, 100)
                    method_frame['rgb_crop_processed_canvas'].delete("all")
                    if photo:
                        method_frame['rgb_crop_processed_canvas'].create_image(50, 50, image=photo)
                
                # RGB corrected crop RAW for this method
                if 'rgb_corrected_crop' in r and r['rgb_corrected_crop'] is not None and r['rgb_corrected_crop'].size > 0:
                    photo = self.cv2_to_tk(r['rgb_corrected_crop'], 100, 100)
                    method_frame['rgb_crop_canvas'].delete("all")
                    if photo:
                        method_frame['rgb_crop_canvas'].create_image(50, 50, image=photo)
        
        # 5. Search region overview
        if best_result:
            orig_offset = (data['search_coords']['original_offset_x'], 
                          data['search_coords']['original_offset_y'])
            bbox_size = (data['original_bbox']['width'], data['original_bbox']['height'])
            
            # Show plain search region without bboxes
            photo = self.cv2_to_tk(data['rgb_search'], 280, 200)
            self.search_canvas.delete("all")
            if photo:
                self.search_canvas.create_image(140, 100, image=photo)
            
            # 6. Heatmap showing exhaustive search
            if 'heatmap' in best_result and best_result['heatmap'] is not None:
                # Draw match location on heatmap
                heatmap_vis = best_result['heatmap'].copy()
                # Mark best location with a white cross
                mx, my = best_result['match_loc']
                cv2.drawMarker(heatmap_vis, (mx, my), (255, 255, 255), 
                              cv2.MARKER_CROSS, 10, 2)
                photo = self.cv2_to_tk(heatmap_vis, 280, 150)
                self.heatmap_canvas.delete("all")
                if photo:
                    self.heatmap_canvas.create_image(140, 75, image=photo)
            
            # Calculate search stats
            search_h, search_w = data['rgb_search'].shape[:2]
            template_h, template_w = data['thermal_crop'].shape[:2]
            positions_tested = (search_w - template_w + 1) * (search_h - template_h + 1)
            
            self.best_method_label.config(
                text=f"Best: {best_result['method']}\n"
                     f"Conf: {best_result['confidence']:.3f}\n"
                     f"Offset: ({best_result['offset_x']:+d}, {best_result['offset_y']:+d}) px"
            )
            
            self.search_stats_label.config(
                text=f"Search: {search_w}x{search_h} px\n"
                     f"Template: {template_w}x{template_h} px\n"
                     f"Positions tested: {positions_tested:,}"
            )
        
        # Update progress - show processed count
        processed = len(self.tool.processed_indices)
        total = len(self.tool.samples)
        self.progress_label.config(text=f"Progress: {processed}/{total}")
        self.stats_label.config(text=f"Good: {len(self.tool.corrections)} | Bad: {len(self.tool.bad_cases)}")
    
    def load_current_sample(self):
        if self.tool.current_idx >= len(self.tool.samples):
            self.show_completion()
            return
        
        sample = self.tool.samples[self.tool.current_idx]
        self.current_data = self.tool.compute_correction(sample)
        self.current_result_idx = 0
        
        if self.current_data is None:
            self.next_sample()  # Skip invalid samples
        else:
            self.update_display()
    
    def mark_good(self):
        if self.current_data is None or not self.current_data['results']:
            return
        
        result = self.current_data['results'][self.current_result_idx]
        sample = self.current_data['sample']
        data = self.current_data
        
        # Compute image characteristics for pattern analysis
        thermal_crop = data['thermal_crop']
        rgb_crop = data['rgb_original_crop']
        
        # Brightness and contrast metrics
        thermal_mean = float(np.mean(thermal_crop))
        thermal_std = float(np.std(thermal_crop))
        rgb_mean = float(np.mean(rgb_crop)) if rgb_crop.size > 0 else 0
        rgb_std = float(np.std(rgb_crop)) if rgb_crop.size > 0 else 0
        
        correction = {
            'flight_key': sample['flight_key'],
            'frame_id': sample['frame_id'],
            'split': sample['split'],
            'original_bbox': sample['bbox'],
            'corrected_bbox': result['corrected_bbox'],
            'method': result['method'],
            'confidence': result['confidence'],
            'offset_x': result['offset_x'],
            'offset_y': result['offset_y'],
            'annotation': sample['annotation'],
            # Image characteristics for pattern analysis
            'thermal_brightness': thermal_mean,
            'thermal_contrast': thermal_std,
            'rgb_brightness': rgb_mean,
            'rgb_contrast': rgb_std,
            'bbox_size': sample['bbox']['width'] * sample['bbox']['height']
        }
        
        self.tool.corrections.append(correction)
        self.tool.good_params.append({
            'method': result['method'],
            'confidence': result['confidence'],
            'thermal_brightness': thermal_mean,
            'rgb_brightness': rgb_mean
        })
        
        # Track as processed
        self.tool.processed_indices.add(self.tool.current_idx)
        
        # Show quick pattern update
        if len(self.tool.corrections) >= 3:
            self.update_quick_pattern()
        
        self.next_sample()
    
    def mark_bad(self):
        if self.current_data is None:
            return
        
        sample = self.current_data['sample']
        self.tool.bad_cases.append({
            'flight_key': sample['flight_key'],
            'frame_id': sample['frame_id'],
            'reason': 'user_marked_bad'
        })
        
        self.next_sample()
    
    def auto_accept_certain(self):
        """
        Auto-accept all remaining samples with high certainty (>= 0.75).
        Shows a preview of how many will be accepted before proceeding.
        """
        # Count high-certainty samples from remaining
        remaining_samples = self.tool.samples[self.tool.current_idx:]
        high_certainty = []
        uncertain = []
        
        print("\nAnalyzing remaining samples for auto-acceptance...")
        for i, sample in enumerate(remaining_samples):
            data = self.tool.compute_correction(sample)
            if data and data.get('certainty', 0) >= 0.75:
                high_certainty.append((sample, data))
            else:
                uncertain.append(sample)
        
        if not high_certainty:
            print("No high-certainty samples found.")
            popup = tk.Toplevel(self.root)
            popup.title("Auto-Certain")
            popup.geometry("350x120")
            popup.configure(bg='#2b2b2b')
            tk.Label(popup, text="No samples with certainty >= 75%", 
                    bg='#2b2b2b', fg='white', font=('Arial', 12)).pack(pady=20)
            tk.Button(popup, text="OK", command=popup.destroy).pack()
            return
        
        # Show confirmation dialog
        msg = f"Found {len(high_certainty)} high-certainty samples\n"
        msg += f"({len(uncertain)} uncertain samples will remain)\n\n"
        msg += "Auto-accept all?"
        
        popup = tk.Toplevel(self.root)
        popup.title("Auto-Certain Confirmation")
        popup.geometry("400x200")
        popup.configure(bg='#2b2b2b')
        
        tk.Label(popup, text=msg, bg='#2b2b2b', fg='white', font=('Arial', 12)).pack(pady=20)
        
        def do_auto_accept():
            popup.destroy()
            self._execute_auto_accept(high_certainty, uncertain)
        
        btn_frame = ttk.Frame(popup)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="✓ Yes, Auto-Accept", command=do_auto_accept,
                 bg='#4CAF50', fg='white', font=('Arial', 12)).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", command=popup.destroy,
                 bg='#666', fg='white', font=('Arial', 12)).pack(side=tk.LEFT, padx=10)
    
    def _execute_auto_accept(self, high_certainty, uncertain):
        """Execute the auto-acceptance of high-certainty samples."""
        print(f"\nAuto-accepting {len(high_certainty)} samples...")
        
        for sample, data in high_certainty:
            # Use the suggested result (consensus-based)
            result = data.get('suggested_result', data['results'][0])
            
            thermal_crop = data['thermal_crop']
            rgb_crop = data['rgb_original_crop']
            
            thermal_mean = float(np.mean(thermal_crop))
            thermal_std = float(np.std(thermal_crop))
            rgb_mean = float(np.mean(rgb_crop)) if rgb_crop.size > 0 else 0
            rgb_std = float(np.std(rgb_crop)) if rgb_crop.size > 0 else 0
            
            correction = {
                'flight_key': sample['flight_key'],
                'frame_id': sample['frame_id'],
                'split': sample['split'],
                'original_bbox': sample['bbox'],
                'corrected_bbox': result['corrected_bbox'],
                'method': result['method'],
                'confidence': result['confidence'],
                'offset_x': result['offset_x'],
                'offset_y': result['offset_y'],
                'annotation': sample['annotation'],
                'thermal_brightness': thermal_mean,
                'thermal_contrast': thermal_std,
                'rgb_brightness': rgb_mean,
                'rgb_contrast': rgb_std,
                'bbox_size': sample['bbox']['width'] * sample['bbox']['height'],
                'auto_accepted': True,
                'certainty': data['certainty']
            }
            
            self.tool.corrections.append(correction)
            self.tool.good_params.append({
                'method': result['method'],
                'confidence': result['confidence'],
                'thermal_brightness': thermal_mean,
                'rgb_brightness': rgb_mean
            })
        
        # Update samples list to only contain uncertain ones
        self.tool.samples = uncertain
        self.tool.current_idx = 0
        
        self.tool.save_session()
        
        print(f"Auto-accepted {len(high_certainty)} samples!")
        print(f"Remaining uncertain: {len(uncertain)}")
        
        # Update progress display
        self.progress_label.config(text=f"Progress: 1/{len(uncertain)} (auto-accepted {len(high_certainty)})")
        
        if uncertain:
            self.load_current_sample()
        else:
            self.show_completion()
    
    def skip(self):
        self.next_sample()
    
    def next_sample(self):
        self.tool.current_idx += 1
        if self.tool.current_idx >= len(self.tool.samples):
            self.show_completion()
        else:
            self.load_current_sample()
    
    def prev_sample(self):
        if self.tool.current_idx > 0:
            self.tool.current_idx -= 1
            self.load_current_sample()
    
    def show_completion(self):
        processed = len(self.tool.processed_indices)
        total = len(self.tool.samples)
        msg = f"Validation Complete!\n\nProcessed: {processed}/{total}\nGood: {len(self.tool.corrections)}\nBad: {len(self.tool.bad_cases)}"
        
        # In demo mode (sample_data), save visualizations
        if self.tool.use_sample_data and len(self.tool.corrections) > 0:
            print("\n[DEMO MODE] Saving visualizations...")
            self._save_demo_visualizations()
            msg += f"\n\nVisualizations saved to:\n{self.tool.viz_output_dir}"
        
        popup = tk.Toplevel(self.root)
        popup.title("Complete")
        popup.geometry("400x200")
        popup.configure(bg='#2b2b2b')
        tk.Label(popup, text=msg, bg='#2b2b2b', fg='white', font=('Arial', 12)).pack(pady=20)
        tk.Button(popup, text="OK", command=popup.destroy).pack()
    
    def _save_demo_visualizations(self):
        """
        Save visualization images for demo mode showing:
        1. Thermal image with all bboxes
        2. RGB image with original (uncorrected) bboxes
        3. RGB image with corrected bboxes
        """
        # Group corrections by frame
        frame_corrections = {}
        for c in self.tool.corrections:
            key = (c['flight_key'], c['frame_id'])
            if key not in frame_corrections:
                frame_corrections[key] = []
            frame_corrections[key].append(c)
        
        print(f"Saving visualizations for {len(frame_corrections)} unique frames...")
        
        for (flight_key, frame_id), corrections in frame_corrections.items():
            # Get image paths from first correction
            first_corr = corrections[0]
            
            # Find matching sample to get image paths
            sample = None
            for s in self.tool.samples:
                if s['flight_key'] == flight_key and s['frame_id'] == frame_id:
                    sample = s
                    break
            
            if not sample:
                continue
            
            # Load images
            thermal_img = cv2.imread(sample['thermal_path'])
            rgb_img = cv2.imread(sample['rgb_path'])
            
            if thermal_img is None or rgb_img is None:
                continue
            
            # Create copies for drawing
            thermal_viz = thermal_img.copy()
            rgb_original_viz = rgb_img.copy()
            rgb_corrected_viz = rgb_img.copy()
            
            # Draw all bboxes
            for c in corrections:
                orig_bbox = c['original_bbox']
                corr_bbox = c['corrected_bbox']
                
                # Thermal: draw bbox in green
                cv2.rectangle(thermal_viz,
                             (orig_bbox['x_min'], orig_bbox['y_min']),
                             (orig_bbox['x_max'], orig_bbox['y_max']),
                             (0, 255, 0), 2)
                
                # RGB original: draw original bbox in red
                cv2.rectangle(rgb_original_viz,
                             (orig_bbox['x_min'], orig_bbox['y_min']),
                             (orig_bbox['x_max'], orig_bbox['y_max']),
                             (0, 0, 255), 2)
                
                # RGB corrected: draw corrected bbox in green
                cv2.rectangle(rgb_corrected_viz,
                             (corr_bbox['x_min'], corr_bbox['y_min']),
                             (corr_bbox['x_max'], corr_bbox['y_max']),
                             (0, 255, 0), 2)
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(thermal_viz, 'THERMAL', (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(rgb_original_viz, 'RGB - ORIGINAL (uncorrected)', (10, 30), font, 0.8, (0, 0, 255), 2)
            cv2.putText(rgb_corrected_viz, 'RGB - CORRECTED', (10, 30), font, 1, (0, 255, 0), 2)
            
            # Combine into a single visualization (horizontal stack)
            # Resize to same height if needed
            h = max(thermal_viz.shape[0], rgb_original_viz.shape[0])
            
            def resize_to_height(img, target_h):
                scale = target_h / img.shape[0]
                return cv2.resize(img, (int(img.shape[1] * scale), target_h))
            
            thermal_viz = resize_to_height(thermal_viz, h)
            rgb_original_viz = resize_to_height(rgb_original_viz, h)
            rgb_corrected_viz = resize_to_height(rgb_corrected_viz, h)
            
            # Stack horizontally
            combined = np.hstack([thermal_viz, rgb_original_viz, rgb_corrected_viz])
            
            # Save
            output_path = self.tool.viz_output_dir / f"manual_frame_{flight_key}_{frame_id}.png"
            cv2.imwrite(str(output_path), combined)
            print(f"  Saved: {output_path.name}")
        
        print(f"Saved {len(frame_corrections)} visualizations to {self.tool.viz_output_dir}")
    
    def auto_correct_all(self):
        if not self.tool.good_params:
            print("No good parameters learned yet!")
            return
        
        method_stats = {}
        for param in self.tool.good_params:
            method = param['method']
            if method not in method_stats:
                method_stats[method] = {'count': 0, 'total_conf': 0}
            method_stats[method]['count'] += 1
            method_stats[method]['total_conf'] += param['confidence']
        
        best_method = max(method_stats.keys(), 
                         key=lambda m: (method_stats[m]['count'], 
                                       method_stats[m]['total_conf'] / method_stats[m]['count']))
        
        print(f"Auto-correcting with: {best_method}")
        
        for split in ['train', 'val', 'test']:
            self.auto_correct_split(split, best_method)
        
        self.tool.save_session()
        print("Done!")
    
    def auto_correct_split(self, split, best_method_name):
        print(f"Processing {split}...")
        
        output_dir = self.tool.output_dir / split
        output_dir.mkdir(exist_ok=True)
        
        method_idx = next((i for i, (name, _) in enumerate(self.tool.matching_methods) 
                          if name == best_method_name), 0)
        method_func = self.tool.matching_methods[method_idx][1]
        
        metadata_path = self.tool.metadata_dir / split
        if not metadata_path.exists():
            return
        
        for meta_file in metadata_path.glob("*_metadata.json"):
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            flight_key = metadata['flight_key']
            corrected_frames = {}
            
            for frame_id, frame_data in metadata.get('frames', {}).items():
                thermal_img_name = frame_data.get('thermal_image')
                rgb_img_name = frame_data.get('rgb_image')
                annotations = frame_data.get('annotations', [])
                
                if not thermal_img_name or not rgb_img_name:
                    corrected_frames[frame_id] = frame_data
                    continue
                
                thermal_path = self.tool.thermal_dir / split / thermal_img_name
                rgb_path = self.tool.rgb_dir / split / rgb_img_name
                
                if not thermal_path.exists() or not rgb_path.exists():
                    corrected_frames[frame_id] = frame_data
                    continue
                
                thermal_img = cv2.imread(str(thermal_path))
                rgb_img = cv2.imread(str(rgb_path))
                
                if thermal_img is None or rgb_img is None:
                    corrected_frames[frame_id] = frame_data
                    continue
                
                corrected_annotations = []
                for ann in annotations:
                    bbox = ann.get('bbox')
                    if not bbox:
                        corrected_annotations.append(ann)
                        continue
                    
                    try:
                        thermal_crop = thermal_img[bbox['y_min']:bbox['y_max'], 
                                                   bbox['x_min']:bbox['x_max']]
                        
                        rgb_search, search_coords = self.tool.expand_search_area(rgb_img, bbox)
                        
                        match_loc, confidence, _, _, _ = self.tool.find_best_match(
                            thermal_crop, rgb_search, method_func
                        )
                        
                        if match_loc:
                            corrected_bbox = {
                                'x_min': search_coords['x_min'] + match_loc[0],
                                'y_min': search_coords['y_min'] + match_loc[1],
                                'x_max': search_coords['x_min'] + match_loc[0] + bbox['width'],
                                'y_max': search_coords['y_min'] + match_loc[1] + bbox['height'],
                                'width': bbox['width'],
                                'height': bbox['height']
                            }
                            
                            corrected_ann = ann.copy()
                            corrected_ann['bbox'] = corrected_bbox
                            corrected_ann['original_bbox'] = bbox
                            corrected_ann['correction_confidence'] = float(confidence)
                            corrected_annotations.append(corrected_ann)
                        else:
                            corrected_annotations.append(ann)
                    except:
                        corrected_annotations.append(ann)
                
                corrected_frame = frame_data.copy()
                corrected_frame['annotations'] = corrected_annotations
                corrected_frames[frame_id] = corrected_frame
            
            corrected_metadata = metadata.copy()
            corrected_metadata['frames'] = corrected_frames
            corrected_metadata['correction_timestamp'] = datetime.now().isoformat()
            
            output_file = output_dir / f"{flight_key}_corrected_metadata.json"
            with open(output_file, 'w') as f:
                json.dump(corrected_metadata, f, indent=2)
            
            print(f"  Saved: {output_file.name}")
    
    def save_and_quit(self):
        self.tool.save_session()
        print("\n" + self.analyze_patterns())
        self.root.quit()
        self.root.destroy()
    
    def show_patterns(self):
        """Show pattern analysis in a popup"""
        analysis = self.analyze_patterns()
        
        popup = tk.Toplevel(self.root)
        popup.title("Pattern Analysis")
        popup.geometry("500x400")
        popup.configure(bg='#2b2b2b')
        
        text = tk.Text(popup, bg='#1a1a1a', fg='white', font=('Courier', 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert('1.0', analysis)
        text.config(state='disabled')
        
        tk.Button(popup, text="Close", command=popup.destroy, 
                  bg='#666', fg='white').pack(pady=5)
    
    def update_quick_pattern(self):
        """Update the pattern label with quick stats"""
        method_counts = {}
        for c in self.tool.corrections:
            method = c['method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        total = len(self.tool.corrections)
        lines = []
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1])[:3]:
            lines.append(f"{method}: {count} ({100*count/total:.0f}%)")
        
        self.pattern_label.config(text="\n".join(lines))
    
    def run(self):
        self.load_current_sample()
        self.root.mainloop()


def main():
    """
    Main entry point for the interactive correction tool.
    
    By default, uses sample_data for quick testing.
    To use full dataset, set USE_SAMPLE_DATA = False below.
    """
    # ==================== CONFIGURATION ====================
    # Set to False to use full dataset (images/, rgb_images/, metadata/)
    USE_SAMPLE_DATA = True
    
    # Data directory (root of the BAMBI_Data folder)
    DATA_DIR = r"c:\Users\hma\Desktop\BAMBI_Data"
    
    # For full dataset: which split to use
    SPLIT = 'test'
    
    # Maximum samples to load
    MAX_SAMPLES = 500
    # =======================================================
    
    print("=" * 60)
    print("Thermal-RGB BBox Correction Tool v2")
    print("INTERACTIVE MODE - All samples")
    print("=" * 60)
    
    if USE_SAMPLE_DATA:
        print(f"\nUsing SAMPLE DATA from: {DATA_DIR}/sample_data/")
        tool = BBoxCorrectionTool(DATA_DIR, use_sample_data=True)
        tool.collect_samples(max_samples=MAX_SAMPLES, use_flat_structure=True)
    else:
        print(f"\nUsing FULL DATASET from: {DATA_DIR}")
        tool = BBoxCorrectionTool(DATA_DIR, use_sample_data=False)
        tool.collect_samples(split=SPLIT, max_samples=MAX_SAMPLES)
    
    if not tool.samples:
        print("No samples found!")
        print(f"  Thermal dir: {tool.thermal_dir}")
        print(f"  RGB dir: {tool.rgb_dir}")
        print(f"  Metadata dir: {tool.metadata_dir}")
        return
    
    print(f"\nLoaded {len(tool.samples)} samples for verification")
    
    print("\nKeyboard shortcuts:")
    print("  1-5       - Accept with method (applies to ALL bboxes in frame)")
    print("  Shift+1-5 - Accept single bbox only (override)")
    print("  R         - Reject")
    print("  S         - Skip")
    print("  ←/→       - Navigate")
    print("  ESC       - Save & Quit")
    print("\nNote: Selecting any method automatically applies the shift")
    print("      to all bboxes in the frame. Progress updates accordingly.")
    
    if USE_SAMPLE_DATA:
        print("\n[DEMO MODE] Visualizations will be saved on completion.")
    
    gui = CorrectionGUI(tool)
    gui.run()


if __name__ == "__main__":
    main()
