"""
Correction Core Engine
======================
Core functionality for thermal-to-RGB bounding box alignment correction.

Uses template matching with multiple image processing methods to find
the best alignment between thermal and RGB imagery.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict


class BBoxCorrectionEngine:
    """
    Engine for correcting bounding box alignment between thermal and RGB images.
    
    Uses template matching with multiple preprocessing methods to find the
    optimal position in RGB that matches the thermal detection.
    """
    
    def __init__(self, expansion_factor: float = 1.0):
        """
        Initialize the correction engine.
        
        Args:
            expansion_factor: How much to expand the search area relative to bbox size.
                              1.0 = search in a 3x3 area (1x expansion on each side)
        """
        self.expansion_factor = expansion_factor
        
        # Matching methods: name -> preprocessing function
        self.matching_methods = [
            ('Grayscale', self._convert_to_grayscale),
            ('CLAHE Contrast', self._enhance_contrast),
            ('Canny Edges', self._edge_detection),
            ('Adaptive Thresh', self._adaptive_threshold),
            ('LAB Luminance', self._color_to_luminance),
        ]
        
        # Learned parameters from corrections
        self.good_params = []
        self.corrections = []
        self.bad_cases = []
    
    def _convert_to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement."""
        gray = self._convert_to_grayscale(img)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    def _edge_detection(self, img: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection."""
        gray = self._convert_to_grayscale(img)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, 50, 150)
    
    def _adaptive_threshold(self, img: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding."""
        gray = self._convert_to_grayscale(img)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    
    def _color_to_luminance(self, img: np.ndarray) -> np.ndarray:
        """Extract luminance channel from LAB color space."""
        if len(img.shape) == 2:
            gray = img
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            gray = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    def expand_search_area(self, img: np.ndarray, bbox: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Expand the search region around a bounding box.
        
        Args:
            img: Source image
            bbox: Bounding box dict with x_min, y_min, x_max, y_max
            
        Returns:
            Tuple of (expanded_region, coordinates_dict)
        """
        h, w = img.shape[:2]
        x_min, y_min = bbox['x_min'], bbox['y_min']
        x_max, y_max = bbox['x_max'], bbox['y_max']
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        expand_x = int(bbox_w * self.expansion_factor)
        expand_y = int(bbox_h * self.expansion_factor)
        
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
    
    def find_best_match(self, thermal_crop: np.ndarray, rgb_search_region: np.ndarray,
                        convert_func: Callable) -> Tuple[Optional[Tuple[int, int]], float,
                                                         np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Find the best matching position using template matching.
        
        Args:
            thermal_crop: The thermal image crop (template)
            rgb_search_region: The RGB region to search within
            convert_func: Preprocessing function to apply
            
        Returns:
            Tuple of (match_location, confidence, thermal_processed, rgb_processed, heatmap)
        """
        thermal_processed = convert_func(thermal_crop)
        rgb_processed = convert_func(rgb_search_region)
        
        if thermal_processed.shape[0] > rgb_processed.shape[0] or \
           thermal_processed.shape[1] > rgb_processed.shape[1]:
            return None, 0, thermal_processed, rgb_processed, None
        
        # Exhaustive template matching
        result = cv2.matchTemplate(rgb_processed, thermal_processed, cv2.TM_CCOEFF_NORMED)
        
        # Find global best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Normalize heatmap for visualization
        heatmap = ((result - result.min()) / (result.max() - result.min() + 1e-8) * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return max_loc, max_val, thermal_processed, rgb_processed, heatmap_colored
    
    def compute_correction(self, thermal_img: np.ndarray, rgb_img: np.ndarray,
                          bbox: Dict) -> Optional[Dict]:
        """
        Compute correction for a single bounding box.
        
        Args:
            thermal_img: Full thermal image
            rgb_img: Full RGB image
            bbox: Bounding box dict with x_min, y_min, x_max, y_max, width, height
            
        Returns:
            Dictionary with correction results, or None if failed
        """
        # Extract thermal crop
        thermal_crop = thermal_img[bbox['y_min']:bbox['y_max'],
                                   bbox['x_min']:bbox['x_max']]
        
        if thermal_crop.size == 0:
            return None
        
        # Get expanded search region from RGB
        rgb_search, search_coords = self.expand_search_area(rgb_img, bbox)
        
        # Get original RGB crop at same position as thermal
        rgb_original_crop = rgb_img[bbox['y_min']:bbox['y_max'],
                                    bbox['x_min']:bbox['x_max']]
        
        results = []
        
        for method_name, convert_func in self.matching_methods:
            try:
                match_loc, confidence, thermal_processed, rgb_processed, heatmap = \
                    self.find_best_match(thermal_crop, rgb_search, convert_func)
                
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
                    'heatmap': heatmap
                })
                
            except Exception as e:
                continue
        
        if not results:
            return None
        
        # Rank results by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Compute certainty score
        certainty, certainty_reason, suggested_result = self._analyze_certainty(results, thermal_crop)
        
        return {
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
    
    def _analyze_certainty(self, results: List[Dict], thermal_crop: np.ndarray) -> Tuple[float, str, Dict]:
        """
        Analyze how certain we are about the correction.
        
        Args:
            results: List of matching results from all methods
            thermal_crop: The thermal crop for analysis
            
        Returns:
            Tuple of (certainty_score, reason_string, suggested_result)
        """
        if not results:
            return 0.0, "No results", None
        
        certainty = 0.5
        reasons = []
        
        # Check method agreement on offset
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
        
        # Check confidence level
        top_conf = results[0]['confidence']
        if top_conf > 0.5:
            certainty += 0.1
            reasons.append(f"High conf {top_conf:.2f}")
        elif top_conf < 0.2:
            certainty -= 0.15
            reasons.append(f"Low conf {top_conf:.2f}")
        
        # Check if top method matches consensus
        top_result = results[0]
        if most_common_offset == (top_result['offset_x'], top_result['offset_y']):
            certainty += 0.1
            reasons.append("Top matches consensus")
        else:
            certainty -= 0.1
            reasons.append("Top differs from consensus")
        
        # Clamp certainty
        certainty = max(0.0, min(1.0, certainty))
        
        # Find result that matches consensus
        suggested_result = top_result
        for r in results:
            if (r['offset_x'], r['offset_y']) == most_common_offset:
                suggested_result = r
                break
        
        reason_str = " | ".join(reasons)
        return certainty, reason_str, suggested_result


def cluster_shifts(all_shifts: List[Dict], tolerance: int = 10) -> List[Dict]:
    """
    Cluster offset shifts that are within ±tolerance pixels of each other.
    
    Args:
        all_shifts: List of shift dictionaries with offset_x, offset_y
        tolerance: Maximum pixel difference to consider same cluster
        
    Returns:
        List of clusters, sorted by size (largest first)
    """
    if not all_shifts:
        return []
    
    clusters = []
    
    for shift in all_shifts:
        ox, oy = shift['offset_x'], shift['offset_y']
        
        # Try to find existing cluster within tolerance
        assigned = False
        for cluster in clusters:
            center_x, center_y = cluster['center']
            if abs(ox - center_x) <= tolerance and abs(oy - center_y) <= tolerance:
                cluster['members'].append(shift)
                # Update center to mean
                all_x = [m['offset_x'] for m in cluster['members']]
                all_y = [m['offset_y'] for m in cluster['members']]
                cluster['center'] = (np.mean(all_x), np.mean(all_y))
                assigned = True
                break
        
        if not assigned:
            clusters.append({
                'center': (ox, oy),
                'members': [shift]
            })
    
    # Sort by number of members (descending)
    clusters.sort(key=lambda c: len(c['members']), reverse=True)
    
    return clusters


def compute_consensus_score(cluster: Dict, num_detections: int, 
                           num_methods: int = 5) -> Tuple[float, Dict]:
    """
    Compute a consensus score for a cluster.
    
    Args:
        cluster: Cluster dictionary with center and members
        num_detections: Total number of detections in the frame
        num_methods: Number of matching methods used
        
    Returns:
        Tuple of (score, details_dict)
    """
    members = cluster['members']
    
    # Which detections contributed?
    detection_ids = set(m['detection_idx'] for m in members)
    detection_coverage = len(detection_ids) / num_detections
    
    # Total votes
    total_votes = len(members)
    max_possible_votes = num_detections * num_methods
    vote_ratio = total_votes / max_possible_votes
    
    # Average confidence
    avg_confidence = np.mean([m['confidence'] for m in members])
    
    # Combined score
    score = (
        0.5 * detection_coverage +  # 50% weight on detection coverage
        0.3 * vote_ratio +          # 30% weight on total votes
        0.2 * avg_confidence        # 20% weight on confidence
    )
    
    return score, {
        'detection_coverage': detection_coverage,
        'detections_voting': len(detection_ids),
        'total_detections': num_detections,
        'total_votes': total_votes,
        'avg_confidence': avg_confidence
    }
