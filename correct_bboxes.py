#!/usr/bin/env python3
"""
Thermal-RGB BBox Automatic Correction Tool
==========================================
Corrects bounding box alignment between thermal and RGB images using
consensus-based template matching with multiple preprocessing methods.

Usage:
    python correct_bboxes.py --thermal-dir ./thermal --rgb-dir ./rgb --labels-dir ./labels

For help:
    python correct_bboxes.py --help
"""

import argparse
import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Import core modules
sys.path.insert(0, str(Path(__file__).parent))
from src.correction_core import BBoxCorrectionEngine, cluster_shifts, compute_consensus_score
from src.label_parsers import YOLOLabelParser, MetadataParser, detect_label_format


class AutoCorrectionTool:
    """Automated batch correction tool using consensus-based approach."""
    
    def __init__(self, thermal_dir: str, rgb_dir: str, labels_dir: str,
                 output_dir: str = None, tolerance: int = 10,
                 min_consensus_score: float = 0.4, min_detection_coverage: float = 0.67):
        """
        Initialize the auto correction tool.
        
        Args:
            thermal_dir: Path to thermal images
            rgb_dir: Path to RGB images
            labels_dir: Path to labels (YOLO .txt or JSON metadata)
            output_dir: Path for output (default: ./output)
            tolerance: Pixel tolerance for clustering shifts (±N pixels)
            min_consensus_score: Minimum score to accept consensus
            min_detection_coverage: Minimum fraction of detections that must agree
        """
        self.thermal_dir = Path(thermal_dir)
        self.rgb_dir = Path(rgb_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.tolerance = tolerance
        self.min_consensus_score = min_consensus_score
        self.min_detection_coverage = min_detection_coverage
        
        self.engine = BBoxCorrectionEngine(expansion_factor=1.0)
        self.label_format = None
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'consensus_found': 0,
            'no_consensus': 0,
            'total_corrections': 0,
            'multi_detection_corrections': 0,
            'single_detection_corrections': 0
        }
    
    def run(self, save_visualizations: bool = True, viz_interval: int = 100):
        """
        Run automatic correction on all data.
        
        Args:
            save_visualizations: Whether to save visualization images
            viz_interval: Save visualization every N corrections
        """
        print("=" * 70)
        print("THERMAL-RGB BBOX AUTOMATIC CORRECTION")
        print(f"Tolerance: ±{self.tolerance}px")
        print(f"Min consensus score: {self.min_consensus_score}")
        print(f"Min detection coverage: {self.min_detection_coverage:.0%}")
        print("=" * 70)
        
        # Detect label format
        try:
            self.label_format = detect_label_format(self.labels_dir)
            print(f"Detected label format: {self.label_format}")
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        # Load all samples
        print("\nLoading samples...")
        samples_by_frame = self._load_samples_by_frame()
        
        if not samples_by_frame:
            print("No samples found! Check your directory paths.")
            return
        
        # Separate multi-detection and single-detection frames
        multi_detection_frames = {k: v for k, v in samples_by_frame.items() if len(v) >= 2}
        single_detection_frames = {k: v for k, v in samples_by_frame.items() if len(v) == 1}
        
        print(f"Total frames: {len(samples_by_frame)}")
        print(f"Frames with 2+ detections: {len(multi_detection_frames)}")
        print(f"Frames with 1 detection: {len(single_detection_frames)}")
        
        # Create visualization directory
        viz_dir = self.output_dir / 'visualizations'
        if save_visualizations:
            viz_dir.mkdir(exist_ok=True)
        
        # Process multi-detection frames with consensus
        print(f"\nProcessing multi-detection frames...")
        multi_corrections = self._process_multi_detection_frames(
            multi_detection_frames, viz_dir, viz_interval, save_visualizations
        )
        
        # Process single-detection frames
        print(f"\nProcessing single-detection frames...")
        single_corrections = self._process_single_detection_frames(single_detection_frames)
        
        # Combine corrections
        all_corrections = multi_corrections + single_corrections
        
        # Save results
        self._save_results(all_corrections)
        
        # Print summary
        self._print_summary(all_corrections)
    
    def _load_samples_by_frame(self) -> dict:
        """Load samples and group by frame."""
        samples_by_frame = defaultdict(list)
        
        if self.label_format == 'yolo':
            samples = self._load_yolo_samples()
        else:
            samples = self._load_metadata_samples()
        
        for sample in samples:
            frame_key = sample.get('frame_id', sample.get('thermal_path', ''))
            samples_by_frame[frame_key].append(sample)
        
        return dict(samples_by_frame)
    
    def _load_yolo_samples(self) -> list:
        """Load samples from YOLO format labels."""
        samples = []
        
        thermal_images = list(self.thermal_dir.glob("*.jpg")) + \
                        list(self.thermal_dir.glob("*.png"))
        
        for thermal_path in thermal_images:
            label_name = thermal_path.stem + ".txt"
            label_path = self.labels_dir / label_name
            
            if not label_path.exists():
                continue
            
            # Find RGB image
            rgb_path = self.rgb_dir / thermal_path.name
            if not rgb_path.exists():
                for ext in ['.jpg', '.png', '.jpeg']:
                    alt_path = self.rgb_dir / (thermal_path.stem + ext)
                    if alt_path.exists():
                        rgb_path = alt_path
                        break
            
            if not rgb_path.exists():
                continue
            
            img = cv2.imread(str(thermal_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            annotations = YOLOLabelParser.parse_file(str(label_path), w, h)
            
            for i, ann in enumerate(annotations):
                samples.append({
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
        
        return samples
    
    def _load_metadata_samples(self) -> list:
        """Load samples from JSON metadata format."""
        samples = []
        
        metadata_files = list(self.labels_dir.glob("*_metadata.json")) + \
                        list(self.labels_dir.glob("*.json"))
        
        # Remove duplicates
        metadata_files = list(set(metadata_files))
        
        for meta_path in metadata_files:
            try:
                metadata = MetadataParser.parse_file(str(meta_path))
                frame_samples = MetadataParser.extract_samples(
                    metadata, self.thermal_dir, self.rgb_dir
                )
                samples.extend(frame_samples)
            except Exception as e:
                print(f"Warning: Could not parse {meta_path}: {e}")
        
        return samples
    
    def _process_multi_detection_frames(self, frames: dict, viz_dir: Path,
                                        viz_interval: int, save_viz: bool) -> list:
        """Process frames with multiple detections using consensus."""
        corrections = []
        processed = 0
        last_viz_count = 0
        
        for frame_key, samples in frames.items():
            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed}/{len(frames)} frames | "
                      f"Corrections: {len(corrections)}")
            
            # Gather all candidate shifts from all methods and detections
            all_shifts = []
            sample_data_cache = {}
            
            for det_idx, sample in enumerate(samples):
                try:
                    thermal_img = cv2.imread(sample['thermal_path'])
                    rgb_img = cv2.imread(sample['rgb_path'])
                    
                    if thermal_img is None or rgb_img is None:
                        continue
                    
                    data = self.engine.compute_correction(
                        thermal_img, rgb_img, sample['bbox']
                    )
                    
                    sample_data_cache[det_idx] = {
                        'sample': sample,
                        'thermal_img': thermal_img,
                        'rgb_img': rgb_img,
                        'data': data
                    }
                    
                    # Collect shifts from all methods
                    for method_name, method_data in data['methods'].items():
                        if method_data.get('shift'):
                            shift = method_data['shift']
                            confidence = method_data.get('confidence', 0)
                            if confidence > 0.3:  # Minimum confidence threshold
                                all_shifts.append({
                                    'dx': shift['dx'],
                                    'dy': shift['dy'],
                                    'method': method_name,
                                    'det_idx': det_idx,
                                    'confidence': confidence
                                })
                except Exception as e:
                    continue
            
            if not all_shifts:
                continue
            
            # Cluster shifts and find consensus
            shift_clusters = cluster_shifts(all_shifts, tolerance=self.tolerance)
            
            if not shift_clusters:
                continue
            
            # Get best cluster
            best_cluster = shift_clusters[0]
            
            # Count unique detections and methods in cluster
            cluster_dets = set(s['det_idx'] for s in best_cluster['shifts'])
            cluster_methods = set(s['method'] for s in best_cluster['shifts'])
            
            # Compute consensus score
            score = compute_consensus_score(
                len(cluster_methods), len(self.engine.matching_methods),
                len(cluster_dets), len(samples)
            )
            
            coverage = len(cluster_dets) / len(samples)
            
            # Check if consensus meets requirements
            if score >= self.min_consensus_score and coverage >= self.min_detection_coverage:
                dx, dy = best_cluster['center']
                
                # Apply correction to all annotations in frame
                for det_idx, sample in enumerate(samples):
                    original_bbox = sample['bbox'].copy()
                    corrected_bbox = {
                        'x_min': original_bbox['x_min'] + dx,
                        'y_min': original_bbox['y_min'] + dy,
                        'x_max': original_bbox['x_max'] + dx,
                        'y_max': original_bbox['y_max'] + dy,
                        'width': original_bbox['width'],
                        'height': original_bbox['height']
                    }
                    
                    corrections.append({
                        'sample': sample,
                        'original_bbox': original_bbox,
                        'corrected_bbox': corrected_bbox,
                        'shift': {'dx': dx, 'dy': dy},
                        'consensus_score': score,
                        'detection_coverage': coverage,
                        'methods_agreeing': len(cluster_methods),
                        'correction_type': 'multi_detection'
                    })
                
                self.stats['consensus_found'] += 1
                self.stats['multi_detection_corrections'] += len(samples)
                
                # Save visualization periodically
                if save_viz and len(corrections) - last_viz_count >= viz_interval:
                    last_viz_count = len(corrections)
                    self._save_visualization(
                        sample_data_cache, dx, dy, viz_dir, 
                        f"batch_{len(corrections):03d}_{frame_key}"
                    )
            else:
                self.stats['no_consensus'] += 1
        
        return corrections
    
    def _process_single_detection_frames(self, frames: dict) -> list:
        """Process frames with single detections (stricter requirements)."""
        corrections = []
        
        for frame_key, samples in frames.items():
            sample = samples[0]
            
            try:
                thermal_img = cv2.imread(sample['thermal_path'])
                rgb_img = cv2.imread(sample['rgb_path'])
                
                if thermal_img is None or rgb_img is None:
                    continue
                
                data = self.engine.compute_correction(
                    thermal_img, rgb_img, sample['bbox']
                )
                
                # Collect high-confidence shifts
                shifts = []
                for method_name, method_data in data['methods'].items():
                    if method_data.get('shift') and method_data.get('confidence', 0) > 0.5:
                        shifts.append({
                            'dx': method_data['shift']['dx'],
                            'dy': method_data['shift']['dy'],
                            'method': method_name,
                            'confidence': method_data['confidence']
                        })
                
                if len(shifts) < 3:  # Require at least 3 methods to agree
                    continue
                
                # Cluster and check consensus
                clusters = cluster_shifts(shifts, tolerance=self.tolerance)
                if not clusters:
                    continue
                
                best = clusters[0]
                if len(best['shifts']) >= 3:  # Strong method agreement
                    dx, dy = best['center']
                    
                    original_bbox = sample['bbox'].copy()
                    corrected_bbox = {
                        'x_min': original_bbox['x_min'] + dx,
                        'y_min': original_bbox['y_min'] + dy,
                        'x_max': original_bbox['x_max'] + dx,
                        'y_max': original_bbox['y_max'] + dy,
                        'width': original_bbox['width'],
                        'height': original_bbox['height']
                    }
                    
                    corrections.append({
                        'sample': sample,
                        'original_bbox': original_bbox,
                        'corrected_bbox': corrected_bbox,
                        'shift': {'dx': dx, 'dy': dy},
                        'methods_agreeing': len(best['shifts']),
                        'correction_type': 'single_detection'
                    })
                    
                    self.stats['single_detection_corrections'] += 1
            
            except Exception as e:
                continue
        
        return corrections
    
    def _save_visualization(self, sample_data_cache: dict, dx: int, dy: int,
                           viz_dir: Path, filename: str):
        """Save a visualization of the correction."""
        if not sample_data_cache:
            return
        
        # Use first sample for visualization
        first_data = list(sample_data_cache.values())[0]
        sample = first_data['sample']
        thermal_img = first_data['thermal_img']
        rgb_img = first_data['rgb_img']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Original
        ax1 = axes[0]
        rgb_display = cv2.cvtColor(rgb_img.copy(), cv2.COLOR_BGR2RGB)
        for det_data in sample_data_cache.values():
            bbox = det_data['sample']['bbox']
            cv2.rectangle(rgb_display,
                         (bbox['x_min'], bbox['y_min']),
                         (bbox['x_max'], bbox['y_max']),
                         (255, 0, 0), 2)
        ax1.imshow(rgb_display)
        ax1.set_title('Original BBox Position')
        ax1.axis('off')
        
        # Corrected
        ax2 = axes[1]
        rgb_display2 = cv2.cvtColor(rgb_img.copy(), cv2.COLOR_BGR2RGB)
        for det_data in sample_data_cache.values():
            bbox = det_data['sample']['bbox']
            cv2.rectangle(rgb_display2,
                         (bbox['x_min'] + dx, bbox['y_min'] + dy),
                         (bbox['x_max'] + dx, bbox['y_max'] + dy),
                         (0, 255, 0), 2)
        ax2.imshow(rgb_display2)
        ax2.set_title(f'Corrected (shift: {dx}, {dy})')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(viz_dir / f"{filename}.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, corrections: list):
        """Save correction results."""
        if not corrections:
            print("No corrections to save.")
            return
        
        # Group by label format
        if self.label_format == 'yolo':
            self._save_yolo_results(corrections)
        else:
            self._save_metadata_results(corrections)
        
        # Save correction log
        log_path = self.output_dir / 'correction_log.json'
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'settings': {
                'tolerance': self.tolerance,
                'min_consensus_score': self.min_consensus_score,
                'min_detection_coverage': self.min_detection_coverage
            },
            'stats': self.stats,
            'corrections': [
                {
                    'frame_id': c['sample']['frame_id'],
                    'original_bbox': c['original_bbox'],
                    'corrected_bbox': c['corrected_bbox'],
                    'shift': c['shift'],
                    'type': c['correction_type']
                }
                for c in corrections
            ]
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\nResults saved to: {self.output_dir}")
    
    def _save_yolo_results(self, corrections: list):
        """Save corrected YOLO format labels."""
        labels_dir = self.output_dir / 'labels'
        labels_dir.mkdir(exist_ok=True)
        
        # Group corrections by label file
        corrections_by_file = defaultdict(list)
        for c in corrections:
            label_path = c['sample']['label_path']
            corrections_by_file[label_path].append(c)
        
        # Write corrected labels
        for label_path, file_corrections in corrections_by_file.items():
            sample = file_corrections[0]['sample']
            img_w, img_h = sample['img_width'], sample['img_height']
            
            # Build corrected annotations
            corrected_anns = []
            for c in file_corrections:
                ann = c['sample']['annotation'].copy()
                ann['bbox'] = c['corrected_bbox']
                corrected_anns.append(ann)
            
            # Write to output
            out_path = labels_dir / Path(label_path).name
            YOLOLabelParser.write_file(str(out_path), corrected_anns, img_w, img_h)
    
    def _save_metadata_results(self, corrections: list):
        """Save corrected JSON metadata."""
        # Group by original metadata file
        corrections_by_meta = defaultdict(list)
        for c in corrections:
            meta_path = c['sample'].get('metadata_path', '')
            if meta_path:
                corrections_by_meta[meta_path].append(c)
        
        for meta_path, file_corrections in corrections_by_meta.items():
            # Load original metadata
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            # Apply corrections
            for c in file_corrections:
                frame_id = str(c['sample'].get('frame_idx', ''))
                ann_idx = c['sample'].get('ann_idx', 0)
                
                if frame_id in metadata.get('frames', {}):
                    frame = metadata['frames'][frame_id]
                    if 'annotations' in frame and len(frame['annotations']) > ann_idx:
                        frame['annotations'][ann_idx]['bbox'] = c['corrected_bbox']
                        frame['annotations'][ann_idx]['corrected'] = True
                        frame['annotations'][ann_idx]['shift'] = c['shift']
            
            # Save corrected metadata
            out_path = self.output_dir / Path(meta_path).name
            with open(out_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def _print_summary(self, corrections: list):
        """Print correction summary."""
        print("\n" + "=" * 70)
        print("CORRECTION SUMMARY")
        print("=" * 70)
        print(f"Total corrections applied: {len(corrections)}")
        print(f"  - Multi-detection frames: {self.stats['multi_detection_corrections']}")
        print(f"  - Single-detection frames: {self.stats['single_detection_corrections']}")
        print(f"Frames with consensus: {self.stats['consensus_found']}")
        print(f"Frames without consensus: {self.stats['no_consensus']}")
        
        if corrections:
            shifts = [(c['shift']['dx'], c['shift']['dy']) for c in corrections]
            avg_dx = sum(s[0] for s in shifts) / len(shifts)
            avg_dy = sum(s[1] for s in shifts) / len(shifts)
            print(f"\nAverage shift: ({avg_dx:.1f}, {avg_dy:.1f}) pixels")


def main():
    parser = argparse.ArgumentParser(
        description='Automatic Thermal-RGB BBox Correction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with YOLO labels
  python correct_bboxes.py --thermal-dir ./thermal --rgb-dir ./rgb --labels-dir ./labels

  # With JSON metadata
  python correct_bboxes.py --thermal-dir ./thermal --rgb-dir ./rgb --labels-dir ./metadata

  # With visualization output
  python correct_bboxes.py --thermal-dir ./thermal --rgb-dir ./rgb --labels-dir ./labels --save-viz
        """
    )
    
    parser.add_argument('--thermal-dir', required=True,
                        help='Path to thermal images directory')
    parser.add_argument('--rgb-dir', required=True,
                        help='Path to RGB images directory')
    parser.add_argument('--labels-dir', required=True,
                        help='Path to labels (YOLO .txt or JSON metadata)')
    parser.add_argument('--output-dir', default='./output',
                        help='Output directory for corrected labels (default: ./output)')
    parser.add_argument('--tolerance', type=int, default=10,
                        help='Pixel tolerance for clustering shifts (default: 10)')
    parser.add_argument('--min-consensus', type=float, default=0.4,
                        help='Minimum consensus score to accept correction (default: 0.4)')
    parser.add_argument('--min-coverage', type=float, default=0.67,
                        help='Minimum fraction of detections that must agree (default: 0.67)')
    parser.add_argument('--save-viz', action='store_true',
                        help='Save visualization samples')
    parser.add_argument('--viz-interval', type=int, default=100,
                        help='Save visualization every N corrections (default: 100)')
    
    args = parser.parse_args()
    
    tool = AutoCorrectionTool(
        thermal_dir=args.thermal_dir,
        rgb_dir=args.rgb_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
        min_consensus_score=args.min_consensus,
        min_detection_coverage=args.min_coverage
    )
    
    tool.run(save_visualizations=args.save_viz, viz_interval=args.viz_interval)


if __name__ == '__main__':
    main()
