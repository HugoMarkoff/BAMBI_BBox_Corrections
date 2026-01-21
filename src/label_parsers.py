"""
Label Parsers for YOLO and Metadata Formats
============================================
Provides unified interface for reading both YOLO label format (.txt) and 
JSON metadata format for bounding box annotations.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class YOLOLabelParser:
    """
    Parser for YOLO format label files.
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All coordinates are normalized [0, 1] relative to image dimensions.
    
    Example:
        2 0.996582 0.841308 0.006835 0.041992
    """
    
    @staticmethod
    def parse_file(label_path: str, img_width: int, img_height: int) -> List[Dict]:
        """
        Parse a YOLO format label file and convert to absolute pixel coordinates.
        
        Args:
            label_path: Path to the .txt label file
            img_width: Width of the corresponding image in pixels
            img_height: Height of the corresponding image in pixels
            
        Returns:
            List of annotation dictionaries with absolute bbox coordinates
        """
        annotations = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert normalized coordinates to absolute pixels
                abs_width = int(width * img_width)
                abs_height = int(height * img_height)
                x_min = int((x_center - width / 2) * img_width)
                y_min = int((y_center - height / 2) * img_height)
                x_max = x_min + abs_width
                y_max = y_min + abs_height
                
                # Clamp to image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_width, x_max)
                y_max = min(img_height, y_max)
                
                annotations.append({
                    'class_id': class_id,
                    'bbox': {
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'width': x_max - x_min,
                        'height': y_max - y_min
                    },
                    'original_normalized': {
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    }
                })
        
        return annotations
    
    @staticmethod
    def write_file(label_path: str, annotations: List[Dict], img_width: int, img_height: int):
        """
        Write annotations to YOLO format label file.
        
        Args:
            label_path: Path to write the .txt label file
            annotations: List of annotation dictionaries with absolute bbox coordinates
            img_width: Width of the corresponding image in pixels
            img_height: Height of the corresponding image in pixels
        """
        with open(label_path, 'w') as f:
            for ann in annotations:
                bbox = ann['bbox']
                class_id = ann.get('class_id', 0)
                
                # Convert absolute coordinates to normalized
                x_center = (bbox['x_min'] + bbox['width'] / 2) / img_width
                y_center = (bbox['y_min'] + bbox['height'] / 2) / img_height
                width = bbox['width'] / img_width
                height = bbox['height'] / img_height
                
                f.write(f"{class_id} {x_center:.10f} {y_center:.10f} {width:.10f} {height:.10f}\n")


class MetadataParser:
    """
    Parser for JSON metadata format with comprehensive annotation information.
    
    Expected structure:
    {
        "flight_key": "123",
        "frames": {
            "frame_id": {
                "thermal_image": "123_456.jpg",
                "rgb_image": "123_456.jpg",
                "annotations": [
                    {
                        "bbox": {"x_min": 100, "y_min": 200, ...},
                        "species": "...",
                        ...
                    }
                ]
            }
        }
    }
    """
    
    @staticmethod
    def parse_file(metadata_path: str) -> Dict:
        """
        Parse a JSON metadata file.
        
        Args:
            metadata_path: Path to the JSON metadata file
            
        Returns:
            Dictionary containing flight metadata and frames
        """
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def write_file(metadata_path: str, metadata: Dict):
        """
        Write metadata to JSON file.
        
        Args:
            metadata_path: Path to write the JSON file
            metadata: Metadata dictionary to write
        """
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def extract_samples(metadata: Dict, thermal_dir: Path, rgb_dir: Path, 
                       split: str = '') -> List[Dict]:
        """
        Extract all annotation samples from metadata.
        
        Args:
            metadata: Parsed metadata dictionary
            thermal_dir: Path to thermal images directory
            rgb_dir: Path to RGB images directory
            split: Dataset split name (train/val/test)
            
        Returns:
            List of sample dictionaries ready for correction
        """
        samples = []
        flight_key = metadata.get('flight_key', '')
        
        for frame_id, frame_data in metadata.get('frames', {}).items():
            thermal_img_name = frame_data.get('thermal_image')
            rgb_img_name = frame_data.get('rgb_image')
            annotations = frame_data.get('annotations', [])
            
            if not thermal_img_name or not rgb_img_name or not annotations:
                continue
            
            thermal_path = thermal_dir / thermal_img_name
            rgb_path = rgb_dir / rgb_img_name
            
            if not thermal_path.exists() or not rgb_path.exists():
                continue
            
            for ann in annotations:
                bbox = ann.get('bbox')
                if not bbox:
                    continue
                
                samples.append({
                    'thermal_path': str(thermal_path),
                    'rgb_path': str(rgb_path),
                    'bbox': bbox,
                    'annotation': ann,
                    'frame_id': frame_id,
                    'flight_key': flight_key,
                    'split': split,
                    'visibility': ann.get('visibility', 0)
                })
        
        return samples


def detect_label_format(labels_path: Path) -> str:
    """
    Detect whether the labels directory contains YOLO format or metadata format.
    
    Args:
        labels_path: Path to the labels directory
        
    Returns:
        'yolo' or 'metadata' depending on detected format
    """
    # Check for .txt files (YOLO format)
    txt_files = list(labels_path.glob("*.txt"))
    if txt_files:
        return 'yolo'
    
    # Check for .json files (metadata format)
    json_files = list(labels_path.glob("*_metadata.json"))
    if json_files:
        return 'metadata'
    
    # Also check for plain .json files
    json_files = list(labels_path.glob("*.json"))
    if json_files:
        return 'metadata'
    
    raise ValueError(f"Could not detect label format in {labels_path}. "
                    "Expected .txt files (YOLO) or .json files (metadata).")
