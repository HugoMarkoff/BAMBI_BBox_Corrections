"""
Thermal-RGB BBox Correction Toolkit
====================================
Tools for correcting bounding box alignment between thermal and RGB imagery.

This package provides both interactive and automated correction capabilities
using template matching with multiple image processing methods.
"""

__version__ = "1.0.0"
__author__ = "BAMBI Project Team"

from .correction_core import BBoxCorrectionEngine
from .label_parsers import YOLOLabelParser, MetadataParser
