
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from enum import Enum
import time


@dataclass(frozen=True)
class Detection:
    """Immutable detection result from a model."""
    
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    frame_id: int
    model_name: str  # 'rtdetr' or 'retinaface'
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate detection data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        
        if not all(isinstance(x, (int, float)) for x in self.bbox):
            raise ValueError(f"Bbox must contain numbers, got {self.bbox}")
        
        if len(self.bbox) != 4:
            raise ValueError(f"Bbox must have 4 values, got {len(self.bbox)}")
    
    @property
    def area(self) -> int:
        """Calculate bbox area."""
        x1, y1, x2, y2 = self.bbox
        return max(0, int((x2 - x1) * (y2 - y1)))
    
    @property
    def center(self) -> Tuple[float, float]:
        """Calculate bbox center point."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def iou(self, other: 'Detection') -> float:
        """Calculate IoU with another detection."""
        x1_1, y1_1, x2_1, y2_1 = self.bbox
        x1_2, y1_2, x2_2, y2_2 = other.bbox
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'frame_id': self.frame_id,
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'area': self.area,
            'center': self.center
        }


@dataclass
class Track:
    """Mutable track representing an object across frames."""
    
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    is_active: bool = True
    frames_since_update: int = 0
    created_at: float = field(default_factory=time.time)
    
    def update(self, detection: Detection):
        """Add new detection to track."""
        self.detections.append(detection)
        self.frames_since_update = 0
        self.is_active = True
    
    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.frames_since_update += 1
    
    @property
    def age(self) -> int:
        """Number of frames this track has existed."""
        return len(self.detections) + self.frames_since_update
    
    @property
    def current_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get most recent bbox."""
        if not self.detections:
            return None
        return self.detections[-1].bbox
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence over track lifetime."""
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)
    
    @property
    def class_name(self) -> Optional[str]:
        """Get class name from first detection."""
        if not self.detections:
            return None
        return self.detections[0].class_name
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'track_id': self.track_id,
            'detections': [d.to_dict() for d in self.detections],
            'is_active': self.is_active,
            'age': self.age,
            'avg_confidence': self.avg_confidence,
            'class_name': self.class_name,
            'created_at': self.created_at
        }


class CensorshipMethod(Enum):
    """Available censorship strategies."""
    BLUR = "blur"
    MOTION_BLUR = "motion_blur"
    PIXELATE = "pixelate"
    BLACKOUT = "blackout"
    EMOJI = "emoji"


@dataclass
class ProcessingConfig:
    """
    Configuration for video processing pipeline.
    
    Enhanced with:
    - Motion blur support
    - Per-class strategy configuration
    - Full intensity controls
    """
    
    # Detection parameters
    nudity_confidence_threshold: float = 0.45
    face_confidence_threshold: float = 0.50
    
    # Tracking parameters
    max_track_age: int = 15
    min_track_hits: int = 1
    iou_threshold: float = 0.5
    
    # Per-class rendering strategies
    class_strategies: Optional[Dict[str, str]] = None
    # Example: {'face': 'blur', 'breast': 'blackbox', 'genitals': 'pixelate'}
    
    # Default strategy (fallback)
    default_censorship_method: CensorshipMethod = CensorshipMethod.BLUR
    
    # Blur settings
    blur_kernel_size: int = 51
    blur_intensity: float = 1.0
    
    # Motion blur settings (NEW!)
    motion_blur_kernel_size: int = 25
    motion_blur_angle: float = 0.0  # Degrees (0-360)
    motion_blur_intensity: float = 1.0
    
    # Pixelate settings
    pixelate_block_size: int = 16
    pixelate_intensity: float = 1.0
    
    # BlackBox settings
    blackbox_color: Tuple[int, int, int] = (0, 0, 0)
    blackbox_opacity: float = 1.0
    blackbox_rounded_corners: int = 0
    
    # Emoji settings (NEW!)
    emoji_scale: float = 1.0
    emoji_intensity: float = 1.0
    
    # Region expansion
    region_padding: int = 10
    
    # Temporal smoothing
    temporal_smoothing: bool = True
    smoothing_alpha: float = 0.7
    
    # Performance parameters
    batch_size: int = 1
    skip_frames: int = 0
    blur_quality_threshold: float = 100.0
    
    # Model paths
    rtdetr_model_path: str = "models/best.pt"
    use_gpu: bool = True
    device: str = "cuda:0"
    
    # Output parameters
    output_resolution: Optional[Tuple[int, int]] = None
    preserve_audio: bool = True
    export_metadata: bool = True
    metadata_format: str = "json"
    
    # Video encoding
    use_nvenc: bool = True
    output_codec: str = "h264"
    crf: int = 23
    use_fp16: bool = False
    
    def __post_init__(self):
        """Validate and set defaults."""
        # Validate thresholds
        if not 0 <= self.nudity_confidence_threshold <= 1:
            raise ValueError("nudity_confidence_threshold must be in [0, 1]")
        
        if not 0 <= self.face_confidence_threshold <= 1:
            raise ValueError("face_confidence_threshold must be in [0, 1]")
        
        # Validate intensities
        if not 0 <= self.blur_intensity <= 1:
            raise ValueError("blur_intensity must be in [0, 1]")
        
        if not 0 <= self.motion_blur_intensity <= 1:
            raise ValueError("motion_blur_intensity must be in [0, 1]")
        
        if not 0 <= self.pixelate_intensity <= 1:
            raise ValueError("pixelate_intensity must be in [0, 1]")
        
        if not 0 <= self.blackbox_opacity <= 1:
            raise ValueError("blackbox_opacity must be in [0, 1]")
        
        # Validate kernel sizes
        if self.blur_kernel_size < 3:
            raise ValueError("blur_kernel_size must be >= 3")
        
        if self.motion_blur_kernel_size < 5:
            raise ValueError("motion_blur_kernel_size must be >= 5")
        
        if self.pixelate_block_size < 1:
            raise ValueError("pixelate_block_size must be >= 1")
        
        # Set default class strategies if not provided
        if self.class_strategies is None:
            self.class_strategies = {}
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ProcessingConfig':
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert censorship_method string to enum if present
        if 'default_censorship_method' in config_dict:
            method_str = config_dict['default_censorship_method']
            config_dict['default_censorship_method'] = CensorshipMethod(method_str)
        
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'nudity_confidence_threshold': self.nudity_confidence_threshold,
            'face_confidence_threshold': self.face_confidence_threshold,
            'max_track_age': self.max_track_age,
            'min_track_hits': self.min_track_hits,
            'iou_threshold': self.iou_threshold,
            'class_strategies': self.class_strategies,
            'default_censorship_method': self.default_censorship_method.value,
            'blur_kernel_size': self.blur_kernel_size,
            'blur_intensity': self.blur_intensity,
            'motion_blur_kernel_size': self.motion_blur_kernel_size,
            'motion_blur_angle': self.motion_blur_angle,
            'motion_blur_intensity': self.motion_blur_intensity,
            'pixelate_block_size': self.pixelate_block_size,
            'pixelate_intensity': self.pixelate_intensity,
            'blackbox_color': self.blackbox_color,
            'blackbox_opacity': self.blackbox_opacity,
            'blackbox_rounded_corners': self.blackbox_rounded_corners,
            'region_padding': self.region_padding,
            'temporal_smoothing': self.temporal_smoothing,
            'batch_size': self.batch_size,
            'output_resolution': self.output_resolution,
            'preserve_audio': self.preserve_audio,
        }


@dataclass
class ProcessingResult:
    """Result of video processing operation."""
    
    input_path: str
    output_path: str
    metadata_path: Optional[str] = None
    
    # Statistics
    total_frames: int = 0
    frames_processed: int = 0
    frames_skipped: int = 0
    processing_time_seconds: float = 0.0
    
    # Detections
    total_detections: int = 0
    nudity_detections: int = 0
    face_detections: int = 0
    total_tracks: int = 0
    
    # Performance metrics
    fps: float = 0.0
    avg_detection_time_ms: float = 0.0
    avg_tracking_time_ms: float = 0.0
    avg_rendering_time_ms: float = 0.0
    
    # Configuration used
    config: Optional[ProcessingConfig] = None
    
    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'input_path': self.input_path,
            'output_path': self.output_path,
            'metadata_path': self.metadata_path,
            'statistics': {
                'total_frames': self.total_frames,
                'frames_processed': self.frames_processed,
                'frames_skipped': self.frames_skipped,
                'processing_time_seconds': self.processing_time_seconds,
                'fps': self.fps,
            },
            'detections': {
                'total': self.total_detections,
                'nudity': self.nudity_detections,
                'faces': self.face_detections,
                'tracks': self.total_tracks,
            },
            'performance': {
                'avg_detection_time_ms': self.avg_detection_time_ms,
                'avg_tracking_time_ms': self.avg_tracking_time_ms,
                'avg_rendering_time_ms': self.avg_rendering_time_ms,
            },
            'config': self.config.to_dict() if self.config else None,
            'errors': self.errors,
            'warnings': self.warnings,
        }