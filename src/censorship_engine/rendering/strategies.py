"""
Censorship strategies with comprehensive intensity controls.

Each strategy implements `apply(frame, bbox)` and returns a modified frame.
Bounding boxes are (x1, y1, x2, y2) in absolute pixel coords.

Production-ready features:
- Full intensity control (0.0 to 1.0) for all strategies
- Robust error handling and validation
- Optimized performance with caching where appropriate
- Comprehensive logging for debugging
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import cv2
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


class RenderStrategy(ABC):
    """Base class for censorship strategies with validation."""

    def __init__(self, intensity: float = 1.0, **kwargs):
        """
        Args:
            intensity: Effect strength (0.0 = no effect, 1.0 = full effect)
        """
        self.intensity = self._validate_intensity(intensity)
        self._validate_parameters(**kwargs)
    
    @staticmethod
    def _validate_intensity(intensity: float) -> float:
        """Validate and clamp intensity to [0.0, 1.0]."""
        try:
            intensity = float(intensity)
            return max(0.0, min(1.0, intensity))
        except (TypeError, ValueError) as e:
            logger.warning(f"Invalid intensity value, defaulting to 1.0: {e}")
            return 1.0
    
    def _validate_parameters(self, **kwargs):
        """Override to validate strategy-specific parameters."""
        pass
    
    @abstractmethod
    def apply(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """
        Apply censorship to `frame` at region `bbox`.
        Returns modified frame (does not mutate input).
        """
        pass
    
    def _clip_bbox(self, bbox: BBox, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Clip bbox to frame boundaries."""
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        
        x1c = max(0, min(x1, w - 1))
        y1c = max(0, min(y1, h - 1))
        x2c = max(x1c + 1, min(x2, w))
        y2c = max(y1c + 1, min(y2, h))
        
        return x1c, y1c, x2c, y2c
    
    def _blend_with_original(self, processed: np.ndarray, original: np.ndarray, intensity: float) -> np.ndarray:
        """Blend processed region with original based on intensity."""
        if intensity >= 1.0:
            return processed
        elif intensity <= 0.0:
            return original
        
        return cv2.addWeighted(
            processed.astype(np.float32), intensity,
            original.astype(np.float32), 1.0 - intensity,
            0
        ).astype(np.uint8)


class BlurStrategy(RenderStrategy):
    """Gaussian blur strategy with intensity control."""

    def __init__(self, kernel_size: int = 51, intensity: float = 1.0, **kwargs):
        """
        Args:
            kernel_size: Blur kernel size (must be odd). Larger = stronger blur.
                        Default: 51 (strong blur)
                        Range: 3-201 (recommended: 15-99)
            intensity: Blur strength multiplier (0.0 to 1.0)
                      1.0 = full blur (default)
                      0.5 = half blur (blend with original)
                      0.0 = no blur (original frame)
        """
        self.kernel_size = self._validate_kernel_size(kernel_size)
        super().__init__(intensity=intensity, **kwargs)
    
    @staticmethod
    def _validate_kernel_size(kernel_size: int) -> int:
        """Validate and fix kernel size."""
        try:
            kernel_size = int(kernel_size)
            kernel_size = max(3, min(201, kernel_size))
            # Ensure odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            return kernel_size
        except (TypeError, ValueError):
            logger.warning(f"Invalid kernel_size, defaulting to 51")
            return 51
    
    def apply(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """Apply Gaussian blur to bbox region."""
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error("Invalid frame provided to BlurStrategy")
            return frame
        
        x1c, y1c, x2c, y2c = self._clip_bbox(bbox, frame.shape)
        
        if x2c <= x1c or y2c <= y1c:
            logger.debug(f"Invalid bbox after clipping: {bbox}")
            return frame
        
        out = frame.copy()
        roi = out[y1c:y2c, x1c:x2c].copy()
        
        # Adapt kernel to ROI size
        roi_h, roi_w = roi.shape[:2]
        max_kernel = min(roi_h, roi_w)
        if max_kernel < 3:
            logger.debug(f"ROI too small for blur: {roi.shape}")
            return frame
        
        # Ensure kernel fits in ROI and is odd
        kernel = min(self.kernel_size, max_kernel)
        if kernel % 2 == 0:
            kernel -= 1
        kernel = max(3, kernel)
        
        try:
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0)
            
            # Blend based on intensity
            result = self._blend_with_original(blurred, roi, self.intensity)
            
            out[y1c:y2c, x1c:x2c] = result
            return out
            
        except cv2.error as e:
            logger.error(f"OpenCV error in BlurStrategy: {e}")
            return frame


class MotionBlurStrategy(RenderStrategy):
    """Motion blur strategy for dynamic censorship effect."""

    def __init__(self, kernel_size: int = 31, angle: float = 45.0, intensity: float = 1.0, **kwargs):
        """
        Args:
            kernel_size: Size of motion blur kernel (larger = stronger effect)
                        Range: 3-101 (recommended: 15-51)
            angle: Direction of motion blur in degrees (0-360)
                  0 = horizontal right, 90 = vertical down
            intensity: Effect strength (0.0 to 1.0)
        """
        self.kernel_size = self._validate_kernel_size(kernel_size)
        self.angle = float(angle) % 360
        super().__init__(intensity=intensity, **kwargs)
        self._kernel_cache = {}
    
    @staticmethod
    def _validate_kernel_size(kernel_size: int) -> int:
        """Validate kernel size."""
        try:
            kernel_size = int(kernel_size)
            return max(3, min(101, kernel_size))
        except (TypeError, ValueError):
            logger.warning(f"Invalid kernel_size, defaulting to 31")
            return 31
    
    def _get_motion_kernel(self, size: int, angle: float) -> np.ndarray:
        """Generate motion blur kernel (cached)."""
        cache_key = (size, angle)
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]
        
        # Create motion blur kernel
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        
        # Calculate line based on angle
        angle_rad = np.deg2rad(angle)
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        
        # Draw line in kernel
        for i in range(-center, center + 1):
            x = int(center + i * cos_val)
            y = int(center + i * sin_val)
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
        
        # Normalize
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum
        
        # Cache for reuse
        if len(self._kernel_cache) < 100:  # Limit cache size
            self._kernel_cache[cache_key] = kernel
        
        return kernel
    
    def apply(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """Apply motion blur to bbox region."""
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error("Invalid frame provided to MotionBlurStrategy")
            return frame
        
        x1c, y1c, x2c, y2c = self._clip_bbox(bbox, frame.shape)
        
        if x2c <= x1c or y2c <= y1c:
            return frame
        
        out = frame.copy()
        roi = out[y1c:y2c, x1c:x2c].copy()
        
        # Adapt kernel to ROI size
        roi_h, roi_w = roi.shape[:2]
        kernel_size = min(self.kernel_size, min(roi_h, roi_w))
        if kernel_size < 3:
            return frame
        
        try:
            # Get motion kernel
            kernel = self._get_motion_kernel(kernel_size, self.angle)
            
            # Apply filter
            blurred = cv2.filter2D(roi, -1, kernel)
            
            # Blend based on intensity
            result = self._blend_with_original(blurred, roi, self.intensity)
            
            out[y1c:y2c, x1c:x2c] = result
            return out
            
        except cv2.error as e:
            logger.error(f"OpenCV error in MotionBlurStrategy: {e}")
            return frame


class PixelateStrategy(RenderStrategy):
    """Pixelation strategy with intensity control."""

    def __init__(self, pixel_block: int = 16, intensity: float = 1.0, **kwargs):
        """
        Args:
            pixel_block: Size of pixel blocks. Larger = stronger pixelation.
                        Default: 16
                        Range: 2-100 (recommended: 8-32)
            intensity: Pixelation strength (0.0 to 1.0)
                      1.0 = full pixelation (default)
                      0.5 = half pixelation (blend with original)
                      0.0 = no pixelation
        """
        self.pixel_block = self._validate_pixel_block(pixel_block)
        super().__init__(intensity=intensity, **kwargs)
    
    @staticmethod
    def _validate_pixel_block(pixel_block: int) -> int:
        """Validate pixel block size."""
        try:
            pixel_block = int(pixel_block)
            return max(2, min(100, pixel_block))
        except (TypeError, ValueError):
            logger.warning(f"Invalid pixel_block, defaulting to 16")
            return 16
    
    def apply(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """Apply pixelation to bbox region."""
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error("Invalid frame provided to PixelateStrategy")
            return frame
        
        x1c, y1c, x2c, y2c = self._clip_bbox(bbox, frame.shape)
        
        if x2c <= x1c or y2c <= y1c:
            return frame
        
        out = frame.copy()
        roi = out[y1c:y2c, x1c:x2c].copy()
        roi_h, roi_w = roi.shape[:2]
        
        if roi_h < 2 or roi_w < 2:
            return frame
        
        try:
            # Compute downscaled size
            small_w = max(1, roi_w // self.pixel_block)
            small_h = max(1, roi_h // self.pixel_block)
            
            # Downscale then upscale for pixelation effect
            temp = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(temp, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            
            # Blend based on intensity
            result = self._blend_with_original(pixelated, roi, self.intensity)
            
            out[y1c:y2c, x1c:x2c] = result
            return out
            
        except cv2.error as e:
            logger.error(f"OpenCV error in PixelateStrategy: {e}")
            return frame


class BlackBoxStrategy(RenderStrategy):
    """Solid-color box strategy with opacity and corner options."""

    def __init__(
        self, 
        color: Tuple[int, int, int] = (0, 0, 0), 
        rounded_corners: int = 0,
        opacity: float = 1.0,
        intensity: float = 1.0,
        **kwargs
    ):
        """
        Args:
            color: BGR tuple (default: black)
            rounded_corners: Corner radius in pixels (0 = sharp corners)
            opacity: Box opacity (0.0 = transparent, 1.0 = opaque)
            intensity: Overall effect strength (0.0 to 1.0)
                      Note: intensity and opacity are different!
                      - opacity controls the box transparency
                      - intensity controls how much the effect is applied
        """
        self.color = self._validate_color(color)
        self.rounded_corners = max(0, int(rounded_corners))
        self.opacity = self._validate_intensity(opacity)
        super().__init__(intensity=intensity, **kwargs)
    
    @staticmethod
    def _validate_color(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Validate and clamp color values."""
        try:
            return tuple(max(0, min(255, int(c))) for c in color)
        except (TypeError, ValueError):
            logger.warning(f"Invalid color, defaulting to black")
            return (0, 0, 0)
    
    def apply(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """Apply solid box to bbox region."""
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error("Invalid frame provided to BlackBoxStrategy")
            return frame
        
        x1c, y1c, x2c, y2c = self._clip_bbox(bbox, frame.shape)
        
        if x2c <= x1c or y2c <= y1c:
            return frame
        
        out = frame.copy()
        roi_h = y2c - y1c
        roi_w = x2c - x1c
        
        try:
            if self.rounded_corners <= 0:
                # Sharp rectangle (fastest path)
                colored_roi = np.full((roi_h, roi_w, 3), self.color, dtype=np.uint8)
                
                # Apply opacity
                if self.opacity < 1.0:
                    original_roi = frame[y1c:y2c, x1c:x2c]
                    colored_roi = cv2.addWeighted(
                        colored_roi, self.opacity,
                        original_roi, 1.0 - self.opacity,
                        0
                    )
                
                # Apply intensity
                if self.intensity < 1.0:
                    original_roi = frame[y1c:y2c, x1c:x2c]
                    colored_roi = self._blend_with_original(colored_roi, original_roi, self.intensity)
                
                out[y1c:y2c, x1c:x2c] = colored_roi
            else:
                # Rounded corners
                out = self._apply_rounded_box(out, x1c, y1c, x2c, y2c, roi_w, roi_h, frame)
            
            return out
            
        except Exception as e:
            logger.error(f"Error in BlackBoxStrategy: {e}")
            return frame
    
    def _apply_rounded_box(self, out, x1c, y1c, x2c, y2c, roi_w, roi_h, frame):
        """Apply rounded rectangle (more complex)."""
        radius = min(
            self.rounded_corners,
            roi_w // 2,
            roi_h // 2
        )
        
        if radius < 1:
            # Fallback to sharp corners
            out[y1c:y2c, x1c:x2c] = self.color
            return out
        
        # Create mask
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        
        # Draw rectangles
        cv2.rectangle(mask, (radius, 0), (roi_w - radius, roi_h), 255, -1)
        cv2.rectangle(mask, (0, radius), (roi_w, roi_h - radius), 255, -1)
        
        # Draw corner circles
        cv2.circle(mask, (radius, radius), radius, 255, -1)
        cv2.circle(mask, (roi_w - radius, radius), radius, 255, -1)
        cv2.circle(mask, (radius, roi_h - radius), radius, 255, -1)
        cv2.circle(mask, (roi_w - radius, roi_h - radius), radius, 255, -1)
        
        # Apply mask
        roi = out[y1c:y2c, x1c:x2c].copy()
        colored = np.full_like(roi, self.color)
        
        # Blend using mask
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        result = (colored * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
        
        # Apply opacity
        if self.opacity < 1.0:
            result = cv2.addWeighted(result, self.opacity, roi, 1.0 - self.opacity, 0)
        
        # Apply intensity
        if self.intensity < 1.0:
            result = self._blend_with_original(result, roi, self.intensity)
        
        out[y1c:y2c, x1c:x2c] = result
        return out


class EmojiStrategy(RenderStrategy):
    """Place an emoji (PNG) over the region."""

    def __init__(self, png_path: Optional[str] = None, scale: float = 1.0, intensity: float = 1.0, **kwargs):
        """
        Args:
            png_path: Path to PNG image with alpha channel
            scale: Scale factor for emoji (1.0 = fit to bbox)
            intensity: Overall effect strength (0.0 to 1.0)
        """
        self.png_path = png_path
        self.scale = max(0.1, min(2.0, float(scale)))
        self._emoji = None
        self._load_emoji()
        super().__init__(intensity=intensity, **kwargs)
    
    def _load_emoji(self):
        """Load emoji image."""
        if not self.png_path:
            logger.warning("No emoji path provided, will fallback to black box")
            return
        
        try:
            self._emoji = cv2.imread(self.png_path, cv2.IMREAD_UNCHANGED)
            if self._emoji is None:
                logger.error(f"Failed to load emoji from: {self.png_path}")
        except Exception as e:
            logger.error(f"Error loading emoji: {e}")
            self._emoji = None

    def apply(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """Apply emoji overlay to bbox region."""
        # Fallback to black box if emoji not loaded
        if self._emoji is None:
            return BlackBoxStrategy(intensity=self.intensity).apply(frame, bbox)
        
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error("Invalid frame provided to EmojiStrategy")
            return frame
        
        x1c, y1c, x2c, y2c = self._clip_bbox(bbox, frame.shape)
        
        if x2c <= x1c or y2c <= y1c:
            return frame
        
        out = frame.copy()
        roi_h = y2c - y1c
        roi_w = x2c - x1c
        
        try:
            # Resize emoji
            target_w = max(1, int(roi_w * self.scale))
            target_h = max(1, int(roi_h * self.scale))
            emoji_rs = cv2.resize(self._emoji, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
            if emoji_rs.shape[2] == 4:  # Has alpha channel
                out = self._apply_with_alpha(out, emoji_rs, x1c, y1c, x2c, y2c, roi_w, roi_h, frame)
            else:
                # No alpha - simple paste
                emoji_bgr = emoji_rs[:, :, :3]
                eh, ew = emoji_bgr.shape[:2]
                
                # Center emoji
                ox = x1c + (roi_w - ew) // 2
                oy = y1c + (roi_h - eh) // 2
                
                # Clip to frame
                ox1, oy1 = max(0, ox), max(0, oy)
                ox2, oy2 = min(frame.shape[1], ox + ew), min(frame.shape[0], oy + eh)
                
                em_x1, em_y1 = ox1 - ox, oy1 - oy
                em_x2, em_y2 = em_x1 + (ox2 - ox1), em_y1 + (oy2 - oy1)
                
                if em_x2 > em_x1 and em_y2 > em_y1:
                    emoji_crop = emoji_bgr[em_y1:em_y2, em_x1:em_x2]
                    original_roi = frame[oy1:oy2, ox1:ox2]
                    blended = self._blend_with_original(emoji_crop, original_roi, self.intensity)
                    out[oy1:oy2, ox1:ox2] = blended
            
            return out
            
        except Exception as e:
            logger.error(f"Error in EmojiStrategy: {e}")
            return frame
    
    def _apply_with_alpha(self, out, emoji_rs, x1c, y1c, x2c, y2c, roi_w, roi_h, frame):
        """Apply emoji with alpha blending."""
        eh, ew = emoji_rs.shape[:2]
        
        # Center emoji
        ox = x1c + (roi_w - ew) // 2
        oy = y1c + (roi_h - eh) // 2
        
        # Clip to frame
        h, w = frame.shape[:2]
        ox1, oy1 = max(0, ox), max(0, oy)
        ox2, oy2 = min(w, ox + ew), min(h, oy + eh)
        
        em_x1, em_y1 = ox1 - ox, oy1 - oy
        em_x2, em_y2 = em_x1 + (ox2 - ox1), em_y1 + (oy2 - oy1)
        
        if em_x2 <= em_x1 or em_y2 <= em_y1:
            return out
        
        # Alpha blending
        alpha = emoji_rs[em_y1:em_y2, em_x1:em_x2, 3:].astype(np.float32) / 255.0
        rgb = emoji_rs[em_y1:em_y2, em_x1:em_x2, :3]
        roi_region = frame[oy1:oy2, ox1:ox2]
        
        # Apply emoji alpha
        blended = (alpha * rgb + (1 - alpha) * roi_region).astype(np.uint8)
        
        # Apply intensity
        if self.intensity < 1.0:
            blended = self._blend_with_original(blended, roi_region, self.intensity)
        
        out[oy1:oy2, ox1:ox2] = blended
        return out


# Strategy registry for easy lookup
STRATEGY_REGISTRY = {
    'blur': BlurStrategy,
    'gaussian': BlurStrategy,
    'gauss': BlurStrategy,
    'motion_blur': MotionBlurStrategy,
    'motion': MotionBlurStrategy,
    'pixelate': PixelateStrategy,
    'pixel': PixelateStrategy,
    'blackout': BlackBoxStrategy,
    'black': BlackBoxStrategy,
    'box': BlackBoxStrategy,
    'blackbox': BlackBoxStrategy,
    'emoji': EmojiStrategy,
}


def get_strategy(name: str, **options) -> RenderStrategy:
    """
    Factory function to create strategy instances.
    
    Args:
        name: Strategy name (e.g., 'blur', 'pixelate', 'blackout')
        **options: Strategy-specific options
    
    Returns:
        RenderStrategy instance
    
    Raises:
        ValueError: If strategy name is not recognized
    """
    name_lower = name.lower().strip()
    
    if name_lower not in STRATEGY_REGISTRY:
        logger.warning(f"Unknown strategy '{name}', defaulting to blur")
        name_lower = 'blur'
    
    strategy_class = STRATEGY_REGISTRY[name_lower]
    
    try:
        return strategy_class(**options)
    except Exception as e:
        logger.error(f"Error creating strategy '{name}': {e}")
        # Fallback to blur with default options
        return BlurStrategy()