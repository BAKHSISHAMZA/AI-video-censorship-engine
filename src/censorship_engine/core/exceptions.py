"""Custom exceptions for the censorship engine."""


class CensorshipEngineError(Exception):
    """Base exception for all engine errors."""
    pass


class ModelLoadError(CensorshipEngineError):
    """Error loading a model."""
    pass


class VideoLoadError(CensorshipEngineError):
    """Error loading video file."""
    pass


class VideoWriteError(CensorshipEngineError):
    """Error writing video file."""
    pass


class DetectionError(CensorshipEngineError):
    """Error during detection."""
    pass


class TrackingError(CensorshipEngineError):
    """Error during tracking."""
    pass


class RenderingError(CensorshipEngineError):
    """Error during rendering."""
    pass


class ConfigurationError(CensorshipEngineError):
    """Invalid configuration."""
    pass


class InsufficientMemoryError(CensorshipEngineError):
    """Not enough GPU/CPU memory."""
    pass