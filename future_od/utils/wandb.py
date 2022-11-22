from dataclasses import dataclass


@dataclass
class WandBConfig:
    """Configuration for Weights and Biases."""

    enabled: bool = False
    entity: str = None
    project: str = "future-od"
    name: str = None
    hyperparams: dict = None
    watch_model: bool = False
    notes: str = None
    num_images: int = 0
    resume_id: str = None
