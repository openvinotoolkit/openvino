"""Configuration management for the LLM optimization pipeline."""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Model-specific configuration."""
    name: str = "distilgpt2"
    max_length: int = 512
    batch_size: int = 1
    trust_remote_code: bool = True
    torch_dtype: str = "float32"


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    precision: str = "fp16"  # fp32, fp16, int8
    quantization_dataset_size: int = 100
    calibration_steps: int = 300
    compression_ratio: float = 0.8


@dataclass
class InferenceConfig:
    """Inference configuration."""
    device: str = "CPU"  # CPU, GPU, AUTO
    num_threads: int = 0  # 0 for auto
    enable_profiling: bool = False
    cache_dir: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Benchmarking configuration."""
    warmup_runs: int = 10
    benchmark_runs: int = 50
    measure_memory: bool = True
    measure_latency: bool = True
    output_tokens: int = 50
    input_length: int = 128


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    
    # Paths
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file or return default config.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config objects from dictionary
        model_config = ModelConfig(**config_dict.get('model', {}))
        opt_config = OptimizationConfig(**config_dict.get('optimization', {}))
        inf_config = InferenceConfig(**config_dict.get('inference', {}))
        bench_config = BenchmarkConfig(**config_dict.get('benchmark', {}))
        
        # Extract top-level config
        top_level = {k: v for k, v in config_dict.items() 
                    if k not in ['model', 'optimization', 'inference', 'benchmark']}
        
        return Config(
            model=model_config,
            optimization=opt_config,
            inference=inf_config,
            benchmark=bench_config,
            **top_level
        )
    
    return Config()


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object
        config_path: Path to save configuration
    """
    config_dict = {
        'model': config.model.__dict__,
        'optimization': config.optimization.__dict__,
        'inference': config.inference.__dict__,
        'benchmark': config.benchmark.__dict__,
        'output_dir': config.output_dir,
        'cache_dir': config.cache_dir,
        'log_level': config.log_level
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)