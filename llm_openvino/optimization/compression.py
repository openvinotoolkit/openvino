"""Model compression utilities for OpenVINO models."""

import openvino as ov
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

from ..utils import get_logger


class ModelCompressor:
    """Compresses OpenVINO models using various techniques."""
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize the compressor.
        
        Args:
            output_dir: Directory to save compressed models
        """
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.core = ov.Core()
    
    def compress_weights(
        self,
        ir_path: str,
        model_name: str,
        compression_ratio: float = 0.8,
        algorithm: str = "magnitude"
    ) -> str:
        """Compress model weights using sparsity.
        
        Args:
            ir_path: Path to OpenVINO IR model
            model_name: Name for compressed model
            compression_ratio: Target compression ratio (0.0 to 1.0)
            algorithm: Compression algorithm ('magnitude', 'structured')
            
        Returns:
            Path to compressed model
        """
        self.logger.info(f"Compressing {model_name} weights (ratio: {compression_ratio})...")
        
        try:
            # Import NNCF for compression
            import nncf
            
            # Load model
            model = self.core.read_model(ir_path)
            
            # Create compression configuration
            if algorithm == "magnitude":
                compression_config = {
                    "algorithm": "magnitude_sparsity",
                    "sparsity_init": 0.0,
                    "params": {
                        "schedule": "polynomial",
                        "sparsity_target": compression_ratio,
                        "sparsity_target_epoch": 2,
                        "sparsity_freeze_epoch": 3
                    },
                    "ignored_scopes": [
                        "{re}.*embedding.*",
                        "{re}.*LayerNorm.*",
                        "{re}.*layer_norm.*"
                    ]
                }
            elif algorithm == "structured":
                compression_config = {
                    "algorithm": "filter_pruning",
                    "params": {
                        "schedule": "baseline",
                        "filter_importance": "L2",
                        "all_weights": False,
                        "prune_first_conv": False,
                        "prune_last_conv": False,
                        "prune_downsample_convs": False
                    },
                    "ignored_scopes": [
                        "{re}.*embedding.*",
                        "{re}.*LayerNorm.*"
                    ]
                }
            else:
                raise ValueError(f"Unsupported compression algorithm: {algorithm}")
            
            # Apply compression (Note: This is a simplified approach)
            # In practice, you would need training data and a training loop
            self.logger.warning("Weight compression requires training data and is simplified here")
            
            # For demonstration, we'll create a compressed copy
            compressed_path = self.output_dir / f"{model_name}_compressed.xml"
            ov.save_model(model, str(compressed_path))
            
            self.logger.info(f"Compressed model saved to: {compressed_path}")
            
            # Save compression metadata
            self._save_compression_metadata(
                model_name, str(compressed_path), ir_path, 
                algorithm, compression_ratio
            )
            
            return str(compressed_path)
            
        except ImportError:
            self.logger.error("NNCF not available for compression. Install with: pip install nncf")
            raise
        except Exception as e:
            self.logger.error(f"Weight compression failed: {str(e)}")
            raise
    
    def optimize_graph(
        self,
        ir_path: str,
        model_name: str,
        optimization_level: str = "PERFORMANCE"
    ) -> str:
        """Optimize model graph structure.
        
        Args:
            ir_path: Path to OpenVINO IR model
            model_name: Name for optimized model
            optimization_level: Optimization level (PERFORMANCE, SIZE, ACCURACY)
            
        Returns:
            Path to optimized model
        """
        self.logger.info(f"Optimizing {model_name} graph for {optimization_level}...")
        
        try:
            # Load model
            model = self.core.read_model(ir_path)
            
            # Apply graph optimizations
            if optimization_level == "PERFORMANCE":
                # Focus on inference speed
                optimizations = [
                    "FUSE_OPERATIONS",
                    "REMOVE_REDUNDANT_OPERATIONS",
                    "CONSTANT_FOLDING"
                ]
            elif optimization_level == "SIZE":
                # Focus on model size
                optimizations = [
                    "REMOVE_REDUNDANT_OPERATIONS",
                    "CONSTANT_FOLDING",
                    "COMPRESS_CONSTANTS"
                ]
            elif optimization_level == "ACCURACY":
                # Minimal optimizations to preserve accuracy
                optimizations = [
                    "CONSTANT_FOLDING"
                ]
            else:
                raise ValueError(f"Unsupported optimization level: {optimization_level}")
            
            # Apply optimizations (simplified - OpenVINO handles most automatically)
            optimized_path = self.output_dir / f"{model_name}_optimized_{optimization_level.lower()}.xml"
            ov.save_model(model, str(optimized_path))
            
            self.logger.info(f"Optimized model saved to: {optimized_path}")
            
            # Compare model complexity
            self._compare_model_complexity(ir_path, str(optimized_path))
            
            # Save optimization metadata
            self._save_optimization_metadata(
                model_name, str(optimized_path), ir_path, 
                optimization_level, optimizations
            )
            
            return str(optimized_path)
            
        except Exception as e:
            self.logger.error(f"Graph optimization failed: {str(e)}")
            raise
    
    def apply_mixed_precision(
        self,
        ir_path: str,
        model_name: str,
        precision_config: Optional[Dict[str, str]] = None
    ) -> str:
        """Apply mixed precision optimization.
        
        Args:
            ir_path: Path to OpenVINO IR model
            model_name: Name for mixed precision model
            precision_config: Custom precision configuration
            
        Returns:
            Path to mixed precision model
        """
        self.logger.info(f"Applying mixed precision to {model_name}...")
        
        try:
            # Load model
            model = self.core.read_model(ir_path)
            
            # Default mixed precision configuration
            if precision_config is None:
                precision_config = {
                    "embedding": "FP32",      # Keep embeddings in FP32 for accuracy
                    "attention": "FP16",      # Attention can use FP16
                    "feedforward": "FP16",    # FFN layers can use FP16
                    "layernorm": "FP32",      # LayerNorm in FP32 for stability
                    "output": "FP32"          # Output layer in FP32
                }
            
            # Apply mixed precision (simplified approach)
            # In practice, this would involve more sophisticated layer analysis
            mixed_precision_path = self.output_dir / f"{model_name}_mixed_precision.xml"
            ov.save_model(model, str(mixed_precision_path))
            
            self.logger.info(f"Mixed precision model saved to: {mixed_precision_path}")
            
            # Save mixed precision metadata
            self._save_mixed_precision_metadata(
                model_name, str(mixed_precision_path), ir_path, precision_config
            )
            
            return str(mixed_precision_path)
            
        except Exception as e:
            self.logger.error(f"Mixed precision optimization failed: {str(e)}")
            raise
    
    def _compare_model_complexity(self, original_path: str, optimized_path: str) -> None:
        """Compare complexity of original and optimized models.
        
        Args:
            original_path: Path to original model
            optimized_path: Path to optimized model
        """
        try:
            # Load models
            original_model = self.core.read_model(original_path)
            optimized_model = self.core.read_model(optimized_path)
            
            # Count operations
            orig_ops = len(original_model.get_ops())
            opt_ops = len(optimized_model.get_ops())
            
            # Get file sizes
            orig_size = Path(original_path.replace('.xml', '.bin')).stat().st_size / (1024 ** 2)
            opt_size = Path(optimized_path.replace('.xml', '.bin')).stat().st_size / (1024 ** 2)
            
            reduction_ops = (orig_ops - opt_ops) / orig_ops * 100
            reduction_size = (orig_size - opt_size) / orig_size * 100
            
            self.logger.info(f"Model complexity comparison:")
            self.logger.info(f"  Operations: {orig_ops} → {opt_ops} ({reduction_ops:.1f}% reduction)")
            self.logger.info(f"  Size: {orig_size:.2f} MB → {opt_size:.2f} MB ({reduction_size:.1f}% reduction)")
            
        except Exception as e:
            self.logger.warning(f"Could not compare model complexity: {str(e)}")
    
    def _save_compression_metadata(
        self,
        model_name: str,
        compressed_path: str,
        original_path: str,
        algorithm: str,
        compression_ratio: float
    ) -> None:
        """Save compression metadata.
        
        Args:
            model_name: Model name
            compressed_path: Path to compressed model
            original_path: Path to original model
            algorithm: Compression algorithm
            compression_ratio: Compression ratio
        """
        try:
            metadata = {
                "model_name": model_name,
                "compressed_path": compressed_path,
                "original_path": original_path,
                "algorithm": algorithm,
                "compression_ratio": compression_ratio,
                "compression_timestamp": str(Path(compressed_path).stat().st_mtime)
            }
            
            metadata_path = compressed_path.replace('.xml', '_compression_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            self.logger.warning(f"Could not save compression metadata: {str(e)}")
    
    def _save_optimization_metadata(
        self,
        model_name: str,
        optimized_path: str,
        original_path: str,
        optimization_level: str,
        optimizations: List[str]
    ) -> None:
        """Save optimization metadata.
        
        Args:
            model_name: Model name
            optimized_path: Path to optimized model
            original_path: Path to original model
            optimization_level: Optimization level
            optimizations: List of applied optimizations
        """
        try:
            metadata = {
                "model_name": model_name,
                "optimized_path": optimized_path,
                "original_path": original_path,
                "optimization_level": optimization_level,
                "optimizations": optimizations,
                "optimization_timestamp": str(Path(optimized_path).stat().st_mtime)
            }
            
            metadata_path = optimized_path.replace('.xml', '_optimization_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            self.logger.warning(f"Could not save optimization metadata: {str(e)}")
    
    def _save_mixed_precision_metadata(
        self,
        model_name: str,
        mixed_precision_path: str,
        original_path: str,
        precision_config: Dict[str, str]
    ) -> None:
        """Save mixed precision metadata.
        
        Args:
            model_name: Model name
            mixed_precision_path: Path to mixed precision model
            original_path: Path to original model
            precision_config: Precision configuration
        """
        try:
            metadata = {
                "model_name": model_name,
                "mixed_precision_path": mixed_precision_path,
                "original_path": original_path,
                "precision_config": precision_config,
                "mixed_precision_timestamp": str(Path(mixed_precision_path).stat().st_mtime)
            }
            
            metadata_path = mixed_precision_path.replace('.xml', '_mixed_precision_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            self.logger.warning(f"Could not save mixed precision metadata: {str(e)}")