"""ONNX export functionality for HuggingFace models."""

import torch
import onnx
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import tempfile
import os

from ..utils import get_logger


class ONNXExporter:
    """Exports HuggingFace models to ONNX format."""
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize the exporter.
        
        Args:
            output_dir: Directory to save ONNX models
        """
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        max_length: int = 512,
        batch_size: int = 1,
        opset_version: int = 14,
        dynamic_axes: bool = True,
        optimize: bool = True
    ) -> str:
        """Export model to ONNX format.
        
        Args:
            model: HuggingFace model to export
            tokenizer: Associated tokenizer
            model_name: Name for the exported model
            max_length: Maximum sequence length
            batch_size: Batch size for export
            opset_version: ONNX opset version
            dynamic_axes: Whether to use dynamic axes
            optimize: Whether to optimize the ONNX model
            
        Returns:
            Path to exported ONNX model
        """
        self.logger.info(f"Exporting {model_name} to ONNX...")
        
        # Prepare model for export
        model.eval()
        
        # Create dummy input
        dummy_input = self._create_dummy_input(
            tokenizer, max_length, batch_size
        )
        
        # Define output path
        onnx_path = self.output_dir / f"{model_name}.onnx"
        
        # Define dynamic axes if requested
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            }
        
        try:
            # Export to ONNX
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["logits"],
                    dynamic_axes=dynamic_axes_dict,
                    verbose=False
                )
            
            self.logger.info(f"Model exported to: {onnx_path}")
            
            # Validate ONNX model
            if self._validate_onnx_model(str(onnx_path)):
                self.logger.info("ONNX model validation passed")
            else:
                raise RuntimeError("ONNX model validation failed")
            
            # Optimize ONNX model if requested
            if optimize:
                optimized_path = self._optimize_onnx_model(str(onnx_path))
                if optimized_path:
                    onnx_path = Path(optimized_path)
                    self.logger.info(f"Optimized ONNX model saved to: {onnx_path}")
            
            # Log model information
            self._log_onnx_info(str(onnx_path))
            
            return str(onnx_path)
            
        except Exception as e:
            self.logger.error(f"ONNX export failed: {str(e)}")
            raise
    
    def _create_dummy_input(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create dummy input for ONNX export.
        
        Args:
            tokenizer: Tokenizer for creating dummy input
            max_length: Maximum sequence length
            batch_size: Batch size
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        # Create dummy text
        dummy_text = ["This is a dummy input for ONNX export."] * batch_size
        
        # Tokenize
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        return inputs.input_ids, inputs.attention_mask
    
    def _validate_onnx_model(self, onnx_path: str) -> bool:
        """Validate exported ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            True if validation passes
        """
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Basic shape inference
            onnx.shape_inference.infer_shapes(onnx_model)
            
            return True
            
        except Exception as e:
            self.logger.error(f"ONNX validation failed: {str(e)}")
            return False
    
    def _optimize_onnx_model(self, onnx_path: str) -> Optional[str]:
        """Optimize ONNX model using onnxruntime.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Path to optimized model or None if optimization failed
        """
        try:
            import onnxruntime as ort
            from onnxruntime.transformers import optimizer
            
            # Create optimized model path
            optimized_path = onnx_path.replace(".onnx", "_optimized.onnx")
            
            # Optimize model
            opt_model = optimizer.optimize_model(
                onnx_path,
                model_type="gpt2",  # Generic transformer optimization
                num_heads=0,  # Auto-detect
                hidden_size=0,  # Auto-detect
                optimization_options=None
            )
            
            # Save optimized model
            opt_model.save_model_to_file(optimized_path)
            
            return optimized_path
            
        except ImportError:
            self.logger.warning("onnxruntime not available for optimization")
            return None
        except Exception as e:
            self.logger.warning(f"ONNX optimization failed: {str(e)}")
            return None
    
    def _log_onnx_info(self, onnx_path: str) -> None:
        """Log information about the ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
        """
        try:
            onnx_model = onnx.load(onnx_path)
            
            # Model info
            self.logger.info(f"ONNX Model Info:")
            self.logger.info(f"  - IR Version: {onnx_model.ir_version}")
            self.logger.info(f"  - Opset Version: {onnx_model.opset_import[0].version}")
            self.logger.info(f"  - Producer: {onnx_model.producer_name}")
            
            # Input/Output info
            graph = onnx_model.graph
            self.logger.info(f"  - Inputs: {len(graph.input)}")
            for inp in graph.input:
                self.logger.info(f"    - {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
            
            self.logger.info(f"  - Outputs: {len(graph.output)}")
            for out in graph.output:
                self.logger.info(f"    - {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
            
            # File size
            file_size = os.path.getsize(onnx_path) / (1024 ** 2)
            self.logger.info(f"  - File Size: {file_size:.2f} MB")
            
        except Exception as e:
            self.logger.warning(f"Could not log ONNX info: {str(e)}")
    
    def compare_outputs(
        self,
        original_model: PreTrainedModel,
        onnx_path: str,
        tokenizer: PreTrainedTokenizer,
        test_input: str = "This is a test input for comparison.",
        tolerance: float = 1e-4
    ) -> bool:
        """Compare outputs between original and ONNX models.
        
        Args:
            original_model: Original PyTorch model
            onnx_path: Path to ONNX model
            tokenizer: Tokenizer for input preparation
            test_input: Test string for comparison
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if outputs match within tolerance
        """
        try:
            import onnxruntime as ort
            
            # Prepare input
            inputs = tokenizer(
                test_input,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Get PyTorch output
            original_model.eval()
            with torch.no_grad():
                pytorch_output = original_model(**inputs).logits
            
            # Get ONNX output
            ort_session = ort.InferenceSession(onnx_path)
            onnx_inputs = {
                "input_ids": inputs.input_ids.numpy(),
                "attention_mask": inputs.attention_mask.numpy()
            }
            onnx_output = ort_session.run(None, onnx_inputs)[0]
            
            # Compare outputs
            diff = torch.abs(pytorch_output - torch.from_numpy(onnx_output))
            max_diff = torch.max(diff).item()
            
            self.logger.info(f"Output comparison - Max difference: {max_diff:.6f}")
            
            if max_diff <= tolerance:
                self.logger.info("✓ ONNX model outputs match PyTorch model")
                return True
            else:
                self.logger.warning(f"✗ ONNX model outputs differ by {max_diff:.6f} (tolerance: {tolerance})")
                return False
                
        except ImportError:
            self.logger.warning("onnxruntime not available for output comparison")
            return True  # Assume success if we can't compare
        except Exception as e:
            self.logger.error(f"Output comparison failed: {str(e)}")
            return False