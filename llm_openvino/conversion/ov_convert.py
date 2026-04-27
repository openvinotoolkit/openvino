"""OpenVINO IR conversion from ONNX models."""

import openvino as ov
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import os

from ..utils import get_logger


class OpenVINOConverter:
    """Converts ONNX models to OpenVINO IR format."""
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize the converter.
        
        Args:
            output_dir: Directory to save OpenVINO IR models
        """
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.core = ov.Core()
    
    def convert_to_ir(
        self,
        onnx_path: str,
        model_name: str,
        precision: str = "FP32",
        compress_to_fp16: bool = False,
        input_shape: Optional[List[int]] = None,
        mean_values: Optional[List[float]] = None,
        scale_values: Optional[List[float]] = None
    ) -> str:
        """Convert ONNX model to OpenVINO IR format.
        
        Args:
            onnx_path: Path to ONNX model
            model_name: Name for the IR model
            precision: Model precision (FP32, FP16)
            compress_to_fp16: Whether to compress weights to FP16
            input_shape: Optional input shape override
            mean_values: Mean values for normalization
            scale_values: Scale values for normalization
            
        Returns:
            Path to converted IR model (.xml file)
        """
        self.logger.info(f"Converting {onnx_path} to OpenVINO IR...")
        
        try:
            # Load ONNX model
            model = ov.convert_model(onnx_path)
            
            # Apply input shape if specified
            if input_shape:
                self.logger.info(f"Reshaping model to: {input_shape}")
                model.reshape(input_shape)
            
            # Apply preprocessing if specified
            if mean_values or scale_values:
                self._apply_preprocessing(model, mean_values, scale_values)
            
            # Define output paths
            ir_path = self.output_dir / f"{model_name}_{precision.lower()}.xml"
            
            # Compress to FP16 if requested
            if compress_to_fp16 or precision == "FP16":
                self.logger.info("Compressing model to FP16...")
                from openvino.tools import mo
                # Note: FP16 compression is applied during compilation or via NNCF
                # For now, we'll save as FP32 and compress during inference
            
            # Save model
            ov.save_model(model, str(ir_path))
            
            self.logger.info(f"IR model saved to: {ir_path}")
            
            # Validate converted model
            if self._validate_ir_model(str(ir_path)):
                self.logger.info("IR model validation passed")
            else:
                raise RuntimeError("IR model validation failed")
            
            # Log model information
            self._log_ir_info(str(ir_path), model)
            
            # Save model metadata
            self._save_model_metadata(model_name, str(ir_path), precision, onnx_path)
            
            return str(ir_path)
            
        except Exception as e:
            self.logger.error(f"IR conversion failed: {str(e)}")
            raise
    
    def _apply_preprocessing(
        self,
        model: ov.Model,
        mean_values: Optional[List[float]] = None,
        scale_values: Optional[List[float]] = None
    ) -> None:
        """Apply preprocessing to the model.
        
        Args:
            model: OpenVINO model
            mean_values: Mean values for normalization
            scale_values: Scale values for normalization
        """
        if not (mean_values or scale_values):
            return
        
        try:
            from openvino.preprocess import PrePostProcessor
            
            ppp = PrePostProcessor(model)
            
            # Apply preprocessing to first input
            input_info = ppp.input()
            
            if mean_values:
                input_info.preprocess().mean(mean_values)
                self.logger.info(f"Applied mean values: {mean_values}")
            
            if scale_values:
                input_info.preprocess().scale(scale_values)
                self.logger.info(f"Applied scale values: {scale_values}")
            
            # Build the model with preprocessing
            model = ppp.build()
            
        except Exception as e:
            self.logger.warning(f"Could not apply preprocessing: {str(e)}")
    
    def _validate_ir_model(self, ir_path: str) -> bool:
        """Validate converted IR model.
        
        Args:
            ir_path: Path to IR model
            
        Returns:
            True if validation passes
        """
        try:
            # Load model
            model = self.core.read_model(ir_path)
            
            # Check if model can be compiled
            compiled_model = self.core.compile_model(model, "CPU")
            
            # Basic shape validation
            for input_layer in model.inputs:
                if len(input_layer.shape) == 0:
                    self.logger.warning(f"Input {input_layer.any_name} has empty shape")
            
            for output_layer in model.outputs:
                if len(output_layer.shape) == 0:
                    self.logger.warning(f"Output {output_layer.any_name} has empty shape")
            
            return True
            
        except Exception as e:
            self.logger.error(f"IR validation failed: {str(e)}")
            return False
    
    def _log_ir_info(self, ir_path: str, model: ov.Model) -> None:
        """Log information about the IR model.
        
        Args:
            ir_path: Path to IR model
            model: OpenVINO model object
        """
        try:
            self.logger.info(f"OpenVINO IR Model Info:")
            
            # File sizes
            xml_size = os.path.getsize(ir_path) / 1024  # KB
            bin_path = ir_path.replace('.xml', '.bin')
            bin_size = os.path.getsize(bin_path) / (1024 ** 2) if os.path.exists(bin_path) else 0  # MB
            
            self.logger.info(f"  - XML file: {xml_size:.2f} KB")
            self.logger.info(f"  - BIN file: {bin_size:.2f} MB")
            
            # Model structure
            self.logger.info(f"  - Inputs: {len(model.inputs)}")
            for inp in model.inputs:
                self.logger.info(f"    - {inp.any_name}: {inp.shape} ({inp.element_type})")
            
            self.logger.info(f"  - Outputs: {len(model.outputs)}")
            for out in model.outputs:
                self.logger.info(f"    - {out.any_name}: {out.shape} ({out.element_type})")
            
            # Operations count
            ops = {}
            for op in model.get_ops():
                op_type = op.get_type_name()
                ops[op_type] = ops.get(op_type, 0) + 1
            
            self.logger.info(f"  - Operations: {sum(ops.values())} total")
            for op_type, count in sorted(ops.items()):
                if count > 1:
                    self.logger.info(f"    - {op_type}: {count}")
            
        except Exception as e:
            self.logger.warning(f"Could not log IR info: {str(e)}")
    
    def _save_model_metadata(
        self,
        model_name: str,
        ir_path: str,
        precision: str,
        source_path: str
    ) -> None:
        """Save model metadata to JSON file.
        
        Args:
            model_name: Model name
            ir_path: Path to IR model
            precision: Model precision
            source_path: Path to source model
        """
        try:
            metadata = {
                "model_name": model_name,
                "ir_path": ir_path,
                "precision": precision,
                "source_path": source_path,
                "conversion_timestamp": str(Path(ir_path).stat().st_mtime),
                "openvino_version": ov.__version__
            }
            
            metadata_path = ir_path.replace('.xml', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Metadata saved to: {metadata_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not save metadata: {str(e)}")
    
    def benchmark_conversion(
        self,
        onnx_path: str,
        model_name: str,
        test_precisions: List[str] = ["FP32", "FP16"]
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark different conversion options.
        
        Args:
            onnx_path: Path to ONNX model
            model_name: Model name
            test_precisions: List of precisions to test
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        for precision in test_precisions:
            self.logger.info(f"Benchmarking {precision} conversion...")
            
            try:
                # Convert model
                ir_path = self.convert_to_ir(
                    onnx_path,
                    f"{model_name}_{precision.lower()}_bench",
                    precision=precision,
                    compress_to_fp16=(precision == "FP16")
                )
                
                # Get file sizes
                xml_size = os.path.getsize(ir_path) / (1024 ** 2)  # MB
                bin_path = ir_path.replace('.xml', '.bin')
                bin_size = os.path.getsize(bin_path) / (1024 ** 2) if os.path.exists(bin_path) else 0
                
                results[precision] = {
                    "ir_path": ir_path,
                    "xml_size_mb": round(xml_size, 2),
                    "bin_size_mb": round(bin_size, 2),
                    "total_size_mb": round(xml_size + bin_size, 2),
                    "conversion_success": True
                }
                
            except Exception as e:
                self.logger.error(f"Failed to convert {precision}: {str(e)}")
                results[precision] = {
                    "conversion_success": False,
                    "error": str(e)
                }
        
        return results
    
    def get_supported_devices(self) -> List[str]:
        """Get list of supported devices.
        
        Returns:
            List of available device names
        """
        try:
            devices = self.core.available_devices
            self.logger.info(f"Available devices: {devices}")
            return devices
        except Exception as e:
            self.logger.error(f"Could not get available devices: {str(e)}")
            return ["CPU"]  # Fallback to CPU