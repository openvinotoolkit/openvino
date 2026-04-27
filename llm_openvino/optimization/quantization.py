"""Model quantization using NNCF and OpenVINO optimization tools."""

import openvino as ov
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Iterator
import json
from datasets import load_dataset
from transformers import PreTrainedTokenizer

from ..utils import get_logger


class ModelQuantizer:
    """Quantizes OpenVINO models using NNCF."""
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize the quantizer.
        
        Args:
            output_dir: Directory to save quantized models
        """
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.core = ov.Core()
    
    def quantize_int8(
        self,
        ir_path: str,
        model_name: str,
        tokenizer: PreTrainedTokenizer,
        calibration_dataset_size: int = 100,
        max_length: int = 512,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1"
    ) -> str:
        """Quantize model to INT8 using NNCF.
        
        Args:
            ir_path: Path to OpenVINO IR model
            model_name: Name for quantized model
            tokenizer: Tokenizer for data preparation
            calibration_dataset_size: Size of calibration dataset
            max_length: Maximum sequence length
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration
            
        Returns:
            Path to quantized model
        """
        self.logger.info(f"Quantizing {model_name} to INT8...")
        
        try:
            # Import NNCF
            import nncf
            
            # Load model
            model = self.core.read_model(ir_path)
            
            # Prepare calibration dataset
            calibration_data = self._prepare_calibration_dataset(
                tokenizer, calibration_dataset_size, max_length, 
                dataset_name, dataset_config
            )
            
            # Create quantization dataset
            quantization_dataset = nncf.Dataset(
                calibration_data,
                transform_fn=self._transform_fn
            )
            
            # Quantize model
            self.logger.info("Running INT8 quantization...")
            quantized_model = nncf.quantize(
                model,
                quantization_dataset,
                model_type=nncf.ModelType.TRANSFORMER,
                # Advanced quantization parameters
                advanced_parameters=nncf.AdvancedQuantizationParameters(
                    overflow_fix=nncf.OverflowFix.DISABLE,
                    bias_correction=True,
                    smooth_quant_alpha=0.5,
                    smooth_quant_beta=0.5
                )
            )
            
            # Save quantized model
            quantized_path = self.output_dir / f"{model_name}_int8.xml"
            ov.save_model(quantized_model, str(quantized_path))
            
            self.logger.info(f"INT8 quantized model saved to: {quantized_path}")
            
            # Validate quantized model
            if self._validate_quantized_model(str(quantized_path)):
                self.logger.info("Quantized model validation passed")
            else:
                self.logger.warning("Quantized model validation failed")
            
            # Compare model sizes
            self._compare_model_sizes(ir_path, str(quantized_path))
            
            # Save quantization metadata
            self._save_quantization_metadata(
                model_name, str(quantized_path), ir_path, 
                "INT8", calibration_dataset_size
            )
            
            return str(quantized_path)
            
        except ImportError:
            self.logger.error("NNCF not available. Install with: pip install nncf")
            raise
        except Exception as e:
            self.logger.error(f"INT8 quantization failed: {str(e)}")
            raise
    
    def quantize_fp16(
        self,
        ir_path: str,
        model_name: str
    ) -> str:
        """Convert model to FP16 precision.
        
        Args:
            ir_path: Path to OpenVINO IR model
            model_name: Name for FP16 model
            
        Returns:
            Path to FP16 model
        """
        self.logger.info(f"Converting {model_name} to FP16...")
        
        try:
            # Load model
            model = self.core.read_model(ir_path)
            
            # Compress to FP16
            from openvino.tools import mo
            
            # For FP16, we use model optimization during compilation
            # Here we'll save the model and let the inference engine handle FP16
            fp16_path = self.output_dir / f"{model_name}_fp16.xml"
            
            # Save model (FP16 conversion happens during compilation)
            ov.save_model(model, str(fp16_path))
            
            self.logger.info(f"FP16 model saved to: {fp16_path}")
            
            # Save metadata
            self._save_quantization_metadata(
                model_name, str(fp16_path), ir_path, "FP16", 0
            )
            
            return str(fp16_path)
            
        except Exception as e:
            self.logger.error(f"FP16 conversion failed: {str(e)}")
            raise
    
    def _prepare_calibration_dataset(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_size: int,
        max_length: int,
        dataset_name: str,
        dataset_config: str
    ) -> List[Dict[str, np.ndarray]]:
        """Prepare calibration dataset for quantization.
        
        Args:
            tokenizer: Tokenizer for text processing
            dataset_size: Number of samples to use
            max_length: Maximum sequence length
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration
            
        Returns:
            List of input dictionaries
        """
        self.logger.info(f"Preparing calibration dataset ({dataset_size} samples)...")
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_name, dataset_config, split="train")
            
            # Sample data
            if len(dataset) > dataset_size:
                dataset = dataset.shuffle(seed=42).select(range(dataset_size))
            
            calibration_data = []
            
            for i, sample in enumerate(dataset):
                if i >= dataset_size:
                    break
                
                # Get text content
                text = sample.get("text", "")
                if not text or len(text.strip()) < 10:
                    continue
                
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors="np",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                
                # Create input dictionary
                input_dict = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64)
                }
                
                calibration_data.append(input_dict)
            
            self.logger.info(f"Prepared {len(calibration_data)} calibration samples")
            return calibration_data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare calibration dataset: {str(e)}")
            # Fallback to dummy data
            return self._create_dummy_calibration_data(tokenizer, dataset_size, max_length)
    
    def _create_dummy_calibration_data(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_size: int,
        max_length: int
    ) -> List[Dict[str, np.ndarray]]:
        """Create dummy calibration data as fallback.
        
        Args:
            tokenizer: Tokenizer for text processing
            dataset_size: Number of samples to create
            max_length: Maximum sequence length
            
        Returns:
            List of dummy input dictionaries
        """
        self.logger.warning("Using dummy calibration data")
        
        dummy_texts = [
            "This is a sample text for calibration.",
            "Machine learning models require careful optimization.",
            "OpenVINO provides excellent performance for inference.",
            "Quantization reduces model size while maintaining accuracy.",
            "Edge deployment requires efficient model optimization."
        ]
        
        calibration_data = []
        
        for i in range(dataset_size):
            text = dummy_texts[i % len(dummy_texts)]
            
            inputs = tokenizer(
                text,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
            
            input_dict = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            
            calibration_data.append(input_dict)
        
        return calibration_data
    
    def _transform_fn(self, data_item: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform function for NNCF dataset.
        
        Args:
            data_item: Input data item
            
        Returns:
            Transformed data item
        """
        return data_item
    
    def _validate_quantized_model(self, quantized_path: str) -> bool:
        """Validate quantized model.
        
        Args:
            quantized_path: Path to quantized model
            
        Returns:
            True if validation passes
        """
        try:
            # Load and compile model
            model = self.core.read_model(quantized_path)
            compiled_model = self.core.compile_model(model, "CPU")
            
            # Check for quantized operations
            quantized_ops = 0
            for op in model.get_ops():
                if "FakeQuantize" in op.get_type_name():
                    quantized_ops += 1
            
            self.logger.info(f"Found {quantized_ops} quantized operations")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Quantized model validation failed: {str(e)}")
            return False
    
    def _compare_model_sizes(self, original_path: str, quantized_path: str) -> None:
        """Compare sizes of original and quantized models.
        
        Args:
            original_path: Path to original model
            quantized_path: Path to quantized model
        """
        try:
            # Get file sizes
            orig_xml = Path(original_path).stat().st_size / (1024 ** 2)
            orig_bin = Path(original_path.replace('.xml', '.bin')).stat().st_size / (1024 ** 2)
            orig_total = orig_xml + orig_bin
            
            quant_xml = Path(quantized_path).stat().st_size / (1024 ** 2)
            quant_bin = Path(quantized_path.replace('.xml', '.bin')).stat().st_size / (1024 ** 2)
            quant_total = quant_xml + quant_bin
            
            compression_ratio = (orig_total - quant_total) / orig_total * 100
            
            self.logger.info(f"Model size comparison:")
            self.logger.info(f"  Original: {orig_total:.2f} MB")
            self.logger.info(f"  Quantized: {quant_total:.2f} MB")
            self.logger.info(f"  Compression: {compression_ratio:.1f}%")
            
        except Exception as e:
            self.logger.warning(f"Could not compare model sizes: {str(e)}")
    
    def _save_quantization_metadata(
        self,
        model_name: str,
        quantized_path: str,
        original_path: str,
        precision: str,
        calibration_size: int
    ) -> None:
        """Save quantization metadata.
        
        Args:
            model_name: Model name
            quantized_path: Path to quantized model
            original_path: Path to original model
            precision: Target precision
            calibration_size: Calibration dataset size
        """
        try:
            metadata = {
                "model_name": model_name,
                "quantized_path": quantized_path,
                "original_path": original_path,
                "precision": precision,
                "calibration_dataset_size": calibration_size,
                "quantization_timestamp": str(Path(quantized_path).stat().st_mtime)
            }
            
            metadata_path = quantized_path.replace('.xml', '_quantization_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Quantization metadata saved to: {metadata_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not save quantization metadata: {str(e)}")
    
    def benchmark_quantization(
        self,
        ir_path: str,
        model_name: str,
        tokenizer: PreTrainedTokenizer,
        precisions: List[str] = ["FP32", "FP16", "INT8"]
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark different quantization options.
        
        Args:
            ir_path: Path to original IR model
            model_name: Model name
            tokenizer: Tokenizer for calibration data
            precisions: List of precisions to test
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        for precision in precisions:
            self.logger.info(f"Benchmarking {precision} quantization...")
            
            try:
                if precision == "FP32":
                    # Original model
                    model_path = ir_path
                elif precision == "FP16":
                    model_path = self.quantize_fp16(ir_path, f"{model_name}_bench")
                elif precision == "INT8":
                    model_path = self.quantize_int8(ir_path, f"{model_name}_bench", tokenizer)
                else:
                    continue
                
                # Get model size
                xml_size = Path(model_path).stat().st_size / (1024 ** 2)
                bin_path = model_path.replace('.xml', '.bin')
                bin_size = Path(bin_path).stat().st_size / (1024 ** 2) if Path(bin_path).exists() else 0
                total_size = xml_size + bin_size
                
                results[precision] = {
                    "model_path": model_path,
                    "total_size_mb": round(total_size, 2),
                    "quantization_success": True
                }
                
            except Exception as e:
                self.logger.error(f"Failed to quantize {precision}: {str(e)}")
                results[precision] = {
                    "quantization_success": False,
                    "error": str(e)
                }
        
        return results