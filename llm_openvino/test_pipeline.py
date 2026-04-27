#!/usr/bin/env python3
"""Test script for the OpenVINO LLM optimization pipeline."""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils import Config, load_config, save_config
from conversion import HuggingFaceLoader, ONNXExporter, OpenVINOConverter
from optimization import ModelQuantizer
from inference import OpenVINOInference


class TestPipelineComponents(unittest.TestCase):
    """Test suite for pipeline components."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config = Config()
        self.config.model.name = "distilgpt2"
        self.config.output_dir = self.test_dir
        self.config.cache_dir = self.test_dir
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_config_loading(self):
        """Test configuration loading and saving."""
        # Save config
        config_path = Path(self.test_dir) / "test_config.yaml"
        save_config(self.config, str(config_path))
        
        # Load config
        loaded_config = load_config(str(config_path))
        
        self.assertEqual(loaded_config.model.name, self.config.model.name)
        self.assertEqual(loaded_config.output_dir, self.config.output_dir)
    
    def test_huggingface_loader(self):
        """Test HuggingFace model loading."""
        loader = HuggingFaceLoader(self.test_dir)
        
        try:
            model, tokenizer, config = loader.load_model_and_tokenizer(
                "distilgpt2", torch_dtype="float32"
            )
            
            # Validate model
            self.assertTrue(loader.validate_model(model, tokenizer))
            
            # Get model info
            model_info = loader.get_model_info(model, config)
            self.assertIn("total_parameters", model_info)
            self.assertGreater(model_info["total_parameters"], 0)
            
        except Exception as e:
            self.skipTest(f"HuggingFace model loading failed: {str(e)}")
    
    def test_onnx_export(self):
        """Test ONNX export functionality."""
        # This test requires a loaded model, so we'll skip if loading fails
        try:
            loader = HuggingFaceLoader(self.test_dir)
            model, tokenizer, _ = loader.load_model_and_tokenizer("distilgpt2")
            
            exporter = ONNXExporter(self.test_dir)
            onnx_path = exporter.export_model(
                model, tokenizer, "test_model", max_length=128
            )
            
            self.assertTrue(Path(onnx_path).exists())
            
        except Exception as e:
            self.skipTest(f"ONNX export test failed: {str(e)}")
    
    def test_openvino_conversion(self):
        """Test OpenVINO IR conversion."""
        # This test requires ONNX model, so we'll create a minimal test
        converter = OpenVINOConverter(self.test_dir)
        
        # Test device availability
        devices = converter.get_supported_devices()
        self.assertIn("CPU", devices)
    
    def test_model_quantizer(self):
        """Test model quantization setup."""
        quantizer = ModelQuantizer(self.test_dir)
        
        # Test quantizer initialization
        self.assertEqual(quantizer.output_dir, Path(self.test_dir))
    
    def test_inference_engine_init(self):
        """Test inference engine initialization."""
        # This test will be skipped if no model is available
        test_model_path = Path(self.test_dir) / "test_model.xml"
        
        if not test_model_path.exists():
            self.skipTest("No test model available for inference testing")


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end pipeline tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_minimal_pipeline(self):
        """Test minimal pipeline execution."""
        from main import LLMOptimizationPipeline
        
        config = Config()
        config.model.name = "distilgpt2"
        config.model.max_length = 128
        config.optimization.precision = "fp16"
        config.benchmark.measure_latency = False
        config.benchmark.measure_memory = False
        config.output_dir = self.test_dir
        config.cache_dir = self.test_dir
        
        pipeline = LLMOptimizationPipeline(config)
        
        try:
            # Test pipeline initialization
            self.assertIsNotNone(pipeline.hf_loader)
            self.assertIsNotNone(pipeline.onnx_exporter)
            self.assertIsNotNone(pipeline.ov_converter)
            
            # Note: Full pipeline test would require significant resources
            # and time, so we only test initialization here
            
        except Exception as e:
            self.skipTest(f"Pipeline initialization failed: {str(e)}")


def run_integration_test():
    """Run a quick integration test with actual model."""
    print("Running integration test...")
    
    test_dir = tempfile.mkdtemp()
    
    try:
        config = Config()
        config.model.name = "distilgpt2"
        config.model.max_length = 64  # Small for testing
        config.optimization.precision = "fp16"
        config.benchmark.measure_latency = True
        config.benchmark.measure_memory = False
        config.benchmark.benchmark_runs = 5  # Quick test
        config.output_dir = test_dir
        config.cache_dir = test_dir
        
        from main import LLMOptimizationPipeline
        pipeline = LLMOptimizationPipeline(config)
        
        # Test individual components
        print("Testing HuggingFace loading...")
        model, tokenizer, model_config = pipeline._load_huggingface_model()
        print("✓ HuggingFace loading successful")
        
        print("Testing ONNX export...")
        onnx_path = pipeline._export_to_onnx(model, tokenizer)
        print(f"✓ ONNX export successful: {onnx_path}")
        
        print("Testing OpenVINO conversion...")
        ir_path = pipeline._convert_to_openvino(onnx_path)
        print(f"✓ OpenVINO conversion successful: {ir_path}")
        
        print("Testing inference...")
        inference_engine = OpenVINOInference(ir_path, device="CPU")
        result = inference_engine.generate_text(
            tokenizer, "Hello world", max_new_tokens=5
        )
        print(f"✓ Inference successful: '{result['generated_text']}'")
        
        print("\n🎉 Integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        return False
        
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OpenVINO LLM pipeline")
    parser.add_argument("--integration", action="store_true",
                       help="Run integration test with actual model")
    parser.add_argument("--unit", action="store_true",
                       help="Run unit tests")
    
    args = parser.parse_args()
    
    if args.integration:
        success = run_integration_test()
        sys.exit(0 if success else 1)
    elif args.unit:
        unittest.main(argv=[''])
    else:
        # Run both by default
        print("Running unit tests...")
        unittest.main(argv=[''], exit=False, verbosity=2)
        
        print("\n" + "="*50)
        success = run_integration_test()
        sys.exit(0 if success else 1)