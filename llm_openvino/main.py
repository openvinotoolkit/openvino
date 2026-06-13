#!/usr/bin/env python3
"""Main pipeline orchestrator for OpenVINO LLM optimization."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from utils import Config, load_config, setup_logger
from conversion import HuggingFaceLoader, ONNXExporter, OpenVINOConverter
from optimization import ModelQuantizer
from inference import OpenVINOInference
from benchmarking import LatencyBenchmark, MemoryBenchmark


class LLMOptimizationPipeline:
    """Main pipeline for LLM optimization and deployment."""
    
    def __init__(self, config: Config):
        """Initialize the pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = setup_logger(
            level=config.log_level,
            log_file=Path(config.output_dir) / "pipeline.log"
        )
        
        # Initialize components
        self.hf_loader = HuggingFaceLoader(config.cache_dir)
        self.onnx_exporter = ONNXExporter(config.output_dir)
        self.ov_converter = OpenVINOConverter(config.output_dir)
        self.quantizer = ModelQuantizer(config.output_dir)
        self.latency_benchmark = LatencyBenchmark(config.output_dir)
        self.memory_benchmark = MemoryBenchmark(config.output_dir)
        
        self.logger.info("Pipeline initialized successfully")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete optimization pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("Starting full LLM optimization pipeline...")
        
        results = {
            "model_name": self.config.model.name,
            "config": self.config.__dict__,
            "stages": {}
        }
        
        try:
            # Stage 1: Load HuggingFace model
            self.logger.info("=== Stage 1: Loading HuggingFace Model ===")
            model, tokenizer, config = self._load_huggingface_model()
            results["stages"]["hf_loading"] = {"status": "success"}
            
            # Stage 2: Export to ONNX
            self.logger.info("=== Stage 2: Exporting to ONNX ===")
            onnx_path = self._export_to_onnx(model, tokenizer)
            results["stages"]["onnx_export"] = {
                "status": "success",
                "onnx_path": onnx_path
            }
            
            # Stage 3: Convert to OpenVINO IR
            self.logger.info("=== Stage 3: Converting to OpenVINO IR ===")
            ir_path = self._convert_to_openvino(onnx_path)
            results["stages"]["ov_conversion"] = {
                "status": "success",
                "ir_path": ir_path
            }
            
            # Stage 4: Apply optimizations
            self.logger.info("=== Stage 4: Applying Optimizations ===")
            optimized_models = self._apply_optimizations(ir_path, tokenizer)
            results["stages"]["optimization"] = {
                "status": "success",
                "optimized_models": optimized_models
            }
            
            # Stage 5: Benchmark performance
            if self.config.benchmark.measure_latency or self.config.benchmark.measure_memory:
                self.logger.info("=== Stage 5: Benchmarking Performance ===")
                benchmark_results = self._benchmark_models(optimized_models, tokenizer)
                results["stages"]["benchmarking"] = {
                    "status": "success",
                    "results": benchmark_results
                }
            
            self.logger.info("Pipeline completed successfully!")
            results["status"] = "success"
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            results["status"] = "failed"
            results["error"] = str(e)
            raise
        
        return results
    
    def _load_huggingface_model(self):
        """Load HuggingFace model and tokenizer."""
        model, tokenizer, config = self.hf_loader.load_model_and_tokenizer(
            self.config.model.name,
            torch_dtype=self.config.model.torch_dtype,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        # Validate model
        if not self.hf_loader.validate_model(model, tokenizer):
            raise RuntimeError("Model validation failed")
        
        # Log model info
        model_info = self.hf_loader.get_model_info(model, config)
        self.logger.info(f"Model info: {model_info}")
        
        return model, tokenizer, config
    
    def _export_to_onnx(self, model, tokenizer):
        """Export model to ONNX format."""
        onnx_path = self.onnx_exporter.export_model(
            model,
            tokenizer,
            self.config.model.name,
            max_length=self.config.model.max_length,
            batch_size=self.config.model.batch_size
        )
        
        # Validate ONNX export
        if not self.onnx_exporter.compare_outputs(model, onnx_path, tokenizer):
            self.logger.warning("ONNX output validation failed")
        
        return onnx_path
    
    def _convert_to_openvino(self, onnx_path):
        """Convert ONNX model to OpenVINO IR."""
        ir_path = self.ov_converter.convert_to_ir(
            onnx_path,
            self.config.model.name,
            precision="FP32"
        )
        
        return ir_path
    
    def _apply_optimizations(self, ir_path, tokenizer):
        """Apply various optimizations to the model."""
        optimized_models = {"fp32": ir_path}
        
        # Apply FP16 optimization
        if self.config.optimization.precision in ["fp16", "both"]:
            try:
                fp16_path = self.quantizer.quantize_fp16(ir_path, self.config.model.name)
                optimized_models["fp16"] = fp16_path
                self.logger.info(f"FP16 optimization completed: {fp16_path}")
            except Exception as e:
                self.logger.error(f"FP16 optimization failed: {str(e)}")
        
        # Apply INT8 quantization
        if self.config.optimization.precision in ["int8", "both"]:
            try:
                int8_path = self.quantizer.quantize_int8(
                    ir_path,
                    self.config.model.name,
                    tokenizer,
                    calibration_dataset_size=self.config.optimization.quantization_dataset_size
                )
                optimized_models["int8"] = int8_path
                self.logger.info(f"INT8 quantization completed: {int8_path}")
            except Exception as e:
                self.logger.error(f"INT8 quantization failed: {str(e)}")
        
        return optimized_models
    
    def _benchmark_models(self, optimized_models, tokenizer):
        """Benchmark all optimized models."""
        benchmark_results = {}
        
        test_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "Machine learning has revolutionized the way we"
        ]
        
        for precision, model_path in optimized_models.items():
            self.logger.info(f"Benchmarking {precision.upper()} model...")
            
            try:
                # Initialize inference engine
                inference_engine = OpenVINOInference(
                    model_path,
                    device=self.config.inference.device,
                    num_threads=self.config.inference.num_threads,
                    enable_profiling=self.config.inference.enable_profiling
                )
                
                # Benchmark latency
                latency_results = {}
                if self.config.benchmark.measure_latency:
                    latency_results = inference_engine.benchmark_inference(
                        tokenizer,
                        test_prompts,
                        max_new_tokens=self.config.benchmark.output_tokens,
                        warmup_runs=self.config.benchmark.warmup_runs,
                        benchmark_runs=self.config.benchmark.benchmark_runs
                    )
                
                # Benchmark memory usage
                memory_results = {}
                if self.config.benchmark.measure_memory:
                    def inference_fn():
                        return inference_engine.generate_text(
                            tokenizer,
                            test_prompts[0],
                            max_new_tokens=self.config.benchmark.output_tokens
                        )
                    
                    memory_results = self.memory_benchmark.measure_memory_usage(inference_fn)
                
                benchmark_results[precision] = {
                    "latency": latency_results,
                    "memory": memory_results,
                    "model_info": inference_engine.get_model_info()
                }
                
            except Exception as e:
                self.logger.error(f"Benchmarking {precision} failed: {str(e)}")
                benchmark_results[precision] = {"error": str(e)}
        
        return benchmark_results
    
    def run_inference_demo(self, model_path: str, precision: str = "fp32"):
        """Run a simple inference demonstration.
        
        Args:
            model_path: Path to OpenVINO model
            precision: Model precision
        """
        self.logger.info(f"Running inference demo with {precision.upper()} model...")
        
        # Load tokenizer
        _, tokenizer, _ = self.hf_loader.load_model_and_tokenizer(self.config.model.name)
        
        # Initialize inference
        inference_engine = OpenVINOInference(
            model_path,
            device=self.config.inference.device,
            num_threads=self.config.inference.num_threads
        )
        
        # Demo prompts
        demo_prompts = [
            "The benefits of artificial intelligence include",
            "In the future, technology will",
            "OpenVINO is a toolkit that"
        ]
        
        for i, prompt in enumerate(demo_prompts, 1):
            self.logger.info(f"\n--- Demo {i}/3 ---")
            self.logger.info(f"Prompt: {prompt}")
            
            result = inference_engine.generate_text(
                tokenizer,
                prompt,
                max_new_tokens=30,
                temperature=0.8,
                do_sample=True
            )
            
            self.logger.info(f"Generated: {result['generated_text']}")
            self.logger.info(f"Tokens/sec: {result['tokens_per_second']:.2f}")
            self.logger.info(f"Latency: {result['generation_time']*1000:.2f} ms")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="OpenVINO LLM Optimization Pipeline")
    parser.add_argument("--model", default="distilgpt2", help="HuggingFace model name")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--optimize", choices=["fp16", "int8", "both"], default="fp16",
                       help="Optimization type")
    parser.add_argument("--device", default="CPU", help="Inference device")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--demo", action="store_true", help="Run inference demo")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = Config()
        config.model.name = args.model
        config.optimization.precision = args.optimize
        config.inference.device = args.device
        config.output_dir = args.output_dir
        config.benchmark.measure_latency = args.benchmark
        config.benchmark.measure_memory = args.benchmark
    
    # Initialize pipeline
    pipeline = LLMOptimizationPipeline(config)
    
    try:
        if args.demo:
            # Run demo with existing model
            model_path = Path(config.output_dir) / f"{config.model.name}_fp32.xml"
            if model_path.exists():
                pipeline.run_inference_demo(str(model_path))
            else:
                print("No existing model found. Run full pipeline first.")
                sys.exit(1)
        else:
            # Run full pipeline
            results = pipeline.run_full_pipeline()
            
            # Save results
            import json
            results_path = Path(config.output_dir) / "pipeline_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\nPipeline completed! Results saved to: {results_path}")
            
            # Print summary
            if "benchmarking" in results["stages"]:
                print("\n=== Performance Summary ===")
                for precision, bench_results in results["stages"]["benchmarking"]["results"].items():
                    if "error" not in bench_results:
                        latency = bench_results.get("latency", {})
                        if latency:
                            print(f"{precision.upper()}: {latency.get('mean_tokens_per_sec', 0):.2f} tokens/sec")
    
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()