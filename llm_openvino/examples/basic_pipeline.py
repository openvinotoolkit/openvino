#!/usr/bin/env python3
"""Basic pipeline example for OpenVINO LLM optimization."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import Config
from main import LLMOptimizationPipeline


def run_basic_pipeline():
    """Run a basic optimization pipeline with distilgpt2."""
    
    # Create configuration
    config = Config()
    config.model.name = "distilgpt2"
    config.model.max_length = 256
    config.optimization.precision = "fp16"
    config.inference.device = "CPU"
    config.benchmark.measure_latency = True
    config.benchmark.measure_memory = True
    config.benchmark.benchmark_runs = 20
    config.output_dir = "outputs/basic_example"
    
    print("=== Basic OpenVINO LLM Pipeline Example ===")
    print(f"Model: {config.model.name}")
    print(f"Optimization: {config.optimization.precision}")
    print(f"Device: {config.inference.device}")
    print(f"Output: {config.output_dir}")
    
    # Initialize and run pipeline
    pipeline = LLMOptimizationPipeline(config)
    
    try:
        results = pipeline.run_full_pipeline()
        
        print("\n=== Pipeline Results ===")
        print(f"Status: {results['status']}")
        
        if results['status'] == 'success':
            # Print model paths
            if 'optimization' in results['stages']:
                models = results['stages']['optimization']['optimized_models']
                print(f"Generated models:")
                for precision, path in models.items():
                    print(f"  {precision.upper()}: {path}")
            
            # Print benchmark results
            if 'benchmarking' in results['stages']:
                print(f"\nPerformance Results:")
                bench_results = results['stages']['benchmarking']['results']
                for precision, metrics in bench_results.items():
                    if 'error' not in metrics:
                        latency = metrics.get('latency', {})
                        if latency:
                            throughput = latency.get('mean_tokens_per_sec', 0)
                            avg_latency = latency.get('mean_ms', 0)
                            print(f"  {precision.upper()}: {throughput:.2f} tokens/sec, {avg_latency:.2f} ms avg latency")
        
        print(f"\nDetailed results saved to: {config.output_dir}/pipeline_results.json")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        return False
    
    return True


def run_inference_demo():
    """Run inference demonstration."""
    
    config = Config()
    config.model.name = "distilgpt2"
    config.output_dir = "outputs/basic_example"
    
    pipeline = LLMOptimizationPipeline(config)
    
    # Check if models exist
    fp32_model = Path(config.output_dir) / f"{config.model.name}_fp32.xml"
    fp16_model = Path(config.output_dir) / f"{config.model.name}_fp16.xml"
    
    print("\n=== Inference Demo ===")
    
    if fp16_model.exists():
        print("Running demo with FP16 model...")
        pipeline.run_inference_demo(str(fp16_model), "fp16")
    elif fp32_model.exists():
        print("Running demo with FP32 model...")
        pipeline.run_inference_demo(str(fp32_model), "fp32")
    else:
        print("No models found. Run the pipeline first.")
        return False
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Basic pipeline example")
    parser.add_argument("--demo-only", action="store_true", 
                       help="Run inference demo only (requires existing models)")
    
    args = parser.parse_args()
    
    if args.demo_only:
        success = run_inference_demo()
    else:
        success = run_basic_pipeline()
        if success:
            run_inference_demo()
    
    sys.exit(0 if success else 1)