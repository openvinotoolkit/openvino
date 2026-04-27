#!/usr/bin/env python3
"""Advanced optimization example with comprehensive benchmarking."""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import Config
from main import LLMOptimizationPipeline


def run_comprehensive_benchmark():
    """Run comprehensive benchmarking across multiple precisions."""
    
    # Create configuration for comprehensive testing
    config = Config()
    config.model.name = "distilgpt2"
    config.model.max_length = 512
    config.optimization.precision = "both"  # Test both FP16 and INT8
    config.inference.device = "CPU"
    config.benchmark.measure_latency = True
    config.benchmark.measure_memory = True
    config.benchmark.benchmark_runs = 50
    config.benchmark.warmup_runs = 10
    config.output_dir = "outputs/advanced_example"
    
    print("=== Advanced OpenVINO LLM Optimization ===")
    print(f"Model: {config.model.name}")
    print(f"Testing: FP32, FP16, INT8")
    print(f"Benchmark runs: {config.benchmark.benchmark_runs}")
    
    # Initialize pipeline
    pipeline = LLMOptimizationPipeline(config)
    
    try:
        results = pipeline.run_full_pipeline()
        
        if results['status'] == 'success':
            # Generate comprehensive report
            generate_performance_report(results, config.output_dir)
            
            # Create visualizations
            create_performance_plots(results, config.output_dir)
            
            print(f"\nComprehensive report generated in: {config.output_dir}")
            
        return results
        
    except Exception as e:
        print(f"Advanced pipeline failed: {str(e)}")
        raise


def generate_performance_report(results, output_dir):
    """Generate detailed performance report."""
    
    output_path = Path(output_dir)
    
    # Extract benchmark data
    benchmark_data = []
    if 'benchmarking' in results['stages']:
        bench_results = results['stages']['benchmarking']['results']
        
        for precision, metrics in bench_results.items():
            if 'error' not in metrics:
                latency = metrics.get('latency', {})
                memory = metrics.get('memory', {})
                
                benchmark_data.append({
                    'precision': precision.upper(),
                    'throughput_tokens_per_sec': latency.get('mean_tokens_per_sec', 0),
                    'latency_mean_ms': latency.get('mean_ms', 0),
                    'latency_p95_ms': latency.get('p95_ms', 0),
                    'memory_peak_mb': getattr(memory, 'peak_rss_mb', 0) if hasattr(memory, 'peak_rss_mb') else 0
                })
    
    # Create DataFrame
    df = pd.DataFrame(benchmark_data)
    
    # Save CSV report
    csv_path = output_path / "performance_report.csv"
    df.to_csv(csv_path, index=False)
    
    # Generate markdown report
    report_path = output_path / "PERFORMANCE_REPORT.md"
    with open(report_path, 'w') as f:
        f.write("# OpenVINO LLM Performance Report\n\n")
        f.write(f"**Model:** {results['model_name']}\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Model Sizes\n\n")
        if 'optimization' in results['stages']:
            models = results['stages']['optimization']['optimized_models']
            for precision, model_path in models.items():
                try:
                    size_mb = Path(model_path.replace('.xml', '.bin')).stat().st_size / (1024**2)
                    f.write(f"- **{precision.upper()}:** {size_mb:.2f} MB\n")
                except:
                    f.write(f"- **{precision.upper()}:** Size unavailable\n")
        
        f.write("\n## Optimization Impact\n\n")
        if len(benchmark_data) > 1:
            fp32_throughput = next((d['throughput_tokens_per_sec'] for d in benchmark_data if d['precision'] == 'FP32'), 0)
            for data in benchmark_data:
                if data['precision'] != 'FP32' and fp32_throughput > 0:
                    speedup = data['throughput_tokens_per_sec'] / fp32_throughput
                    f.write(f"- **{data['precision']} vs FP32:** {speedup:.2f}x speedup\n")
    
    print(f"Performance report saved to: {report_path}")


def create_performance_plots(results, output_dir):
    """Create performance visualization plots."""
    
    try:
        output_path = Path(output_dir)
        
        # Extract data for plotting
        precisions = []
        throughputs = []
        latencies = []
        
        if 'benchmarking' in results['stages']:
            bench_results = results['stages']['benchmarking']['results']
            
            for precision, metrics in bench_results.items():
                if 'error' not in metrics:
                    latency = metrics.get('latency', {})
                    if latency:
                        precisions.append(precision.upper())
                        throughputs.append(latency.get('mean_tokens_per_sec', 0))
                        latencies.append(latency.get('mean_ms', 0))
        
        if len(precisions) > 1:
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Throughput plot
            bars1 = ax1.bar(precisions, throughputs, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(precisions)])
            ax1.set_title('Throughput Comparison')
            ax1.set_ylabel('Tokens per Second')
            ax1.set_xlabel('Precision')
            
            # Add value labels on bars
            for bar, value in zip(bars1, throughputs):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}', ha='center', va='bottom')
            
            # Latency plot
            bars2 = ax2.bar(precisions, latencies, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(precisions)])
            ax2.set_title('Latency Comparison')
            ax2.set_ylabel('Average Latency (ms)')
            ax2.set_xlabel('Precision')
            
            # Add value labels on bars
            for bar, value in zip(bars2, latencies):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_path / "performance_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Performance plots saved to: {plot_path}")
    
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Could not create plots: {str(e)}")


if __name__ == "__main__":
    run_comprehensive_benchmark()