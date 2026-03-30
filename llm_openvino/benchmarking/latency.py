"""Latency benchmarking utilities."""

import time
import statistics
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import json
from pathlib import Path

from ..utils import get_logger


@dataclass
class LatencyResult:
    """Container for latency measurement results."""
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    samples: List[float]


class LatencyBenchmark:
    """Comprehensive latency benchmarking for inference operations."""
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize latency benchmark.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def measure_inference_latency(
        self,
        inference_fn: Callable,
        warmup_runs: int = 10,
        benchmark_runs: int = 50,
        **inference_kwargs
    ) -> LatencyResult:
        """Measure inference latency with statistical analysis.
        
        Args:
            inference_fn: Function to benchmark
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
            **inference_kwargs: Arguments for inference function
            
        Returns:
            LatencyResult with statistics
        """
        self.logger.info(f"Measuring latency: {warmup_runs} warmup + {benchmark_runs} benchmark runs")
        
        # Warmup
        self.logger.info("Running warmup...")
        for _ in range(warmup_runs):
            inference_fn(**inference_kwargs)
        
        # Benchmark
        self.logger.info("Running benchmark...")
        latencies = []
        
        for i in range(benchmark_runs):
            start_time = time.perf_counter()
            inference_fn(**inference_kwargs)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Completed {i + 1}/{benchmark_runs} runs")
        
        # Calculate statistics
        result = LatencyResult(
            mean_ms=statistics.mean(latencies),
            std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            min_ms=min(latencies),
            max_ms=max(latencies),
            p50_ms=np.percentile(latencies, 50),
            p95_ms=np.percentile(latencies, 95),
            p99_ms=np.percentile(latencies, 99),
            samples=latencies
        )
        
        self.logger.info(f"Latency results: {result.mean_ms:.2f}±{result.std_ms:.2f} ms")
        return result
    
    def measure_end_to_end_latency(
        self,
        pipeline_fn: Callable,
        test_inputs: List[Any],
        warmup_runs: int = 5,
        **pipeline_kwargs
    ) -> Dict[str, LatencyResult]:
        """Measure end-to-end pipeline latency.
        
        Args:
            pipeline_fn: Pipeline function to benchmark
            test_inputs: List of test inputs
            warmup_runs: Number of warmup runs
            **pipeline_kwargs: Arguments for pipeline function
            
        Returns:
            Dictionary with latency results per input
        """
        results = {}
        
        for i, test_input in enumerate(test_inputs):
            self.logger.info(f"Benchmarking input {i+1}/{len(test_inputs)}")
            
            def run_pipeline():
                return pipeline_fn(test_input, **pipeline_kwargs)
            
            result = self.measure_inference_latency(
                run_pipeline, warmup_runs=warmup_runs, benchmark_runs=20
            )
            
            results[f"input_{i}"] = result
        
        return results
    
    def save_results(
        self,
        results: Dict[str, Any],
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save benchmark results to JSON file.
        
        Args:
            results: Benchmark results
            filename: Output filename
            metadata: Optional metadata
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{filename}.json"
        
        # Convert LatencyResult objects to dictionaries
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, LatencyResult):
                serializable_results[key] = {
                    "mean_ms": value.mean_ms,
                    "std_ms": value.std_ms,
                    "min_ms": value.min_ms,
                    "max_ms": value.max_ms,
                    "p50_ms": value.p50_ms,
                    "p95_ms": value.p95_ms,
                    "p99_ms": value.p99_ms,
                    "sample_count": len(value.samples)
                }
            else:
                serializable_results[key] = value
        
        # Add metadata
        output_data = {
            "results": serializable_results,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Results saved to: {output_path}")
        return str(output_path)