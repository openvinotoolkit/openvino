"""Memory usage benchmarking utilities."""

import psutil
import os
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import json
from pathlib import Path

from ..utils import get_logger


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float  # Available system memory


@dataclass
class MemoryResult:
    """Container for memory measurement results."""
    peak_rss_mb: float
    peak_vms_mb: float
    avg_rss_mb: float
    avg_vms_mb: float
    memory_increase_mb: float
    snapshots: List[MemorySnapshot]


class MemoryBenchmark:
    """Memory usage benchmarking for inference operations."""
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize memory benchmark.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.process = psutil.Process(os.getpid())
    
    def measure_memory_usage(
        self,
        operation_fn: Callable,
        sampling_interval: float = 0.1,
        **operation_kwargs
    ) -> MemoryResult:
        """Measure memory usage during operation execution.
        
        Args:
            operation_fn: Function to monitor
            sampling_interval: Memory sampling interval in seconds
            **operation_kwargs: Arguments for operation function
            
        Returns:
            MemoryResult with usage statistics
        """
        self.logger.info("Starting memory usage measurement...")
        
        snapshots = []
        monitoring = True
        
        # Get baseline memory
        baseline = self._get_memory_snapshot()
        snapshots.append(baseline)
        
        def memory_monitor():
            while monitoring:
                snapshot = self._get_memory_snapshot()
                snapshots.append(snapshot)
                time.sleep(sampling_interval)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()
        
        try:
            # Run the operation
            start_time = time.time()
            result = operation_fn(**operation_kwargs)
            end_time = time.time()
            
        finally:
            # Stop monitoring
            monitoring = False
            monitor_thread.join(timeout=1.0)
        
        # Get final snapshot
        final_snapshot = self._get_memory_snapshot()
        snapshots.append(final_snapshot)
        
        # Calculate statistics
        rss_values = [s.rss_mb for s in snapshots]
        vms_values = [s.vms_mb for s in snapshots]
        
        memory_result = MemoryResult(
            peak_rss_mb=max(rss_values),
            peak_vms_mb=max(vms_values),
            avg_rss_mb=sum(rss_values) / len(rss_values),
            avg_vms_mb=sum(vms_values) / len(vms_values),
            memory_increase_mb=final_snapshot.rss_mb - baseline.rss_mb,
            snapshots=snapshots
        )
        
        self.logger.info(f"Memory usage - Peak RSS: {memory_result.peak_rss_mb:.2f} MB, "
                        f"Increase: {memory_result.memory_increase_mb:.2f} MB")
        
        return memory_result
    
    def _get_memory_snapshot(self) -> MemorySnapshot:
        """Get current memory usage snapshot.
        
        Returns:
            MemorySnapshot with current usage
        """
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            system_memory = psutil.virtual_memory()
            
            return MemorySnapshot(
                timestamp=time.time(),
                rss_mb=memory_info.rss / (1024 ** 2),
                vms_mb=memory_info.vms / (1024 ** 2),
                percent=memory_percent,
                available_mb=system_memory.available / (1024 ** 2)
            )
        except Exception as e:
            self.logger.warning(f"Could not get memory snapshot: {str(e)}")
            return MemorySnapshot(0, 0, 0, 0, 0)
    
    def benchmark_model_loading(
        self,
        load_fn: Callable,
        **load_kwargs
    ) -> MemoryResult:
        """Benchmark memory usage during model loading.
        
        Args:
            load_fn: Model loading function
            **load_kwargs: Arguments for loading function
            
        Returns:
            MemoryResult for model loading
        """
        self.logger.info("Benchmarking model loading memory usage...")
        
        return self.measure_memory_usage(load_fn, sampling_interval=0.05, **load_kwargs)
    
    def benchmark_inference_memory(
        self,
        inference_fn: Callable,
        num_runs: int = 10,
        **inference_kwargs
    ) -> MemoryResult:
        """Benchmark memory usage during inference.
        
        Args:
            inference_fn: Inference function
            num_runs: Number of inference runs
            **inference_kwargs: Arguments for inference function
            
        Returns:
            MemoryResult for inference
        """
        self.logger.info(f"Benchmarking inference memory usage ({num_runs} runs)...")
        
        def run_multiple_inference():
            for _ in range(num_runs):
                inference_fn(**inference_kwargs)
        
        return self.measure_memory_usage(run_multiple_inference, sampling_interval=0.02)
    
    def save_results(
        self,
        results: Dict[str, Any],
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save memory benchmark results to JSON file.
        
        Args:
            results: Benchmark results
            filename: Output filename
            metadata: Optional metadata
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{filename}.json"
        
        # Convert MemoryResult objects to dictionaries
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, MemoryResult):
                serializable_results[key] = {
                    "peak_rss_mb": value.peak_rss_mb,
                    "peak_vms_mb": value.peak_vms_mb,
                    "avg_rss_mb": value.avg_rss_mb,
                    "avg_vms_mb": value.avg_vms_mb,
                    "memory_increase_mb": value.memory_increase_mb,
                    "snapshot_count": len(value.snapshots)
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
        
        self.logger.info(f"Memory results saved to: {output_path}")
        return str(output_path)