"""OpenVINO inference engine for LLM models."""

import openvino as ov
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import PreTrainedTokenizer
import torch

from ..utils import get_logger


class OpenVINOInference:
    """OpenVINO inference engine for language models."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        num_threads: Optional[int] = None,
        enable_profiling: bool = False
    ):
        """Initialize the inference engine.
        
        Args:
            model_path: Path to OpenVINO IR model (.xml file)
            device: Target device (CPU, GPU, AUTO)
            num_threads: Number of threads for CPU inference
            enable_profiling: Enable performance profiling
        """
        self.logger = get_logger(self.__class__.__name__)
        self.model_path = model_path
        self.device = device
        self.enable_profiling = enable_profiling
        
        # Initialize OpenVINO
        self.core = ov.Core()
        
        # Set device configuration
        if device == "CPU" and num_threads:
            self.core.set_property("CPU", {"INFERENCE_NUM_THREADS": str(num_threads)})
        
        # Load and compile model
        self._load_model()
        
        self.logger.info(f"Initialized OpenVINO inference on {device}")
    
    def _load_model(self) -> None:
        """Load and compile the OpenVINO model."""
        try:
            # Load model
            self.logger.info(f"Loading model from {self.model_path}")
            self.model = self.core.read_model(self.model_path)
            
            # Get model information
            self.input_names = [inp.any_name for inp in self.model.inputs]
            self.output_names = [out.any_name for out in self.model.outputs]
            
            self.logger.info(f"Model inputs: {self.input_names}")
            self.logger.info(f"Model outputs: {self.output_names}")
            
            # Compile model
            config = {}
            if self.enable_profiling:
                config["PERF_COUNT"] = "YES"
            
            self.compiled_model = self.core.compile_model(
                self.model, self.device, config
            )
            
            # Create inference request
            self.infer_request = self.compiled_model.create_infer_request()
            
            self.logger.info("Model compiled successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def generate_text(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate text using the model.
        
        Args:
            tokenizer: Tokenizer for text processing
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: Padding token ID
            
        Returns:
            Dictionary with generated text and metadata
        """
        start_time = time.time()
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="np",
            padding=True,
            truncation=True
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        
        # Generate tokens
        generated_tokens = []
        current_input_ids = input_ids.copy()
        current_attention_mask = attention_mask.copy()
        
        for step in range(max_new_tokens):
            # Run inference
            logits = self._forward(current_input_ids, current_attention_mask)
            
            # Get next token
            next_token = self._sample_next_token(
                logits, temperature, top_k, top_p, do_sample
            )
            
            # Check for EOS token
            if next_token == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token)
            
            # Update input for next iteration
            current_input_ids = np.concatenate([
                current_input_ids, 
                np.array([[next_token]])
            ], axis=1)
            
            current_attention_mask = np.concatenate([
                current_attention_mask,
                np.array([[1]])
            ], axis=1)
        
        # Decode generated text
        if generated_tokens:
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            generated_text = ""
        
        full_text = prompt + generated_text
        
        generation_time = time.time() - start_time
        tokens_per_second = len(generated_tokens) / generation_time if generation_time > 0 else 0
        
        return {
            "generated_text": generated_text,
            "full_text": full_text,
            "generated_tokens": len(generated_tokens),
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
            "prompt_tokens": input_ids.shape[1]
        }
    
    def _forward(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Model logits
        """
        # Prepare inputs
        inputs = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64)
        }
        
        # Run inference
        self.infer_request.infer(inputs)
        
        # Get output
        logits = self.infer_request.get_output_tensor(0).data
        
        return logits
    
    def _sample_next_token(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> int:
        """Sample next token from logits.
        
        Args:
            logits: Model logits
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Next token ID
        """
        # Get logits for last position
        last_logits = logits[0, -1, :]  # [vocab_size]
        
        if not do_sample:
            # Greedy decoding
            return int(np.argmax(last_logits))
        
        # Apply temperature
        if temperature != 1.0:
            last_logits = last_logits / temperature
        
        # Convert to probabilities
        probs = self._softmax(last_logits)
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_indices = np.argpartition(last_logits, -top_k)[-top_k:]
            filtered_probs = np.zeros_like(probs)
            filtered_probs[top_k_indices] = probs[top_k_indices]
            probs = filtered_probs / np.sum(filtered_probs)
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            cumulative_probs = np.cumsum(probs[sorted_indices])
            cutoff_index = np.searchsorted(cumulative_probs, top_p) + 1
            
            filtered_probs = np.zeros_like(probs)
            filtered_probs[sorted_indices[:cutoff_index]] = probs[sorted_indices[:cutoff_index]]
            probs = filtered_probs / np.sum(filtered_probs)
        
        # Sample from distribution
        return int(np.random.choice(len(probs), p=probs))
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities.
        
        Args:
            x: Input logits
            
        Returns:
            Softmax probabilities
        """
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)
    
    def benchmark_inference(
        self,
        tokenizer: PreTrainedTokenizer,
        test_prompts: List[str],
        max_new_tokens: int = 50,
        warmup_runs: int = 5,
        benchmark_runs: int = 20
    ) -> Dict[str, Any]:
        """Benchmark inference performance.
        
        Args:
            tokenizer: Tokenizer for text processing
            test_prompts: List of test prompts
            max_new_tokens: Maximum tokens to generate
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        self.logger.info("Running inference benchmark...")
        
        # Warmup
        self.logger.info(f"Warming up with {warmup_runs} runs...")
        for _ in range(warmup_runs):
            self.generate_text(
                tokenizer, test_prompts[0], max_new_tokens=10, do_sample=False
            )
        
        # Benchmark
        self.logger.info(f"Running {benchmark_runs} benchmark runs...")
        latencies = []
        throughputs = []
        total_tokens = 0
        
        for i in range(benchmark_runs):
            prompt = test_prompts[i % len(test_prompts)]
            
            result = self.generate_text(
                tokenizer, prompt, max_new_tokens=max_new_tokens, do_sample=False
            )
            
            latencies.append(result["generation_time"])
            throughputs.append(result["tokens_per_second"])
            total_tokens += result["generated_tokens"]
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        avg_throughput = np.mean(throughputs)
        std_throughput = np.std(throughputs)
        
        results = {
            "device": self.device,
            "model_path": self.model_path,
            "benchmark_runs": benchmark_runs,
            "total_tokens_generated": total_tokens,
            "latency": {
                "mean_ms": avg_latency * 1000,
                "std_ms": std_latency * 1000,
                "min_ms": min(latencies) * 1000,
                "max_ms": max(latencies) * 1000,
                "p50_ms": np.percentile(latencies, 50) * 1000,
                "p95_ms": np.percentile(latencies, 95) * 1000,
                "p99_ms": np.percentile(latencies, 99) * 1000
            },
            "throughput": {
                "mean_tokens_per_sec": avg_throughput,
                "std_tokens_per_sec": std_throughput,
                "min_tokens_per_sec": min(throughputs),
                "max_tokens_per_sec": max(throughputs)
            }
        }
        
        self.logger.info(f"Benchmark completed:")
        self.logger.info(f"  Average latency: {avg_latency*1000:.2f} ms")
        self.logger.info(f"  Average throughput: {avg_throughput:.2f} tokens/sec")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_path": self.model_path,
            "device": self.device,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "input_shapes": {inp.any_name: list(inp.shape) for inp in self.model.inputs},
            "output_shapes": {out.any_name: list(out.shape) for out in self.model.outputs}
        }
        
        return info
    
    def get_performance_counters(self) -> Optional[Dict[str, Any]]:
        """Get performance counters if profiling is enabled.
        
        Returns:
            Performance counters or None
        """
        if not self.enable_profiling:
            return None
        
        try:
            counters = self.infer_request.get_profiling_info()
            return {
                "counters": [
                    {
                        "node_name": counter.node_name,
                        "exec_type": counter.exec_type,
                        "status": counter.status.name,
                        "real_time_us": counter.real_time.total_seconds() * 1e6,
                        "cpu_time_us": counter.cpu_time.total_seconds() * 1e6
                    }
                    for counter in counters
                ]
            }
        except Exception as e:
            self.logger.warning(f"Could not get performance counters: {str(e)}")
            return None