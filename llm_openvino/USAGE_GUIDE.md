# OpenVINO LLM Optimization Pipeline - Usage Guide

This guide provides comprehensive instructions for using the OpenVINO LLM optimization pipeline.

## Quick Start

### 1. Installation

```bash
# Clone or download the pipeline
cd llm_openvino

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_pipeline.py --unit
```

### 2. Basic Usage

```bash
# Run with default settings (distilgpt2, FP16 optimization)
python main.py --benchmark

# Specify model and optimization
python main.py --model distilgpt2 --optimize fp16 --benchmark

# Use custom configuration
python main.py --config config/distilgpt2_int8.yaml
```

### 3. Run Examples

```bash
# Basic pipeline example
python examples/basic_pipeline.py

# Advanced optimization with comprehensive benchmarking
python examples/advanced_optimization.py

# Inference demo only (requires existing models)
python examples/basic_pipeline.py --demo-only
```

## Detailed Usage

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --model MODEL         HuggingFace model name (default: distilgpt2)
  --config CONFIG       Path to YAML configuration file
  --optimize {fp16,int8,both}  Optimization type (default: fp16)
  --device DEVICE       Inference device: CPU, GPU, AUTO (default: CPU)
  --benchmark           Run performance benchmarks
  --demo               Run inference demonstration
  --output-dir DIR      Output directory (default: outputs)
```

### Configuration Files

Create YAML configuration files for reproducible experiments:

```yaml
# config/my_model.yaml
model:
  name: "distilgpt2"
  max_length: 512
  batch_size: 1

optimization:
  precision: "int8"
  quantization_dataset_size: 200

inference:
  device: "CPU"
  num_threads: 4

benchmark:
  warmup_runs: 10
  benchmark_runs: 50
  measure_memory: true
  measure_latency: true
```

### Supported Models

The pipeline supports HuggingFace causal language models:

- **Tested Models:**
  - `distilgpt2` (recommended for testing)
  - `gpt2`
  - `microsoft/DialoGPT-small`

- **Model Requirements:**
  - Must be a causal language model
  - Compatible with `AutoModelForCausalLM`
  - Reasonable size for your hardware

### Optimization Options

#### FP16 Optimization
- Reduces model size by ~50%
- Faster inference on supported hardware
- Minimal accuracy loss

```bash
python main.py --model distilgpt2 --optimize fp16
```

#### INT8 Quantization
- Reduces model size by ~75%
- Significant speedup on CPU
- Requires calibration dataset

```bash
python main.py --model distilgpt2 --optimize int8
```

#### Both Optimizations
- Creates FP32, FP16, and INT8 versions
- Comprehensive performance comparison

```bash
python main.py --model distilgpt2 --optimize both --benchmark
```

## Pipeline Stages

### Stage 1: HuggingFace Loading
- Downloads and caches model
- Validates model compatibility
- Extracts model information

### Stage 2: ONNX Export
- Converts PyTorch model to ONNX
- Validates output consistency
- Optimizes ONNX graph

### Stage 3: OpenVINO Conversion
- Converts ONNX to OpenVINO IR
- Applies graph optimizations
- Validates IR model

### Stage 4: Optimization
- Applies FP16/INT8 quantization
- Uses calibration data for INT8
- Compares model sizes

### Stage 5: Benchmarking
- Measures inference latency
- Tracks memory usage
- Generates performance reports

## Output Structure

```
outputs/
├── distilgpt2.onnx                    # ONNX model
├── distilgpt2_fp32.xml/.bin          # OpenVINO FP32 model
├── distilgpt2_fp16.xml/.bin          # OpenVINO FP16 model
├── distilgpt2_int8.xml/.bin          # OpenVINO INT8 model
├── pipeline_results.json             # Complete results
├── performance_report.csv            # Benchmark data
├── performance_comparison.png         # Performance plots
└── pipeline.log                      # Execution log
```

## Performance Analysis

### Interpreting Results

The pipeline generates comprehensive performance metrics:

```json
{
  "latency": {
    "mean_ms": 45.2,
    "p95_ms": 52.1,
    "tokens_per_second": 22.1
  },
  "memory": {
    "peak_rss_mb": 324.5,
    "memory_increase_mb": 156.2
  }
}
```

### Expected Performance Gains

Typical improvements with DistilGPT-2:

| Precision | Size Reduction | Speed Improvement | Memory Reduction |
|-----------|----------------|-------------------|------------------|
| FP16      | ~50%           | 1.5-2x           | ~50%             |
| INT8      | ~75%           | 2-3x             | ~75%             |

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce model size or batch size
   python main.py --model distilgpt2 --config config/small_model.yaml
   ```

2. **NNCF Import Error**
   ```bash
   pip install nncf>=2.6.0
   ```

3. **Model Download Fails**
   ```bash
   # Use offline mode or different model
   export HF_DATASETS_OFFLINE=1
   python main.py --model gpt2
   ```

4. **OpenVINO Device Error**
   ```bash
   # Check available devices
   python -c "import openvino as ov; print(ov.Core().available_devices)"
   ```

### Performance Optimization Tips

1. **CPU Optimization**
   ```yaml
   inference:
     device: "CPU"
     num_threads: 4  # Match your CPU cores
   ```

2. **Memory Optimization**
   ```yaml
   model:
     max_length: 256  # Reduce sequence length
     batch_size: 1    # Use batch size 1
   ```

3. **Speed Optimization**
   ```yaml
   optimization:
     precision: "int8"  # Use INT8 for maximum speed
   ```

## Advanced Usage

### Custom Models

```python
from main import LLMOptimizationPipeline
from utils import Config

config = Config()
config.model.name = "your-custom-model"
config.model.trust_remote_code = True  # If needed

pipeline = LLMOptimizationPipeline(config)
results = pipeline.run_full_pipeline()
```

### Programmatic Usage

```python
from conversion import HuggingFaceLoader, ONNXExporter
from optimization import ModelQuantizer
from inference import OpenVINOInference

# Load model
loader = HuggingFaceLoader()
model, tokenizer, config = loader.load_model_and_tokenizer("distilgpt2")

# Export to ONNX
exporter = ONNXExporter()
onnx_path = exporter.export_model(model, tokenizer, "my_model")

# Quantize
quantizer = ModelQuantizer()
int8_path = quantizer.quantize_int8(onnx_path, "my_model", tokenizer)

# Run inference
inference = OpenVINOInference(int8_path)
result = inference.generate_text(tokenizer, "Hello world")
```

### Integration Testing

```bash
# Run comprehensive tests
python test_pipeline.py --integration

# Unit tests only
python test_pipeline.py --unit
```

## Best Practices

1. **Start Small**: Begin with `distilgpt2` for testing
2. **Use Configuration Files**: Create reusable configs for experiments
3. **Monitor Resources**: Watch memory usage during quantization
4. **Validate Results**: Always check model outputs after optimization
5. **Benchmark Thoroughly**: Use sufficient warmup and benchmark runs
6. **Save Results**: Keep detailed logs and results for analysis

## Contributing

This pipeline is designed as a GSoC-level contribution to OpenVINO. Key areas for enhancement:

- Support for more model architectures
- Advanced quantization techniques
- GPU optimization
- Distributed inference
- Model serving integration

For questions or contributions, refer to the main OpenVINO repository guidelines.