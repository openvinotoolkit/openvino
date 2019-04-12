# OpenVINO™ Benchmark Python* package
Inference Engine `openvino.tools.benchmark` Python\* package consists types to measure synchronous mode latency.  
The package depends on `openvino.tools.accuracy_checker` the package.

Please, refer to https://docs.openvinotoolkit.org for details.

## Usage
You can use the `openvino.tools.calibration` package in a simple way:
```Python
import openvino.tools.benchmark as benchmark

config = benchmark.CommandLineReader.read()
result = benchmark.Benchmark(config).run()
print("{0}: {1:.4} ms".format(config.model, result.latency * 1000.0))
```
### Explanation
1. Import `openvino.tools.benchmark` types:
```Python
import openvino.tools.benchmark as benchmark
```

2. Read configuration and execute the benchmark:
```Python
config = benchmark.CommandLineReader.read()
result = benchmark.Benchmark(config).run()
```

3. Print results:
```Python
print("{0}: {1:.4} ms".format(config.model, result.latency * 1000.0))
```