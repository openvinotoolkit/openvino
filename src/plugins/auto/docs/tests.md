# AUTO Plugin tests

## AUTO plugin Unit Tests

Auto unit test is a set of unit tests using gmock, each of which is for testing selection logic, device list parser, meta device filter and all other internal functions of Auto plugin.

## How to run tests

### Build unit test

1. Turn on `ENABLE_TESTS` in cmake option:

   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release \
       -DENABLE_TESTS=ON \
   ```

2. Build

   ```bash
   make ov_auto_unit_tests
   ```

3. You can find `ov_auto_unit_tests` in *bin* directory after build

### Run unit test

You can run _`ov_auto_unit_tests`_ in *bin* directory which is the output of OpenVINO build

If you want to run a specific unit test, you can use `gtest_filter` option as follows:

```
./ov_auto_unit_tests --gtest_filter='*filter_name*'
```

Then, you can get the result similar to:

```bash
openvino/bin/intel64/Release$ ./ov_auto_unit_tests --gtest_filter=*AutoReleaseHelperTest*cpuLoadFailure_accelerateorLoadFailure*
Running main() from /home/openvino/thirdparty/gtest/gtest/googletest/src/gtest_main.cc
Note: Google Test filter = *AutoReleaseHelperTest*cpuLoadFailure_accelerateorLoadFailure*
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from smoke_Auto_BehaviorTests/AutoReleaseHelperTest
[ RUN      ] smoke_Auto_BehaviorTests/AutoReleaseHelperTest.releaseResource/cpuLoadFailure_accelerateorLoadFailure
[       OK ] smoke_Auto_BehaviorTests/AutoReleaseHelperTest.releaseResource/cpuLoadFailure_accelerateorLoadFailure (732 ms)
[----------] 1 test from smoke_Auto_BehaviorTests/AutoReleaseHelperTest (732 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (732 ms total)
[  PASSED  ] 1 test.
```

## Tests AUTO plugin with benchmark_app

### Performance mode
Benchmark app provides various options for configuring execution parameters on supported devices. This section convers all of the supported performance hints options for AUTO plugin tests. AUTO plugin supports three performance modes setting: including latency, throughput and cumulative_throughput.

#### Latency

Example of Running benchmark_app with ``-hint latency`` on AUTO plugin is shown below:

```bash
openvino/bin/intel64/Release$ ./benchark_app -m openvino/src/core/tests/models/ir/add_abc.xml -d AUTO -hint latency
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
[Step 2/11] Loading OpenVINO Runtime
[ INFO ] OpenVINO:
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
[ INFO ] 
[ INFO ] Device info:
[ INFO ] AUTO
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
...
[Step 8/11] Querying optimal runtime parameters
[ INFO ] Model:
[ INFO ]   NETWORK_NAME: add_abc
[ INFO ]   EXECUTION_DEVICES: (CPU)
[ INFO ]   PERFORMANCE_HINT: ``LATENCY``
[ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
[ INFO ]   MULTI_DEVICE_PRIORITIES: GPU,CPU
[ INFO ]   CPU: 
...
[ INFO ]     PERFORMANCE_HINT: LATENCY
...
[Step 11/11] Dumping statistics report
[ INFO ] Execution Devices: [ GPU ]
[ INFO ] Count:               76254 iterations
[ INFO ] Duration:            120002.81 ms
[ INFO ] Latency:
[ INFO ]    Median:           1.54 ms
[ INFO ]    Average:          1.54 ms
[ INFO ]    Min:              0.14 ms
[ INFO ]    Max:              3.71 ms
[ INFO ] Throughput:          635.44 FPS
```

#### Throughput

Example of Running benchmark_app with ``-hint throughput`` on AUTO plugin is shown below:
```bash
openvino/bin/intel64/Release$ ./benchark_app -m openvino/src/core/tests/models/ir/add_abc.xml -d AUTO -hint throughput
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
[Step 2/11] Loading OpenVINO Runtime
[ INFO ] OpenVINO:
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
[ INFO ] 
[ INFO ] Device info:
[ INFO ] AUTO
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
[ INFO ] 
[ INFO ] 
...
[Step 8/11] Querying optimal runtime parameters
[ INFO ] Model:
[ INFO ]   NETWORK_NAME: add_abc
[ INFO ]   EXECUTION_DEVICES: (CPU)
[ INFO ]   PERFORMANCE_HINT: ``THROUGHPUT``
[ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 4
[ INFO ]   MULTI_DEVICE_PRIORITIES: GPU,CPU
...
[Step 11/11] Dumping statistics report
[ INFO ] Execution Devices: [ GPU ]
[ INFO ] Count:               168284 iterations
[ INFO ] Duration:            120004.81 ms
[ INFO ] Latency:
[ INFO ]    Median:           2.79 ms
[ INFO ]    Average:          2.81 ms
[ INFO ]    Min:              0.44 ms
[ INFO ]    Max:              12.11 ms
[ INFO ] Throughput:          1402.31 FPS
```
#### Cumulative throughput

Example of Running benchmark_app with ``-hint cumulative_throughput`` on AUTO plugin is shown below:
```bash
openvino/bin/intel64/Release$ ./benchark_app -m openvino/src/core/tests/models/ir/add_abc.xml -d AUTO -hint cumulative_throughput
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
[Step 2/11] Loading OpenVINO Runtime
[ INFO ] OpenVINO:
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
[ INFO ] 
[ INFO ] Device info:
[ INFO ] AUTO
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
[ INFO ] 
[ INFO ] 
...
[Step 8/11] Querying optimal runtime parameters
[ INFO ] Model:
[ INFO ]   NETWORK_NAME: add_abc
[ INFO ]   EXECUTION_DEVICES: CPU GPU
[ INFO ]   PERFORMANCE_HINT: ``CUMULATIVE_THROUGHPUT``
[ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 8
...
[Step 11/11] Dumping statistics report
[ INFO ] Execution Devices: [ CPU GPU ]
[ INFO ] Count:               468448 iterations
[ INFO ] Duration:            120001.31 ms
[ INFO ] Latency:
[ INFO ]    Median:           0.36 ms
[ INFO ]    Average:          0.36 ms
[ INFO ]    Min:              0.22 ms
[ INFO ]    Max:              10.48 ms
[ INFO ] Throughput:          3903.69 FPS
```

### Enable/Disable CPU as acceleration

This section shows the setting to AUTO plugin that enables/disables CPU as acceleration (or helper device) at the beginning via the benchmark APP. Configure the property ``ENABLE_STARTUP_FALLBACK`` first in the JSON file ``config.json``.

Running benchmark_APP with enabling the property ``ENABLE_STARTUP_FALLBACK`` in JSON file ``config.json``.

```bash
{
   "AUTO": {
            "ENABLE_STARTUP_FALLBACK": "YES"
   }
}
```

The retrieved property ``EXECUTION_DEVICE`` from AUTO will be CPU accelerator (``(CPU)``).

```bash
openvino/bin/intel64/Release$ ./benchark_app -m openvino/src/core/tests/models/ir/add_abc.xml -d AUTO -load_config ./config.json
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
[Step 2/11] Loading OpenVINO Runtime
[ INFO ] OpenVINO:
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
[ INFO ] 
[ INFO ] Device info:
[ INFO ] AUTO
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
[ INFO ] 
[ INFO ] 
...
[ INFO ]   EXECUTION_DEVICES: (CPU)
...
[ INFO ] First inference took 0.65 ms
...
[ INFO ] Count:               169420 iterations
[ INFO ] Duration:            120004.85 ms
[ INFO ] Latency:
[ INFO ]    Median:           2.76 ms
[ INFO ]    Average:          2.78 ms
[ INFO ]    Min:              0.51 ms
[ INFO ]    Max:              8.39 ms
[ INFO ] Throughput:          1411.78 FPS
```

Running benchmark_APP with disabling the property ``ENABLE_STARTUP_FALLBACK`` in JSON file ``config.json``.

```bash
{
   "AUTO": {
            "ENABLE_STARTUP_FALLBACK": "NO"
   }
}
```

The retrieved property ``EXECUTION_DEVICE`` from AUTO will be CPU accelerator (``GPU``).

```bash
openvino/bin/intel64/Release$ ./benchark_app -m openvino/src/core/tests/models/ir/add_abc.xml -d AUTO -load_config ./config.json
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
[Step 2/11] Loading OpenVINO Runtime
[ INFO ] OpenVINO:
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
[ INFO ] 
[ INFO ] Device info:
[ INFO ] AUTO
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
[ INFO ] 
[ INFO ] 
...
[ INFO ]   EXECUTION_DEVICES: GPU
...
[ INFO ] First inference took 3.97 ms
...
[ INFO ] Count:               167560 iterations
[ INFO ] Duration:            120003.96 ms
[ INFO ] Latency:
[ INFO ]    Median:           2.76 ms
[ INFO ]    Average:          2.81 ms
[ INFO ]    Min:              0.78 ms
[ INFO ]    Max:              5.99 ms
[ INFO ] Throughput:          1396.29 FPS
```

### Device selection fallback of the AUTO

This section will show the fallback of device selection within the AUTO plugin if the device with high priority doesn't support the precision of the inputting model. For example, CPU supports both FP16 and FP32 precision model, while GNA doesn't FP32 precision. Although GNA has higher priority, AUTO plugin will ultimately choose the CPU plugin to load model with FP32 precision.  

AUTO will select GNA if no other device is specified in the device candidate list.

```bash
openvino/bin/intel64/Release$ ./benchmark_app -m openvino/src/core/tests/models/ir/add_abc.xml -d AUTO:GNA -t 10
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
[Step 2/11] Loading OpenVINO Runtime
[ INFO ] OpenVINO:
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
[ INFO ] 
[ INFO ] Device info:
[ INFO ] AUTO
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
[ INFO ] 
[ INFO ] GNA
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
...
[ INFO ]   GNA: 
[ INFO ]     OPTIMIZATION_CAPABILITIES: INT16 INT8 EXPORT_IMPORT
...
[Step 11/11] Dumping statistics report
[ INFO ] Execution Devices: [ GNA ]
...
[ INFO ] Latency:
[ INFO ]    Median:           0.01 ms
[ INFO ]    Average:          0.01 ms
[ INFO ]    Min:              0.01 ms
[ INFO ]    Max:              0.20 ms
[ INFO ] Throughput:          69131.99 FPS
```

Device selection fallback will happen here. CPU will be selected by AUTO as GNA doesn't support FP32 precision model.

```bash
openvino/bin/intel64/Release$ ./benchmark_app -m openvino/src/core/tests/models/ir/add_abc.xml -d AUTO:GNA,CPU -t 10
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
[Step 2/11] Loading OpenVINO Runtime
[ INFO ] OpenVINO:
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
[ INFO ] 
[ INFO ] Device info:
[ INFO ] AUTO
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
[ INFO ] 
[ INFO ] CPU
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
[ INFO ] 
[ INFO ] GNA
[ INFO ] Build ................................. <OpenVINO version>-<Branch name>
...
[Step 11/11] Dumping statistics report
[ INFO ] Execution Devices: [ CPU ]
...
[ INFO ] Latency:
[ INFO ]    Median:           8.90 ms
[ INFO ]    Average:          8.95 ms
[ INFO ]    Min:              5.16 ms
[ INFO ]    Max:              32.78 ms
[ INFO ] Throughput:          446.08 FPS
```
