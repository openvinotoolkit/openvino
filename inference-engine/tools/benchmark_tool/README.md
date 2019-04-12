# OpenVINO™ Benchmark Tool
Inference Engine Benchmark Tool is a Python\* command-line tool, which measures latency for synchronous mode.

Please, refer to https://docs.openvinotoolkit.org for details.

## Usage

In general, the Benchmark Tool is configured in the same way as the Accuracy Checker. You can also use additional command line arguments to define benchmark-specific parameters:

| Argument                                     | Type   | Description                                              |
| -------------------------------------------- | ------ | -------------------------------------------------------- |
| -c, --config                                 | string | Required. Path to the YML file with local configuration  |
| -ic, --benchmark_iterations_count            | string | Optional. Benchmark itertations count. (1000 is default) |

## Hardware requirements
Hardware requirements depend on a model. Typically for public models RAM memory size has to be not less then 16Gb independently on operation system.