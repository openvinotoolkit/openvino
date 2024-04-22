# Performance Hint and Threads Scheduling 

## Contents
- [Introduction](#introduction)
- [Latency Hint on Hybrid Core Systems](#latency-hint-on-hybrid-core-systems)
- [Throughput Hint on Hybrid Core Systems](#throughput-hint-on-hybrid-core-systems)
- [Latency Hint on Non-Hybrid Core Systems or Single-Socket XEON platforms](#latency-hint-on-non-hybrid-core-systems-or-single-socket-xeon-platforms)
- [Throughput Hint on Non-Hybrid Core Systems or Single-Socket XEON platforms](#throughput-hint-on-non-hybrid-core-systems-or-single-socket-xeon-platforms)
- [Latency Hint on Dual-Sockert XEON platforms](#latency-hint-on-dual-sockert-xeon-platforms)
- [Throughput Hint on Dual-Sockert XEON platforms](#throughput-hint-on-dual-sockert-xeon-platforms)

## Introduction

Even though all supported devices in OpenVINO™ support low-level performance settings, these settings are not recommended for wide useage unless the application is targeted for specific platforms and models. The recommended approach is to configure performance in OpenVINO Runtime is using high level performance hint property `ov::hint::performance_mode`. Using the perforamnce hints will ensure the best portability and scability of the applications for various platforms and models.

In order to ease the configuration of hardware devices, OpenVINO offers two dedicated performance hints, namely latency hint `ov::hint::PerformanceMode::LATENCY` and throughput hint `ov::hint::PerformanceMode::THROUGHPUT`.

Using CPU as example, with these two performance hints, automatic configuration of the following low-level performance properties are applied on the threads scheduling side. Please note that these configuration details may subject to change among releases to offer the best overall performance on various platforms and a large set of models. The overall performance is usually measured with GEOMean calculation of performance difference for hundreds of models in various size and precisions.  
- `ov::num_streams`
- `ov::inference_num_threads`
- `ov::hint::scheduling_core_type`
- `ov::hint::enable_hyper_threading`
- `ov::hint::enable_cpu_pinning`

For additional details regarding the above configurations, please refer to:
https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html#multi-stream-execution
https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html#optimization-guide

## Latency Hint on Hybrid Core Systems

In this scenario, default setting of `ov::hint::scheduling_core_type` is decided by model precision and ratio of P-cores and E-cores.

> **NOTE**: P-cores is short for Performance-cores and E-cores is for Efficient-cores. These are available in Intel 12th, 13th, and 14th generations of Intel® Core™ processor (code name: Alder Lake, Raptor Lake, Raptor Lake refresh), and Intel® Core™ Ultra Processors (code name Meteor Lake). 

|                            | INT8 model          | FP32 model          |
| -------------------------- | ------------------- | ------------------- |
| P-cores / E-cores < 2      | P-cores             | P-cores             |
| 2 <= P-cores / E-cores < 4 | P-cores             | P-cores and E-cores |
| 4 <= P-cores / E-cores     | P-cores and E-cores | P-cores and E-cores |

Then the default settings of low-level performance properties on Windows and Linux is as follows.

| Property                         | Windows                           | Linux                                         |
| -------------------------------- | --------------------------------- | --------------------------------------------- |
| ov::num_streams                  | 1                                 | 1                                             |
| ov::inference_num_threads        | Dependent on scheduling_core_type | Dependent on scheduling_core_type             |
| ov::hint::scheduling_core_type   | Above table                       | Above table                                   |
| ov::hint::enable_hyper_threading | No                                | No                                            |
| ov::hint::enable_cpu_pinning     | No                                | Yes except using P-cores and E-cores together |

> **NOTE**: Both P-cores and E-cores are used for Latency Hint on Intel® Core™ Ultra Processors with Windows, except for large language models.

## Throughput Hint on Hybrid Core Systems

In this scenario, thread scheduling first evaluates the memory pressure of the model being inferred on the current platform, and determines the number of threads per stream, as shown below.

| Memory Pressure | Threads per stream    |
| --------------- | --------------------- |
| least           | 1 P-core or 2 E-cores |
| less            | 2                     |
| normal          | 3 or 4 or 5           |

Then the value of `ov::num_streams` is `ov::inference_num_threads` divided by the number of threads per stream. And the default settings of low-level performance properties on Windows and Linux are as follows.

| Property                         | Windows                       | Linux                         |
| -------------------------------- | ----------------------------- | ----------------------------- |
| ov::num_streams                  | Calculate as above            | Calculate as above            |
| ov::inference_num_threads        | Number of P-cores and E-cores | Number of P-cores and E-cores |
| ov::hint::scheduling_core_type   | P-cores and E-cores           | P-cores and E-cores           |
| ov::hint::enable_hyper_threading | Yes                           | Yes                           |
| ov::hint::enable_cpu_pinning     | No                            | Yes                           |

## Latency Hint on Non-Hybrid Core Systems or Single-Socket XEON platforms

In this case, the logic is the same as the case where `ov::hint::scheduling_core_type` is P-cores in [Latency Hint on Hybrid Core Systems](#latency-hint-on-hybrid-core-systems).

## Throughput Hint on Non-Hybrid Core Systems or Single-Socket XEON platforms

In this case, the logic is the same as the case where `ov::hint::scheduling_core_type` is P-cores in [Throughput Hint on Hybrid Core Systems](#throughput-hint-on-hybrid-core-systems).

## Latency Hint on Dual-Sockert XEON platforms

In this scenario, thread scheduling only create 1 stream, currently pinned to one socket. And the default settings of low-level performance properties on Windows and Linux is as follows.

| Property                         | Windows                       | Linux                         |
| -------------------------------- | ----------------------------- | ----------------------------- |
| ov::num_streams                  | 1                             | 1                             |
| ov::inference_num_threads        | Number of cores on one socket | Number of cores on one socket |
| ov::hint::scheduling_core_type   | P-cores or E-cores            | P-cores or E-cores            |
| ov::hint::enable_hyper_threading | No                            | No                            |
| ov::hint::enable_cpu_pinning     | Not Support                   | Yes                           |

## Throughput Hint on Dual-Sockert XEON platforms

In this scenario, thread scheduling first evaluates the memory pressure of the model being inferred on the current platform, and determines the number of threads per stream, as shown below.

| Memory Pressure | Threads per stream |
| --------------- | ------------------ |
| least           | 1                  |
| less            | 2                  |
| normal          | 3 or 4 or 5        |

Then the value of `ov::num_streams` is `ov::inference_num_threads` divided by the number of threads per stream. And the default settings of low-level performance properties on Windows and Linux are as follows.

| Property                         | Windows                         | Linux                           |
| -------------------------------- | ------------------------------- | ------------------------------- |
| ov::num_streams                  | Calculate as above              | Calculate as above              |
| ov::inference_num_threads        | Number of cores on dual sockets | Number of cores on dual sockets |
| ov::hint::scheduling_core_type   | P-cores or E-cores              | P-cores or E-cores              |
| ov::hint::enable_hyper_threading | No                              | No                              |
| ov::hint::enable_cpu_pinning     | Not Support                     | Yes                             |
