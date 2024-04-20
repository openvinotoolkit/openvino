# Performance Hint and Threads Scheduling 

## Contents
- [Introduction](#introduction)
- [Default Setting of Latency Hint on Hybrid Core](#default-setting-of-latency-hint-on-hybrid-core)
- [Default Setting of Throughput Hint on Hybrid Core](#default-setting-of-throughput-hint-on-hybrid-core)
- [Default Setting of Latency Hint on Dual Sockerts XEON](#default-setting-of-latency-hint-on-dual-sockerts-xeon)
- [Default Setting of Throughput Hint on Dual Sockerts XEON](#default-setting-of-throughput-hint-on-dual-sockerts-xeon)

## Introduction

Even though all supported devices in OpenVINO™ offer low-level performance settings, utilizing them is not recommended outside of very few cases. The preferred way to configure performance in OpenVINO Runtime is using high level performance hint property `ov::hint::performance_mode`. This is a future-proof solution fully compatible with the automatic device selection inference mode and designed with portability in mind.

In order to ease the configuration of the device, OpenVINO offers two dedicated performance hints, namely latency hint `ov::hint::PerformanceMode::LATENCY` and throughput hint `ov::hint::PerformanceMode::THROUGHPUT`.

For threads scheduling in CPU inference, a certain level of automatic configuration of the following low-level performance properties is a result of performance hints.
- `ov::num_streams`
- `ov::inference_num_threads`
- `ov::hint::scheduling_core_type`
- `ov::hint::enable_hyper_threading`
- `ov::hint::enable_cpu_pinning`

## Default Setting of Latency Hint on Hybrid Core

In this scenario, default setting of `ov::hint::scheduling_core_type` is decided by model precision and ratio of P-cores and E-cores.

> **NOTE**: P-cores is Performance-cores and E-cores is Efficient-cores.

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

> **NOTE**: Meteor Lake on Windows use P-cores and E-cores (except for large language models).

## Default Setting of Throughput Hint on Hybrid Core

In this scenario, threads scheduling first evaluates the memory pressure of the current model on the current platform and determines the number of threads per stream, as shown below.

| Memory Pressure | Threads per stream    |
| --------------- | --------------------- |
| least           | 1 P-core or 2 E-cores |
| less            | 2                     |
| normal          | 3 or 4 or 5           |

Then `ov::num_streams` is `ov::inference_num_threads` divided by threads per stream. And the default settings of low-level performance properties on Windows and Linux is as follows.

| Property                         | Windows                       | Linux                         |
| -------------------------------- | ----------------------------- | ----------------------------- |
| ov::num_streams                  | Calculate as above            | Calculate as above            |
| ov::inference_num_threads        | Number of P-cores and E-cores | Number of P-cores and E-cores |
| ov::hint::scheduling_core_type   | P-cores and E-cores           | P-cores and E-cores           |
| ov::hint::enable_hyper_threading | Yes                           | Yes                           |
| ov::hint::enable_cpu_pinning     | No                            | Yes                           |

## Default Setting of Latency Hint on Dual Sockerts XEON

In this scenario, threads scheduling only create 1 stream on one socket. And the default settings of low-level performance properties on Windows and Linux is as follows.

| Property                         | Windows                       | Linux                         |
| -------------------------------- | ----------------------------- | ----------------------------- |
| ov::num_streams                  | 1                             | 1                             |
| ov::inference_num_threads        | Number of cores on one socket | Number of cores on one socket |
| ov::hint::scheduling_core_type   | P-cores or E-cores            | P-cores or E-cores            |
| ov::hint::enable_hyper_threading | No                            | No                            |
| ov::hint::enable_cpu_pinning     | Not Support                   | Yes                           |

## Default Setting of Throughput Hint on Dual Sockerts XEON

In this scenario, threads scheduling first evaluates the memory pressure of the current model on the current platform and determines the number of threads per stream, as shown below.

| Memory Pressure | Threads per stream |
| --------------- | ------------------ |
| least           | 1                  |
| less            | 2                  |
| normal          | 3 or 4 or 5        |

Then `ov::num_streams` is `ov::inference_num_threads` divided by threads per stream. And the default settings of low-level performance properties on Windows and Linux is as follows.

| Property                         | Windows                         | Linux                           |
| -------------------------------- | ------------------------------- | ------------------------------- |
| ov::num_streams                  | Calculate as above              | Calculate as above              |
| ov::inference_num_threads        | Number of cores on dual sockets | Number of cores on dual sockets |
| ov::hint::scheduling_core_type   | P-cores or E-cores              | P-cores or E-cores              |
| ov::hint::enable_hyper_threading | No                              | No                              |
| ov::hint::enable_cpu_pinning     | Not Support                     | Yes                             |
