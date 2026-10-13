---
sidebar_position: 5
sidebar_label: "Optimize Inference"
description: "Improving inference performance involves model and runtime optimizations that can be done independently. Inference speed depends on latency and throughput."
---

# Optimize Inference


Runtime, or deployment optimization focuses on tuning inference and execution parameters. Unlike
model-level optimization, it is highly specific to the hardware you use and the goal you want
to achieve. You need to plan whether to prioritize accuracy or performance,
[throughput](optimizing-throughput/index.md) or [latency](optimizing-latency/index.md),
or aim at the golden mean. You should also predict how scalable your application needs to be
and how exactly it is going to work with the inference component. This way, you will be able
to achieve the best results for your product.

:::note
For more information on this topic, see the following articles:

- [Inference Devices and Modes](../inference-devices-and-modes/index.md)
- [Inputs Pre-processing with the OpenVINO](general-optimizations.md#inputs-pre-processing)
- [Async API](general-optimizations.md#async-api)
- [The 'get_tensor' Idiom](general-optimizations.md#tensor-idiom)
- For variably-sized inputs, consider [dynamic shapes](../model-input-output/dynamic-shapes/index.md)
:::

## Performance-Portable Inference

To make configuration easier and performance optimization more portable, OpenVINO offers the
[Performance Hints](high-level-performance-hints.md) feature. It comprises
two high-level “presets” focused on latency **(default)** or throughput.

Although inference with OpenVINO Runtime can be configured with a multitude
of low-level performance settings, it is not recommended, as:

- It requires deep understanding of device architecture and the inference engine.
- It may not translate well to other device-model combinations. For example:
  
  - CPU and GPU deduce their optimal number of streams differently.
  - Different devices of the same type, favor different execution configurations.
  - Different models favor different parameter configurations (e.g., compute vs memory-bandwidth,
    inference precision, and possible model quantization).
  - Execution “scheduling” impacts performance strongly and is highly device specific. GPU-oriented
    optimizations [do not always map well to the CPU](optimizing-low-level-implementation.md).

## Additional Resources

- [Using Async API and running multiple inference requests in parallel to leverage throughput](optimizing-throughput/index.md#throughput-app-design).
- [The throughput approach implementation details for specific devices](optimizing-low-level-implementation.md)
- [Details on throughput](optimizing-throughput/index.md)
- [Details on latency](optimizing-latency/index.md)
- [API examples and details](high-level-performance-hints.md)
