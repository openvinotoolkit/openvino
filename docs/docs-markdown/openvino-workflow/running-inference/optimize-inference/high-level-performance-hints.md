---
sidebar_position: 2
sidebar_label: "High-level Performance Hints"
description: "OpenVINO Runtime offers two dedicated high-level performance hints, namely throughput and latency, that help to configure an inference device."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# High-level Performance Hints

Even though all [supported devices](../../../documentation/compatibility-and-support/supported-devices.md) in OpenVINO™ offer low-level performance settings, utilizing them is not recommended outside of very few cases.
The preferred way to configure performance in OpenVINO Runtime is using performance hints. This is a future-proof solution fully compatible with the [automatic device selection inference mode](../inference-devices-and-modes/auto-device-selection/index.md) and designed with *portability* in mind.

The hints also set the direction of the configuration in the right order. Instead of mapping the application needs to the low-level performance settings, and keeping an associated application logic to configure each possible device separately, the hints express a target scenario with a single config key and let the *device* configure itself in response.

Previously, a certain level of automatic configuration was the result of the *default* values of the parameters. For example, the number of CPU streams was deduced from the number of CPU cores, when `ov::streams::AUTO` was set. However, the resulting number of streams did not account for actual compute requirements of the model to be inferred.
The hints, in contrast, respect the actual model, so the parameters for optimal throughput are calculated for each model individually (based on its compute versus memory bandwidth requirements and capabilities of the device).

## Performance Hints: Latency and Throughput

As discussed in the [Optimization Guide](index.md) there are a few different metrics associated with inference speed. Latency and throughput are some of the most widely used metrics that measure the overall performance of an application.

Therefore, in order to ease the configuration of the device, OpenVINO offers two dedicated hints, namely `ov::hint::PerformanceMode::LATENCY` **(default)** and `ov::hint::PerformanceMode::THROUGHPUT`.

For more information on conducting performance measurements with the `benchmark_app`, refer to the last section in this document.

Models may load slower and consume more memory when using `ov::hint::PerformanceMode::THROUGHPUT` compared to `ov::hint::PerformanceMode::LATENCY`. `THROUGHPUT` and `LATENCY` hints can improve performance in asynchronous inference and support different use cases in synchronous inference. For more details on asynchronous inference, see the [Working with Sync/Async API](#working-with-sync-async-api) section of this document.

## Performance Hints: How It Works

Internally, every device "translates" the value of the hint to the actual performance settings.
For example, the `ov::hint::PerformanceMode::THROUGHPUT` selects the number of CPU or GPU streams.
Additionally, the optimal batch size is selected for the GPU and the [automatic batching](../inference-devices-and-modes/automatic-batching.md) is applied whenever possible. To check whether the device supports it, refer to the [Supported devices](../../../documentation/compatibility-and-support/supported-devices.md) article.

The resulting (device-specific) settings can be queried back from the instance of the `ov:Compiled_Model`.
Be aware that the `benchmark_app` outputs the actual settings for the `THROUGHPUT` hint. See the example of the output below:

```bash
$benchmark_app -hint tput -d CPU -m 'path to your favorite model'
...
[Step 8/11] Setting optimal runtime parameters
[ INFO ] Device: CPU
[ INFO ]   { PERFORMANCE_HINT , THROUGHPUT }
...
[ INFO ]   { OPTIMAL_NUMBER_OF_INFER_REQUESTS , 4 }
[ INFO ]   { NUM_STREAMS , 4 }
...
```

## Using the Performance Hints: Basic API

In the example code snippet below, `ov::hint::PerformanceMode::THROUGHPUT` is specified for the `ov::hint::performance_mode` property for `compile_model`:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      import openvino.properties as props
      import openvino.properties.hint as hints

      config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
      compiled_model = core.compile_model(model, "GPU", config)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  auto compiled_model = core.compile_model(model, "GPU",
      ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
  ```
  </TabItem>
</Tabs>

## Additional (Optional) Hints from the App

For an application that processes 4 video streams, the most future-proof way to communicate the limitation of the parallel slack is to equip the performance hint with the optional `ov::hint::num_requests` configuration key set to 4.
As mentioned earlier, this will limit the batch size for the GPU and the number of inference streams for the CPU. Thus, each device uses the `ov::hint::num_requests` while converting the hint to the actual device configuration options:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
                hints.num_requests: "4"}
      # limiting the available parallel slack for the 'throughput'
      # so that certain parameters (like selected batch size) are automatically accommodated accordingly
      compiled_model = core.compile_model(model, "GPU", config)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // limiting the available parallel slack for the 'throughput' hint via the ov::hint::num_requests
  // so that certain parameters (like selected batch size) are automatically accommodated accordingly 
  auto compiled_model = core.compile_model(model, "GPU",
      ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
      ov::hint::num_requests(4));
  ```
  </TabItem>
</Tabs>

## Optimal Number of Inference Requests

The hints are used on the presumption that the application queries `ov::optimal_number_of_infer_requests` to create and run the returned number of requests simultaneously:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      # when the batch size is automatically selected by the implementation
      # it is important to query/create and run the sufficient requests
      config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
      compiled_model = core.compile_model(model, "GPU", config)
      num_requests = compiled_model.get_property(props.optimal_number_of_infer_requests)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // when the batch size is automatically selected by the implementation
  // it is important to query/create and run the sufficient #requests
  auto compiled_model = core.compile_model(model, "GPU",
      ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
  auto num_requests = compiled_model.get_property(ov::optimal_number_of_infer_requests);
  ```
  </TabItem>
</Tabs>

While an application is free to create more requests if needed (for example to support asynchronous inputs population) **it is very important to at least run the** `ov::optimal_number_of_infer_requests` **of the inference requests in parallel**. It is recommended for efficiency, or device utilization, reasons.


## Working with Sync/Async API {#working-with-sync-async-api}

The API of the inference requests offers Sync and Async execution. The `ov::InferRequest::infer()` is inherently synchronous and simple to operate (as it serializes the execution flow in the current application thread). The Async "splits" the `infer()` into `ov::InferRequest::start_async()` and `ov::InferRequest::wait()` (or callbacks). For more information on synchronous and asynchronous modes, refer to the [OpenVINO Inference Request](../inference-request/index.md).

Use the `LATENCY` hint with synchronous inference (a blocking inference call with a single inference request) for low latency. Use the `THROUGHPUT` hint with asynchronous inference (callbacks-based) for high throughput when the applications handles multiple concurrent inference requests at the same time. `THROUGHPUT` and `LATENCY` hints automatically configure the inference pipeline for the optimal number of streams and requests.

## Combining the Hints and Individual Low-Level Settings

While sacrificing the portability to some extent, it is possible to combine the hints with individual device-specific settings.
For example, use `ov::hint::PerformanceMode::THROUGHPUT` to prepare a general configuration and override any of its specific values:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
                props.inference_num_threads: "4"}
      # limiting the available parallel slack for the 'throughput'
      # so that certain parameters (like selected batch size) are automatically accommodated accordingly
      compiled_model = core.compile_model(model, "CPU", config)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  {
      // high-level performance hints are compatible with low-level device-specific settings 
  auto compiled_model = core.compile_model(model, "CPU",
      ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
      ov::inference_num_threads(4));
  }
  ```
  </TabItem>
</Tabs>

## Testing Performance of the Hints with the Benchmark\_App

Using the [benchmark_app sample](../../../get-started/learn-openvino/openvino-samples/benchmark-tool.md) is the best way to evaluate the functionality of the performance hints for a particular device:

- benchmark\_app **-hint tput** -d 'device' -m 'path to your model'
- benchmark\_app **-hint latency** -d 'device' -m 'path to your model'

Disabling the hints to emulate the pre-hints era (highly recommended before trying the individual low-level settings, such as the number of streams as below, threads, etc):

- benchmark\_app **-hint none -nstreams 1**  -d 'device' -m 'path to your model'
