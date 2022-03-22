# Optimizing for Throughput {#openvino_docs_deployment_optimization_guide_tput}

## General Throughput Considerations
As described in the section on the [latency-specific considerations](./dldt_deployment_optimization_latency.md) one possible use-case is delivering every single request at the minimal delay.
Throughput on the other hand, is about inference scenarios in which potentially large **number of inference requests are served simultaneously to improve the device utilization**.

Here, the overall application inference rate can be significantly improved with the right performance configuration.
Also, if the model is not already memory bandwidth-limited, the associated increase in latency is not linearly dependent on the number of requests executed in parallel.
With the OpenVINO there are two major means of processing multiple inputs simultaneously: **batching** and "**streams**", explained in this document.

## OpenVINO Streams
As detailed in the [common-optimizations section](ref @openvino_docs_deployment_optimization_guide_common) running multiple inference requests asynchronously is important for general application efficiency.
The [Asynchronous API](./dldt_deployment_optimization_common.md) is in fact the "application side" of scheduling, as every device internally implements a queue. The queue acts as a buffer, storing the inference requests until retrieved by the device at its own pace. 

Further, the devices may actually process multiple inference requests in parallel in order to improve the device utilization and overall throughput. This parallelism is commonly referred as **'streams'**. Some devices (like GPU) may run several requests per stream to amortize the host-side costs.
Notice that streams are **really executing the requests in parallel, but not in the lock step** (as e.g. the batching does), which makes the streams fully compatible with [dynamically-shaped inputs](../OV_Runtime_UG/ov_dynamic_shapes.md) when individual requests can have different shapes. 

For efficient asynchronous execution, the streams are actually handling inference with special pool of the threads.
So each time you start inference requests (potentially from different application threads), they are actually muxed into a inference queue of the particular `ov:Compiled_Model`. 
If there is a vacant stream, it pops the request from the queue and actually expedites that to the on-device execution.

The usage of multiple streams is an inherently throughput-oriented approach, as every stream requires a dedicated device memory to do inference in parallel to the rest of streams.
Although similar, the streams are always preferable compared to creating multiple `ov:Compiled_Model` instances for the same model, as weights memory is shared across streams, reducing the overall memory consumption.
Notice that the streams inflate the model load/compilation time.
Finally, using streams does increase the latency of an individual request, this is why for example the [latency hint](./dldt_deployment_optimization_hints.md) governs a device to create a bare minimum of streams (usually just one).

## Batching
There are two primary ways batching can help your performance. You may configure your clients to send batched requests to TensorFlow Serving, or you may send individual requests and configure TensorFlow Serving to wait up to a predetermined period of time, and per
To leverage Batching It might be necessary to redesign the application logic in a task if it's adapted from communicating directly with a service to using a message queue.

## Choosing the Batch Size and Number of Streams

Few considerations for selecting **number of streams**:
* Usually, the Number of service instances deployed only have to be adequate to meet average load rather than the peak load. 
* Notice that [high-level performance hints](../OV_Runtime_UG/performance_hints.md) allows the implementation to select the optimal number of the streams, _depending on the model's compute demands_, application needs (e.g. number of input sources) and device capabilities (e.g. int8 hardware acceleration, number of cores, tiles, etc).

* Your application may send explicitly batched requests to the OpenVINO
   * Yet different devices behave differently with the batch sizes. The optimal batch size depends on the model, inference precision and other factors.
* Streams doesn't require the explicit application logic to collect the data, as they don't execute in a lock step, as explained in the next section 
   * But just like with the batch size, different devices require different number of execution streams to maximize the throughput.

Predicting inference performance is difficult and finding optimal execution parameters requires direct experiments with measurements.
One possible throughput optimization strategy is to **set an upper bound for latency and then increase the batch size or number of the streams until that tail latency is met (or the throughput is not growing anymore)**.
Also, consider [Deep Learning Workbench](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Introduction.html).

## OpenVINO Hints as a Way to Select the Execution Parameters Automatically

Overall, the latency-throughput is not linearly dependent and very _device_ specific. It is also tightly integrated with _model_ characteristics.
As for the possible inference devices the scenery had already become pretty diverse, the OpenVINO has introduced the dedicated notion of the high-level performance configuration "hints" to describe the target application scenarios.
The hints are described [here](./dldt_deployment_optimization_hints.md). 

The hints also obviates the need for explicit (application-side) batching. With the hints, the only requirement for the application is to run multiple individual requests using [Async API](./dldt_deployment_optimization_common.md) and let the OpenVINO decide whether to collect the requests and execute them in batch, streams, or both.

> **NOTE**: [OpenVINO performance hints](./dldt_deployment_optimization_hints.md) is a recommended way for performance configuration, which is both device-agnostic and future-proof. 

## Multi-Device Execution
OpenVINO offers _automatic_, [scalable multi-device inference](../OV_Runtime_UG/multi_device.md). This is simple _application-transparent_ way to improve the throughput. No need to re-architecture existing applications for any explicit multi-device support: no explicit network loading to each device, no separate per-device queues, no additional logic to balance the inference requests between devices, etc. From the application point of view, it is communicating to the single device that internally handles the actual machinery.
Just like with other throughput-oriented scenarios, there are two major pre-requisites for optimal multi-device performance:
*	Using the [Asynchronous API](ref @openvino_docs_deployment_optimization_guide_common) and [callbacks](../OV_Runtime_UG/ov_infer_request.md) in particular
*	Providing the multi-device (and hence the underlying devices) with enough data to crunch. As the inference requests are naturally independent data pieces, the multi-device performs load-balancing at the “requests” (outermost) level to minimize the scheduling overhead.

Notice that the resulting performance is usually a fraction of the “ideal” (plain sum) value, when the devices compete for a certain resources, like the memory-bandwidth which is shared between CPU and iGPU.
> **NOTE**: While the legacy approach of optimizing the parameters of each device separately works, the [OpenVINO performance hints](./dldt_deployment_optimization_hints.md) allow to configure all devices (that are part of the specific multi-device configuration) at once.
