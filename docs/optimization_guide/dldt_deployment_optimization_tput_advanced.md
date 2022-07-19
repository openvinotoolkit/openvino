# Using Advanced Throughput Options: Streams and Batching {#openvino_docs_deployment_optimization_guide_tput_advanced}

## OpenVINO Streams
As detailed in the [common-optimizations section](@ref openvino_docs_deployment_optimization_guide_common) running multiple inference requests asynchronously is important for general application efficiency.
Internally, every device implements a queue. The queue acts as a buffer, storing the inference requests until retrieved by the device at its own pace. 
The devices may actually process multiple inference requests in parallel in order to improve the device utilization and overall throughput.
This configurable mean of this device-side parallelism is commonly referred as **streams**.

> **NOTE**: Notice that streams are **really executing the requests in parallel, but not in the lock step** (as e.g. the batching does), which makes the streams fully compatible with [dynamically-shaped inputs](../OV_Runtime_UG/ov_dynamic_shapes.md) when individual requests can have different shapes.

> **NOTE**: Most OpenVINO devices (including CPU, GPU and VPU) support the streams, yet the _optimal_ number of the streams is deduced very differently, please see the a dedicated section below.

Few general considerations:
* Using the streams does increase the latency of an individual request
   * When no number of streams is not specified, a device creates a bare minimum of streams (usually just one), as the latency-oriented case is default
   * Please find further tips for the optimal number of the streams [below](@ref throughput_advanced)
* Streams are memory-hungry, as every stream duplicates the intermediate buffers to do inference in parallel to the rest of streams
   * Always prefer streams over creating multiple `ov:Compiled_Model` instances for the same model, as weights memory is shared across streams, reducing the memory consumption
* Notice that the streams also inflate the model load (compilation) time.

For efficient asynchronous execution, the streams are actually handling the inference with a special pool of the threads (a thread per stream).
Each time you start inference requests (potentially from different application threads), they are actually muxed into a inference queue of the particular `ov:Compiled_Model`. 
If there is a vacant stream, it pops the request from the queue and actually expedites that to the on-device execution.
There are further device-specific details e.g. for the CPU, that you may find in the [internals](dldt_deployment_optimization_internals.md) section.

## Batching
Hardware accelerators like GPUs are optimized for massive compute parallelism, so the batching helps to saturate the device and leads to higher throughput.
While the streams (described earlier) already help to hide the communication overheads and certain bubbles in the scheduling, running multiple OpenCL kernels simultaneously is less GPU-efficient, compared to calling a kernel on the multiple inputs at once.   
As explained in the next section, the batching is a must to leverage maximum throughput on the GPUs.

There are two primary ways of using the batching to help application performance:
* Collecting the inputs explicitly on the application side and then _sending these batched requests to the OpenVINO_
   * Although this gives flexibility with the possible batching strategies, the approach requires redesigning the application logic
* _Sending individual requests_, while configuring the OpenVINO to collect and perform inference on the requests in batch [automatically](../OV_Runtime_UG/automatic_batching.md).
In both cases, optimal batch size is very device-specific. Also as explained below, the optimal batch size depends on the model, inference precision and other factors.

@anchor throughput_advanced
## Choosing the Number of Streams and/or Batch Size
Predicting the inference performance is difficult and finding optimal execution parameters requires direct experiments with measurements.
Run performance testing in the scope of development, and make sure to validate overall (end-to-end) application performance.

Different devices behave differently with the batch sizes. The optimal batch size depends on the model, inference precision and other factors.
Similarly, different devices require different number of execution streams to saturate.
Finally, in some cases  combination of streams and batching may be required to maximize the throughput.

One possible throughput optimization strategy is to **set an upper bound for latency and then increase the batch size and/or number of the streams until that tail latency is met (or the throughput is not growing anymore)**.
Also, consider [OpenVINO Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction) that builds handy latency vs throughput charts, iterating over possible values of the batch size and number of streams.

> **NOTE**: When playing with [dynamically-shaped inputs](../OV_Runtime_UG/ov_dynamic_shapes.md) use only the streams (no batching), as they tolerate individual requests having different shapes. 

> **NOTE**: Using the [High-Level Performance Hints](../OV_Runtime_UG/performance_hints.md) is the alternative,  portable and future-proof option, allowing the OpenVINO to find best combination of streams and batching for a given scenario and model. 

### Number of Streams Considerations
* Select the number of streams is it is **less or equal** to the number of requests that your application would be able to runs simultaneously
* To avoid wasting resources, the number of streams should be enough to meet the _average_ parallel slack rather than the peak load
* As a more portable option (that also respects the underlying hardware configuration) use the `ov::streams::AUTO`
* It is very important to keep these streams busy, by running as many inference requests as possible (e.g. start the newly-arrived inputs immediately)
   * Bare minimum of requests to saturate the device can be queried as `ov::optimal_number_of_infer_requests` of the  `ov:Compiled_Model` 
* _Maximum number of streams_ for the device (per model) can be queried as the `ov::range_for_streams`

### Batch Size Considerations
* Select the batch size that is **equal** to the number of requests that your application is able to runs simultaneously
   * Otherwise (or if the number of "available" requests fluctuates), you may need to keep several instances of the network (reshaped to the different batch size) and select the properly sized instance in the runtime accordingly 
* For OpenVINO devices that internally implement a dedicated heuristic, the `ov::optimal_batch_size` is a _device_ property (that accepts the actual model as a parameter) to query the recommended batch size for the model.


### Few Device Specific Details
* For the **GPU**:
   * When the parallel slack is small (e.g. only 2-4 requests executed simultaneously), then using only the streams for the GPU may suffice
      * Notice that the GPU runs 2 request per stream, so 4 requests can be served by 2 streams
      * Alternatively, consider single stream with with 2 requests (each with a small batch size like 2), which would total the same 4 inputs in flight
   * Typically, for 4 and more requests the batching delivers better throughput
   * Batch size can be calculated as "number of inference requests executed in parallel" divided by the "number of requests that the streams consume"
      * E.g. if you process 16 cameras (by 16 requests inferenced _simultaneously_) by the two GPU streams (each can process two requests), the batch size per request is 16/(2*2)=4 

* For the **CPU always use the streams first**
   * On the high-end CPUs, using moderate (2-8) batch size _in addition_ to the maximum number of streams, may further improve the performance.
