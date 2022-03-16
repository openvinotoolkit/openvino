# Optimizing for Throughput {#openvino_docs_deployment_optimization_guide_tput}

### General Throughput Considerations
As described in the section on the [latency-specific considerations](./dldt_deployment_optimization_latency.md) one possible use-case is focused on delivering the every single request at the minimal delay.
Throughput on the other hand, is about inference scenarios in which potentially large number of inference requests are served simultaneously.
Here, the overall application throughput can be significantly improved  with the right performance configuration.
Also, if the model is not already compute- or memory bandwidth-limited, the associated increase in latency is not linearly dependent on the number of requests executed in parallel.

With the OpenVINO there two major means of running the multiple requests simultaneously: batching and "streams", explained in this document. 
Yet, different GPUs behave differently with batch sizes, just like different CPUs require different number of execution streams to maximize the throughput.
Predicting inference performance is difficult and and finding optimal execution parameters require direct experiments measurements.
One possible throughput optimization strategy is to set an upper bound for latency and then increase the batch size or number of the streams until that tail latency is met (or the throughput is not growing anymore).
Also, consider [Deep Learning Workbench](https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Introduction.html).

Finally, the [automatic multi-device execution](../OV_Runtime_UG/multi_device.md) helps to improve the throughput, please also see the section below. 
While earlier approach of optimizing the parameters of each device separately does work, the resulting multi-device performance is a fraction of the “ideal” (plain sum) performance that is different for different topologies. 

Overall, the latency-throughput is not linearly dependent and very _device_ specific. It is also tightly integrated with _model_ characteristics.
As for the possible inference devices the scenery had already become pretty diverse, the OpenVINO has introduced the dedicated notion of the high-level performance configuration "hints" to describe the target application scenarios.
The hints are described [next](./dldt_deployment_optimization_hints.md). 

The rest of the document provides low-level details on the OpenVINO's low-level ways to optimize the throughput.

## Low-Level Implementation Details
### OpenVINO Streams <a name="ov-streams"></a>
As detailed in the section <a href="#ov-async-api">OpenVINO Async API</a> running multiple inference requests asynchronously is important for general application efficiency.
Additionally, most devices support running multiple inference requests in parallel in order to improve the device utilization. The _level_ of the parallelism (i.e. how many requests are really executed in parallel on the device) is commonly referred as a number of 'streams'. Some devices run several requests per stream to amortize the host-side costs.
Notice that streams (that can be considered as independent queues) are really executing the requests in parallel, but not in the lock step (as e.g. the batching does). 

Also, notice that for efficient asynchronous execution, the streams are actually handling inference with special pool of the threads.
So each time you start inference requests (potentially from different application threads), they are actually muxed into a inference queue of the particular `ov:compiled_model`. 
If there is a vacant stream, it pops the request from the queue and actually expedites that to the on-device execution.

The usage of multiple streams is an inherently throughput-oriented approach, as every stream requires a dedicated memory to operate in parallel to the rest streams (read-only data like weights are usually shared between all streams).
Also, the streams inflate the load/compilation time.
This is why the [latency hint](./dldt_deployment_optimization_hints.md) governs a device to create a bare minimum of streams (usually just one).

Finally, the streams are always preferable compared to creating  multiple instances of the same model, as weights memory is shared across streams, reducing possible  memory consumption.

### Throughput on the CPU: Internals <a name="cpu-streams"></a>
In order to best serve multiple inference requests simultaneously, the inference threads are grouped/pinned to the particular CPU cores, constituting the CPU streams.
This provides much better performance for the networks than batching especially for the many-core machines:
![](../img/cpu_streams_explained_1.png)

Compared with the batching, the parallelism is somewhat transposed (i.e. performed over inputs, with much less synchronization within CNN ops):
![](../img/cpu_streams_explained.png)

Notice that [high-level performance hints](../OV_Runtime_UG/performance_hints.md) allows the implementation to select the optimal number of the streams, _depending on the model compute demands_ and CPU capabilities (including [int8 inference](../OV_Runtime_UG/Int8Inference.md) hardware acceleration, number of cores, etc).

### Automatic Batching Internals <a name="ov-auto-batching"></a>
While the GPU plugin fully supports general notion of the streams, the associated performance (throughput) improvements are usually modest.
The primary reason is that, while the streams allow to hide the communication overheads and hide certain bubbles in device utilization, running multiple OpenCL kernels on the GPU simultaneously is less efficient, compared to calling a kernel on the multiple inputs at once.   

When the parallel slack is small (e.g. only 2-4 requests executed simultaneously), then using the streams for the GPU may suffice. 
Typically, for 4 and more requests the batching delivers better throughput for the GPUs. Using the [High-Level Performance Hints](../OV_Runtime_UG/performance_hints.md) is the most portable and future-proof option, allowing the OpenVINO to find best combination of streams and batching for a given scenario. 
As explained in the section on the [automatic batching](../OV_Runtime_UG/automatic_batching.md), the feature performs on-the-fly grouping of the inference requests to improve device utilization.
The Automatic Batching relaxes the requirement for an application to saturate devices like GPU by _explicitly_ using a large batch. It essentially it performs transparent inputs gathering from 
individual inference requests followed by the actual batched execution, with no programming effort from the user:
![](../img/BATCH_device.PNG)

Essentially, the Automatic Batching shifts the asynchronousity from the individual requests to the groups of requests that constitute the batches. Thus, for the execution to be efficient it is very important that the requests arrive timely, without causing a batching timeout. 
Normally, the timeout should never be hit. It is rather a graceful way to handle the application exit (when the inputs are not arriving anymore, so the full batch is not possible to collect).

So if your workload experiences the timeouts (resulting in the performance drop, as the timeout value adds itself to the latency of every request), consider balancing the timeout value vs the batch size. For example in many cases having smaller timeout value/batch size may yield better performance than large batch size, but coupled with the timeout value that is cannot guarantee accommodating the full number of the required requests.

Finally, as explained in the "get_tensor idiom" section below the Automatic Batching saves on inputs/outputs copies when the application code prefers the "get" versions of the tensor data access APIs. 
