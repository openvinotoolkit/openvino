Advanced Throughput Options: Streams and Batching
=======================================================


.. meta::
   :description: With OpenVINO streams a device may handle processing multiple
                 inference requests and the batching helps to saturate the
                 device and leads to higher throughput.


OpenVINO Streams
####################

As explained in the :doc:`common-optimizations section <../general-optimizations>`, running multiple inference requests asynchronously is important for general application efficiency.
Internally, every device implements a queue, which acts as a buffer, storing the inference requests until retrieved by the device at its own pace.
The devices may actually process multiple inference requests in parallel in order to improve the device utilization and overall throughput.
This configurable method of this device-side parallelism is commonly referred as **streams**.

.. note::

   Be aware that streams are **really executing the requests in parallel, but not in the lock step** (as the batching does), which makes the streams fully compatible with :doc:`dynamically-shaped inputs <../../model-input-output/dynamic-shapes>`, while individual requests can have different shapes.

.. note::

   Most OpenVINO devices (including CPU and GPU) support the streams, yet the *optimal* number of the streams is deduced very differently. More information on this topic can be found in the section `below <#number-of-streams-considerations>`__.

A few general considerations:

* Using the streams does increase the latency of an individual request:

  * When the number of streams is not specified, a device creates a bare minimum of streams (usually, just one), as the latency-oriented case is default.
  * See further tips for the optimal number of the streams `below <#choosing-the-number-of-streams-and-or-batch-size>`__.

* Streams are memory-intensive, as every stream duplicates the intermediate buffers to do inference in parallel to the rest of the streams:

  * Always prefer streams over creating multiple ``ov:Compiled_Model`` instances for the same model, as weights memory is shared across streams, reducing the memory consumption.

* Keep in mind that the streams also inflate the model load (compilation) time.

For efficient asynchronous execution, the streams are actually handling the inference with a special pool of the threads (a thread per stream).
Each time you start inference requests (potentially from different application threads), they are actually muxed into an inference queue of the particular ``ov:Compiled_Model``.
If there is a vacant stream, it pulls the request from the queue and actually expedites that to the on-device execution.
There are further device-specific details, like for the CPU, in the :doc:`internals <../optimizing-low-level-implementation>` section.

Batching
####################

Hardware accelerators such as GPUs are optimized for a massive compute parallelism, so the batching helps to saturate the device and leads to higher throughput.
While the streams (described in previous section) already help to hide the communication overheads and certain bubbles in the scheduling, running multiple OpenCL kernels simultaneously is less GPU-efficient compared to calling a kernel on the multiple inputs at once.
As explained in the next section, the batching is a must to leverage maximum throughput on the GPU.

There are several primary methods of using the batching to help application performance:

* Collecting the inputs explicitly on the application side and then **sending the batch requests to OpenVINO**:

  * Although this gives flexibility with the possible batching strategies, the approach requires redesigning the application logic.

* **Sending individual requests**, while configuring OpenVINO to collect and perform inference on the requests in batch :doc:`automatically <../../inference-devices-and-modes/automatic-batching>`.

In both cases, the optimal batch size is very device-specific. As explained below, the optimal batch size also depends on the model, inference precision and other factors.


Choosing the Number of Streams and/or Batch Size
################################################

Predicting the inference performance is difficult and finding optimal execution parameters requires direct experiments with measurements.
Run performance testing in the scope of development, and make sure to validate overall (*end-to-end*) application performance.

Different devices behave differently with the batch sizes. The optimal batch size depends on the model, inference precision and other factors.
Similarly, different devices require a different number of execution streams to saturate.
In some cases, combination of streams and batching may be required to maximize the throughput.

One possible throughput optimization strategy is to **set an upper bound for latency and then increase the batch size and/or number of the streams until that tail latency is met (or the throughput is not growing anymore)**.

.. note::

   When playing with :doc:`dynamically-shaped inputs <../../model-input-output/dynamic-shapes>`, use only the streams (no batching), as they tolerate individual requests having different shapes.

.. note::

   Using the :doc:`High-Level Performance Hints <../high-level-performance-hints>` is the alternative, portable and future-proof option, allowing OpenVINO to find the best combination of streams and batching for a given scenario and a model.

Number of Streams Considerations
++++++++++++++++++++++++++++++++

* Select the number of streams that is **less or equal** to the number of requests that the application would be able to run simultaneously.
* To avoid wasting resources, the number of streams should be enough to meet the *average* parallel slack rather than the peak load.
* Use the `ov::streams::AUTO <../../../../api/c_cpp_api/group__ov__runtime__cpp__prop__api.html#_CPPv44AUTO>`__ as a more portable option (that also respects the underlying hardware configuration).
* It is very important to keep these streams busy, by running as many inference requests as possible (for example, start the newly-arrived inputs immediately):

  * A bare minimum of requests to saturate the device can be queried as the `ov::optimal_number_of_infer_requests <../../../../api/c_cpp_api/group__ov__runtime__cpp__prop__api.html#_CPPv432optimal_number_of_infer_requests>`__ of the  ``ov:Compiled_Model``.

* *The maximum number of streams* for the device (per model) can be queried as the `ov::range_for_streams <../../../../api/c_cpp_api/group__ov__runtime__cpp__prop__api.html#_CPPv417range_for_streams>`__.

Batch Size Considerations
+++++++++++++++++++++++++

* Select the batch size that is **equal** to the number of requests that your application is able to run simultaneously:

  * Otherwise (or if the number of "available" requests fluctuates), you may need to keep several instances of the network (reshaped to the different batch size) and select the properly sized instance in the runtime accordingly.

* For OpenVINO devices that implement a dedicated heuristic internally, the `ov::optimal_batch_size <../../../../api/c_cpp_api/group__ov__runtime__cpp__prop__api.html#_CPPv418optimal_batch_size>`__ is a *device* property (that accepts the actual model as a parameter) to query the recommended batch size for the model.


A Few Device-specific Details
+++++++++++++++++++++++++++++

* For the **GPU**:

  * When the parallel slack is small, for example, only 2-4 requests executed simultaneously, then using only the streams for the GPU may suffice:

    * The GPU runs 2 requests per stream, so 4 requests can be served by 2 streams.
    * Alternatively, consider a single stream with 2 requests (each with a small batch size like 2), which would total the same 4 inputs in flight.

  * Typically, for 4 and more requests the batching delivers better throughput.
  * A batch size can be calculated as "a number of inference requests executed in parallel" divided by the "number of requests that the streams consume":

    * For example, if you process 16 cameras (by 16 requests inferenced *simultaneously*) by 2 GPU streams (each can process two requests), the batch size per request is 16/(2*2)=4.

* For the **CPU, always use the streams first!**:

  * On high-end CPUs, using moderate (2-8) batch size *in addition* to the maximum number of streams may further improve the performance.


