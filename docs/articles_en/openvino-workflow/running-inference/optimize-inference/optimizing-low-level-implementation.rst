Further Low-Level Implementation Details
========================================


.. meta::
   :description: Automatic Batching moves asynchronicity from individual
                 requests to groups of requests, and the CPU streams are
                 inference threads grouped by CPU cores.


Throughput on the CPU: Internals
################################

As explained in the :doc:`throughput-related section <optimizing-throughput>`, the OpenVINO streams are means of running multiple requests in parallel.
In order to best serve multiple inference requests executed simultaneously, the inference threads are grouped/pinned to the particular CPU cores, constituting the "CPU" streams.
This provides much better performance for the networks than batching, especially for the multiple-core systems:

.. list-table::
   :header-rows: 1

   * - Conventional Approach
     - Streams
   * - | Every CNN op is internally parallelized over a full number of CPU cores and it is detrimental for non-scalable ops.
       | A lot of synchronization between many threads results in overhead.
       | An only option to improve efficiency is batching.
     - | CPU cores are evenly distributed between execution streams (each 1-4 threads).
       | Less threads per stream means less synchronization, better locality, and finer granularity.
   * - |conventional-approach|
     - | |execution-streams|
       | Requests are executed in parallel with a small number of threads.
       | Layer-wise, the streams imply much less synchronization.

.. |conventional-approach| image:: ../../../assets/images/cpu_execution_conventional_approach.svg

.. |execution-streams| image:: ../../../assets/images/cpu_execution_streams.svg

Compared to the batching, the parallelism is somewhat transposed (performed over inputs with much less synchronization within CNN ops):

.. list-table::
   :header-rows: 1

   * - Large Batch Approach
     - Streams
   * - | All threads process all inputs at once.
       | Assumes all layers are parallelized well.
       | “Fat” requests are executed one by one.
     - | CPU cores are evenly distributed between execution streams.
       | “Parallelize the outermost loop” rule of thumb.
       | Individual requests are executed in parallel.
   * - |large-batch-approach|
     - | |execution-streams-2|
       | Inputs-wise the streams are the “transposed” batch.

.. |large-batch-approach| image:: ../../../assets/images/large_batch_approach.svg

.. |execution-streams-2| image:: ../../../assets/images/cpu_execution_streams_2.svg


Keep in mind that :doc:`high-level performance hints <high-level-performance-hints>` allow the implementation to select the optimal number of streams depending on model's compute demands and CPU capabilities, including :doc:`int8 inference <../../model-optimization>` hardware acceleration, number of cores, etc.

Automatic Batching Internals
############################

:doc:`Automatic batching <../inference-devices-and-modes/automatic-batching>` performs on-the-fly grouping of inference requests to improve device utilization.
It relaxes the requirement for an application to saturate devices such as GPU by using a large batch "explicitly". It performs transparent input gathering from individual inference requests followed by the actual batched execution, with no programming effort from the user:

.. image:: ../../../assets/images/batch_device.svg

Essentially, Automatic Batching shifts asynchronicity from individual requests to groups of requests that constitute the batches. Furthermore, for the execution to be efficient, it is very important that the requests arrive timely, without causing a batching timeout.
Normally, the timeout should never be hit. It is rather a graceful way to handle the application exit (when the inputs are not arriving anymore, so the full batch is not possible to collect).

If a workload experiences timeouts, which lead to a drop in performance due to increased latency of every request, consider balancing its value against the batch size. For example, a smaller batch size and timeout value may yield better results than a large batch size coupled with a timeout value that cannot guarantee accommodating all the required requests.

Finally, following the ``get_tensor`` idiom section from the :doc:`general optimizations <general-optimizations>` helps Automatic Batching to save on inputs/outputs copies. According to that, you should always prefer the "get" versions of the tensors' data access APIs in your applications.

