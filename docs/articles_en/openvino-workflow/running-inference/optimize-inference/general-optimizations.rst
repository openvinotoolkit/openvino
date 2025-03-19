General Optimizations
=====================


.. meta::
   :description: General optimizations include application-level optimization
                 methods that improve data pipelining, pre-processing
                 acceleration and both latency and throughput.


This article covers application-level optimization techniques, such as asynchronous execution, to improve data pipelining, pre-processing acceleration and so on.
While the techniques (e.g. pre-processing) can be specific to end-user applications, the associated performance improvements are general and shall improve any target scenario -- both latency and throughput.

.. _inputs_pre_processing:

Inputs Pre-Processing with OpenVINO
###################################

In many cases, a network expects a pre-processed image. It is advised not to perform any unnecessary steps in the code:

* Model conversion API can efficiently incorporate the mean and normalization (scale) values into a model (for example, to the weights of the first convolution). For more details, see the :doc:`relevant model conversion API command-line parameters <../../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/preprocessing-api-details>`.
* Let OpenVINO accelerate other means of :doc:`Image Pre-processing and Conversion <optimize-preprocessing>`
* Data which is already in the "on-device" memory can be input directly by using the :doc:`remote tensors API of the GPU Plugin <../inference-devices-and-modes/gpu-device/remote-tensor-api-gpu-plugin>`.

.. _async_api:

Prefer OpenVINO Async API
#########################

The API of the inference requests offers Sync and Async execution. While the ``ov::InferRequest::infer()`` is inherently synchronous and executes immediately (effectively serializing the execution flow in the current application thread), the Async "splits" the ``infer()`` into ``ov::InferRequest::start_async()`` and ``ov::InferRequest::wait()``. For more information, see the :doc:`API examples <../inference-request>`.

A typical use case for the ``ov::InferRequest::infer()`` is running a dedicated application thread per source of inputs (e.g. a camera), so that every step (frame capture, processing, parsing the results, and associated logic) is kept serial within the thread.
In contrast, the ``ov::InferRequest::start_async()`` and ``ov::InferRequest::wait()`` allow the application to continue its activities and poll or wait for the inference completion when really needed. Therefore, one reason for using an asynchronous code is "efficiency".

.. note::

   Although the Synchronous API can be somewhat easier to start with, prefer to use the Asynchronous (callbacks-based, below) API in the production code. The reason is that it is the most general and scalable way to implement the flow control for any possible number of requests (and hence both latency and throughput scenarios).


The key advantage of the Async approach is that when a device is busy with the inference, the application can do other things in parallel (e.g. populating inputs or scheduling other requests) rather than wait for the current inference to complete first.

In the example below, inference is applied to the results of the video decoding. It is possible to keep two parallel infer requests, and while the current one is processed, the input frame for the next one is being captured. This essentially hides the latency of capturing, so that the overall frame rate is rather determined only by the slowest part of the pipeline (decoding vs inference) and not by the sum of the stages.

.. image:: ../../../assets/images/synch-vs-asynch.svg
   :alt: Intel® VTune™ screenshot

Below are example-codes for the regular and async-based approaches to compare:

* Normally, the frame is captured with OpenCV and then immediately processed:

  .. doxygensnippet:: docs/articles_en/assets/snippets/dldt_optimization_guide8.cpp
     :language: cpp
     :fragment: [part8]

* In the "true" async mode, the ``NEXT`` request is populated in the main (application) thread, while the ``CURRENT`` request is processed:

  .. doxygensnippet:: docs/articles_en/assets/snippets/dldt_optimization_guide9.cpp
     :language: cpp
     :fragment: [part9]


The technique can be generalized to any available parallel slack. For example, you can do inference and simultaneously encode the resulting or previous frames or run further inference, like emotion detection on top of the face detection results.
Refer to the :doc:`Benchmark App Sample <../../../get-started/learn-openvino/openvino-samples/benchmark-tool>` for complete examples of the Async API in action.

.. note::

   Using the Asynchronous API is a must for :doc:`throughput-oriented scenarios <optimizing-throughput>`.

Notes on Callbacks
++++++++++++++++++++

Keep in mind that the ``ov::InferRequest::wait()`` of the Async API waits for the specific request only. However, running multiple inference requests in parallel provides no guarantees on the completion order. This may complicate a possible logic based on the ``ov::InferRequest::wait``. The most scalable approach is using callbacks (set via the ``ov::InferRequest::set_callback``) that are executed upon completion of the request. The callback functions will be used by OpenVINO Runtime to notify you of the results (or errors).
This is a more event-driven approach.

A few important points on the callbacks:

* It is the job of the application to ensure that any callback function is thread-safe.
* Although executed asynchronously by a dedicated threads, the callbacks should NOT include heavy operations (e.g. I/O) and/or blocking calls. Work done by any callback should be kept to a minimum.

.. _tensor_idiom:

The "get_tensor" Idiom
######################

Each device within OpenVINO may have different internal requirements on the memory padding, alignment, etc., for intermediate tensors. The **input/output tensors** are also accessible by the application code.
As every ``ov::InferRequest`` is created by the particular instance of the ``ov::CompiledModel`` (that is already device-specific) the requirements are respected and the input/output tensors of the requests are still device-friendly.
To sum it up:

* The ``get_tensor`` (that offers the ``data()`` method to get a system-memory pointer to the content of a tensor), is a recommended way to populate the inference inputs (and read back the outputs) **from/to the host memory**:

  * For example, for the GPU device, the **input/output tensors** are mapped to the host (which is fast) only when the ``get_tensor`` is used, while for the ``set_tensor`` a copy into the internal GPU structures may happen.

* In contrast, when the input tensors are already in the **on-device memory** (e.g. as a result of the video-decoding), prefer the ``set_tensor`` as a zero-copy way to proceed. For more details, see the :doc:`GPU device Remote tensors API <../inference-devices-and-modes/gpu-device/remote-tensor-api-gpu-plugin>`.


Consider the :ref:`API examples <in_out_tensors>` for the ``get_tensor`` and ``set_tensor``.

