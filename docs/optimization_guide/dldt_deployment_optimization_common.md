# General Optimizations {#openvino_docs_deployment_optimization_guide_common}

## Inputs Pre-processing with OpenVINO

In many cases, a network expects a pre-processed image, so make sure you do not perform unnecessary steps in your code:
- Model Optimizer can efficiently bake the mean and normalization (scale) values into the model (for example, to the weights of the first convolution). Please see [relevant Model Optimizer command-line options](../MO_DG/prepare_model/Additional_Optimizations.md).
- Let the OpenVINO accelerate other means of [Image Pre-processing and Conversion](../OV_Runtime_UG/preprocessing_overview.md).
- Note that in many cases, you can directly share the (input) data with the OpenVINO, for example consider [remote tensors API of the GPU Plugin](../OV_Runtime_UG//supported_plugins/GPU_RemoteTensor_API.md).

## Prefer OpenVINO Async API <a name="ov-async-api"></a>
The API of the inference requests offers Sync and Async execution. While the `ov::InferRequest::infer()` is inherently synchronous and executes immediately (effectively serializing the execution flow in the current application thread), the Async "splits" the `infer()` into `ov::InferRequest::start_async()` and `ov::InferRequest::wait()`. Please consider the [API examples](../OV_Runtime_UG/ov_infer_request.md).

A typical use-case for the `ov::InferRequest::infer()` is running a dedicated application thread per source of inputs (e.g. a camera), so that every step (frame capture, processing, results parsing and associated logic) is kept serial within the thread.
In contrast, the `ov::InferRequest::start_async()` and `ov::InferRequest::wait()` allow the application to continue its activities and poll or wait for the inference completion when really needed. So one reason for using asynchronous code is _efficiency_.

> **NOTE**: Although the Synchronous API can be somewhat easier to start with, in the production code always prefer to use the Asynchronous (callbacks-based, below) API, as it is the most general and scalable way to implement the flow control for any possible number of requests (and hence both latency and throughput scenarios).

Let's see how the OpenVINO Async API can improve overall throughput rate of the application. The key advantage of the Async approach is as follows:  while a device is busy with the inference, the application can do other things in parallel (e.g. populating inputs or scheduling other requests) rather than wait for the inference to complete.

In the example below, inference is applied to the results of the video decoding. So it is possible to keep two parallel infer requests, and while the current is processed, the input frame for the next is being captured. This essentially hides the latency of capturing, so that the overall frame rate is rather determined only by the slowest part of the pipeline (decoding IR inference) and not by the sum of the stages.

You can compare the pseudo-codes for the regular and async-based approaches:

-	In the regular way, the frame is captured with OpenCV and then immediately processed:<br>

@snippet snippets/dldt_optimization_guide8.cpp part8

![Intel&reg; VTune&trade; screenshot](../img/vtune_regular.png)

-	In the "true" async mode, the `NEXT` request is populated in the main (application) thread, while the `CURRENT` request is processed:<br>

@snippet snippets/dldt_optimization_guide9.cpp part9

![Intel&reg; VTune&trade; screenshot](../img/vtune_async.png)

The technique can be generalized to any available parallel slack. For example, you can do inference and simultaneously encode the resulting or previous frames or run further inference, like emotion detection on top of the face detection results.
Refer to the [Object Detection С++ Demo](@ref omz_demos_object_detection_demo_cpp), [Object Detection Python Demo](@ref omz_demos_object_detection_demo_python)(latency-oriented Async API showcase) and [Benchmark App Sample](../../samples/cpp/benchmark_app/README.md) for complete examples of the Async API in action.

### Notes on Callbacks
Notice that the Async's `ov::InferRequest::wait()` waits for the specific request only. However, running multiple inference requests in parallel provides no guarantees on the completion order. This may complicate a possible logic based on the `ov::InferRequest::wait`. The most scalable approach is using callbacks (set via the `ov::InferRequest::set_callback`) that are executed upon completion of the request. The callback functions will be used by the OpenVINO runtime to notify on the results (or errors. 
This is more event-driven approach.

Few important points on the callbacks:
- It is the application responsibility to ensure that any callback function is thread-safe
- Although executed asynchronously by a dedicated threads the callbacks should NOT include heavy operations (e.g. I/O) and/or blocking calls. Keep the work done by any callback to a minimum.

## "get_tensor" Idiom <a name="new-request-based-api"></a>

`get_tensor` is a recommended way to populate the inference inputs (and read back the outputs), as it internally allocates the data with right padding/alignment for the device. For example, the GPU inputs/outputs tensors are mapped to the host (which is fast) only when the `get_tensor` is used, while for the `set_tensor` a copy into the internal GPU structures may happen.
Please consider the [API examples](../OV_Runtime_UG/ov_infer_request.md).
In contrast, the `set_tensor` is a preferable way to handle remote tensors, [for example with the GPU device](../OV_Runtime_UG//supported_plugins/GPU_RemoteTensor_API.md).
