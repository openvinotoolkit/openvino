# Asynchronous Inference Request {#async_infer_request}

Asynchronous Inference Request runs an inference pipeline asynchronously in one or several task executors depending on a device pipeline structure.
Inference Engine Plugin API provides the base InferenceEngine::AsyncInferRequestThreadSafeDefault class:

- The class has the `_pipeline` field of `std::vector<std::pair<ITaskExecutor::Ptr, Task> >`, which contains pairs of an executor and executed task.
- All executors are passed as arguments to a class constructor and they are in the running state and ready to run tasks.
- The class has the InferenceEngine::AsyncInferRequestThreadSafeDefault::StopAndWait method, which waits for `_pipeline` to finish in a class destructor. The method does not stop task executors and they are still in the running stage, because they belong to the executable network instance and are not destroyed.

`AsyncInferRequest` Class
------------------------

Inference Engine Plugin API provides the base InferenceEngine::AsyncInferRequestThreadSafeDefault class for a custom asynchronous inference request implementation:

@snippet src/template_async_infer_request.hpp async_infer_request:header

#### Class Fields

- `_inferRequest` - a reference to the [synchronous inference request](@ref infer_request) implementation. Its methods are reused in the `AsyncInferRequest` constructor to define a device pipeline.
- `_waitExecutor` - a task executor that waits for a response from a device about device tasks completion

> **NOTE**: If a plugin can work with several instances of a device, `_waitExecutor` must be device-specific. Otherwise, having a single task executor for several devices does not allow them to work in parallel.

### `AsyncInferRequest()`

The main goal of the `AsyncInferRequest` constructor is to define a device pipeline `_pipeline`. The example below demonstrates `_pipeline` creation with the following stages:

- `inferPreprocess` is a CPU compute task.
- `startPipeline` is a CPU ligthweight task to submit tasks to a remote device.
- `waitPipeline` is a CPU non-compute task that waits for a response from a remote device.
- `inferPostprocess` is a CPU compute task.

@snippet src/template_async_infer_request.cpp async_infer_request:ctor

The stages are distributed among two task executors in the following way:

- `inferPreprocess` and `startPipeline` are combined into a single task and run on `_requestExecutor`, which computes CPU tasks.
- You need at least two executors to overlap compute tasks of a CPU and a remote device the plugin works with. Otherwise, CPU and device tasks are executed serially one by one.
- `waitPipeline` is sent to `_waitExecutor`, which works with the device.

> **NOTE**: `callbackExecutor` is also passed to the constructor and it is used in the base InferenceEngine::AsyncInferRequestThreadSafeDefault class, which adds a pair of `callbackExecutor` and a callback function set by the user to the end of the pipeline.

Inference request stages are also profiled using IE_PROFILING_AUTO_SCOPE, which shows how pipelines of multiple asynchronous inference requests are run in parallel via the [Intel® VTune™ Profiler](https://software.intel.com/en-us/vtune) tool.

### `~AsyncInferRequest()`

In the asynchronous request destructor, it is necessary to wait for a pipeline to finish. It can be done using the InferenceEngine::AsyncInferRequestThreadSafeDefault::StopAndWait method of the base class.

@snippet src/template_async_infer_request.cpp async_infer_request:dtor
