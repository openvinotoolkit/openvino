---
sidebar_position: 4
sidebar_label: "Asynchronous Inference Request"
description: "Use the base ov::IAsyncInferRequest class to implement a custom asynchronous inference request in OpenVINO."
---

# Asynchronous Inference Request

Asynchronous Inference Request runs an inference pipeline asynchronously in one or several task executors depending on a device pipeline structure.
OpenVINO Runtime Plugin API provides the base ov::IAsyncInferRequest class:

- The class has the `m_pipeline` field of `std::vector<std::pair<std::shared_ptr<ov::threading::ITaskExecutor>, ov::threading::Task> >`, which contains pairs of an executor and executed task.
- All executors are passed as arguments to a class constructor and they are in the running state and ready to run tasks.
- The class has the ov::IAsyncInferRequest::stop\_and\_wait method, which waits for `m_pipeline` to finish in a class destructor. The method does not stop task executors and they are still in the running stage, because they belong to the compiled model instance and are not destroyed.

## AsyncInferRequest Class

OpenVINO Runtime Plugin API provides the base ov::IAsyncInferRequest class for a custom asynchronous inference request implementation:

```cpp
class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    AsyncInferRequest(const std::shared_ptr<InferRequest>& request,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);

    ~AsyncInferRequest();
    void cancel() override;

private:
    std::function<void()> m_cancel_callback;
    std::shared_ptr<ov::threading::ITaskExecutor> m_wait_executor;
};
```

### Class Fields

- `m_cancel_callback` - a callback which allows to interrupt the execution
- `m_wait_executor` - a task executor that waits for a response from a device about device tasks completion

:::note
If a plugin can work with several instances of a device, `m_wait_executor` must be device-specific. Otherwise, having a single task executor for several devices does not allow them to work in parallel.
:::

### AsyncInferRequest()

The main goal of the `AsyncInferRequest` constructor is to define a device pipeline `m_pipeline`. The example below demonstrates `m_pipeline` creation with the following stages:

- `infer_preprocess_and_start_pipeline` is a CPU lightweight task to submit tasks to a remote device.
- `wait_pipeline` is a CPU non-compute task that waits for a response from a remote device.
- `infer_postprocess` is a CPU compute task.

```cpp
ov::template_plugin::AsyncInferRequest::AsyncInferRequest(
    const std::shared_ptr<ov::template_plugin::InferRequest>& request,
    const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
    const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
    const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ov::IAsyncInferRequest(request, task_executor, callback_executor),
      m_wait_executor(wait_executor) {
    // In current implementation we have CPU only tasks and no needs in 2 executors
    // So, by default single stage pipeline is created.
    // This stage executes InferRequest::infer() using cpuTaskExecutor.
    // But if remote asynchronous device is used the pipeline can by splitted tasks that are executed by cpuTaskExecutor
    // and waiting tasks. Waiting tasks can lock execution thread so they use separate threads from other executor.
    constexpr const auto remoteDevice = false;

    m_cancel_callback = [request] {
        request->cancel();
    };
    if (remoteDevice) {
        m_pipeline = {{task_executor,
                       [this, request] {
                           OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin,
                                              "TemplatePlugin::AsyncInferRequest::infer_preprocess_and_start_pipeline");
                           request->infer_preprocess();
                           request->start_pipeline();
                       }},
                      {m_wait_executor,
                       [this, request] {
                           OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin,
                                              "TemplatePlugin::AsyncInferRequest::wait_pipeline");
                           request->wait_pipeline();
                       }},
                      {task_executor, [this, request] {
                           OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin,
                                              "TemplatePlugin::AsyncInferRequest::infer_postprocess");
                           request->infer_postprocess();
                       }}};
    }
}
```

The stages are distributed among two task executors in the following way:

- `infer_preprocess_and_start_pipeline` prepare input tensors and run on `m_request_executor`, which computes CPU tasks.
- You need at least two executors to overlap compute tasks of a CPU and a remote device the plugin works with. Otherwise, CPU and device tasks are executed serially one by one.
- `wait_pipeline` is sent to `m_wait_executor`, which works with the device.

:::note
`m_callback_executor` is also passed to the constructor and it is used in the base ov::IAsyncInferRequest class, which adds a pair of `callback_executor` and a callback function set by the user to the end of the pipeline.
:::

### ~AsyncInferRequest()

In the asynchronous request destructor, it is necessary to wait for a pipeline to finish. It can be done using the ov::IAsyncInferRequest::stop\_and\_wait method of the base class.

```cpp
ov::template_plugin::AsyncInferRequest::~AsyncInferRequest() {
    ov::IAsyncInferRequest::stop_and_wait();
}
```

### cancel()

The method allows to cancel the infer request execution:

```cpp
void ov::template_plugin::AsyncInferRequest::cancel() {
    ov::IAsyncInferRequest::cancel();
    m_cancel_callback();
}
```
