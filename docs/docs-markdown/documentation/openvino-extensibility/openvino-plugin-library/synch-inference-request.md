---
sidebar_position: 3
sidebar_label: "Synchronous Inference Request"
description: "Use the ov::ISyncInferRequest interface as the base class to implement a synchronous inference request in OpenVINO."
---

# Synchronous Inference Request

`InferRequest` class functionality:

- Allocate input and output tensors needed for a backend-dependent network inference.
- Define functions for inference process stages (for example, `preprocess`, `upload`, `infer`, `download`, `postprocess`). These functions can later be used to define an execution pipeline during [Asynchronous Inference Request](asynch-inference-request.md) implementation.
- Call inference stages one by one synchronously.

## InferRequest Class

OpenVINO Plugin API provides the interface ov::ISyncInferRequest which should be
used as a base class for a synchronous inference request implementation. Based of that, a declaration
of a synchronous request class can look as follows:

```cpp
class InferRequest : public ov::ISyncInferRequest {
public:
    explicit InferRequest(const std::shared_ptr<const ov::template_plugin::CompiledModel>& compiled_model);
    ~InferRequest();

    void infer() override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    // pipeline methods-stages which are used in async infer request implementation and assigned to particular executor
    void infer_preprocess();
    void start_pipeline();
    void wait_pipeline();
    void infer_postprocess();
    void cancel();

    void set_tensors_impl(const ov::Output<const ov::Node> port,
                          const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

private:
    std::shared_ptr<const CompiledModel> get_template_model() const;

    enum { Preprocess, Postprocess, StartPipeline, WaitPipeline, numOfStages };

    std::array<openvino::itt::handle_t, numOfStages> m_profiling_task;
    // for performance counters
    std::array<std::chrono::duration<float, std::micro>, numOfStages> m_durations;

    std::vector<ov::Tensor> m_backend_input_tensors;
    std::vector<ov::Tensor> m_backend_output_tensors;
    std::shared_ptr<ov::runtime::Executable> m_executable;
    ov::EvaluationContext m_eval_context;
    std::vector<ov::SoPtr<ov::IVariableState>> m_variable_states;
};
```

### Class Fields

The example class has several fields:

- `m_profiling_task` - array of the `std::array<openvino::itt::handle_t, numOfStages>` type. Defines names for pipeline stages. Used to profile an inference pipeline execution with the Intel® instrumentation and tracing technology (ITT).
- `m_durations` - array of durations of each pipeline stage.
- backend-specific fields:
  
  - `m_backend_input_tensors` - input backend tensors.
  - `m_backend_output_tensors` - output backend tensors.
  - `m_executable` - an executable object / backend computational graph.
  - `m_eval_context` - an evaluation context to save backend states after the inference.
  - `m_variable_states` - a vector of variable states.

### InferRequest Constructor

The constructor initializes helper fields and calls methods which allocate tensors:

```cpp
ov::template_plugin::InferRequest::InferRequest(const std::shared_ptr<const ov::template_plugin::CompiledModel>& model)
    : ov::ISyncInferRequest(model) {
    // TODO: allocate infer request device and host buffers if needed, fill actual list of profiling tasks

    auto requestID = std::to_string(get_template_model()->m_request_id.fetch_add(1));

    std::string name = get_template_model()->m_model->get_friendly_name() + "_Req" + requestID;
    m_profiling_task = {
        openvino::itt::handle("Template" + std::to_string(get_template_model()->m_cfg.device_id) + "_" + name +
                              "_Preprocess"),
        openvino::itt::handle("Template" + std::to_string(get_template_model()->m_cfg.device_id) + "_" + name +
                              "_Postprocess"),
        openvino::itt::handle("Template" + std::to_string(get_template_model()->m_cfg.device_id) + "_" + name +
                              "_StartPipeline"),
        openvino::itt::handle("Template" + std::to_string(get_template_model()->m_cfg.device_id) + "_" + name +
                              "_WaitPipline"),
    };
    m_durations = {};
    m_executable = get_template_model()->get_template_plugin()->m_backend->compile(get_template_model()->m_model);

    // Allocate plugin backend specific memory handles
    m_backend_input_tensors.resize(get_inputs().size());
    m_backend_output_tensors.resize(get_outputs().size());

    // Allocate input/output tensors
    for (const auto& input : get_inputs()) {
        allocate_tensor(input, [input](ov::SoPtr<ov::ITensor>& tensor) {
            // Can add a check to avoid double work in case of shared tensors
            allocate_tensor_impl(tensor,
                                 input.get_element_type(),
                                 input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape());
        });
    }
    for (const auto& output : get_outputs()) {
        allocate_tensor(output, [output](ov::SoPtr<ov::ITensor>& tensor) {
            // Can add a check to avoid double work in case of shared tensors
            allocate_tensor_impl(tensor,
                                 output.get_element_type(),
                                 output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
        });
    }

    // Save variable states
    ov::op::util::VariableContext variable_context;
    const auto& ov_model = m_executable->get_model();
    collect_variables(ov_model, variable_context, m_variable_states);
    m_eval_context.emplace("VariableContext", variable_context);
}
```

:::note
Use inputs/outputs information from the compiled model to understand shape and element type of tensors, which you can set with ov::InferRequest::set\_tensor and get with ov::InferRequest::get\_tensor. A plugin uses these hints to determine its internal layouts and element types for input and output tensors if needed.
:::

### ~InferRequest Destructor

Destructor can contain plugin specific logic to finish and destroy infer request.

```cpp
ov::template_plugin::InferRequest::~InferRequest() = default;
```

### set\_tensors\_impl()

The method allows to set batched tensors in case if the plugin supports it.

```cpp
void ov::template_plugin::InferRequest::set_tensors_impl(const ov::Output<const ov::Node> port,
                                                         const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    for (const auto& input : get_inputs()) {
        if (input == port) {
            m_batched_tensors[input.get_tensor_ptr()] = tensors;
            return;
        }
    }
    OPENVINO_THROW("Cannot find input tensors for port ", port);
}
```

### query\_state()

The method returns variable states from the model.

```cpp
std::vector<ov::SoPtr<ov::IVariableState>> ov::template_plugin::InferRequest::query_state() const {
    return m_variable_states;
}
```

### infer()

The method calls actual pipeline stages synchronously. Inside the method plugin should check input/output tensors, move external tensors to backend and run the inference.

```cpp
void ov::template_plugin::InferRequest::infer() {
    // TODO: fill with actual list of pipeline stages, which are executed synchronously for sync infer requests
    infer_preprocess();
    start_pipeline();
    wait_pipeline();  // does nothing in current implementation
    infer_postprocess();
}
```

#### 1. infer\_preprocess()

Below is the code of the `infer_preprocess()` method. The method checks user input/output tensors and demonstrates conversion from user tensor to backend specific representation:

```cpp
void ov::template_plugin::InferRequest::infer_preprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[Preprocess]);
    auto start = Time::now();
    convert_batched_tensors();
    check_tensors();

    // Allocate backend tensors
    OPENVINO_ASSERT(get_inputs().size() == m_backend_input_tensors.size());
    for (size_t i = 0; i < get_inputs().size(); i++) {
        auto tensor = get_tensor(get_inputs()[i]);
        if (std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr)) {
            auto vector_tensor = std::dynamic_pointer_cast<ov::template_plugin::VectorImpl>(tensor._ptr);
            OPENVINO_ASSERT(vector_tensor, "Template plugin supports only VectorTensor with remote context.");
            auto element_type = vector_tensor->get_element_type();
            void* data = vector_tensor->get_data();
            OPENVINO_ASSERT(data != nullptr);
            // Create backend tenor
            m_backend_input_tensors[i] =
                get_template_model()->get_template_plugin()->m_backend->create_tensor(element_type,
                                                                                      vector_tensor->get_shape(),
                                                                                      data);
        } else if (tensor->is_continuous()) {
            // No ROI extraction is needed
            m_backend_input_tensors[i] =
                get_template_model()->get_template_plugin()->m_backend->create_tensor(tensor->get_element_type(),
                                                                                      tensor->get_shape(),
                                                                                      tensor->data());
        } else {
            OPENVINO_ASSERT(tensor->get_element_type().bitwidth() % 8 == 0,
                            "Template plugin: Unsupported ROI tensor with element type having ",
                            std::to_string(tensor->get_element_type().bitwidth()),
                            " bits size");
            // Perform manual extraction of ROI tensor
            // Basic implementation doesn't take axis order into account `desc.getBlockingDesc().getOrder()`
            // Performance of manual extraction is not optimal, but it is ok for template implementation
            m_backend_input_tensors[i] =
                get_template_model()->get_template_plugin()->m_backend->create_tensor(tensor->get_element_type(),
                                                                                      tensor->get_shape());
            auto backend_tensor_impl = ov::get_tensor_impl(m_backend_input_tensors[i]);
            OPENVINO_ASSERT(backend_tensor_impl, "Failed to create backend tensor");
            tensor->copy_to(backend_tensor_impl._ptr);
        }
    }
    // Tensors can be dynamic, so in this case we need to allocate tensors with right shape
    OPENVINO_ASSERT(get_outputs().size() == m_backend_output_tensors.size());
    for (size_t i = 0; i < get_outputs().size(); i++) {
        const auto& result = get_template_model()->m_model->get_results()[i];
        if (result->get_output_partial_shape(0).is_dynamic()) {
            m_backend_output_tensors[i] = get_template_model()->get_template_plugin()->m_backend->create_tensor();
            continue;
        }
        auto tensor = make_tensor(get_tensor(get_outputs()[i]));
        if (tensor.is_continuous() && !tensor.is<ov::RemoteTensor>())
            m_backend_output_tensors[i] =
                get_template_model()->get_template_plugin()->m_backend->create_tensor(tensor.get_element_type(),
                                                                                      tensor.get_shape(),
                                                                                      tensor.data());
        else
            m_backend_output_tensors[i] =
                get_template_model()->get_template_plugin()->m_backend->create_tensor(tensor.get_element_type(),
                                                                                      tensor.get_shape());
    }
    m_durations[Preprocess] = Time::now() - start;
}
```

#### 2. start\_pipeline()

Executes a pipeline synchronously using `m_executable` object:

```cpp
void ov::template_plugin::InferRequest::start_pipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[StartPipeline])
    auto start = Time::now();
    m_executable->call(m_backend_output_tensors,
                       m_backend_input_tensors,
                       m_eval_context,
                       get_template_model()->m_cfg.perf_count);
    m_durations[StartPipeline] = Time::now() - start;
}
```

#### 3. wait\_pipeline()

Waits a pipeline in case of plugin asynchronous execution:

```cpp
void ov::template_plugin::InferRequest::wait_pipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[WaitPipeline])
    auto start = Time::now();
    // TODO: Wait pipeline using driver API or other synchronizations methods
    // NOTE: not used in current implementation since `startPipeline` executes pipiline synchronously
    m_durations[WaitPipeline] = Time::now() - start;
}
```

#### 4. infer\_postprocess()

Converts backend specific tensors to tensors passed by user:

```cpp
void ov::template_plugin::InferRequest::infer_postprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[Postprocess]);
    auto start = Time::now();
    OPENVINO_ASSERT(get_outputs().size() == m_backend_output_tensors.size());
    for (size_t i = 0; i < get_outputs().size(); i++) {
        const auto& result = get_template_model()->m_model->get_results()[i];
        const auto& host_tensor = m_backend_output_tensors[i];
        auto tensor = get_tensor(get_outputs()[i]);
        // The outer tensor may have been allocated earlier from a still-dynamic shape, before this
        // plugin's post-transform model resolved it to static - so is_dynamic() alone can wrongly
        // skip the copy below. Also copy whenever the shapes don't already match.
        if (result->get_output_partial_shape(0).is_dynamic() || tensor->get_shape() != host_tensor.get_shape()) {
            ov::Output<const ov::Node> output{result->output(0).get_node(), result->output(0).get_index()};
            allocate_tensor(output, [&host_tensor](ov::SoPtr<ov::ITensor>& tensor) {
                allocate_tensor_impl(tensor, host_tensor.get_element_type(), host_tensor.get_shape());
                host_tensor.copy_to(ov::make_tensor(tensor));
            });
        } else if (!tensor->is_continuous()) {
            host_tensor.copy_to(ov::make_tensor(tensor));
        } else if (std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr)) {
            auto vector_tensor = std::dynamic_pointer_cast<ov::template_plugin::VectorImpl>(tensor._ptr);
            OPENVINO_ASSERT(vector_tensor, "Template plugin supports only VectorTensor with remote context.");
            void* data = vector_tensor->get_data();
            // Copy to vector
            std::memcpy(data, host_tensor.data(), tensor->get_byte_size());
        }
    }
    m_durations[Postprocess] = Time::now() - start;
}
```

### get\_profiling\_info()

The method returns the profiling info which was measured during pipeline stages execution:

```cpp
std::vector<ov::ProfilingInfo> ov::template_plugin::InferRequest::get_profiling_info() const {
    std::vector<ov::ProfilingInfo> info;
    const auto fill_profiling_info = [](const std::string& name,
                                        const std::chrono::duration<float, std::micro>& time) -> ov::ProfilingInfo {
        ov::ProfilingInfo p_info;
        p_info.status = ov::ProfilingInfo::Status::EXECUTED;
        p_info.node_name = name;
        p_info.cpu_time = p_info.real_time = std::chrono::duration_cast<std::chrono::milliseconds>(time);
        return p_info;
    };

    info.emplace_back(fill_profiling_info("input preprocessing", m_durations[Preprocess]));
    info.emplace_back(fill_profiling_info("execution time", m_durations[StartPipeline]));
    auto template_model = get_template_model();
    for (const auto& op : template_model->get_runtime_model()->get_ops()) {
        const auto& rt_info = op->get_rt_info();
        const auto& it = rt_info.find(ov::runtime::interpreter::PERF_COUNTER_NAME);
        OPENVINO_ASSERT(it != rt_info.end(), "Operation ", op, " doesn't contain performance counter");
        auto counter = it->second.as<std::shared_ptr<ov::runtime::interpreter::PerfCounter>>();
        info.emplace_back(fill_profiling_info(op->get_friendly_name(), counter->duration()));
    }
    info.emplace_back(fill_profiling_info("output postprocessing", m_durations[Postprocess]));

    return info;
}
```

### cancel()

The plugin specific method allows to interrupt the synchronous execution from the AsyncInferRequest:

```cpp
void ov::template_plugin::InferRequest::cancel() {
    m_executable->cancel();
}
```

The next step in the plugin library implementation is the [Asynchronous Inference Request](asynch-inference-request.md) class.
