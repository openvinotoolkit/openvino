---
sidebar_position: 2
sidebar_label: "Compiled Model"
description: "Use the ov::CompiledModel class as the base class for a compiled model and to create an arbitrary number of ov::InferRequest objects."
---

# Compiled Model

`ov::CompiledModel` class functionality:

- Compile an `ov::Model` instance to a backend specific graph representation
- Create an arbitrary number of `ov::InferRequest` objects
- Hold some common resources shared between different instances of `ov::InferRequest`. For example:
  
  - `ov::ICompiledModel::m_task_executor` task executor to implement asynchronous execution
  - `ov::ICompiledModel::m_callback_executor` task executor to run an asynchronous inference request callback in a separate thread

## CompiledModel Class

OpenVINO Plugin API provides the interface `ov::ICompiledModel` which should be used as a base class
for a compiled model. Based on that, a declaration of an compiled model class can look as follows:

```cpp
class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const ov::SoPtr<ov::IRemoteContext>& context,
                  const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                  const Configuration& cfg,
                  bool loaded_from_cache = false);

    // Methods from a base class ov::ICompiledModel
    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    friend class InferRequest;
    friend class Plugin;

    void compile_model(const std::shared_ptr<ov::Model>& model);
    std::shared_ptr<const Plugin> get_template_plugin() const;

    mutable std::atomic<std::size_t> m_request_id = {0};
    Configuration m_cfg;
    std::shared_ptr<ov::Model> m_model;
    const bool m_loaded_from_cache;
};
```

### Class Fields

The example class has several fields:

- `m_request_id` - Tracks a number of created inference requests, which is used to distinguish
  different inference requests during profiling via the Intel® Instrumentation and Tracing Technology (ITT) library.
- `m_cfg` - Defines a configuration a compiled model was compiled with.
- `m_model` - Keeps a reference to transformed `ov::Model` which is used in OpenVINO reference
  backend computations. Note, in case of other backends with backend specific graph representation
  `m_model` has different type and represents backend specific graph or just a set of computational kernels to perform an inference.
- `m_loaded_from_cache` - Allows to understand that model was loaded from cache.

### CompiledModel Constructor

This constructor accepts a generic representation of a model as an ov::Model and is compiled into a backend specific device graph:

```cpp
ov::template_plugin::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                                  const std::shared_ptr<const ov::IPlugin>& plugin,
                                                  const ov::SoPtr<ov::IRemoteContext>& context,
                                                  const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                                  const Configuration& cfg,
                                                  bool loaded_from_cache)
    : ov::ICompiledModel(model, plugin, context, task_executor),  // Disable default threads creation
      m_cfg(cfg),
      m_model(model),
      m_loaded_from_cache(loaded_from_cache) {
    // TODO: if your plugin supports device ID (more that single instance of device can be on host machine)
    // you should select proper device based on KEY_DEVICE_ID or automatic behavior
    // In this case, m_wait_executor should also be created per device.
    try {
        compile_model(m_model);
    } catch (const std::exception& e) {
        OPENVINO_THROW("Standard exception from compilation library: ", e.what());
    } catch (...) {
        OPENVINO_THROW("Generic exception is thrown");
    }
}
```

The implementation `compile_model()` is fully device-specific.

### compile\_model()

The function accepts a const shared pointer to `ov::Model` object and applies OpenVINO passes
using `transform_model()` function, which defines plugin-specific conversion pipeline. To support
low precision inference, the pipeline can include Low Precision Transformations. These
transformations are usually hardware specific. You can find how to use and configure Low Precisions
Transformations in [Low Precision Transformations](advanced-guides/low-precision-transformations/index.md) guide.

```cpp
// forward declaration
void transform_model(const std::shared_ptr<ov::Model>& model);

void ov::template_plugin::CompiledModel::compile_model(const std::shared_ptr<ov::Model>& model) {
    // apply plugins transformations
    if (!m_cfg.disable_transformations)
        transform_model(model);

    // Integrate performance counters to the compiled model
    for (const auto& op : model->get_ops()) {
        auto& rt_info = op->get_rt_info();
        rt_info[ov::runtime::interpreter::PERF_COUNTER_NAME] =
            std::make_shared<ov::runtime::interpreter::PerfCounter>();
    }

    // Perform any other steps like allocation and filling backend specific memory handles and so on
}
```

:::note
After all these steps, the backend specific graph is ready to create inference requests and perform inference.
:::

### export\_model()

The implementation of the method should write all data to the `model_stream`, which is required
to import a backend specific graph later in the `Plugin::import_model` method:

```cpp
void ov::template_plugin::CompiledModel::export_model(std::ostream& model_stream) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "CompiledModel::export_model");

    std::stringstream xmlFile, binFile;
    ov::pass::Serialize serializer(xmlFile, binFile);
    serializer.run_on_model(m_model);

    auto m_constants = binFile.str();
    auto m_model = xmlFile.str();

    auto dataSize = static_cast<std::uint64_t>(m_model.size());
    model_stream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    model_stream.write(m_model.c_str(), dataSize);

    if (m_cfg.cache_mode == CacheMode::OPTIMIZE_SPEED) {
        dataSize = static_cast<std::uint64_t>(m_constants.size());
        model_stream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
        model_stream.write(reinterpret_cast<char*>(&m_constants[0]), dataSize);
    }
}
```

### create\_sync\_infer\_request()

The method creates an synchronous inference request and returns it.

```cpp
std::shared_ptr<ov::ISyncInferRequest> ov::template_plugin::CompiledModel::create_sync_infer_request() const {
    return std::make_shared<InferRequest>(
        std::static_pointer_cast<const ov::template_plugin::CompiledModel>(shared_from_this()));
}
```

While the public OpenVINO API has a single interface for inference request, which can be executed
in synchronous and asynchronous modes, a plugin library implementation has two separate classes:

- [Synchronous inference request](synch-inference-request.md), which defines pipeline stages and runs them synchronously in the `infer` method.
- [Asynchronous inference request](asynch-inference-request.md), which is a wrapper for a synchronous
  inference request and can run a pipeline asynchronously. Depending on a device pipeline structure,
  it can have one or several stages:
  
  - For single-stage pipelines, there is no need to define this method and create a class derived
    from `ov::IAsyncInferRequest`. For single stage pipelines, a default implementation of this
    method creates `ov::IAsyncInferRequest` wrapping a synchronous inference request and runs
    it asynchronously in the `m_request_executor` executor.
  - For pipelines with multiple stages, such as performing some preprocessing on host, uploading
    input data to a device, running inference on a device, or downloading and postprocessing output
    data, schedule stages on several task executors to achieve better device use and performance.
    You can do it by creating a sufficient number of inference requests running in parallel.
    In this case, device stages of different inference requests are overlapped with preprocessing
    and postprocessing stage giving better performance.

:::info[Important]
It is up to you to decide how many task executors you need to optimally execute a device pipeline.
:::

### create\_infer\_request()

The method creates an asynchronous inference request and returns it.

```cpp
std::shared_ptr<ov::IAsyncInferRequest> ov::template_plugin::CompiledModel::create_infer_request() const {
    auto internal_request = create_sync_infer_request();
    auto async_infer_request = std::make_shared<AsyncInferRequest>(
        std::static_pointer_cast<ov::template_plugin::InferRequest>(internal_request),
        get_task_executor(),
        get_template_plugin()->m_waitExecutor,
        get_callback_executor());

    return async_infer_request;
}
```

### get\_property()

Returns a current value for a property with the name `name`. The method extracts configuration values a compiled model is compiled with.

```cpp
ov::Any ov::template_plugin::CompiledModel::get_property(const std::string& name) const {
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::model_name,
                                                    ov::supported_properties,
                                                    ov::execution_devices,
                                                    ov::loaded_from_cache,
                                                    ov::optimal_number_of_infer_requests,
                                                    ov::runtime_requirements};
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{ov::device::id, ov::enable_profiling, ov::hint::performance_mode};
        return rw_properties;
    };
    if (ov::model_name == name) {
        auto& model_name = m_model->get_friendly_name();
        return decltype(ov::model_name)::value_type(model_name);
    } else if (ov::loaded_from_cache == name) {
        return m_loaded_from_cache;
    } else if (ov::execution_devices == name) {
        return decltype(ov::execution_devices)::value_type{get_plugin()->get_device_name() + "." +
                                                           std::to_string(m_cfg.device_id)};
    } else if (ov::optimal_number_of_infer_requests == name) {
        unsigned int value = m_cfg.streams;
        return decltype(ov::optimal_number_of_infer_requests)::value_type(value);
    } else if (ov::runtime_requirements == name) {
        // A real plugin would encode compilation capabilities.
        return std::string(get_template_plugin()->get_runtime_requirements());
    } else if (ov::supported_properties == name) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        auto supported_properties = decltype(ov::supported_properties)::value_type();
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return supported_properties;
    }

    return m_cfg.Get(name);
}
```

This function is the only way to get configuration values when a model is imported and compiled by other developers and tools.

### set\_property()

The methods allows to set compiled model specific properties.

```cpp
void ov::template_plugin::CompiledModel::set_property(const ov::AnyMap& properties) {
    m_cfg = Configuration{properties, m_cfg};
}
```

### get\_runtime\_model()

The methods returns the runtime model with backend specific information.

```cpp
std::shared_ptr<const ov::Model> ov::template_plugin::CompiledModel::get_runtime_model() const {
    auto model = m_model->clone();
    // Add execution information into the model
    size_t exec_order = 0;
    for (const auto& op : model->get_ordered_ops()) {
        auto& info = op->get_rt_info();
        const auto& it = info.find(ov::runtime::interpreter::PERF_COUNTER_NAME);
        OPENVINO_ASSERT(it != info.end(), "Operation ", op, " doesn't contain performance counter");
        auto perf_count = it->second.as<std::shared_ptr<ov::runtime::interpreter::PerfCounter>>();
        OPENVINO_ASSERT(perf_count, "Performance counter is empty");
        info[ov::exec_model_info::LAYER_TYPE] = op->get_type_info().name;
        info[ov::exec_model_info::EXECUTION_ORDER] = std::to_string(exec_order++);
        info[ov::exec_model_info::IMPL_TYPE] = "ref";
        info[ov::exec_model_info::PERF_COUNTER] = m_cfg.perf_count && perf_count && perf_count->avg() != 0
                                                      ? std::to_string(perf_count->avg())
                                                      : "not_executed";

        std::string original_names = ov::getFusedNames(op);
        if (original_names.empty()) {
            original_names = op->get_friendly_name();
        } else if (original_names.find(op->get_friendly_name()) == std::string::npos) {
            original_names = op->get_friendly_name() + "," + original_names;
        }
        info[ov::exec_model_info::ORIGINAL_NAMES] = original_names;
    }
    return model;
}
```

The next step in plugin library implementation is the [Synchronous Inference Request](synch-inference-request.md) class.
