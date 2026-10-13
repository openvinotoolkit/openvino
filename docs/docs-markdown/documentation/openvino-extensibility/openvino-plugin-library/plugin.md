---
sidebar_position: 1
sidebar_label: "Plugin"
description: "Explore OpenVINO Plugin API, which includes functions and helper classes that simplify the development of new plugins."
---

# Plugin

OpenVINO Plugin usually represents a wrapper around a backend. Backends can be:

- OpenCL-like backend (e.g. clDNN library) for GPU devices.
- oneDNN backend for Intel CPU devices.
- NVIDIA cuDNN for NVIDIA GPUs.

The responsibility of OpenVINO Plugin:

- Initializes a backend and throw exception in `Engine` constructor if backend cannot be initialized.
- Provides information about devices enabled by a particular backend, e.g. how many devices, their properties and so on.
- Loads or imports [compiled model](compiled-model.md) objects.

In addition to the OpenVINO Public API, the OpenVINO provides the Plugin API, which is a set of functions and helper classes that simplify new plugin development:

- header files in the `src/inference/dev_api/openvino` directory
- implementations in the `src/inference/src/dev/` directory
- symbols in the OpenVINO shared library

To build an OpenVINO plugin with the Plugin API, see the [OpenVINO Plugin Building](build-plugin-using-cmake.md) guide.

## Plugin Class

OpenVINO Plugin API provides the helper ov::IPlugin class recommended to use as a base class for a plugin.
Based on that, declaration of a plugin class can look as follows:

```cpp
namespace ov {
namespace template_plugin {

class Plugin : public ov::IPlugin {
public:
    Plugin();
    ~Plugin();

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

private:
    friend class CompiledModel;
    friend class InferRequest;
    std::string_view get_runtime_requirements() const;

    std::shared_ptr<ov::runtime::Backend> m_backend;
    Configuration m_cfg;
    std::shared_ptr<ov::threading::ITaskExecutor> m_waitExecutor;
};

}  // namespace template_plugin
}  // namespace ov
```

### Class Fields

The provided plugin class also has several fields:

- `m_backend` - a backend engine that is used to perform actual computations for model inference. For `Template` plugin `ov::runtime::Backend` is used which performs computations using OpenVINO™ reference implementations.
- `m_waitExecutor` - a task executor that waits for a response from a device about device tasks completion.
- `m_cfg` of type `Configuration`:

```cpp

struct Configuration {
    Configuration();
    Configuration(const Configuration&) = default;
    Configuration(Configuration&&) = default;
    Configuration& operator=(const Configuration&) = default;
    Configuration& operator=(Configuration&&) = default;

    explicit Configuration(const ov::AnyMap& config,
                           const Configuration& defaultCfg = {},
                           const bool throwOnUnsupported = true);

    ov::Any Get(const std::string& name) const;

    // Plugin configuration parameters

    int device_id = 0;
    bool perf_count = false;
    ov::threading::IStreamsExecutor::Config streams_executor_config{};
    int streams = 1;
    int threads = 0;
    int threads_per_stream = 0;
    ov::hint::PerformanceMode performance_mode = ov::hint::PerformanceMode::LATENCY;
    uint32_t num_requests = 1;
    bool disable_transformations = false;
    bool exclusive_async_requests = false;

    // unused
    ov::element::Type inference_precision = ov::element::dynamic;
    ov::hint::ExecutionMode execution_mode = ov::hint::ExecutionMode::ACCURACY;
    ov::log::Level log_level = ov::log::Level::NO;

    ov::hint::Priority model_priority = ov::hint::Priority::DEFAULT;

    ov::hint::SchedulingCoreType schedulingCoreType = ov::hint::SchedulingCoreType::ANY_CORE;
    bool enableCpuPinning = false;
    bool enableHyperThreading = false;
    int compilation_thread_num = 1;
    EncryptionCallbacks encryption_callbacks{};
    std::filesystem::path weights_path{};
    AnyMap compiled_model_runtime_properties{};
    CacheMode cache_mode{CacheMode::OPTIMIZE_SPEED};
};
```

As an example, a plugin configuration has three value parameters:

- `device_id` - particular device ID to work with. Applicable if a plugin supports more than one `Template` device. In this case, some plugin methods, like `set_property`, `query_model`, and `compile_model`, must support the ov::device::id property.
- `perf_counts` - boolean value to identify whether to collect performance counters during [Inference Request](synch-inference-request.md) execution.
- `streams_executor_config` - configuration of `ov::threading::IStreamsExecutor` to handle settings of multi-threaded context.
- `performance_mode` - configuration of `ov::hint::PerformanceMode` to set the performance mode.
- `disable_transformations` - allows to disable transformations which are applied in the process of model compilation.
- `exclusive_async_requests` - allows to use exclusive task executor for asynchronous infer requests.

### Plugin Constructor

A plugin constructor must contain code that checks the ability to work with a device of the `Template`
type. For example, if some drivers are required, the code must check
driver availability. If a driver is not available (for example, OpenCL runtime is not installed in
case of a GPU device or there is an improper version of a driver is on a host machine), an exception
must be thrown from a plugin constructor.

A plugin must define a device name enabled via the `set_device_name()` method of a base class:

```cpp
ov::template_plugin::Plugin::Plugin() {
    // TODO: fill with actual device name, backend engine
    set_device_name("TEMPLATE");

    // create backend which performs inference using openvino reference implementations
    m_backend = ov::runtime::Backend::create();

    // create default stream executor with a given name
    m_waitExecutor = get_executor_manager()->get_idle_cpu_streams_executor({wait_executor_name});
}
```

### Plugin Destructor

A plugin destructor must stop all plugins activities, and clean all allocated resources.

```cpp
ov::template_plugin::Plugin::~Plugin() {
    // Plugin should remove executors from executor cache to avoid threads number growth in the whole application
    get_executor_manager()->clear(stream_executor_name);
    get_executor_manager()->clear(wait_executor_name);
}
```

### compile\_model()

The plugin should implement two `compile_model()` methods: the first one compiles model without remote context, the second one with remote context if plugin supports.

This is the most important function of the `Plugin` class is to create an instance of compiled `CompiledModel`,
which holds a backend-dependent compiled model in an internal representation:

```cpp
std::shared_ptr<ov::ICompiledModel> ov::template_plugin::Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties) const {
    return compile_model(model, properties, {});
}
```

```cpp
std::shared_ptr<ov::ICompiledModel> ov::template_plugin::Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::SoPtr<ov::IRemoteContext>& context) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::compile_model");

    Configuration fullConfig;
    {
        auto _properties = properties;
        // remove not supported properties which are consumed by compile_model
        _properties.erase(ov::loaded_from_cache.name());
        _properties.erase(ov::hint::compiled_blob.name());
        fullConfig = Configuration{_properties, m_cfg};
    }

    fullConfig.streams_executor_config = ov::threading::IStreamsExecutor::Config{stream_executor_name,
                                                                                 fullConfig.streams,
                                                                                 fullConfig.threads_per_stream};
    auto streamsExecutorConfig =
        ov::threading::IStreamsExecutor::Config::make_default_multi_threaded(fullConfig.streams_executor_config);
    fullConfig.streams = streamsExecutorConfig.get_streams();
    fullConfig.threads = streamsExecutorConfig.get_threads();
    fullConfig.threads_per_stream = streamsExecutorConfig.get_threads_per_stream();

    return std::make_shared<CompiledModel>(
        model->clone(),
        shared_from_this(),
        context,
        fullConfig.exclusive_async_requests
            ? get_executor_manager()->get_executor(template_exclusive_executor)
            : get_executor_manager()->get_idle_cpu_streams_executor(streamsExecutorConfig),
        fullConfig,
        false);
}
```

Before a creation of an `CompiledModel` instance via a constructor, a plugin may check if a provided
ov::Model object is supported by a device if it is needed.

Actual model compilation is done in the `CompiledModel` constructor. Refer to the [CompiledModel Implementation Guide](compiled-model.md) for details.

:::note
Actual configuration map used in `CompiledModel` is constructed as a base plugin configuration set via `Plugin::set_property`, where some values are overwritten with `config` passed to `Plugin::compile_model`. Therefore, the config of  `Plugin::compile_model` has a higher priority.
:::

### transform\_model()

The function accepts a const shared pointer to ov::Model object and applies common and device-specific transformations on a copied model to make it more friendly to hardware operations. For details how to write custom device-specific transformation, refer to [Writing OpenVINO™ transformations](../transformation-api/index.md) guide. See detailed topics about model representation:

- [Intermediate Representation and Operation Sets](../../openvino-ir-format/operation-sets.md)
- [Quantized models](advanced-guides/quantized-models.md).

```cpp
void transform_model(const std::shared_ptr<ov::Model>& model) {
    // Perform common optimizations and device-specific transformations
    ov::pass::Manager passManager("Plugin:Template");

    // Defaults are overwritten below from the actual PA node's input types (inputs 3/4 for
    // cache, input 0 for query). f32 is only used if there is no PA node in the model.
    ov::pass::ConvertPagedAttnInputs::KVCacheConfig cacheConfig;
    cacheConfig.keyCachePrecision = ov::element::f32;
    cacheConfig.valueCachePrecision = ov::element::f32;
    cacheConfig.inferencePrecision = ov::element::f32;
    for (const auto& op : model->get_ops()) {
        if (auto pa = std::dynamic_pointer_cast<ov::op::PagedAttentionExtension>(op)) {
            const auto key_et = pa->get_input_element_type(3);
            const auto val_et = pa->get_input_element_type(4);
            if (!key_et.is_dynamic())
                cacheConfig.keyCachePrecision = key_et;
            if (!val_et.is_dynamic())
                cacheConfig.valueCachePrecision = val_et;
            const auto qry_et = pa->get_input_element_type(0);
            if (!qry_et.is_dynamic())
                cacheConfig.inferencePrecision = qry_et;
            break;
        }
    }
    auto noop_shape = [](const ov::element::Type&, const bool, const size_t, int64_t&, int64_t&) {};
    passManager.register_pass<ov::pass::ConvertPagedAttnInputs>(cacheConfig, noop_shape);

    // Example: register CommonOptimizations transformation from transformations library
    passManager.register_pass<ov::pass::CommonOptimizations>();
    // Disable some transformations
    passManager.get_pass_config()->disable<ov::pass::UnrollIf>();
    passManager.get_pass_config()->disable<ov::pass::ConvertMaxPool14ToMaxPool8>();
    passManager.get_pass_config()->disable<ov::pass::ConvertAvgPool14ToAvgPool1>();
    // This transformation changes output name
    passManager.get_pass_config()->disable<ov::pass::ConvertReduceSumToPooling>();
    // Register any other transformations
    // ..

    const auto& pass_config = passManager.get_pass_config();

    // Allow FP16 Converts to be folded and FP16 constants to be upgraded to FP32 data type
    pass_config->disable<ov::pass::DisableDecompressionConvertConstantFolding>();
    pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();

    // Disabled SDPA transformation, since there is ref SDPA op.
    pass_config->disable<ov::pass::ScaledDotProductAttentionDecomposition>();

    // Attach cache manager to PA nodes for KV cache block management
    passManager.register_pass<ov::pass::AttachCacheManagerToPagedAttention>();

    // After `run_passes`, we have the transformed function, where operations match device operations,
    // and we can create device backend-dependent graph
    passManager.run_passes(model);
}
```

:::note
After all these transformations, an `ov::Model` object contains operations which can be perfectly mapped to backend kernels. E.g. if backend has kernel computing `A + B` operations at once, the `transform_model` function should contain a pass which fuses operations `A` and `B` into a single custom operation A + B which fits backend kernels set.
:::

### query\_model()

Use the method with the `HETERO` mode, which allows to distribute model execution between different
devices based on the `ov::Node::get_rt_info()` map, which can contain the `affinity` key.
The `query_model` method analyzes operations of provided `model` and returns a list of supported
operations via the ov::SupportedOpsMap structure. The `query_model` firstly applies `transform_model` passes to input `ov::Model` argument. After this, the transformed model in ideal case contains only operations are 1:1 mapped to kernels in computational backend. In this case, it's very easy to analyze which operations is supposed (`m_backend` has a kernel for such operation or extensions for the operation is provided) and not supported (kernel is missed in `m_backend`):

1. Store original names of all operations in input `ov::Model`.
2. Apply `transform_model` passes. Note, the names of operations in a transformed model can be different and we need to restore the mapping in the steps below.
3. Construct `supported` map which contains names of original operations. Note that since the inference is performed using OpenVINO™ reference backend, the decision whether the operation is supported or not depends on whether the latest OpenVINO opset contains such operation.
4. `ov.SupportedOpsMap` contains only operations which are fully supported by `m_backend`.

```cpp
ov::SupportedOpsMap ov::template_plugin::Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                                             const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::query_model");

    Configuration fullConfig{properties, m_cfg, false};

    OPENVINO_ASSERT(model, "OpenVINO Model is empty!");

    auto supported = ov::get_supported_nodes(
        model,
        [&](std::shared_ptr<ov::Model>& model) {
            // skip transformations in case of user config
            if (fullConfig.disable_transformations)
                return;
            // 1. It is needed to apply all transformations as it is done in compile_model
            transform_model(model);
        },
        [&](std::shared_ptr<ov::Node> node) {
            // 2. Сheck whether node is supported
            ov::OpSet op_super_set;
#define _OPENVINO_OP_REG(NAME, NAMESPACE) op_super_set.insert<NAMESPACE::NAME>();
        // clang-format off
#include "openvino/opsets/opset1_tbl.hpp"
#include "openvino/opsets/opset2_tbl.hpp"
#include "openvino/opsets/opset3_tbl.hpp"
#include "openvino/opsets/opset4_tbl.hpp"
#include "openvino/opsets/opset5_tbl.hpp"
#include "openvino/opsets/opset6_tbl.hpp"
#include "openvino/opsets/opset7_tbl.hpp"
#include "openvino/opsets/opset8_tbl.hpp"
#include "openvino/opsets/opset9_tbl.hpp"
#include "openvino/opsets/opset10_tbl.hpp"
#include "openvino/opsets/opset11_tbl.hpp"
#include "openvino/opsets/opset12_tbl.hpp"
#include "openvino/opsets/opset13_tbl.hpp"
#include "openvino/opsets/opset14_tbl.hpp"
#include "openvino/opsets/opset15_tbl.hpp"
#include "openvino/opsets/opset16_tbl.hpp"
#include "openvino/opsets/opset17_tbl.hpp"
        // clang-format on
#undef _OPENVINO_OP_REG
            return op_super_set.contains_type(node->get_type_info());
        });

    // 3. Produce the result
    ov::SupportedOpsMap res;
    for (auto&& layerName : supported) {
        res.emplace(layerName, get_device_name() + "." + std::to_string(m_cfg.device_id));
    }

    return res;
}
```

### set\_property()

Sets new values for plugin property keys:

```cpp
void ov::template_plugin::Plugin::set_property(const ov::AnyMap& properties) {
    m_cfg = Configuration{properties, m_cfg};
}
```

In the snippet above, the `Configuration` class overrides previous configuration values with the new
ones. All these values are used during backend specific model compilation and execution of inference requests.

:::note
The function must throw an exception if it receives an unsupported configuration key.
:::

### get\_property()

Returns a current value for a specified property key:

```cpp
ov::Any ov::template_plugin::Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::available_devices,
                                                    ov::supported_properties,
                                                    ov::device::full_name,
                                                    ov::device::architecture,
                                                    ov::device::capabilities,
                                                    ov::device::type,
                                                    ov::range_for_async_infer_requests,
                                                    ov::execution_devices,
                                                    ov::compatibility_check};
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{
            ov::device::id,
            ov::enable_profiling,
            ov::hint::performance_mode,
            ov::hint::num_requests,
            ov::hint::inference_precision,
            ov::hint::execution_mode,
            ov::num_streams,
            ov::template_plugin::disable_transformations,
            ov::log::level,
            ov::hint::model_priority,
            ov::hint::enable_hyper_threading,
            ov::hint::enable_cpu_pinning,
            ov::hint::scheduling_core_type,
            ov::compilation_num_threads,
            ov::inference_num_threads,
            ov::weights_path,
            ov::cache_mode,
        };
        return rw_properties;
    };
    if (ov::supported_properties == name) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        std::vector<ov::PropertyName> supported_properties;
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return supported_properties;
    } else if (ov::internal::supported_properties == name) {
        return decltype(ov::internal::supported_properties)::value_type{
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::inference_num_threads.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::internal::threads_per_stream.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::internal::compiled_model_runtime_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::cache_header_alignment.name(), ov::PropertyMutability::RO},
        };
    } else if (ov::available_devices == name) {
        // TODO: fill list of available devices
        return decltype(ov::available_devices)::value_type{{""}};
    } else if (ov::device::full_name == name) {
        return decltype(ov::device::full_name)::value_type{"Template Device Full Name"};
    } else if (ov::device::architecture == name) {
        // TODO: return device architecture for device specified by DEVICE_ID config
        return decltype(ov::device::architecture)::value_type{get_device_name()};
    } else if (ov::device::type == name) {
        return decltype(ov::device::type)::value_type{ov::device::Type::INTEGRATED};
    } else if (ov::internal::caching_properties == name) {
        return decltype(ov::internal::caching_properties)::value_type{ov::device::architecture};
    } else if (ov::device::capabilities == name) {
        // TODO: fill actual list of supported capabilities: e.g. Template device supports only FP32 and EXPORT_IMPORT
        return decltype(ov::device::capabilities)::value_type{ov::device::capability::FP32,
                                                              ov::device::capability::EXPORT_IMPORT};
    } else if (ov::execution_devices == name) {
        return decltype(ov::execution_devices)::value_type{get_device_name()};
    } else if (ov::range_for_async_infer_requests == name) {
        return decltype(ov::range_for_async_infer_requests)::value_type{1, 1, 1};
    } else if (ov::compatibility_check == name) {
        if (auto it = arguments.find(ov::runtime_requirements.name()); it != arguments.end()) {
            const auto& requirements = it->second.as<std::string>();
            if (!requirements.empty()) {
                if (const auto pos = requirements.find(get_runtime_requirements()); pos == 0) {
                    return ov::CompatibilityCheck::SUPPORTED;
                } else {
                    return ov::CompatibilityCheck::UNSUPPORTED;
                }
            }
        }
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    } else {
        return m_cfg.Get(name);
    }
}
```

The function is implemented with the `Configuration::Get` method, which wraps an actual configuration
key value to the ov::Any and returns it.

:::note
The function must throw an exception if it receives an unsupported configuration key.
:::

### import\_model()

The importing of compiled model mechanism allows to import a previously exported backend specific model and wrap it
using an [CompiledModel](compiled-model.md) object. This functionality is useful if
backend specific model compilation takes significant time and/or cannot be done on a target host
device due to other reasons.

During export of backend specific model using `CompiledModel::export_model`, a plugin may export any
type of information it needs to import a compiled model properly and check its correctness.
For example, the export information may include:

- Compilation options (state of `Plugin::m_cfg` structure).
- Information about a plugin and a device type to check this information later during the import and throw an exception if the `model` stream contains wrong data. For example, if devices have different capabilities and a model compiled for a particular device cannot be used for another, such type of information must be stored and checked during the import.
- Compiled backend specific model itself.

```cpp
std::shared_ptr<ov::ICompiledModel> ov::template_plugin::Plugin::import_model(std::istream& model,
                                                                              const ov::AnyMap& properties) const {
    return import_model(model, {}, properties);
}
```

```cpp
std::shared_ptr<ov::ICompiledModel> ov::template_plugin::Plugin::import_model(
    std::istream& model,
    const ov::SoPtr<ov::IRemoteContext>& context,
    const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::import_model");

    // check ov::loaded_from_cache property and erase it due to not needed any more.
    auto _properties = properties;
    const auto& it = _properties.find(ov::loaded_from_cache.name());
    bool loaded_from_cache = false;
    if (it != _properties.end()) {
        loaded_from_cache = it->second.as<bool>();
        _properties.erase(it);
    }
    _properties.erase(ov::hint::compiled_blob.name());

    auto fullConfig = Configuration{_properties, m_cfg};
    fullConfig.streams_executor_config = ov::threading::IStreamsExecutor::Config{stream_executor_name,
                                                                                 fullConfig.streams,
                                                                                 fullConfig.threads_per_stream};
    auto weights = get_model_weights(properties);
    if (!weights) {
        if (auto model_hint = properties.find(ov::hint::model.name()); model_hint != properties.end()) {
            if (auto m = model_hint->second.as<std::shared_ptr<ov::Model>>()) {
                if (m->has_rt_info("__weights_path")) {
                    AnyMap rt_info;
                    rt_info[ov::weights_path.name()] = m->get_rt_info<ov::Any>("__weights_path");
                    weights = get_model_weights(rt_info);
                }
            }
        }
    }
    auto ov_model = get_ov_model_from_blob(*this, weights, model.tellg(), properties);
    if (!ov_model) {
        // read XML content
        std::string xmlString = get_model_str(model);

        // read blob content
        if (!weights) {
            weights = get_model_weights(model);
        }

        ov_model = get_core()->read_model(xmlString, weights);
    }
    auto streamsExecutorConfig =
        ov::threading::IStreamsExecutor::Config::make_default_multi_threaded(fullConfig.streams_executor_config);
    fullConfig.streams = streamsExecutorConfig.get_streams();
    fullConfig.threads = streamsExecutorConfig.get_threads();
    fullConfig.threads_per_stream = streamsExecutorConfig.get_threads_per_stream();
    auto compiled_model =
        std::make_shared<CompiledModel>(ov_model,
                                        shared_from_this(),
                                        context,
                                        get_executor_manager()->get_idle_cpu_streams_executor(streamsExecutorConfig),
                                        fullConfig,
                                        loaded_from_cache);
    return compiled_model;
}
```

### create\_context()

The Plugin should implement `Plugin::create_context()` method which returns `ov::RemoteContext` in case if plugin supports remote context, in other case the plugin can throw an exception that this method is not implemented.

```cpp
ov::SoPtr<ov::IRemoteContext> ov::template_plugin::Plugin::create_context(const ov::AnyMap& remote_properties) const {
    return std::make_shared<ov::template_plugin::RemoteContext>();
}
```

### get\_default\_context()

`Plugin::get_default_context()` also needed in case if plugin supports remote context, if the plugin doesn't support it, this method can throw an exception that functionality is not implemented.

```cpp
ov::SoPtr<ov::IRemoteContext> ov::template_plugin::Plugin::get_default_context(
    const ov::AnyMap& remote_properties) const {
    return std::make_shared<ov::template_plugin::RemoteContext>();
}
```

## Create Instance of Plugin Class

OpenVINO plugin library must export only one function creating a plugin instance using OV\_DEFINE\_PLUGIN\_CREATE\_FUNCTION macro:

```cpp
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_template_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::template_plugin::Plugin, version)
```

Next step in a plugin library implementation is the [CompiledModel](compiled-model.md) class.
