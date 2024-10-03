// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <fstream>
#include <string_view>

#include "async_infer_request.hpp"
#include "intel_npu/al/config/config.hpp"
#include "intel_npu/al/config/options.hpp"
#include "intel_npu/al/icompiler.hpp"
#include "intel_npu/al/itt.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "transformations/utils/utils.hpp"

namespace {

constexpr std::string_view NO_EXECUTOR_FOR_INFERENCE =
    "Can't create infer request!\n"
    "Please make sure that the device is available. Only exports can be made.";

std::uint32_t hash(const std::vector<uint8_t>& data) {
    std::uint32_t result = 1171117u;
    for (const auto& c : data)
        result = ((result << 7) + result) + static_cast<uint32_t>(c);
    return result;
}

}  // namespace

namespace intel_npu {

// Macro for registering simple get<> properties which have everything defined in their optionBase
#define REGISTER_SIMPLE_PROPERTY(option_name, config_type)                                                          \
    do {                                                                                                            \
        std::string o_name = option_name.name();                                                                    \
        if (_config.hasOpt(o_name)) {                                                                               \
            _properties.emplace(                                                                                    \
                o_name,                                                                                             \
                std::make_tuple(_config.isOptPublic(o_name), ov::PropertyMutability::RO, [](const Config& config) { \
                    return config.get<config_type>();                                                               \
                }));                                                                                                \
        }                                                                                                           \
    } while (0)

// Macro for registering properties which have custom function
#define REGISTER_CUSTOM_PROPERTY(option_name, __func)                                                              \
    do {                                                                                                           \
        std::string o_name = option_name.name();                                                                   \
        if (_config.hasOpt(o_name)) {                                                                              \
            _properties.emplace(o_name,                                                                            \
                                std::make_tuple(_config.isOptPublic(o_name), ov::PropertyMutability::RO, __func)); \
        }                                                                                                          \
    } while (0)

// Macro for defining full custom properties
#define REGISTER_PROPERTY(option_name, __isPublic, __isMutable, __func)                    \
    do {                                                                                   \
        std::string o_name = option_name.name();                                           \
        if (_config.hasOpt(o_name)) {                                                      \
            _properties.emplace(o_name, std::make_tuple(__isPublic, __isMutable, __func)); \
        }                                                                                  \
    } while (0)

using intel_npu::envVarStrToBool;

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const std::shared_ptr<IDevice>& device,
                             const ov::SoPtr<ICompiler>& compiler,
                             const bool profiling,
                             const Config& config)
    : ICompiledModel(model, plugin),
      _model(model),
      _config(config),
      _logger("CompiledModel", config.get<LOG_LEVEL>()),
      _device(device),
      _compiler(profiling ? std::optional(compiler) : std::nullopt) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::CompiledModel");
    OPENVINO_ASSERT(compiler != nullptr, "NPU CompiledModel: the pointer towards the compiler object is null");

    try {
        _networkPtr = std::make_shared<const NetworkDescription>(compiler->compile(model, config));
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    } catch (...) {
        _logger.error("Unexpected exception");
        OPENVINO_THROW("NPU CompiledModel: got an unexpected exception from compiler");
    }

    OV_ITT_TASK_CHAIN(COMPILED_MODEL, itt::domains::NPUPlugin, "CompiledModel::CompiledModel", "initialize_properties");
    _properties = std::make_unique<Properties>(PropertiesType::COMPILED_MODEL, _config);
    _properties->registerProperties();

    configure_stream_executors();

    OV_ITT_TASK_NEXT(COMPILED_MODEL, "create_executor");
    create_executor();

    OV_ITT_TASK_SKIP(COMPILED_MODEL);
}

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const std::shared_ptr<const NetworkDescription>& networkDescription,
                             const std::shared_ptr<IDevice>& device,
                             const std::optional<ov::SoPtr<ICompiler>>& compiler,
                             const Config& config)
    : ICompiledModel(model, plugin),
      _networkPtr(networkDescription),
      _model(model),
      _config(config),
      _logger("CompiledModel", config.get<LOG_LEVEL>()),
      _device(device),
      _compiler(compiler) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::CompiledModel");
    OPENVINO_ASSERT(_networkPtr != nullptr,
                    "NPU CompiledModel: the pointer towards the NetworkDescription object is null");

    OV_ITT_TASK_CHAIN(COMPILED_MODEL, itt::domains::NPUPlugin, "CompiledModel::CompiledModel", "initialize_properties");
    _properties = std::make_unique<Properties>(PropertiesType::COMPILED_MODEL, _config);
    _properties->registerProperties();

    configure_stream_executors();

    OV_ITT_TASK_NEXT(COMPILED_MODEL, "create_executor");
    create_executor();

    OV_ITT_TASK_SKIP(COMPILED_MODEL);
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::create_infer_request");

    if (_executorPtr == nullptr && _device != nullptr) {
        _executorPtr = _device->createExecutor(_networkPtr, _config);
    }
    if (_executorPtr == nullptr) {
        OPENVINO_THROW(NO_EXECUTOR_FOR_INFERENCE);
    }

    const std::shared_ptr<SyncInferRequest>& syncInferRequest =
        _device->createInferRequest(shared_from_this(), _executorPtr, _config);
    syncInferRequest->initialize_states();

    return std::make_shared<AsyncInferRequest>(syncInferRequest,
                                               get_task_executor(),
                                               _resultExecutor,
                                               get_callback_executor());
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    OPENVINO_THROW_NOT_IMPLEMENTED(
        "The synchronous inference request structure implemented by the NPU plugin does not inherit "
        "the \"ov::ISyncInferRequest\" class");
}

void CompiledModel::export_model(std::ostream& stream) const {
    const auto& blob = _networkPtr->compiledNetwork;
    stream.write(reinterpret_cast<const char*>(blob.data()), blob.size());

    std::stringstream str;
    str << "Blob size: " << blob.size() << ", hash: " << std::hex << hash(blob);
    _logger.info(str.str().c_str());
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    return _model;
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    // 1. Set the property via Properties interface
    _properties->set_property(properties);

    // 2. Extra hooks
    if (properties.count(std::string(WORKLOAD_TYPE::key())) != 0) {
        if (_executorPtr != nullptr) {
            const auto workloadType = properties.at(ov::workload_type.name()).as<ov::WorkloadType>();
            _executorPtr->setWorkloadType(workloadType);
        }
    }
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    ov::AnyMap dummy;
    return _properties->get_property(name, dummy);
}

const std::shared_ptr<const NetworkDescription>& CompiledModel::get_network_description() const {
    return _networkPtr;
}

const Config& CompiledModel::get_config() const {
    return _config;
}

const ov::SoPtr<ICompiler>& CompiledModel::get_compiler() const {
    if (_compiler.has_value()) {
        return _compiler.value();
    }
    OPENVINO_THROW("PERF_COUNT property is not set");
}

void CompiledModel::configure_stream_executors() {
    std::shared_ptr<ov::threading::ITaskExecutor> task_executor;
    if (get_plugin()->get_property(ov::internal::exclusive_async_requests.name(), {}).as<bool>()) {
        task_executor = ov::threading::executor_manager()->get_executor("NPU");
    } else if (get_property(ov::hint::enable_cpu_pinning.name()).as<bool>()) {
        auto executor_config = ov::threading::IStreamsExecutor::Config{
            "Intel NPU plugin executor",
            get_plugin()->get_property(ov::num_streams.name(), {}).as<ov::streams::Num>(),
            1,
            ov::hint::SchedulingCoreType::PCORE_ONLY,
            true};
        task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(executor_config);
    } else {
        task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(
            ov::threading::IStreamsExecutor::Config{"NPUPlugin executor"});
    }

    set_task_executor(std::move(task_executor));
    const auto executorId = _networkPtr->metadata.name + "_NPUResultExecutor";
    _resultExecutor = ov::threading::executor_manager()->get_executor(executorId);
}

// void CompiledModel::initialize_properties() {
//     const auto pluginSupportedProperties =
//         get_plugin()->get_property(ov::supported_properties.name(), {}).as<std::vector<ov::PropertyName>>();
//     const auto isPropertySupported = [&pluginSupportedProperties](const std::string& name) {
//         return std::any_of(pluginSupportedProperties.begin(),
//                            pluginSupportedProperties.end(),
//                            [&name](const ov::PropertyName& property) {
//                                return property == name;
//                            });
//     };
// }

void CompiledModel::create_executor() {
    if (_config.get<CREATE_EXECUTOR>()) {
        _logger.info("Creating the executor inside the \"CompiledModel\" constructor");

        // If no device has been defined, the executor shall keep the default value of "nullptr". In this scenario,
        // only export operations will be allowed
        if (_device != nullptr) {
            _executorPtr = _device->createExecutor(_networkPtr, _config);
        }
    } else {
        _logger.info("Executor will not be created inside the \"CompiledModel\" constructor");
    }
}

}  // namespace intel_npu
