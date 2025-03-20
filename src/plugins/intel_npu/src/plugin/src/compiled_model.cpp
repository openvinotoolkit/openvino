// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <fstream>
#include <string_view>

#include "async_infer_request.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/common.hpp"
#include "intel_npu/config/compiler.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/runtime.hpp"
#include "metadata.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "transformations/utils/utils.hpp"

namespace {

const std::vector<size_t> CONSTANT_NODE_DUMMY_SHAPE{1};

}

namespace intel_npu {

using intel_npu::envVarStrToBool;

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const std::shared_ptr<IDevice>& device,
                             const std::shared_ptr<IGraph>& graph,
                             const Config& config)
    : ICompiledModel(model, plugin),
      _config(config),
      _logger("CompiledModel", config.get<LOG_LEVEL>()),
      _device(device),
      _graph(graph) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::CompiledModel");

    OV_ITT_TASK_CHAIN(COMPILED_MODEL, itt::domains::NPUPlugin, "CompiledModel::CompiledModel", "initialize_properties");
    initialize_properties();
    configure_stream_executors();

    OV_ITT_TASK_SKIP(COMPILED_MODEL);
}

CompiledModel::~CompiledModel() {
    _logger.debug("~CompiledModel()");
    std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(get_task_executor())->cpu_reset();
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::create_infer_request");

    if (!_config.get<CREATE_EXECUTOR>() || _config.get<DEFER_WEIGHTS_LOAD>()) {
        _graph->initialize(_config);
    }

    const std::shared_ptr<SyncInferRequest>& syncInferRequest =
        _device->createInferRequest(shared_from_this(), _config);
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
    _logger.debug("CompiledModel::export_model");
    size_t blobSizeBeforeVersioning = _graph->export_blob(stream);

    auto meta = Metadata<CURRENT_METADATA_VERSION>(blobSizeBeforeVersioning, CURRENT_OPENVINO_VERSION);
    meta.write(stream);
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    ov::ParameterVector parameters;
    ov::NodeVector results;

    for (const IODescriptor& inputDescriptor : _graph->get_metadata().inputs) {
        if (inputDescriptor.isStateInput || inputDescriptor.isStateOutput || inputDescriptor.isShapeTensor) {
            continue;
        }

        std::shared_ptr<ov::op::v0::Parameter> parameter =
            std::make_shared<ov::op::v0::Parameter>(inputDescriptor.precision, inputDescriptor.shapeFromCompiler);

        parameter->set_friendly_name(inputDescriptor.nodeFriendlyName);
        parameter->output(0).get_tensor().set_names(inputDescriptor.outputTensorNames);
        parameters.push_back(std::move(parameter));
    }

    // The "result" nodes require a parent node in order to satisfy the API conventions. Additionally, a dummy shape for
    // the "Constant" node was required since the specific constructor does not accept "ov::PartialShape" values (a
    // constant can't have dynamic shape). The dummy tensor was also brought in order to register the correct,
    // potentially dynamic, output shape.
    for (const IODescriptor& outputDescriptor : _graph->get_metadata().outputs) {
        if (outputDescriptor.isStateInput || outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor) {
            continue;
        }

        std::shared_ptr<ov::Node> constantDummy = std::make_shared<ov::op::v0::Constant>(
            outputDescriptor.precision,
            outputDescriptor.shapeFromCompiler.to_shape().empty() ? CONSTANT_NODE_DUMMY_SHAPE
                                                                  : outputDescriptor.shapeFromCompiler.to_shape());

        const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy =
            std::make_shared<ov::descriptor::Tensor>(outputDescriptor.precision,
                                                     outputDescriptor.shapeFromCompiler,
                                                     outputDescriptor.outputTensorNames);

        std::shared_ptr<ov::Node> result = std::make_shared<ov::op::v0::Result>(constantDummy);
        result->output(0).set_tensor_ptr(tensorDummy);
        result->set_friendly_name(outputDescriptor.nodeFriendlyName);
        results.push_back(std::move(result));
    }

    _logger.warning("Returning a dummy ov::Model object that contains only the given parameter and result nodes");

    return std::make_shared<ov::Model>(results, parameters);
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    std::map<std::string, std::string> config;
    for (auto&& value : properties) {
        config.emplace(value.first, value.second.as<std::string>());
    }
    for (const auto& configEntry : config) {
        if (_properties.find(configEntry.first) == _properties.end()) {
            OPENVINO_THROW("Unsupported configuration key: ", configEntry.first);
        } else {
            if (std::get<1>(_properties[configEntry.first]) == ov::PropertyMutability::RO) {
                OPENVINO_THROW("READ-ONLY configuration key: ", configEntry.first);
            }
        }
    }

    _config.update(config);
    if (config.find(ov::workload_type.name()) != config.end()) {
        const auto workloadType = properties.at(ov::workload_type.name()).as<ov::WorkloadType>();
        _graph->set_workload_type(workloadType);
    }
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    auto configIterator = _properties.find(name);
    if (configIterator != _properties.cend()) {
        return std::get<2>(configIterator->second)(_config);
    }

    OPENVINO_THROW("Unsupported property ", name);
}

const std::shared_ptr<IGraph>& CompiledModel::get_graph() const {
    return _graph;
}

const Config& CompiledModel::get_config() const {
    return _config;
}

void CompiledModel::configure_stream_executors() {
    std::shared_ptr<ov::threading::ITaskExecutor> task_executor;
    if (get_plugin()->get_property(ov::internal::exclusive_async_requests.name(), {}).as<bool>()) {
        task_executor = ov::threading::executor_manager()->get_executor("NPU");
    } else if (get_property(ov::hint::enable_cpu_pinning.name()).as<bool>()) {
        auto executor_config = ov::threading::IStreamsExecutor::Config{
            /* name = */ "Intel NPU plugin executor",
            /* streams = */ get_plugin()->get_property(ov::num_streams.name(), {}).as<ov::streams::Num>(),
            /* threads_per_stream = */ 1,
            /* thread_preferred_core_type = */ ov::hint::SchedulingCoreType::PCORE_ONLY,
            /* cpu_reservation = */ true};
        task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(executor_config);
    } else {
        task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(
            ov::threading::IStreamsExecutor::Config{"NPUPlugin executor"});
    }

    set_task_executor(std::move(task_executor));
    const auto executorId = _graph->get_metadata().name + "_NPUResultExecutor";
    _resultExecutor = ov::threading::executor_manager()->get_executor(executorId);
}

void CompiledModel::initialize_properties() {
    const auto pluginSupportedProperties =
        get_plugin()->get_property(ov::supported_properties.name(), {}).as<std::vector<ov::PropertyName>>();
    const auto isPropertySupported = [&pluginSupportedProperties](const std::string& name) {
        return std::any_of(pluginSupportedProperties.begin(),
                           pluginSupportedProperties.end(),
                           [&name](const ov::PropertyName& property) {
                               return property == name;
                           });
    };
    _properties = {
        // OV Public
        // =========
        {ov::supported_properties.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              return _supportedProperties;
          }}},
        {ov::device::id.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<DEVICE_ID>();
          }}},
        {ov::enable_profiling.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<PERF_COUNT>();
          }}},
        {ov::model_name.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              OPENVINO_ASSERT(_graph != nullptr, "Missing graph");
              return _graph->get_metadata().name;
          }}},
        {ov::optimal_number_of_infer_requests.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              // value is allowed to be queried prior the network is compiled
              return static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(config));
          }}},
        {ov::execution_devices.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              return std::string("NPU");
          }}},
        {ov::loaded_from_cache.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<LOADED_FROM_CACHE>();
          }}},
        {ov::workload_type.name(),
         {isPropertySupported(ov::workload_type.name()),
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<WORKLOAD_TYPE>();
          }}},
        // OV Public Hints
        // =========
        {ov::hint::performance_mode.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<PERFORMANCE_HINT>();
          }}},
        {ov::hint::execution_mode.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<EXECUTION_MODE_HINT>();
          }}},
        {ov::hint::num_requests.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<PERFORMANCE_HINT_NUM_REQUESTS>();
          }}},
        {ov::hint::inference_precision.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<INFERENCE_PRECISION_HINT>();
          }}},
        {ov::hint::enable_cpu_pinning.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<ENABLE_CPU_PINNING>();
          }}},
        {ov::hint::model_priority.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<MODEL_PRIORITY>();
          }}},
        // OV Internals
        // =========
        {ov::internal::supported_properties.name(),
         {false,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              static const std::vector<ov::PropertyName> supportedProperty{
                  ov::PropertyName(ov::internal::caching_properties.name(), ov::PropertyMutability::RO),
              };
              return supportedProperty;
          }}},
        // NPU Public
        // =========
        {ov::intel_npu::compilation_mode_params.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<COMPILATION_MODE_PARAMS>();
          }}},
        {ov::intel_npu::compiler_dynamic_quantization.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<COMPILER_DYNAMIC_QUANTIZATION>();
          }}},
        {ov::intel_npu::qdq_optimization.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<QDQ_OPTIMIZATION>();
          }}},
        {ov::intel_npu::turbo.name(),
         {isPropertySupported(ov::intel_npu::turbo.name()),
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<TURBO>();
          }}},
        // NPU Private
        // =========
        {ov::intel_npu::tiles.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<TILES>();
          }}},
        {ov::intel_npu::profiling_type.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.getString<PROFILING_TYPE>();
          }}},
        {ov::intel_npu::platform.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<PLATFORM>();
          }}},
        {ov::intel_npu::dynamic_shape_to_static.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.getString<DYNAMIC_SHAPE_TO_STATIC>();
          }}},
        {ov::intel_npu::create_executor.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<CREATE_EXECUTOR>();
          }}},
        {ov::intel_npu::defer_weights_load.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<DEFER_WEIGHTS_LOAD>();
          }}},
        {ov::intel_npu::batch_mode.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.getString<BATCH_MODE>();
          }}},
        {ov::intel_npu::run_inferences_sequentially.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<RUN_INFERENCES_SEQUENTIALLY>();
          }}},
    };

    for (auto& property : _properties) {
        if (std::get<0>(property.second)) {
            _supportedProperties.emplace_back(property.first, std::get<1>(property.second));
        }
    }
}

}  // namespace intel_npu
