// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <cinttypes>
#include <fstream>
#include <string_view>

#include "async_infer_request.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "metadata.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "transformations/utils/utils.hpp"

namespace {

enum class ExecutorKind { CALLBACK, REQUEST };

struct ExecutorCreationOptions {
    ExecutorKind kind = ExecutorKind::REQUEST;
    bool enableExclusiveAsyncRequests = false;
    bool enableCpuPinning = false;
};

std::shared_ptr<ov::threading::ITaskExecutor> create_stream_executors(const std::string& executorName,
                                                                      const ExecutorCreationOptions& options = {}) {
    // Callback executor always uses simple config
    if (options.kind == ExecutorKind::CALLBACK) {
        return std::make_shared<ov::threading::CPUStreamsExecutor>(
            ov::threading::IStreamsExecutor::Config{executorName});
    }

    // Task executor - check for special configurations
    if (options.enableExclusiveAsyncRequests) {
        return ov::threading::executor_manager()->get_executor("NPU Plugin Exclusive Async Requests Executor");
    }

    if (options.enableCpuPinning) {
        auto executor_config = ov::threading::IStreamsExecutor::Config{
            /* name = */ executorName,
            /* streams = */ 1,
            /* threads_per_stream = */ 1,
            /* thread_preferred_core_type = */ ov::hint::SchedulingCoreType::PCORE_ONLY,
            /* cpu_reservation = */ true,
            /* cpu_pinning = */ true};
        return std::make_shared<ov::threading::CPUStreamsExecutor>(executor_config);
    }

    return std::make_shared<ov::threading::CPUStreamsExecutor>(ov::threading::IStreamsExecutor::Config{executorName});
}

}  // namespace

namespace intel_npu {

using intel_npu::envVarStrToBool;

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const std::shared_ptr<IDevice>& device,
                             const std::shared_ptr<IGraph>& graph,
                             const FilteredConfig& config,
                             const std::optional<int64_t>& batchSize)
    : ICompiledModel(model,
                     plugin,
                     nullptr,
                     create_stream_executors("Intel NPU plugin callback executor",
                                             ExecutorCreationOptions{ExecutorKind::CALLBACK, false, false})),
      _logger("CompiledModel", config.get<LOG_LEVEL>()),
      _device(device),
      _graph(graph),
      _batchSize(batchSize) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::CompiledModel");

    // Support for specific properties might depend on the characteristics of the compiled model.
    // Adjust lower level config availability to influence the supported properties list if needed
    FilteredConfig localConfig = config;
    if (!_graph->get_compatibility_descriptor().has_value()) {
        _logger.debug("Graph's compatibility descriptor has no value. Disabling RUNTIME_REQUIREMENTS property.");
        localConfig.enable(ov::runtime_requirements.name(), false);
    }

    OV_ITT_TASK_CHAIN(COMPILED_MODEL, itt::domains::NPUPlugin, "CompiledModel::CompiledModel", "initialize_properties");
    _propertiesManager = std::make_unique<Properties>(PropertiesType::COMPILED_MODEL, localConfig);

    if (_propertiesManager->getConfig().get<RUN_INFERENCES_SEQUENTIALLY>()) {
        _logger.warning("The configuration option \"RUN_INFERENCES_SEQUENTIALLY\" is set to \"true\". This will cause "
                        "all inference requests for this Compiled Model to run sequentially.");

        set_task_executor(create_stream_executors(
            "Intel NPU plugin start inferences sequential executor",
            ExecutorCreationOptions{
                ExecutorKind::REQUEST,
                get_plugin()->get_property(ov::internal::exclusive_async_requests.name(), {}).as<bool>(),
                _propertiesManager->getConfig().get<ENABLE_CPU_PINNING>()}));

        _wait_seq_executor = create_stream_executors(
            "Intel NPU plugin wait inferences sequential executor",
            ExecutorCreationOptions{
                ExecutorKind::REQUEST,
                get_plugin()->get_property(ov::internal::exclusive_async_requests.name(), {}).as<bool>(),
                _propertiesManager->getConfig().get<ENABLE_CPU_PINNING>()});
    }

    OV_ITT_TASK_SKIP(COMPILED_MODEL);
}

CompiledModel::~CompiledModel() {
    _logger.debug("~CompiledModel()");

    if (!get_plugin()->get_property(ov::internal::exclusive_async_requests.name(), {}).as<bool>() &&
        _propertiesManager->getConfig().get<ENABLE_CPU_PINNING>()) {
        auto stream_executor = std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(get_task_executor());
        if (stream_executor) {
            stream_executor->cpu_reset();
        }

        stream_executor = std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(_wait_seq_executor);
        if (stream_executor) {
            stream_executor->cpu_reset();
        }
    }
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::create_infer_request");

    // sanity check
    OPENVINO_ASSERT(_device != nullptr, "No available devices. Failed to create infer request!");

    if (!_propertiesManager->getConfig().get<CREATE_EXECUTOR>() ||
        _propertiesManager->getConfig().get<DEFER_WEIGHTS_LOAD>()) {
        OPENVINO_ASSERT(_graph != nullptr, "Invalid graph handle! Failed to create infer request!");
        _graph->initialize(_propertiesManager->getConfig());
    }

    OPENVINO_ASSERT(_graph != nullptr && _graph->init_completed(),
                    "Graph is unavailable or failed to initialize. The driver may be missing or too old to run "
                    "inference for this blob.");

    const auto& syncInferRequest = _device->createInferRequest(shared_from_this(), _propertiesManager->getConfig());

    if (_propertiesManager->getConfig().get<RUN_INFERENCES_SEQUENTIALLY>()) {
        return std::make_shared<AsyncInferRequest>(syncInferRequest,
                                                   get_task_executor(),
                                                   get_callback_executor(),
                                                   _wait_seq_executor,
                                                   nullptr);
    }

    const auto executorCreationOptions = ExecutorCreationOptions{
        ExecutorKind::REQUEST,
        get_plugin()->get_property(ov::internal::exclusive_async_requests.name(), {}).as<bool>(),
        _propertiesManager->getConfig().get<ENABLE_CPU_PINNING>()};
    auto requestExecutor = create_stream_executors("Intel NPU plugin executor", executorCreationOptions);

    return std::make_shared<AsyncInferRequest>(
        syncInferRequest,
        requestExecutor,
        get_callback_executor(),
        nullptr,
        [requestExecutor, executorCreationOptions]() {
            if (!executorCreationOptions.enableExclusiveAsyncRequests && executorCreationOptions.enableCpuPinning) {
                auto stream_executor = std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(requestExecutor);
                if (stream_executor) {
                    stream_executor->cpu_reset();
                }
            }
        });
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    OPENVINO_THROW_NOT_IMPLEMENTED(
        "The synchronous inference request structure implemented by the NPU plugin does not inherit "
        "the \"ov::ISyncInferRequest\" class");
}

void CompiledModel::export_model(std::ostream& stream) const {
    _logger.debug("CompiledModel::export_model");

    uint64_t blobSizesBeforeVersioning;
    std::optional<uint64_t> blobSizeAfterEncryption = std::nullopt;
    std::optional<std::vector<uint64_t>> initBlobSizes;

    if (_propertiesManager->getConfig().has(CACHE_ENCRYPTION_CALLBACKS::key().data()) &&
        _propertiesManager->getConfig().get<CACHE_ENCRYPTION_CALLBACKS>().encrypt != nullptr) {
        std::string encryptedBlobStr;
        {
            std::string tmpBlobStr;
            {
                std::stringstream tmpStringStream;
                std::tie(blobSizesBeforeVersioning, initBlobSizes) =
                    _graph->export_blob(tmpStringStream);  // +1x blob size
                tmpBlobStr = tmpStringStream.str();        // +2x blob size
            }  // -1x blob size when deallocating temporary stringstream
            encryptedBlobStr =
                _propertiesManager->getConfig().get<CACHE_ENCRYPTION_CALLBACKS>().encrypt(tmpBlobStr);  // +2x blob size
            blobSizeAfterEncryption = encryptedBlobStr.size();
            if (blobSizeAfterEncryption.value() % utils::STANDARD_PAGE_SIZE != 0) {
                _logger.warning("Encrypted blob size %" PRIu64
                                " is not page aligned, memory optimization when reading this blob "
                                "won't be applied",
                                blobSizeAfterEncryption.value());
            }
        }  // -1x blob size when deallocating temporary blob string
        stream.write(encryptedBlobStr.c_str(), encryptedBlobStr.size());
    }  // -1x blob size when deallocating encrypted blob string
    else {
        //  Write blob directly to user's output stream
        std::tie(blobSizesBeforeVersioning, initBlobSizes) = _graph->export_blob(stream);
    }

    if (!_propertiesManager->getConfig().get<EXPORT_RAW_BLOB>()) {
        std::optional<std::vector<ov::Layout>> inputLayouts = std::vector<ov::Layout>();
        std::optional<std::vector<ov::Layout>> outputLayouts = std::vector<ov::Layout>();

        for (const ov::Output<const ov::Node>& nodeOutput : inputs()) {
            inputLayouts->push_back(
                std::dynamic_pointer_cast<const ov::op::v0::Parameter>(nodeOutput.get_node_shared_ptr())->get_layout());
        }
        for (const ov::Output<const ov::Node>& nodeOutput : outputs()) {
            outputLayouts->push_back(
                std::dynamic_pointer_cast<const ov::op::v0::Result>(nodeOutput.get_node_shared_ptr())->get_layout());
        }

        std::optional<uint32_t> compilerVersion = std::nullopt;
        if (_propertiesManager->getConfig().has(ov::intel_npu::compiler_version.name())) {
            compilerVersion = _propertiesManager->getConfig().get<COMPILER_VERSION>();
        }

        Metadata<CURRENT_METADATA_VERSION>(blobSizesBeforeVersioning,
                                           CURRENT_OPENVINO_VERSION,
                                           initBlobSizes,
                                           _batchSize,
                                           inputLayouts,
                                           outputLayouts,
                                           compilerVersion,
                                           blobSizeAfterEncryption,
                                           _graph->get_compatibility_descriptor())
            .write(stream);
    }
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    ov::ParameterVector parameters;
    ov::ResultVector results;
    std::shared_ptr<const ov::Model> dummyModel;

    try {
        for (const ov::Output<const ov::Node>& nodeOutput : inputs()) {
            std::shared_ptr<ov::Node> clonedParameter =
                std::dynamic_pointer_cast<const ov::op::v0::Parameter>(nodeOutput.get_node_shared_ptr())
                    ->clone_with_new_inputs({});
            parameters.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(clonedParameter));
        }

        for (const ov::Output<const ov::Node>& nodeOutput : outputs()) {
            const auto resultOriginal =
                std::dynamic_pointer_cast<const ov::op::v0::Result>(nodeOutput.get_node_shared_ptr());

            // A dummy node is required for constructing and populating the Result node. A Constant one is perhaps the
            // most fitting choice here.
            std::shared_ptr<ov::Node> constantDummy =
                std::make_shared<ov::op::v0::Constant>(nodeOutput.get_element_type(),
                                                       nodeOutput.get_partial_shape().get_max_shape());
            // Attached to the Result node as output tensor in order to provide the correct tensor names. Additionally,
            // the dummy Constant node could use only static shapes. If the shape is dynamic, this construct can provide
            // the correct shape to the Result node.
            const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy =
                std::make_shared<ov::descriptor::Tensor>(nodeOutput.get_element_type(),
                                                         nodeOutput.get_partial_shape(),
                                                         nodeOutput.get_names());

            auto& resultCopy = results.emplace_back(std::make_shared<ov::op::v0::Result>(constantDummy));
            resultCopy->output(0).set_tensor_ptr(tensorDummy);
            resultCopy->set_friendly_name(resultOriginal->get_friendly_name());

            dummyModel = std::make_shared<ov::Model>(results, parameters);
        }
    } catch (const std::exception& e) {
        OPENVINO_THROW("Failed to construct a dummy ov::Model object as runtime model. ", e.what());
    }

    _logger.warning("Returning a dummy ov::Model object that contains only the given parameter and result nodes");

    return dummyModel;
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    // 1. Set the property via Properties interface
    _propertiesManager->setProperty(properties);

    // 2. Extra hooks
    if (properties.count(std::string(WORKLOAD_TYPE::key())) != 0) {
        if (_graph != nullptr) {
            const auto workloadType = properties.at(ov::workload_type.name()).as<ov::WorkloadType>();
            _graph->set_workload_type(workloadType);
        }
    }

    if (properties.count(std::string(MODEL_PRIORITY::key())) != 0) {
        if (_graph != nullptr) {
            const auto modelPriority = properties.at(ov::hint::model_priority.name()).as<ov::hint::Priority>();
            _graph->set_model_priority(modelPriority);
        }
    }
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    // special cases
    if (name == ov::model_name.name()) {
        OPENVINO_ASSERT(_graph != nullptr, "Missing graph");
        return _graph->get_metadata().name;
    } else if (name == ov::runtime_requirements.name()) {
        // Reading the (dummy) property content to check if it is supported
        _propertiesManager->getProperty(name);

        _logger.debug("Runtime requirements from the graph %s length: %zu",
                      _graph->get_compatibility_descriptor().value(),
                      _graph->get_compatibility_descriptor().value().size());

        std::ostringstream requirementsString;
        Metadata<CURRENT_METADATA_VERSION>(
            0,  // no real blob
            CURRENT_OPENVINO_VERSION,
            std::nullopt,  // weightless blobs are not supported
            _batchSize,
            std::nullopt,  // input_layouts are not relevant for the compatibility check
            std::nullopt,  // output_layouts are not relevant for the compatibility check
            std::nullopt,  // skip compiler version as well since it is already included in runtime requirements string
            std::nullopt,  // skip encrypted blob size since it is not relevant for the compatibility check
            _graph->get_compatibility_descriptor())
            .write_as_text(requirementsString);
        _logger.debug("Runtime requirements string: %s length: %zu",
                      requirementsString.str().c_str(),
                      requirementsString.str().length());

        return requirementsString.str();
    }

    // default behaviour
    return _propertiesManager->getProperty(name);
}

const std::shared_ptr<IGraph>& CompiledModel::get_graph() const {
    return _graph;
}

const FilteredConfig& CompiledModel::get_config() const {
    return _propertiesManager->getConfig();
}

void CompiledModel::release_memory() {
    if (_graph != nullptr) {
        _graph->evict_memory();
    }
}

}  // namespace intel_npu
