// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <cinttypes>
#include <fstream>
#include <string_view>

#include "async_infer_request.hpp"
#include "executor.hpp"
#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "metadata.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/runtime/properties.hpp"
#include "transformations/utils/utils.hpp"

namespace intel_npu {

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const std::shared_ptr<IDevice>& device,
                             const std::shared_ptr<IGraph>& graph,
                             const FilteredConfig& config,
                             const std::optional<int64_t>& batchSize)
    : ICompiledModel(model, plugin, nullptr, nullptr),
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

    OV_ITT_TASK_SKIP(COMPILED_MODEL);
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::create_infer_request");

    std::call_once(_streamExecutorsInitFlag, [this] {
        const_cast<CompiledModel*>(this)->configure_stream_executors();
    });

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

    const std::shared_ptr<InferRequest>& inferRequest =
        _device->createInferRequest(shared_from_this(), _propertiesManager->getConfig());

    return std::make_shared<AsyncInferRequest>(inferRequest,
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

        auto compatibilityDescriptor = _graph->get_compatibility_descriptor();
        if (compatibilityDescriptor.has_value()) {
            const auto descriptorView = compatibilityDescriptor.value();
            _logger.debug("Runtime requirements from the graph %.*s length: %zu",
                          static_cast<int>(descriptorView.size()),
                          descriptorView.data(),
                          descriptorView.size());
        }

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
            compatibilityDescriptor)
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

void CompiledModel::configure_stream_executors() {
    const FilteredConfig& config = get_config();

    // In case of sequential execution of async requests for the same compiled model, the compiled model must use
    // dedicated executors with a single thread to ensure sequential execution of its async requests.
    if (config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        set_task_executor(make_executor("Intel NPU plugin start inferences executor", 1));
        _resultExecutor = make_executor("Intel NPU plugin wait inferences executor", 1);

        return;
    }

    const auto numStreams = config.get<NUM_STREAMS>();
    if (numStreams > 0) {
        // Use a single thread for start executors to reduce contention on the shared task queue, while scaling wait
        // executor workers with num_streams to improve result fetch throughput. Callbacks intentionally run on wait
        // threads.
        const size_t workers = static_cast<size_t>(numStreams);

        set_task_executor(make_executor("Intel NPU plugin start inferences executor", 1));
        _resultExecutor = make_executor("Intel NPU plugin wait inferences executor", workers);
    } else if (numStreams == 0) {
        // For special case when num_streams is explicitly set to 0, start inference will happen in the same thread as
        // the call to InferRequest::start_async, while wait executor will still be created with a single worker.
        // Callback execution is intentionally done on that wait thread.
        set_task_executor(make_executor("Intel NPU plugin start inferences executor", 0));
        _resultExecutor = make_executor("Intel NPU plugin wait inferences executor", 1);
    } else {
        // Auto mode (default): workers are created on demand. The baseline number of workers that stay alive during
        // idle periods (30 s timeout) is derived from the optimal number of parallel infer requests recommended for
        // the current platform in THROUGHPUT mode. The pool can then grow dynamically to match runtime workload.
        const size_t keepWorkers = static_cast<size_t>(
            utils::getOptimalNumberOfInferRequestsInParallel(config.get<PLATFORM>(),
                                                             ov::hint::PerformanceMode::THROUGHPUT));

        set_task_executor(make_executor("Intel NPU plugin run inferences executor", keepWorkers, true));
        _resultExecutor = nullptr;
    }
}

}  // namespace intel_npu
