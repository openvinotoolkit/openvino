// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph.hpp"

#include <iterator>

#include "compiler_impl.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_cmd_queue_pool.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace intel_npu {

Graph::Graph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
             const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
             const GraphDescriptor& graphDesc,
             NetworkMetadata metadata,
             std::optional<ov::Tensor> blob,
             const FilteredConfig& config,
             const std::optional<std::string>& compatibilityDescriptor,
             const bool blobIsPersistent,
             const bool calledFromWeightlessGraph)
    : IGraph(),
      _zeGraphExt(zeGraphExt),
      _zeroInitStruct(zeroInitStruct),
      _graphDesc(graphDesc),
      _metadata(std::move(metadata)),
      _blob(std::move(blob)),
      _compatibilityDescriptor(compatibilityDescriptor),
      _blobIsPersistent(blobIsPersistent),
      _logger("Graph", config.get<LOG_LEVEL>()) {
    if (!config.get<CREATE_EXECUTOR>() || config.get<DEFER_WEIGHTS_LOAD>()) {
        _logger.info("Graph initialize is deferred from the \"Graph\" constructor");
        return;
    }

    if (!calledFromWeightlessGraph) {
        // Will be called at a later stage from WeightlessGraph::initialize() in order to save some memory
        initialize(config);
    }
}

const NetworkMetadata& Graph::get_metadata() const {
    return _metadata;
}

void Graph::update_network_name(std::string_view name) {
    _metadata.name = name;
}

CommandQueueDesc Graph::get_command_queue_desc() const {
    std::lock_guard<std::mutex> lock(_commandQueueDescMutex);
    return _commandQueueDesc;
}

void Graph::set_workload_type(const ov::WorkloadType workloadType) {
    if (_zeroInitStruct == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(_commandQueueDescMutex);
    auto zeWorkloadType = zeroUtils::toZeQueueWorkloadType(workloadType);
    if (_commandQueueDesc.workload() == zeWorkloadType) {
        return;
    }
    _commandQueueDesc.set_workload(zeWorkloadType);
}

void Graph::set_model_priority(const ov::hint::Priority modelPriority) {
    if (_zeroInitStruct == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(_commandQueueDescMutex);
    auto zeModelPriority = zeroUtils::toZeQueuePriority(modelPriority);
    if (_commandQueueDesc.priority() == zeModelPriority) {
        return;
    }
    _commandQueueDesc.set_priority(zeModelPriority);
}

ze_graph_handle_t Graph::get_handle() const {
    return _graphDesc._handle;
}

std::pair<uint64_t, std::optional<std::vector<uint64_t>>> Graph::export_blob(std::ostream& stream) const {
    const uint8_t* blobPtr = nullptr;
    size_t blobSize;
    std::vector<uint8_t> blobVec;  // plugin needs to keep a copy of the blob for older drivers

    if (_blobIsReleased) {
        OPENVINO_THROW("Model was imported and released after initialization. Model export is not allowed anymore.");
    }

    if (_blob == std::nullopt) {
        OPENVINO_ASSERT(_zeGraphExt != nullptr, "Zero compiler adapter wasn't initialized");
        // when compiling the model using Compiler in Driver, the blob is handled by the driver
        _zeGraphExt->getGraphBinary(_graphDesc, blobVec, blobPtr, blobSize);
    } else {  // in all other cases, the blob is handled by the plugin
        blobPtr = static_cast<const uint8_t*>(_blob->data());
        blobSize = _blob->get_byte_size();
    }

    if (blobSize > static_cast<decltype(blobSize)>(std::numeric_limits<std::streamsize>::max())) {
        OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
    }
    stream.write(reinterpret_cast<const char*>(blobPtr), static_cast<std::streamsize>(blobSize));

    if (!stream) {
        _logger.error("Write blob to stream failed. Blob is broken!");
        return std::make_pair(0, std::nullopt);
    }

    if (_logger.level() >= ov::log::Level::INFO) {
        std::uint32_t result = 1171117u;
        for (const uint8_t* it = blobPtr; it != blobPtr + blobSize; ++it) {
            result = ((result << 7) + result) + static_cast<uint32_t>(*it);
        }

        std::stringstream str;
        str << "Blob size: " << blobSize << ", hash: " << std::hex << result;
        _logger.info(str.str().c_str());
    }

    size_t size = utils::align_size_to_standard_page_size(blobSize);
    size_t paddingSize = size - blobSize;
    if (paddingSize > 0) {
        std::fill_n(std::ostream_iterator<char>(stream), paddingSize, 0);

        if (!stream) {
            _logger.error("Write padding to stream failed. Blob is broken!");
            return std::make_pair(0, std::nullopt);
        }

        _logger.info("Blob size with padding: %ld", size);
    }

    _logger.info("Write blob to stream successfully.");
    return std::make_pair(size, std::nullopt);
}

std::vector<ov::ProfilingInfo> Graph::process_profiling_output(const std::vector<uint8_t>& profData) const {
    auto compiler = VCLCompilerImpl::getInstance();
    OPENVINO_ASSERT(compiler != nullptr, "Profiling post-processing requires the NPU plugin compiler library");

    std::vector<uint8_t> blob(_blob->get_byte_size());
    blob.assign(reinterpret_cast<const uint8_t*>(_blob->data()),
                reinterpret_cast<const uint8_t*>(_blob->data()) + _blob->get_byte_size());
    return compiler->process_profiling_output(profData, blob);
}

void Graph::set_argument_value(uint32_t id, const void* data) const {
    if (_zeGraphExt == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }
    _zeGraphExt->setGraphArgumentValue(_graphDesc, id, data);
}

void Graph::set_argument_value_with_strides(uint32_t id, const void* data, const std::vector<size_t>& strides) const {
    if (_zeGraphExt == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }
    _zeGraphExt->setGraphArgumentValueWithStrides(_graphDesc, id, data, strides);
}

void Graph::initialize_impl(const FilteredConfig& config) {
    _logger.debug("Graph initialize start");

    if (_zeGraphExt == nullptr || _graphDesc._handle == nullptr || _zeroInitStruct == nullptr) {
        // To ensure that no issues are thrown during subsequent calls.
        return;
    }

    uint32_t commandQueueOptions = 0;
    if (config.has<TURBO>() && config.get<TURBO>()) {
        if (_zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 0)) {
            _logger.debug("Set ZE_NPU_COMMAND_QUEUE_OPTION_TURBO in command queue options");
            commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
        }
    }
    if (_zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1) &&
        config.has<RUN_INFERENCES_SEQUENTIALLY>() && config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        _logger.debug("Set ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC in command queue options");
        commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC;
    }

    {
        std::lock_guard<std::mutex> lock(_commandQueueDescMutex);
        _commandQueueDesc = CommandQueueDesc{
            zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
            config.has<WORKLOAD_TYPE>() ? zeroUtils::toZeQueueWorkloadType(config.get<WORKLOAD_TYPE>()) : std::nullopt,
            commandQueueOptions,
            this,
            config.get<SHARED_COMMON_QUEUE>(),
        };
    }

    _zeGraphExt->initializeGraph(_graphDesc);
    _logger.debug("Graph initialize finish");

    //  We are allowed to release the original blob because weights were loaded in NPU memory during
    //  _zeGraphExt->initializeGraph(). The driver will not access the original blob from this moment on, so we are
    //  releasing it here to avoid unnecessary memory usage.
    _blobIsReleased = release_blob(config);

    if (!_batchSize.has_value()) {
        _batchSize = determine_batch_size();
    }

    if (_zeroInitStruct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
        config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        auto numberOfCommandLists = _batchSize.has_value() ? *_batchSize : 1;

        _lastSubmittedEvent.resize(numberOfCommandLists);
    }
    // To ensure that the initialization of the graph does not exit prematurely due to nullptrs
    _init_completed.store(true, std::memory_order_release);
}

bool Graph::release_blob(const FilteredConfig& config) {
    if ((_zeGraphExt != nullptr && _zeGraphExt->isBlobDataImported(_graphDesc)) || _blobIsPersistent ||
        _blob == std::nullopt || _zeroInitStruct->getGraphDdiTable().version() < ZE_MAKE_VERSION(1, 8) ||
        config.get<PERF_COUNT>()) {
        return false;
    }

    ze_graph_properties_2_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES_2;
    _zeroInitStruct->getGraphDdiTable().pfnGetProperties2(_graphDesc._handle, &properties);

    if (~properties.initStageRequired & ZE_GRAPH_STAGE_INITIALIZE) {
        return false;
    }

    _blob = std::nullopt;
    _logger.debug("Blob is released");

    return true;
};

void Graph::set_last_submitted_event(const std::shared_ptr<Event>& event, size_t indexOfCommandList) {
    _lastSubmittedEvent[indexOfCommandList] = event;
}

const std::shared_ptr<Event>& Graph::get_last_submitted_event(size_t indexOfCommandList) const {
    return _lastSubmittedEvent[indexOfCommandList];
}

void Graph::resize_last_submitted_event(size_t batch) {
    _lastSubmittedEvent.resize(batch);
}

void Graph::set_batch_size(std::size_t batch) {
    _batchSize = batch;
}

uint32_t Graph::get_unique_id() {
    return _uniqueId++;
}

void Graph::set_last_submitted_id(uint32_t id_index) {
    _lastSubmittedId = id_index;
}

uint32_t Graph::get_last_submitted_id() const {
    return _lastSubmittedId;
}

std::optional<std::string> Graph::get_compatibility_descriptor() const {
    return _compatibilityDescriptor;
}

std::optional<bool> Graph::is_profiling_blob() const {
    if (_zeroInitStruct->getGraphDdiTable().version() < ZE_MAKE_VERSION(1, 16)) {
        _logger.debug("Cannot determine if the blob was compiled for profiling");
        return std::nullopt;
    }
    ze_graph_properties_3_t graphProperties = {};
    graphProperties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES_3;

    auto result = _zeroInitStruct->getGraphDdiTable().pfnGetProperties3(get_handle(), &graphProperties);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetArgumentProperties3", result, _zeroInitStruct->getGraphDdiTable());

    return graphProperties.flags & ZE_GRAPH_PROPERTIES_FLAG_PROFILING_ENABLED;
}

std::optional<size_t> Graph::determine_batch_size() {
    if (!_metadata.outputs.at(0).shapeFromIRModel.has_value()) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    const ov::PartialShape& firstShape = *_metadata.outputs.at(0).shapeFromIRModel;
    if (firstShape.is_dynamic() || firstShape.rank().get_length() == 0) {
        return std::nullopt;
    }

    const size_t candidateBatchSize = firstShape[utils::BATCH_AXIS].get_max_length();
    if (candidateBatchSize == 0 || candidateBatchSize == utils::DEFAULT_BATCH_SIZE) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    auto checkDescriptorsUseCandidateBatchSize = [candidateBatchSize](const std::vector<IODescriptor>& descriptors) {
        for (const IODescriptor& descriptor : descriptors) {
            OPENVINO_ASSERT(descriptor.shapeFromIRModel.has_value(),
                            "Missing value for the \"shapeFromIRModel\" attribute, I/O descriptor");

            const ov::PartialShape& shapeFromCompiler = descriptor.shapeFromCompiler;
            const ov::PartialShape& shapeFromIRModel = *descriptor.shapeFromIRModel;

            if (shapeFromCompiler.is_dynamic() || shapeFromCompiler.rank().get_length() == 0 ||
                *shapeFromCompiler.begin() != utils::DEFAULT_BATCH_SIZE) {
                return false;
            }

            if (!descriptor.isStateInput && !descriptor.isStateOutput && !descriptor.isShapeTensor) {
                if (shapeFromIRModel.is_dynamic() || shapeFromIRModel.rank().get_length() == 0 ||
                    *shapeFromIRModel.begin() != candidateBatchSize) {
                    return false;
                }
            }
        }

        return true;
    };

    if (!checkDescriptorsUseCandidateBatchSize(_metadata.inputs) ||
        !checkDescriptorsUseCandidateBatchSize(_metadata.outputs)) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    _logger.debug("Batching is handled by the plugin");

    return candidateBatchSize;
}

const std::optional<std::size_t> Graph::get_batch_size() const {
    return _batchSize;
}

void Graph::evict_memory() {
    if (_zeGraphExt != nullptr) {
        _zeGraphExt->evict_memory(_graphDesc);
    }
}

Graph::~Graph() {
    // make sure all the context-dependent components are destroyed before the zero context is destroyed
    if (_zeGraphExt != nullptr) {
        _zeGraphExt->destroyGraph(_graphDesc);
    }

    if (!_lastSubmittedEvent.empty()) {
        _lastSubmittedEvent.clear();
    }
}

}  // namespace intel_npu
