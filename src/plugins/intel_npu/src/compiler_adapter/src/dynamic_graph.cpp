// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_graph.hpp"

#include <array>
#include <iterator>
#include <ostream>

#include "compiler_impl.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_cmd_queue_pool.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "ze_graph_ext_wrappers.hpp"

namespace intel_npu {

void DynamicGraph::create_execution_engine() {
    npu_vm_runtime_blob_desc_t blobDesc;
    blobDesc.pInput = reinterpret_cast<const uint8_t*>(_blob.value().data());
    blobDesc.inputSize = _blob.value().get_byte_size();

    if (npuVMRuntimeCreate(&blobDesc, &_engine, &_engineProperties) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to create VM runtime engine");
    }
}

void DynamicGraph::prepare_metadata() {
    _metadata.inputs.clear();
    _metadata.outputs.clear();
    for (uint32_t i = 0; i < _engineProperties.numOfGraphArgs; ++i) {
        // TODO: follow graph ext to support Optional metadata for weightless model
        ze_graph_argument_properties_3_t arg;
        ze_graph_argument_metadata_t meta;
        std::array<int64_t, ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE> upperBound = {};
        if (npuVMRuntimeGetMetadata(_engine, i, &arg, &meta, upperBound.data()) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
            OPENVINO_THROW("Failed to get VM runtime metadata");
        }
        IODescriptor ioDesc = createIODescriptorFromLevelZero(i, arg, meta);
        ioDesc.supportsStridedLayout = true;
        switch (arg.type) {
        case ZE_GRAPH_ARGUMENT_TYPE_INPUT: {
            _metadata.inputs.push_back(std::move(ioDesc));
        } break;
        case ZE_GRAPH_ARGUMENT_TYPE_OUTPUT: {
            _metadata.outputs.push_back(std::move(ioDesc));
        } break;
        default: {
            OPENVINO_THROW("Invalid ze_graph_argument_type_t found in ze_graph_argument_properties_3_t object: ",
                           arg.type);
        }
        }
    }
    _metadata.bindRelatedDescriptors();
}

void DynamicGraph::initialize_engine() {
    if (!_engineInitialized) {
        create_execution_engine();
        prepare_metadata();
        _engineInitialized = true;
        _metadata.numberOfSubgraphs = _engineProperties.numOfSubGraphs;

        _logger.debug("num of subgraphs: %d inputs: %d outputs: %d",
                      _engineProperties.numOfSubGraphs,
                      _metadata.inputs.size(),
                      _metadata.outputs.size());
    }

    if (_logger.level() >= ov::log::Level::DEBUG) {
        _logger.debug("Dump metadata info from blob");
        _logger.debug("Metadata inputs: %d", _metadata.inputs.size());
        for (const auto& input : _metadata.inputs) {
            _logger.debug("Input compiler name: %s input node name: %s shapeFromCompiler: %s shapeFromIRModel: %s",
                          input.nameFromCompiler.c_str(),
                          input.nodeFriendlyName.c_str(),
                          input.shapeFromCompiler.to_string().c_str(),
                          input.shapeFromIRModel.has_value() ? input.shapeFromIRModel->to_string().c_str() : "N/A");
        }
        _logger.debug("Metadata outputs: %d", _metadata.outputs.size());
        for (const auto& output : _metadata.outputs) {
            _logger.debug("Output compiler name: %s output node name: %s shapeFromCompiler: %s shapeFromIRModel: %s",
                          output.nameFromCompiler.c_str(),
                          output.nodeFriendlyName.c_str(),
                          output.shapeFromCompiler.to_string().c_str(),
                          output.shapeFromIRModel.has_value() ? output.shapeFromIRModel->to_string().c_str() : "N/A");
        }
    }
}

DynamicGraph::DynamicGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                           ov::Tensor blob,
                           const FilteredConfig& config,
                           BlobType blobType)
    : _zeroInitStruct(zeroInitStruct),
      _blob(std::move(blob)),
      _blobType(blobType),
      _logger("DynamicGraph", config.get<LOG_LEVEL>()) {
    _logger.info("Create DynamicGraph");
    // Metadata comes from the VM runtime parsing the blob; unlike a regular Graph, it is not prefetched by the
    // compiler/parser and must be available before plugin builds a dummy ov::Model for the CompiledModel.
    // This is CPU-side parsing only - no L0/device setup.
    initialize_engine();
}

std::pair<uint64_t, std::optional<std::vector<uint64_t>>> DynamicGraph::export_blob(std::ostream& stream) const {
    const uint8_t* blobPtr = nullptr;
    size_t blobSize = 0;

    std::vector<uint8_t> blobVec;  // plugin needs to keep a copy of the blob for older drivers

    if (_blobIsReleased) {
        OPENVINO_THROW("Model was optimized away. Try importing it using `ov::hint::compiled_blob` property to extend "
                       "its lifetime.");
    }

    if (_blob ==
        std::nullopt) {  // when compiling the model using Compiler in Driver, the blob is handled by the driver
        OPENVINO_THROW("No CiD is supported yet!");
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

        _logger.info("Blob size: %zu, hash: %x", blobSize, result);
    }

    size_t size = utils::align_size_to_standard_page_size(blobSize);
    size_t paddingSize = size - blobSize;
    if (paddingSize > 0) {
        std::fill_n(std::ostream_iterator<char>(stream), paddingSize, 0);
        if (!stream) {
            _logger.error("Write padding to stream failed. Blob is broken!");
            return std::make_pair(0, std::nullopt);
        }
        _logger.info("Blob size with padding: %zu", size);
    }
    _logger.info("Write blob to stream successfully.");
    return std::make_pair(size, std::nullopt);
}

const NetworkMetadata& DynamicGraph::get_metadata() const {
    return _metadata;
}

void DynamicGraph::update_network_name(std::string_view name) {
    _metadata.name = name;
}

CommandQueueDesc DynamicGraph::get_command_queue_desc() const {
    std::lock_guard<std::mutex> lock(_commandQueueDescMutex);
    return _commandQueueDesc;
}

void DynamicGraph::set_workload_type(const ov::WorkloadType workloadType) {
    if (_zeroInitStruct == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(_commandQueueDescMutex);
    auto zeWorkloadType = zeroUtils::toZeQueueWorkloadType(workloadType);

    if (_commandQueue && zeWorkloadType.has_value()) {
        // When shared common queue is disabled, workload type is set per command queue.
        // Update the existing queue if it has already been created.
        _commandQueue->setWorkloadType(zeWorkloadType.value());
        _workloadType = workloadType;

        return;
    }

    if (_commandQueueDesc.workload() == zeWorkloadType) {
        return;
    }
    _commandQueueDesc.set_workload(zeWorkloadType);
}

void DynamicGraph::set_model_priority(const ov::hint::Priority modelPriority) {
    if (_zeroInitStruct == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(_commandQueueDescMutex);
    auto zeModelPriority = zeroUtils::toZeQueuePriority(modelPriority);
    if (_commandQueueDesc.priority() == zeModelPriority) {
        return;
    }
    _commandQueueDesc.set_priority(zeModelPriority);

    if (_commandQueue) {
        // When shared common queue is disabled, workload type is set per command queue.
        // Recreate the queue with the new priority while preserving the current workload type.
        if (_workloadType.has_value()) {
            auto zeWorkloadType = zeroUtils::toZeQueueWorkloadType(_workloadType.value());
            _commandQueueDesc.set_workload(zeWorkloadType);
            _workloadType = std::nullopt;  // Clear the cached workload type after applying it to the new queue
        }

        _commandQueue = ZeroCmdQueuePool::getInstance().getCommandQueue(_zeroInitStruct, _commandQueueDesc);
    }
}

void* DynamicGraph::get_handle() const {
    return _engine;
}

void DynamicGraph::initialize_impl(const FilteredConfig& config) {
    _logger.debug("Graph initialize start");

    if (!_engineInitialized) {
        // initialize VM execution engine, metadata, input&output descriptors
        initialize_engine();
    }

    if (!_zeroInitStruct) {
        _logger.warning("Zero device is not available, skip graph initialize!");
        return;
    }

    _logger.debug("Graph initialize without graph handle");

    uint32_t commandQueueOptions = 0;
    if (config.has<TURBO>() && config.get<TURBO>()) {
        OPENVINO_ASSERT(_zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 0),
                        "Turbo is not supported by the current driver");
        _logger.debug("Set ZE_NPU_COMMAND_QUEUE_OPTION_TURBO in command queue options");
        commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
    }
    if (config.has<RUN_INFERENCES_SEQUENTIALLY>() && config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        OPENVINO_ASSERT(_zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1),
                        "Running inferences sequentially is not supported by the current driver");
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
            config.get<SHARED_COMMON_QUEUE>()};

        if (config.get<SHARED_COMMON_QUEUE>() == false) {
            // Keep it alive per compiled model when the shared common queue feature is disabled.
            _commandQueue = ZeroCmdQueuePool::getInstance().getCommandQueue(_zeroInitStruct, _commandQueueDesc);
        }
    }

    _logger.debug("Graph initialize finish");

    // To ensure that the initialization of the graph does not exit prematurely due to nullptrs
    _init_completed.store(true, std::memory_order_release);
}

bool DynamicGraph::release_blob(const FilteredConfig& config) {
    _logger.warning("Release blob is skipped, no handle for DynamicGraph");
    return false;
};

uint32_t DynamicGraph::get_unique_id() {
    return _uniqueId++;
}

void DynamicGraph::set_last_submitted_id(uint32_t id_index) {
    _lastSubmittedId = id_index;
}

uint32_t DynamicGraph::get_last_submitted_id() const {
    return _lastSubmittedId;
}

DynamicGraph::~DynamicGraph() {
    if (!_lastSubmittedEvent.empty()) {
        _lastSubmittedEvent.clear();
    }
    if (_engine != nullptr) {
        npuVMRuntimeDestroy(_engine);
        _engine = nullptr;
    }
}

std::optional<bool> DynamicGraph::is_profiling_blob() const {
    _logger.warning("Profiling is not supported for DynamicGraph");
    return std::nullopt;
}

std::optional<std::string_view> DynamicGraph::get_compatibility_descriptor() const {
    _logger.warning("Compatibility descriptor is not supported for DynamicGraph");
    return std::nullopt;
}

BlobType DynamicGraph::get_blob_type() const {
    return _blobType;
}

}  // namespace intel_npu
