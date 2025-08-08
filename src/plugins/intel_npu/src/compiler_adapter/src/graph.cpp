// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph.hpp"

#include <iterator>

#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace {
constexpr std::size_t BATCH_AXIS = 0;
constexpr std::size_t DEFAULT_BATCH_SIZE = 1;

std::optional<size_t> extract_batch_impl(const ov::Shape& shape,
                                         std::optional<ov::PartialShape> partial_shape,
                                         size_t index) {
    if (partial_shape.has_value()) {
        if (partial_shape.value()[index].is_dynamic()) {
            return shape[index];
        }
        return std::nullopt;
    }
    return shape[index];
}

std::optional<size_t> extract_batch(const ze_graph_argument_layout_t& nLayout,
                                    const ov::Shape& shape,
                                    std::optional<ov::PartialShape> partial_shape = std::nullopt) {
    if (nLayout == ZE_GRAPH_ARGUMENT_LAYOUT_NCHW || nLayout == ZE_GRAPH_ARGUMENT_LAYOUT_NHWC ||
        nLayout == ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW || nLayout == ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC ||
        nLayout == ZE_GRAPH_ARGUMENT_LAYOUT_NC) {
        return extract_batch_impl(shape, partial_shape, 0);
    } else if (nLayout == ZE_GRAPH_ARGUMENT_LAYOUT_CN) {
        return extract_batch_impl(shape, partial_shape, 1);
    }

    return std::nullopt;
}

}  // namespace

namespace intel_npu {

Graph::Graph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
             const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
             const GraphDescriptor& graphDesc,
             NetworkMetadata metadata,
             std::optional<ov::Tensor> blob,
             const Config& config,
             const bool blobIsPersistent,
             const ov::SoPtr<ICompiler>& compiler,
             const bool calledFromWeightlessGraph)
    : IGraph(),
      _zeGraphExt(zeGraphExt),
      _zeroInitStruct(zeroInitStruct),
      _graphDesc(graphDesc),
      _metadata(std::move(metadata)),
      _blob(std::move(blob)),
      _blobIsPersistent(blobIsPersistent),
      _compiler(compiler),
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

const std::vector<ArgumentDescriptor>& Graph::get_input_descriptors() const {
    return _inputDescriptors;
}

const std::vector<ArgumentDescriptor>& Graph::get_output_descriptors() const {
    return _outputDescriptors;
}

const std::shared_ptr<CommandQueue>& Graph::get_command_queue() const {
    return _commandQueue;
}

uint32_t Graph::get_command_queue_group_ordinal() const {
    return _commandQueueGroupOrdinal;
}

void Graph::set_workload_type(const ov::WorkloadType workloadType) const {
    if (_commandQueue == nullptr) {
        return;
    }

    ze_command_queue_workload_type_t zeWorkloadType;
    switch (workloadType) {
    case ov::WorkloadType::DEFAULT:
        zeWorkloadType = ze_command_queue_workload_type_t::ZE_WORKLOAD_TYPE_DEFAULT;
        break;
    case ov::WorkloadType::EFFICIENT:
        zeWorkloadType = ze_command_queue_workload_type_t::ZE_WORKLOAD_TYPE_BACKGROUND;
        break;
    default:
        OPENVINO_THROW("Unknown value for WorkloadType!");
    }

    _commandQueue->setWorkloadType(zeWorkloadType);
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

    if (_blob ==
        std::nullopt) {  // when compiling the model using Compiler in Driver, the blob is handled by the driver
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

    size_t size = utils::align_size_to_standarg_page_size(blobSize);
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

std::vector<ov::ProfilingInfo> Graph::process_profiling_output(const std::vector<uint8_t>& profData,
                                                               const Config& config) const {
    if (_compiler == nullptr) {
        OPENVINO_THROW("Profiling post-processing is not supported.");
    }

    std::vector<uint8_t> blob(_blob->get_byte_size());
    blob.assign(reinterpret_cast<const uint8_t*>(_blob->data()),
                reinterpret_cast<const uint8_t*>(_blob->data()) + _blob->get_byte_size());
    return _compiler->process_profiling_output(profData, blob, config);
}

void Graph::set_argument_value(uint32_t argi, const void* argv) const {
    if (_zeGraphExt == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }
    _zeGraphExt->setGraphArgumentValue(_graphDesc, argi, argv);
}

void Graph::initialize(const Config& config) {
    _logger.debug("Graph initialize start");

    if (_zeGraphExt == nullptr || _graphDesc._handle == nullptr) {
        return;
    }

    _logger.debug("performing pfnGetProperties");
    ze_graph_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    auto result = _zeroInitStruct->getGraphDdiTable().pfnGetProperties(_graphDesc._handle, &props);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetProperties", result, _zeroInitStruct->getGraphDdiTable());

    _logger.debug("performing pfnGetArgumentProperties3");
    for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
        ze_graph_argument_properties_3_t arg3{};
        arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        auto result = _zeroInitStruct->getGraphDdiTable().pfnGetArgumentProperties3(_graphDesc._handle, index, &arg3);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetArgumentProperties3", result, _zeroInitStruct->getGraphDdiTable());

        if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            _inputDescriptors.push_back(ArgumentDescriptor{arg3, index});
            _logger.debug("got pfnGetArgumentProperties3 for input: %s", _inputDescriptors.back().to_string().c_str());
        } else {
            _outputDescriptors.push_back(ArgumentDescriptor{arg3, index});
            _logger.debug("got pfnGetArgumentProperties3 for output: %s",
                          _outputDescriptors.back().to_string().c_str());
        }
    }

    _inputDescriptors.shrink_to_fit();
    _outputDescriptors.shrink_to_fit();

    _commandQueueGroupOrdinal = zeroUtils::findCommandQueueGroupOrdinal(_zeroInitStruct->getDevice(),
                                                                        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);

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

    _commandQueue = std::make_shared<CommandQueue>(_zeroInitStruct,
                                                   zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
                                                   _commandQueueGroupOrdinal,
                                                   commandQueueOptions);

    if (config.has<WORKLOAD_TYPE>()) {
        set_workload_type(config.get<WORKLOAD_TYPE>());
    }

    _zeGraphExt->initializeGraph(_graphDesc, _commandQueueGroupOrdinal);
    _logger.debug("Graph initialize finish");

    //  We are allowed to release the original blob because weights were loaded in NPU memory during
    //  _zeGraphExt->initializeGraph(). The driver will not access the original blob from this moment on, so we are
    //  releasing it here to avoid unnecessary memory usage.
    _blobIsReleased = release_blob(config);

    _batchSize = determine_batch_size();

    if (_zeroInitStruct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
        config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        auto numberOfCommandLists = _batchSize.has_value() ? *_batchSize : 1;

        _lastSubmittedEvent.resize(numberOfCommandLists);
    }
}

bool Graph::release_blob(const Config& config) {
    if (_graphDesc._data || _blobIsPersistent || _blob == std::nullopt ||
        _zeroInitStruct->getGraphDdiTable().version() < ZE_MAKE_VERSION(1, 8) || config.get<PERF_COUNT>()) {
        return false;
    }

    ze_graph_properties_2_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
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

std::optional<size_t> Graph::determine_dynamic_batch_size(const std::shared_ptr<ov::ITensor>& tensor,
                                                          const std::optional<size_t> batchSize,
                                                          const std::optional<size_t> index,
                                                          const bool isInput) const {
    if (tensor == nullptr && !batchSize.has_value()) {
        _logger.debug("Batch size is not provided and tensor is null, returning std::nullopt");
        return std::nullopt;
    }

    if (!_metadata.inputs.at(0).shapeFromIRModel.has_value() ||
        !_metadata.outputs.at(0).shapeFromIRModel.value().is_dynamic() || !index.has_value()) {
        return std::nullopt;
    }

    if (batchSize.has_value()) {
        return batchSize.value();
    }

    if (tensor == nullptr || tensor->get_shape().empty()) {
        return std::nullopt;
    }

    auto& desc = isInput ? _inputDescriptors.at(*index) : _outputDescriptors.at(*index);
    const ov::PartialShape& firstPartialShape = *_metadata.inputs.at(0).shapeFromIRModel;

    return extract_batch(desc.info.networkLayout, tensor->get_shape(), firstPartialShape);
}

std::optional<size_t> Graph::determine_batch_size() {
    if (!_metadata.outputs.at(0).shapeFromIRModel.has_value()) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    const ov::PartialShape& firstOutputShape = *_metadata.outputs.at(0).shapeFromIRModel;
    if (firstOutputShape.is_dynamic() || firstOutputShape.rank().get_length() == 0) {
        return std::nullopt;
    }

    const size_t candidateBatchSize = firstOutputShape[BATCH_AXIS].get_max_length();
    if (candidateBatchSize == 0 || candidateBatchSize == DEFAULT_BATCH_SIZE) {
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
                *shapeFromCompiler.begin() != DEFAULT_BATCH_SIZE) {
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

Graph::~Graph() {
    // make sure all the context-dependent components are destroyed before the zero context is destroyed
    if (_zeGraphExt != nullptr) {
        _zeGraphExt->destroyGraph(_graphDesc);
    }

    if (!_lastSubmittedEvent.empty()) {
        _lastSubmittedEvent.clear();
    }

    if (_commandQueue != nullptr) {
        _commandQueue.reset();
    }
}

}  // namespace intel_npu
