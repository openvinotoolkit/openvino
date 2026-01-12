// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "irgraph.hpp"

#include <iostream>
#include <iterator>

#include "intel_npu/config/options.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace intel_npu {

void IRGraph::MemRefType::setArg(const void* arg) {
    basePtr = data = arg;
}

void IRGraph::MemRefType::setSize(const intel_npu::IODescriptor& desc) {
    // Note: check difference between shape from compiler and shape from IR.
    const auto& shape = desc.shapeFromCompiler.get_shape();
    sizes.resize(shape.size());
    strides.resize(shape.size());
    dimsCount = static_cast<uint32_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        sizes[i] = shape[i];
        strides[i] = 0;  // Strides are not set yet, will be updated
    }
}

void IRGraph::MemRefType::updateStride() {
    // Note: NCHW layout style
    uint64_t stride = 1;
    for (int64_t i = dimsCount - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= sizes[i];
    }
}

void IRGraph::MemRefType::alignWithHandle() {
    if (memRef == nullptr) {
        return;
    }

    if(npuMLIRRuntimeParseMemRef(
        memRef, &basePtr, &data, &offset, sizes.data(), strides.data(), &dimsCount) != NPU_MLIR_RUNTIME_RESULT_SUCCESS) {
        throw std::runtime_error("Failed to parse MemRef handle");
    }
}

void IRGraph::GraphArguments::setArgumentValue(uint32_t argi, const void* argv) {
    if (argi < _inputs.size()) {
        _logger.debug("setArgumentValue for index %d (input %d)", argi, argi);
        _inputs[argi].basePtr = _inputs[argi].data = const_cast<void*>(argv);
    } else {
        auto idx = argi - _inputs.size();
        _logger.debug("setArgumentValue for index %d (output %d)", argi, idx);
        if (idx < _outputs.size()) {
            _outputs[idx].basePtr = _outputs[idx].data = const_cast<void*>(argv);
        }
    }
}

void IRGraph::GraphArguments::setArgumentValueWithStrides(uint32_t argi,
                                                          const void* argv,
                                                          const std::vector<size_t>& strides) {
    _logger.debug("setArgumentValueWithStrides for index %d", argi);
    if (argi < _inputs.size()) {
        std::ostringstream oss;
        oss << _inputs[argi];
        _logger.debug("setArgumentValueWithStrides for index %d (input %d)", argi, argi);
        _logger.debug("Before change: %s", oss.str().c_str());
        _inputs[argi].basePtr = _inputs[argi].data = const_cast<void*>(argv);

        for (int64_t i = 0; i < _inputs[argi].dimsCount; i++) {
            _inputs[argi].strides[i] = strides[i];
        }

        oss.clear();
        oss.str("");
        oss << _inputs[argi];
        _logger.debug("After change: %s", oss.str().c_str());

    } else {
        auto idx = argi - _inputs.size();
        _logger.debug("setArgumentValueWithStrides for index %d (output %d)", argi, idx);
        if (idx < _outputs.size()) {
            std::ostringstream oss;
            oss << _outputs[idx];
            _logger.debug("Before change: %s", oss.str().c_str());
            _outputs[idx].basePtr = _outputs[idx].data = const_cast<void*>(argv);

            // size_t stridesSize = strides.size();
            for (int64_t i = 0; i < _outputs[idx].dimsCount; i++) {
                _outputs[idx].strides[i] = strides[i];
            }

            oss.clear();
            oss.str("");
            oss << _outputs[idx];
            _logger.debug("After change: %s", oss.str().c_str());
        }
    }
}
class IRGraphImpl : public IRGraph::Impl {
public:
    using MemRefType = IRGraph::MemRefType;

public:
    IRGraphImpl() : _logger("IRGraphImpl", Logger::global().level()) {}
    void initialize(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata) override;
    void createExecutionEngine(std::optional<ov::Tensor>& blob);
    void prepareMetadata(NetworkMetadata& metadata);
    void initializeIRGraphExecution(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata);
    void setArgumentValue(uint32_t argi, const void* argv) override;
    void setArgumentValueWithStrides(uint32_t argi, const void* argv, const std::vector<size_t>& strides) override;
    void initializeGraph(uint64_t command_queue_group_ordinal) override;
    uint64_t getNumSubgraphs() override {
        return _engineProperties.numOfSubGraphs;
    }
    void executeGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                      IRGraph::GraphArguments& args,
                      std::vector<ze_command_list_handle_t>& commandLists,
                      ze_command_queue_handle_t commandQueue,
                      ze_fence_handle_t inferenceFence,
                      ze_event_handle_t event,
                      ze_graph_profiling_pool_handle_t profiling) override;
    void getBinding(IRGraph::GraphArguments& binding) override;

    virtual ~IRGraphImpl() {
        destroy();
    }

    void destroy() {
        if (_engine != nullptr) {
            npuMLIRRuntimeDestroy(_engine);
            _engine = nullptr;
        }
    }

    void predictOutputShape(std::vector<MemRefType>& inputDescriptors,
                            std::vector<MemRefType>& outputDescriptors) override;

public:
    npu_mlir_runtime_handle_t _engine = nullptr;
    npu_mlir_runtime_properties_t _engineProperties;
    IRGraph::GraphArguments _binding;
    bool _initializedMLIR = false;
    Logger _logger;
};

//bool IRGraphImpl::_initializedMLIR = false;

void IRGraphImpl::initialize(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata) {
    if (!_initializedMLIR) {
        initializeIRGraphExecution(blob, metadata);
        _initializedMLIR = true;
    }

    _binding._inputs.resize(metadata.inputs.size());

    // dump output of _metadata
    _logger.debug("Dump metadata info from blob");
    _logger.debug("Metadata inputs: %d", metadata.inputs.size());
    for (const auto& input : metadata.inputs) {
        _logger.debug("Input compiler name: %s input node name: %s shapeFromCompiler: %s shapeFromIRModel: %s",
                      input.nameFromCompiler.c_str(),
                      input.nodeFriendlyName.c_str(),
                      input.shapeFromCompiler.to_string().c_str(),
                      input.shapeFromIRModel.has_value() ? input.shapeFromIRModel->to_string().c_str() : "N/A");
    }
    _logger.debug("Metadata outputs: %d", metadata.outputs.size());
    for (const auto& output : metadata.outputs) {
        _logger.debug("Output compiler name: %s output node name: %s shapeFromCompiler: %s shapeFromIRModel: %s",
                      output.nameFromCompiler.c_str(),
                      output.nodeFriendlyName.c_str(),
                      output.shapeFromCompiler.to_string().c_str(),
                      output.shapeFromIRModel.has_value() ? output.shapeFromIRModel->to_string().c_str() : "N/A");
    }

    _logger.debug("Dump MemRefType from initial metadata:");
    _logger.debug("Inputs:");
    auto& inputs = _binding._inputs;
    inputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        // Use size as placeholder of stride
        const auto& shape = metadata.inputs[i].shapeFromCompiler.get_shape();
        std::vector<int64_t> shapeVec(shape.begin(), shape.end());
        inputs[i] = MemRefType(nullptr, nullptr, 0, shapeVec, shapeVec, shapeVec.size());
        // Calc real stride
        inputs[i].updateStride();
        std::ostringstream oss;
        oss << inputs[i];
        _logger.debug("MemRefType for input %d : %s", i, oss.str().c_str());
    }

    _logger.debug("Outputs:");
    _binding._outputs.resize(metadata.outputs.size());
    auto& outputs = _binding._outputs;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& shape = metadata.outputs[i].shapeFromCompiler.get_shape();
        std::vector<int64_t> shapeVec(shape.begin(), shape.end());
        outputs[i] = MemRefType(nullptr, nullptr, 0, shapeVec, shapeVec, shapeVec.size());
        outputs[i].updateStride();
        std::ostringstream oss;
        oss << outputs[i];
        _logger.debug("MemRefType for output %d : %s", i, oss.str().c_str());
    }
}

void IRGraphImpl::createExecutionEngine(std::optional<ov::Tensor>& blob) {
    _npu_mlir_runtime_blob_desc_t blobDesc;
    blobDesc.pInput = reinterpret_cast<const uint8_t*>(blob.value().data());
    blobDesc.inputSize = blob.value().get_byte_size();

    if (npuMLIRRuntimeCreate(&blobDesc, &_engine, &_engineProperties) != NPU_MLIR_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to create MLIR runtime engine");
    }
}

/**
 * @brief Extracts the I/O metadata from Level Zero specific structures and converts them into OpenVINO specific
 * ones.
 *
 * @param arg The main Level Zero structure from which most metadata will be extracted.
 * @param metadata The secondary Level Zero structure from which metadata will be extracted. More specifically, the
 * argument is used for populating "shapeFromIRModel". Not providing this argument will lead to an empty value for
 * the referenced attribute.
 * @returns A descriptor object containing the metadata converted in OpenVINO specific structures.
 */
static IODescriptor getIODescriptor(const ze_graph_argument_properties_3_t& arg,
                                    const std::optional<ze_graph_argument_metadata_t>& metadata) {
    auto logger = Logger::global().clone("getIODescriptor");
    ov::element::Type_t precision = zeroUtils::toOVElementType(arg.devicePrecision);
    ov::Shape shapeFromCompiler;
    ov::PartialShape shapeFromIRModel;
    std::unordered_set<std::string> outputTensorNames;

    for (uint32_t id = 0; id < arg.associated_tensor_names_count; id++) {
        outputTensorNames.insert(arg.associated_tensor_names[id]);
    }
    for (uint32_t id = 0; id < arg.dims_count; id++) {
        shapeFromCompiler.push_back(arg.dims[id]);
    }
    if (metadata.has_value()) {
        const auto dynamicDim = std::numeric_limits<uint64_t>::max();
        shapeFromIRModel.reserve(metadata->shape_size);
        for (uint32_t id = 0; id < metadata->shape_size; id++) {
            if (metadata->shape[id] != dynamicDim) {
                shapeFromIRModel.push_back(metadata->shape[id]);
            } else {
                // lower bound is ignored, so we set it to 1 just to satisfy the Dimension constructor,
                // upper bound is set to the value from shapeFromCompiler as it is filled with upper bounds
                // in case of dynamic dimensions
                if (id == utils::BATCH_AXIS && shapeFromCompiler[id] == utils::DEFAULT_BATCH_SIZE) {
                    logger.info("Ignore dynamic batch size upper limit, but keep the dimension dynamic as a metadata "
                                "from compiler has been lost.");
                    // We need to kepp batch dimension dynamic
                    shapeFromIRModel.push_back(ov::Dimension(1, dynamicDim));
                } else {
                    shapeFromIRModel.push_back(ov::Dimension(1, shapeFromCompiler[id]));
                    // shapeFromIRModel.push_back(-1);
                }
            }
        }
    }

    // Flags will be used instead of indices for informing the type of the current entry
    std::string nameFromCompiler = arg.name;
    const bool isInput = (arg.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT);
    bool isStateInput = false;
    bool isStateOutput = false;
    bool isShapeTensor = false;
    bool isInitInputWeights = false;
    bool isInitOutputWeights = false;
    bool isMainInputWeights = false;
    if (isInput && isStateInputName(nameFromCompiler)) {
        nameFromCompiler = nameFromCompiler.substr(READVALUE_PREFIX.length());
        isStateInput = true;
    } else if (!isInput && isStateOutputName(nameFromCompiler)) {
        nameFromCompiler = nameFromCompiler.substr(ASSIGN_PREFIX.length());
        isStateOutput = true;
    } else if (isShapeTensorName(nameFromCompiler)) {
        nameFromCompiler = nameFromCompiler.substr(SHAPE_TENSOR_PREFIX.length());
        isShapeTensor = true;
    } else if (isInput && isInitInputWeightsName(nameFromCompiler)) {
        nameFromCompiler = nameFromCompiler.substr(INIT_INPUT_WEIGHTS_PREFIX.length());
        isInitInputWeights = true;
    } else if (!isInput && isInitOutputWeightsName(nameFromCompiler)) {
        nameFromCompiler = nameFromCompiler.substr(INIT_OUTPUT_WEIGHTS_PREFIX.length());
        isInitOutputWeights = true;
    } else if (isInput && isMainInputWeightsName(nameFromCompiler)) {
        nameFromCompiler = nameFromCompiler.substr(MAIN_INPUT_WEIGHTS_PREFIX.length());
        isMainInputWeights = true;
    }

    return {std::move(nameFromCompiler),
            precision,
            shapeFromCompiler,
            isStateInput,
            isStateOutput,
            isShapeTensor,
            isInitInputWeights,
            isInitOutputWeights,
            isMainInputWeights,
            std::nullopt,
            arg.debug_friendly_name,
            std::move(outputTensorNames),
            metadata.has_value() ? std::optional(shapeFromIRModel) : std::nullopt};
}

void IRGraphImpl::prepareMetadata(NetworkMetadata& metadata) {
    metadata.inputs.clear();
    metadata.outputs.clear();
    for (uint32_t i = 0; i < _engineProperties.numOfGraphArgs; ++i) {
        // TODO: follow graph ext to support Optional metadata for weightless model
        ze_graph_argument_properties_3_t arg;
        ze_graph_argument_metadata_t meta;
        std::vector<int64_t> upperBound;
        upperBound.reserve(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE);
        if (npuMLIRRuntimeGetMetadata(_engine, i, &arg, &meta, upperBound.data()) != NPU_MLIR_RUNTIME_RESULT_SUCCESS) {
            OPENVINO_THROW("Failed to get MLIR runtime metadata");
        }
        IODescriptor ioDesc = getIODescriptor(arg, meta);
        // TODO: Once runtime returns right value, can remove change on index and layout
        ioDesc.indexUsedByDriver = i;
        ioDesc.supportsStridedLayout = true;
        switch (arg.type) {
        case ZE_GRAPH_ARGUMENT_TYPE_INPUT: {
            metadata.inputs.push_back(ioDesc);
        } break;
        case ZE_GRAPH_ARGUMENT_TYPE_OUTPUT: {
            metadata.outputs.push_back(ioDesc);
        } break;
        default: {
            OPENVINO_THROW("Invalid ze_graph_argument_type_t found in ze_graph_argument_properties_3_t object: ",
                           arg.type);
        }
        }
    }
    metadata.bindRelatedDescriptors();
}

void IRGraphImpl::getBinding(IRGraph::GraphArguments& binding) {
    binding = _binding;
}

void IRGraphImpl::initializeIRGraphExecution(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata) {
    createExecutionEngine(blob);
    prepareMetadata(metadata);

    _logger.debug("num of subgraphs: %d inputs: %d outputs: %d",
                  _engineProperties.numOfSubGraphs,
                  metadata.inputs.size(),
                  metadata.outputs.size());
}

void IRGraphImpl::setArgumentValue(uint32_t argi, const void* argv) {
    auto inputs = _binding._inputs;
    if (argi < inputs.size()) {
        _logger.debug("setArgumentValue for index %d (input %d)", argi, argi);
        inputs[argi].basePtr = inputs[argi].data = const_cast<void*>(argv);
    } else {
        auto outputs = _binding._outputs;
        auto idx = argi - inputs.size();
        _logger.debug("setArgumentValue for index %d (output %d)", argi, idx);
        if (idx < outputs.size()) {
            outputs[idx].basePtr = outputs[idx].data = const_cast<void*>(argv);
        }
    }
}

void IRGraphImpl::setArgumentValueWithStrides(uint32_t argi, const void* argv, const std::vector<size_t>& strides) {
    _logger.debug("setArgumentProperty for index %d", argi);
    auto& inputs = _binding._inputs;
    if (argi < inputs.size()) {
        std::ostringstream oss;
        oss << inputs[argi];
        _logger.debug("setArgumentProperty for index %d (input %d)", argi, argi);
        _logger.debug("Before change: %s", oss.str().c_str());
        inputs[argi].basePtr = inputs[argi].data = const_cast<void*>(argv);

        // size_t stridesSize = strides.size();
        for (int64_t i = 0; i < inputs[argi].dimsCount; i++) {
            inputs[argi].strides[i] = strides[i];
        }

        oss.clear();
        oss.str("");
        oss << inputs[argi];
        _logger.debug("After change: %s", oss.str().c_str());

    } else {
        auto& outputs = _binding._outputs;
        auto idx = argi - inputs.size();
        _logger.debug("setArgumentValue for index %d (output %d)", argi, idx);
        if (idx < outputs.size()) {
            std::ostringstream oss;
            oss << outputs[idx];
            _logger.debug("Before change: %s", oss.str().c_str());
            outputs[idx].basePtr = outputs[idx].data = const_cast<void*>(argv);

            // size_t stridesSize = strides.size();
            for (int64_t i = 0; i < outputs[idx].dimsCount; i++) {
                outputs[idx].strides[i] = strides[i];
            }

            oss.clear();
            oss.str("");
            oss << outputs[idx];
            _logger.debug("After change: %s", oss.str().c_str());
        }
    }
}

void IRGraphImpl::initializeGraph(uint64_t ordinal) {
    // TODO
}

void IRGraphImpl::executeGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                               IRGraph::GraphArguments& args,
                               std::vector<ze_command_list_handle_t>& commandLists,
                               ze_command_queue_handle_t commandQueue,
                               ze_fence_handle_t fence,
                               ze_event_handle_t event,
                               ze_graph_profiling_pool_handle_t profiling) {
    npu_mlir_runtime_execute_params_t* params = &args._executeParams;

    for (auto& in : args._inputs) {
        in.UpdateMemRefHandleStatus();
        args._inputMemRefs.push_back(in.memRef);
    }
    for (auto& out : args._outputs) {
        out.UpdateMemRefHandleStatus();
        args._outputMemRefs.push_back(out.memRef);
    }

    params->pInputs = args._inputMemRefs.data();
    params->numOfInputs = static_cast<uint32_t>(args._inputMemRefs.size());
    params->pOutputs = args._outputMemRefs.data();
    params->numOfOutputs = static_cast<uint32_t>(args._outputMemRefs.size());
    params->ctx = zeroInitStruct->getContext();
    params->device = zeroInitStruct->getDevice();
    params->graphDdiTableExt = zeroInitStruct->getGraphDdiTable().getImpl();
    params->commandLists = commandLists.data();
    params->numCommandLists = static_cast<uint64_t>(commandLists.size());
    params->commandQueue = commandQueue;
    params->inferenceFence = fence;
    params->event = event;

    if (npuMLIRRuntimeExecute(_engine, params) != NPU_MLIR_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to execute MLIR runtime engine");
    }
}

void IRGraphImpl::predictOutputShape(std::vector<MemRefType>& inputDescriptors,
                                     std::vector<MemRefType>& outputDescriptors) {
    std::vector<npu_mlir_runtime_mem_ref_handle_t> inputs;
    for (auto& in : inputDescriptors) {
        in.UpdateMemRefHandleStatus();
        inputs.push_back(in.memRef);
    }
    std::vector<npu_mlir_runtime_mem_ref_handle_t> outputs;
    for (auto& out : outputDescriptors) {
        out.UpdateMemRefHandleStatus();
        outputs.push_back(out.memRef);
    }

    npu_mlir_runtime_predict_output_shape_params_t params;
    params.pInputs = inputs.data();
    params.numOfInputs = static_cast<uint32_t>(inputs.size());
    params.pOutputs = outputs.data();
    params.numOfOutputs = static_cast<uint32_t>(outputs.size());

    if (npuMLIRRuntimePredictOutputShape(_engine, &params) != NPU_MLIR_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to execute MLIR runtime engine");
    } else {
        // Update MemRefType with the info from handle
        // for (auto& in : inputDescriptors) {
        //     in.alignWithHandle();
        // }
        for (auto& out : outputDescriptors) {
            out.alignWithHandle();
        }
        _logger.debug("Output shape prediction is done successfully.");
    }
}

IRGraph::IRGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                 std::optional<ov::Tensor> blob,
                 bool blobAllocatedByPlugin,
                 const Config& config,
                 const ov::SoPtr<ICompiler>& compiler)
    : _zeroInitStruct(zeroInitStruct),
      _blob(std::move(blob)),
      _blobAllocatedByPlugin(blobAllocatedByPlugin),
      _compiler(compiler),
      _logger("Graph", config.get<LOG_LEVEL>()) {
    if (!config.get<CREATE_EXECUTOR>() || config.get<DEFER_WEIGHTS_LOAD>()) {
        _logger.info("Graph initialize is deferred from the \"Graph\" constructor");
        return;
    }

    _impl = std::make_unique<IRGraphImpl>();

    // initialize MLIR execution engine, metadata, input&output descriptors
    _impl->initialize(_blob, _metadata);

    _num_of_subgraphs = _impl->getNumSubgraphs();

    initialize(config);
}

std::pair<uint64_t, std::optional<std::vector<uint64_t>>> IRGraph::export_blob(std::ostream& stream) const {
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

const NetworkMetadata& IRGraph::get_metadata() const {
    return _metadata;
}

void IRGraph::update_network_name(std::string_view name) {
    _metadata.name = name;
}

const std::shared_ptr<CommandQueue>& IRGraph::get_command_queue() const {
    return _commandQueue;
}

uint32_t IRGraph::get_command_queue_group_ordinal() const {
    return _commandQueueGroupOrdinal;
}

void IRGraph::set_workload_type(const ov::WorkloadType workloadType) const {
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

std::vector<ov::ProfilingInfo> IRGraph::process_profiling_output(const std::vector<uint8_t>& profData,
                                                                 const Config& config) const {
    if (_compiler == nullptr) {
        OPENVINO_THROW("Profiling post-processing is not supported.");
    }

    std::vector<uint8_t> blob(_blob->get_byte_size());
    blob.assign(reinterpret_cast<const uint8_t*>(_blob->data()),
                reinterpret_cast<const uint8_t*>(_blob->data()) + _blob->get_byte_size());
    return _compiler->process_profiling_output(profData, blob, config);
}

void IRGraph::set_argument_value(uint32_t argi, const void* argv) const {
    if (_impl == nullptr) {
        _logger.warning("Graph handle is null, dynamic pipeline to handle set_argument_value");
        return;
    }

    _impl->setArgumentValue(argi, argv);
}

void IRGraph::set_argument_value_with_strides(uint32_t id, const void* data, const std::vector<size_t>& strides) const {
    if (_impl == nullptr) {
        _logger.warning("Graph handle is null, dynamic pipeline to handle set_argument_value");
        return;
    }

    _impl->setArgumentValueWithStrides(id, data, strides);
}

ze_graph_handle_t IRGraph::get_handle() const {
    _logger.warning("IRGraph does not support get_handle() method.");
    return nullptr;
}

void IRGraph::initialize(const Config& config) {
    _logger.debug("Graph initialize start");

    if(!_zeroInitStruct) {
        _logger.warning("Zero device is not available, skip graph initialize!");
    }

    if (_commandQueue = nullptr) {
        _logger.debug("Graph initialize without graph handle");

        _commandQueueGroupOrdinal =
            zeroUtils::findCommandQueueGroupOrdinal(_zeroInitStruct->getDevice(),
                                                    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);

        uint32_t commandQueueOptions = 0;

        if (config.has<TURBO>() && config.get<TURBO>()) {
            if (_zeroInitStruct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 0)) {
                OPENVINO_THROW("Turbo is not supported by the current driver");
            }
            commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
        }

        if (_zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1) &&
            config.has<RUN_INFERENCES_SEQUENTIALLY>() && config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
            commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC;
        }

        _commandQueue = std::make_shared<CommandQueue>(_zeroInitStruct,
                                                       zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
                                                       _commandQueueGroupOrdinal,
                                                       commandQueueOptions);

        if (config.has<WORKLOAD_TYPE>()) {
            set_workload_type(config.get<WORKLOAD_TYPE>());
        }

        _impl->initializeGraph(_commandQueueGroupOrdinal);

        _logger.debug("Graph initialize finish");

        //  We are allowed to release the original blob because weights were loaded in NPU memory during
        //  _zeGraphExt->initializeGraph(). The driver will not access the original blob from this moment on, so we are
        //  releasing it here to avoid unnecessary memory usage.
        //_blobIsReleased = release_blob(config);

        _batchSize = determine_batch_size();

        if (_zeroInitStruct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
            config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
            auto numberOfCommandLists = _batchSize.has_value() ? *_batchSize : 1;

            _lastSubmittedEvent.resize(numberOfCommandLists);
        }
        return;
    }

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

    // TODO
    // invoke for graph intialization
    // engine->invoke("initialization")

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

    // To ensure that the initialization of the graph does not exit prematurely due to nullptrs
    _init_completed = true;
}

bool IRGraph::release_blob(const Config& config) {
    _logger.warning("Release blob is skipped, no handle for IRGraph");
    // if (!_blobAllocatedByPlugin) {
    //     return false;
    // }

    // if (_blob == std::nullopt || _zeroInitStruct->getGraphDdiTable().version() < ZE_GRAPH_EXT_VERSION_1_8 ||
    //     config.get<PERF_COUNT>()) {
    //     return false;
    // }

    // ze_graph_properties_2_t properties = {};
    // properties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    // _zeroInitStruct->getGraphDdiTable().pfnGetProperties2(_handle, &properties);

    // if (~properties.initStageRequired & ZE_GRAPH_STAGE_INITIALIZE) {
    //     return false;
    // }

    // _blob = std::nullopt;
    // _logger.debug("Blob is released");

    return false;
};

void IRGraph::set_last_submitted_event(const std::shared_ptr<Event>& event, size_t indexOfCommandList) {
    _lastSubmittedEvent[indexOfCommandList] = event;
}

const std::shared_ptr<Event>& IRGraph::get_last_submitted_event(size_t indexOfCommandList) const {
    return _lastSubmittedEvent[indexOfCommandList];
}

void IRGraph::resize_last_submitted_event(size_t batch) {
    _lastSubmittedEvent.resize(batch);
}

void IRGraph::set_batch_size(std::size_t batch) {
    _batchSize = batch;
}

uint32_t IRGraph::get_unique_id() {
    return _uniqueId++;
}

void IRGraph::set_last_submitted_id(uint32_t id_index) {
    _lastSubmittedId = id_index;
}

uint32_t IRGraph::get_last_submitted_id() const {
    return _lastSubmittedId;
}

std::optional<size_t> IRGraph::determine_batch_size() {
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

const std::optional<std::size_t> IRGraph::get_batch_size() const {
    return _batchSize;
}

IRGraph::~IRGraph() {
    // make sure all the context-dependent components are destroyed before the zero context is destroyed
    // if (_handle != nullptr) {
    //     _handle = nullptr;
    // }

    if (!_lastSubmittedEvent.empty()) {
        _lastSubmittedEvent.clear();
    }

    if (_commandQueue != nullptr) {
        _commandQueue.reset();
    }
}

void IRGraph::execute(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                      IRGraph::GraphArguments& args,
                      std::vector<ze_command_list_handle_t>& commandLists,
                      ze_command_queue_handle_t commandQueue,
                      ze_fence_handle_t inferenceFence,
                      ze_event_handle_t event,
                      ze_graph_profiling_pool_handle_t profiling) {
    auto impl = reinterpret_cast<IRGraphImpl*>(_impl.get());

    if (impl == nullptr)
        return;

    impl->executeGraph(zeroInitStruct, args, commandLists, commandQueue, inferenceFence, event, profiling);
}

void IRGraph::getBinding(GraphArguments& args) {
    auto impl = reinterpret_cast<IRGraphImpl*>(_impl.get());

    if (impl == nullptr)
        return;

    impl->getBinding(args);
}

uint64_t IRGraph::get_num_subgraphs() const {
    return _num_of_subgraphs;
}

void IRGraph::predict_output_shape(std::vector<MemRefType>& inputDescriptors,
                                   std::vector<MemRefType>& outputDescriptors) {
    auto impl = reinterpret_cast<IRGraphImpl*>(_impl.get());

    if (impl == nullptr)
        return;

    impl->predictOutputShape(inputDescriptors, outputDescriptors);
}

}  // namespace intel_npu
