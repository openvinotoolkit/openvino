// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_graph.hpp"

#include <iostream>
#include <iterator>

#include "compiler_impl.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_cmd_queue_pool.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace intel_npu {

class DynamicGraphImpl : public DynamicGraph::Impl {
public:
    using MemRefType = DynamicGraph::MemRefType;

public:
    DynamicGraphImpl() : _engineProperties{}, _logger("DynamicGraphImpl", Logger::global().level()) {}
    void initialize(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata) override;
    void createExecutionEngine(std::optional<ov::Tensor>& blob);
    void prepareMetadata(NetworkMetadata& metadata);
    void initializeDynamicGraphExecution(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata);
    void setArgumentValue(uint32_t argi, const void* argv) override;
    void setArgumentValueWithStrides(uint32_t argi, const void* argv, const std::vector<size_t>& strides) override;
    uint64_t getNumSubgraphs() override {
        return _engineProperties.numOfSubGraphs;
    }
    void executeGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                      DynamicGraph::GraphArguments& args,
                      std::vector<ze_command_list_handle_t>& commandLists,
                      ze_command_queue_handle_t commandQueue,
                      ze_fence_handle_t inferenceFence,
                      ze_event_handle_t event,
                      ze_graph_profiling_pool_handle_t profiling) override;
    void getBinding(DynamicGraph::GraphArguments& binding) override;

    virtual ~DynamicGraphImpl() {
        destroy();
    }

    void destroy() {
        if (_engine != nullptr) {
            npuVMRuntimeDestroy(_engine);
            _engine = nullptr;
        }
    }

    void predictOutputShape(std::vector<DynamicGraph::MemRefType>& inputDescriptors,
                            std::vector<DynamicGraph::MemRefType>& outputDescriptors) override;

public:
    npu_vm_runtime_handle_t _engine = nullptr;
    npu_vm_runtime_properties_t _engineProperties;
    DynamicGraph::GraphArguments _binding;
    bool _initialized = false;
    Logger _logger;
};

void DynamicGraphImpl::initialize(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata) {
    if (!_initialized) {
        initializeDynamicGraphExecution(blob, metadata);
        _initialized = true;
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

    auto& inputs = _binding._inputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
        // Use size as placeholder of stride
        const auto& shape = metadata.inputs[i].shapeFromCompiler.get_shape();
        std::vector<int64_t> shapeVec(shape.begin(), shape.end());
        inputs[i] = MemRefType(nullptr, nullptr, 0, shapeVec, shapeVec, shapeVec.size());
        // Calc real stride
        inputs[i].updateStride();
    }

    _binding._outputs.resize(metadata.outputs.size());
    auto& outputs = _binding._outputs;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& shape = metadata.outputs[i].shapeFromCompiler.get_shape();
        std::vector<int64_t> shapeVec(shape.begin(), shape.end());
        outputs[i] = MemRefType(nullptr, nullptr, 0, shapeVec, shapeVec, shapeVec.size());
        outputs[i].updateStride();
    }
}

void DynamicGraphImpl::createExecutionEngine(std::optional<ov::Tensor>& blob) {
    npu_vm_runtime_blob_desc_t blobDesc;
    blobDesc.pInput = reinterpret_cast<const uint8_t*>(blob.value().data());
    blobDesc.inputSize = blob.value().get_byte_size();

    if (npuVMRuntimeCreate(&blobDesc, &_engine, &_engineProperties) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to create VM runtime engine");
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

void DynamicGraphImpl::prepareMetadata(NetworkMetadata& metadata) {
    metadata.inputs.clear();
    metadata.outputs.clear();
    for (uint32_t i = 0; i < _engineProperties.numOfGraphArgs; ++i) {
        // TODO: follow graph ext to support Optional metadata for weightless model
        ze_graph_argument_properties_3_t arg;
        ze_graph_argument_metadata_t meta;
        std::vector<int64_t> upperBound;
        upperBound.reserve(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE);
        if (npuVMRuntimeGetMetadata(_engine, i, &arg, &meta, upperBound.data()) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
            OPENVINO_THROW("Failed to get VM runtime metadata");
        }
        IODescriptor ioDesc = getIODescriptor(arg, meta);
        // TODO: Once runtime returns right value, can remove change on index and layout
        ioDesc.indexUsedByDriver = i;
        ioDesc.supportsStridedLayout = true;
        switch (arg.type) {
        case ZE_GRAPH_ARGUMENT_TYPE_INPUT: {
            metadata.inputs.push_back(std::move(ioDesc));
        } break;
        case ZE_GRAPH_ARGUMENT_TYPE_OUTPUT: {
            metadata.outputs.push_back(std::move(ioDesc));
        } break;
        default: {
            OPENVINO_THROW("Invalid ze_graph_argument_type_t found in ze_graph_argument_properties_3_t object: ",
                           arg.type);
        }
        }
    }
    metadata.bindRelatedDescriptors();
}

void DynamicGraphImpl::getBinding(DynamicGraph::GraphArguments& binding) {
    binding = _binding;
}

void DynamicGraphImpl::initializeDynamicGraphExecution(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata) {
    createExecutionEngine(blob);
    prepareMetadata(metadata);

    _logger.debug("num of subgraphs: %d inputs: %d outputs: %d",
                  _engineProperties.numOfSubGraphs,
                  metadata.inputs.size(),
                  metadata.outputs.size());
}

void DynamicGraphImpl::setArgumentValue(uint32_t argi, const void* argv) {
    auto& inputs = _binding._inputs;
    if (argi < inputs.size()) {
        _logger.debug("setArgumentValue for index %d (input %d)", argi, argi);
        inputs[argi].setArg(argv);
    } else {
        auto& outputs = _binding._outputs;
        auto idx = argi - inputs.size();
        _logger.debug("setArgumentValue for index %d (output %d)", argi, idx);
        if (idx < outputs.size()) {
            outputs[idx].setArg(argv);
        }
    }
}

void DynamicGraphImpl::setArgumentValueWithStrides(uint32_t argi,
                                                   const void* argv,
                                                   const std::vector<size_t>& strides) {
    _logger.debug("setArgumentValueWithStrides for index %d", argi);
    auto& inputs = _binding._inputs;
    if (argi < inputs.size()) {
        _logger.debug("setArgumentValueWithStrides for index %d (input %d)", argi, argi);
        inputs[argi].setArg(argv);

        for (int64_t i = 0; i < inputs[argi]._dimsCount; i++) {
            inputs[argi]._strides[i] = strides[i];
        }
    } else {
        auto& outputs = _binding._outputs;
        auto idx = argi - inputs.size();
        _logger.debug("setArgumentValueWithStrides for index %d (output %d)", argi, idx);
        if (idx < outputs.size()) {
            outputs[idx].setArg(argv);

            for (int64_t i = 0; i < outputs[idx]._dimsCount; i++) {
                outputs[idx]._strides[i] = strides[i];
            }
        }
    }
}

void DynamicGraphImpl::executeGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                                    IDynamicGraph::GraphArguments& args,
                                    std::vector<ze_command_list_handle_t>& commandLists,
                                    ze_command_queue_handle_t commandQueue,
                                    ze_fence_handle_t fence,
                                    ze_event_handle_t event,
                                    ze_graph_profiling_pool_handle_t profiling) {
    std::shared_ptr<DynamicGraph::GraphArgumentsImpl> argsImpl =
        args._impl ? std::static_pointer_cast<DynamicGraph::GraphArgumentsImpl>(args._impl)
                   : std::make_shared<DynamicGraph::GraphArgumentsImpl>();

    npu_vm_runtime_execute_params_t* params = &argsImpl->_executeParams;

    for (auto& in : args._inputs) {
        std::shared_ptr<DynamicGraph::MemRefTypeImpl> inImpl =
            std::static_pointer_cast<DynamicGraph::MemRefTypeImpl>(in._impl);
        if (inImpl == nullptr) {
            inImpl = std::make_shared<DynamicGraph::MemRefTypeImpl>();
            in._impl = inImpl;
        }
        inImpl->UpdateMemRefHandleStatus(in);
        if (args._impl == nullptr) {
            argsImpl->_inputMemRefs.push_back(inImpl->_memRef);
        }
    }
    for (auto& out : args._outputs) {
        std::shared_ptr<DynamicGraph::MemRefTypeImpl> outImpl =
            std::static_pointer_cast<DynamicGraph::MemRefTypeImpl>(out._impl);
        if (outImpl == nullptr) {
            outImpl = std::make_shared<DynamicGraph::MemRefTypeImpl>();
            out._impl = outImpl;
        }
        outImpl->UpdateMemRefHandleStatus(out);
        if (args._impl == nullptr) {
            argsImpl->_outputMemRefs.push_back(outImpl->_memRef);
        }
    }

    params->pInputs = argsImpl->_inputMemRefs.data();
    params->numOfInputs = static_cast<uint32_t>(argsImpl->_inputMemRefs.size());
    params->pOutputs = argsImpl->_outputMemRefs.data();
    params->numOfOutputs = static_cast<uint32_t>(argsImpl->_outputMemRefs.size());
    params->ctx = zeroInitStruct->getContext();
    params->device = zeroInitStruct->getDevice();
    params->graphDdiTableExt = zeroInitStruct->getGraphDdiTable().getImpl();
    params->commandLists = commandLists.data();
    params->numCommandLists = static_cast<uint64_t>(commandLists.size());
    params->commandQueue = commandQueue;
    params->inferenceFence = fence;
    params->event = event;

    if (npuVMRuntimeExecute(_engine, params) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to execute VM runtime engine");
    }

    if (args._impl == nullptr) {
        args._impl = argsImpl;
    }
}

void DynamicGraphImpl::predictOutputShape(std::vector<MemRefType>& inputDescriptors,
                                          std::vector<MemRefType>& outputDescriptors) {
    std::vector<npu_vm_runtime_mem_ref_handle_t> inputs;
    for (auto& in : inputDescriptors) {
        std::shared_ptr<DynamicGraph::MemRefTypeImpl> inImpl =
            std::static_pointer_cast<DynamicGraph::MemRefTypeImpl>(in._impl);
        if (inImpl == nullptr) {
            inImpl = std::make_shared<DynamicGraph::MemRefTypeImpl>();
            in._impl = inImpl;
        }
        inImpl->UpdateMemRefHandleStatus(in);
        inputs.push_back(inImpl->_memRef);
    }
    std::vector<npu_vm_runtime_mem_ref_handle_t> outputs;
    for (auto& out : outputDescriptors) {
        std::shared_ptr<DynamicGraph::MemRefTypeImpl> outImpl =
            std::static_pointer_cast<DynamicGraph::MemRefTypeImpl>(out._impl);
        if (outImpl == nullptr) {
            outImpl = std::make_shared<DynamicGraph::MemRefTypeImpl>();
            out._impl = outImpl;
        }
        outImpl->UpdateMemRefHandleStatus(out);
        outputs.push_back(outImpl->_memRef);
    }

    npu_vm_runtime_predict_output_shape_params_t params;
    params.pInputs = inputs.data();
    params.numOfInputs = static_cast<uint32_t>(inputs.size());
    params.pOutputs = outputs.data();
    params.numOfOutputs = static_cast<uint32_t>(outputs.size());

    if (npuVMRuntimePredictOutputShape(_engine, &params) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to execute VM runtime engine");
    } else {
        for (auto& out : outputDescriptors) {
            std::shared_ptr<DynamicGraph::MemRefTypeImpl> outImpl =
                std::static_pointer_cast<DynamicGraph::MemRefTypeImpl>(out._impl);
            if (outImpl == nullptr) {
                OPENVINO_THROW("MemRefType implementation is broken, unkown error happens in shape prediction.");
            }
            outImpl->alignWithHandle(out);
        }
        _logger.debug("Output shape prediction is done successfully.");
    }
}

DynamicGraph::DynamicGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                           ov::Tensor blob,
                           bool blobAllocatedByPlugin,
                           const FilteredConfig& config)
    : _zeroInitStruct(zeroInitStruct),
      _blob(std::move(blob)),
      _logger("DynamicGraph", config.get<LOG_LEVEL>()) {
    _logger.info("Create DynamicGraph");
    if (!config.get<CREATE_EXECUTOR>() || config.get<DEFER_WEIGHTS_LOAD>()) {
        _logger.info("Graph initialize is deferred from the \"Graph\" constructor");
        return;
    }

    _impl = std::make_unique<DynamicGraphImpl>();
    // TODO: metadata needs to be parsed even when CREATE_EXECUTOR is 0 or DEFER_WEIGHTS_LOAD is YES, keep here to
    // support pure compilation without vm runtime initialize VM execution engine, metadata, input&output
    // descriptors
    _impl->initialize(_blob, _metadata);

    _num_of_subgraphs = _impl->getNumSubgraphs();

    initialize(config);
}

std::pair<uint64_t, std::optional<std::vector<uint64_t>>> DynamicGraph::export_blob(
    std::ostream& stream,
    const std::optional<std::function<std::string(const std::string&)>>& encryptionCallbackOpt) const {
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

    size_t size = utils::align_size_to_standard_page_size(blobSize);
    size_t paddingSize = size - blobSize;
    if (encryptionCallbackOpt.has_value()) {
        std::string tmpBlobStr(reinterpret_cast<const char*>(blobPtr), blobSize);
        if (paddingSize > 0) {
            // Pad plaintext before encryption so decrypting the full serialized buffer remains symmetric.
            std::fill_n(std::back_inserter(tmpBlobStr), paddingSize, 0);
        }

        auto encryptedBlobStr = encryptionCallbackOpt.value()(tmpBlobStr);
        tmpBlobStr.clear();

        if (encryptedBlobStr.size() >
            static_cast<decltype(encryptedBlobStr.size())>(std::numeric_limits<std::streamsize>::max())) {
            OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
        }

        if (encryptedBlobStr.size() % utils::STANDARD_PAGE_SIZE != 0) {
            _logger.warning("Encrypted blob size %zu is not page aligned, memory optimization when reading this blob "
                            "won't be applied",
                            encryptedBlobStr.size());
        }

        stream.write(encryptedBlobStr.c_str(), static_cast<std::streamsize>(encryptedBlobStr.size()));
    } else {
        if (blobSize > static_cast<decltype(blobSize)>(std::numeric_limits<std::streamsize>::max())) {
            OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
        }
        stream.write(reinterpret_cast<const char*>(blobPtr), static_cast<std::streamsize>(blobSize));
    }

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

    if (paddingSize > 0) {
        if (!encryptionCallbackOpt.has_value()) {
            std::fill_n(std::ostream_iterator<char>(stream), paddingSize, 0);

            if (!stream) {
                _logger.error("Write padding to stream failed. Blob is broken!");
                return std::make_pair(0, std::nullopt);
            }

            _logger.info("Blob size with padding: %zu", size);
        }
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
}

void DynamicGraph::set_argument_value(uint32_t argi, const void* argv) const {
    if (_impl == nullptr) {
        _logger.warning("Graph handle is null, dynamic pipeline to handle set_argument_value");
        return;
    }

    _impl->setArgumentValue(argi, argv);
}

void DynamicGraph::set_argument_value_with_strides(uint32_t id,
                                                   const void* data,
                                                   const std::vector<size_t>& strides) const {
    if (_impl == nullptr) {
        _logger.warning("Graph handle is null, dynamic pipeline to handle set_argument_value");
        return;
    }

    _impl->setArgumentValueWithStrides(id, data, strides);
}

ze_graph_handle_t DynamicGraph::get_handle() const {
    _logger.warning("DynamicGraph does not support get_handle() method.");
    return nullptr;
}

void DynamicGraph::initialize_impl(const FilteredConfig& config) {
    _logger.debug("Graph initialize start");

    if (!_impl) {
        _impl = std::make_unique<DynamicGraphImpl>();
        // initialize VM execution engine, metadata, input&output descriptors
        _impl->initialize(_blob, _metadata);
        _num_of_subgraphs = _impl->getNumSubgraphs();
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
    }

    _logger.debug("Graph initialize finish");

    _batchSize = determine_batch_size();

    // To ensure that the initialization of the graph does not exit prematurely due to nullptrs
    _init_completed.store(true, std::memory_order_release);
}

bool DynamicGraph::release_blob(const FilteredConfig& config) {
    _logger.warning("Release blob is skipped, no handle for DynamicGraph");
    return false;
};

void DynamicGraph::set_batch_size(std::size_t batch) {
    _batchSize = batch;
}

uint32_t DynamicGraph::get_unique_id() {
    return _uniqueId++;
}

void DynamicGraph::set_last_submitted_id(uint32_t id_index) {
    _lastSubmittedId = id_index;
}

uint32_t DynamicGraph::get_last_submitted_id() const {
    return _lastSubmittedId;
}

std::optional<size_t> DynamicGraph::determine_batch_size() {
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

const std::optional<std::size_t> DynamicGraph::get_batch_size() const {
    return _batchSize;
}

DynamicGraph::~DynamicGraph() {
    if (!_lastSubmittedEvent.empty()) {
        _lastSubmittedEvent.clear();
    }
}

void DynamicGraph::execute(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                           DynamicGraph::GraphArguments& args,
                           std::vector<ze_command_list_handle_t>& commandLists,
                           ze_command_queue_handle_t commandQueue,
                           ze_fence_handle_t inferenceFence,
                           ze_event_handle_t event,
                           ze_graph_profiling_pool_handle_t profiling) {
    auto impl = reinterpret_cast<DynamicGraphImpl*>(_impl.get());

    if (impl == nullptr)
        return;

    impl->executeGraph(zeroInitStruct, args, commandLists, commandQueue, inferenceFence, event, profiling);
}

void DynamicGraph::getBinding(GraphArguments& args) {
    auto impl = reinterpret_cast<DynamicGraphImpl*>(_impl.get());

    if (impl == nullptr)
        return;

    impl->getBinding(args);
}

uint64_t DynamicGraph::get_num_subgraphs() const {
    return _num_of_subgraphs;
}

void DynamicGraph::predict_output_shape(std::vector<MemRefType>& inputDescriptors,
                                        std::vector<MemRefType>& outputDescriptors) {
    auto impl = reinterpret_cast<DynamicGraphImpl*>(_impl.get());

    if (impl == nullptr)
        return;

    impl->predictOutputShape(inputDescriptors, outputDescriptors);
}

std::optional<bool> DynamicGraph::is_profiling_blob() const {
    _logger.warning("Profiling is not supported for DynamicGraph");
    return std::nullopt;
}

}  // namespace intel_npu
