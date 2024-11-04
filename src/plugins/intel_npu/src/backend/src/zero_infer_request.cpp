// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_infer_request.hpp"

#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/common.hpp"
#include "intel_npu/config/compiler.hpp"
#include "intel_npu/config/runtime.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "zero_memory.hpp"

using namespace intel_npu;

namespace {

constexpr std::size_t SINGLE_TENSOR = 0;
constexpr std::size_t BATCH_AXIS = 0;
constexpr std::size_t DEFAULT_BATCH_SIZE = 1;
constexpr bool INPUT = true;
constexpr bool OUTPUT = false;

/**
 * @brief Checks that the metadata of the provided descriptor corresponds to the values registered in the Level Zero
 * structure.
 * @param ioDescriptor The OpenVINO API specific I/O descriptor which shall be compared.
 * @param zeDescriptor The Level Zero specific structure used for comparison.
 */
void check_level_zero_attributes_match(const IODescriptor& ioDescriptor, const ArgumentDescriptor& zeDescriptor) {
    std::string zeDescriptorName = zeDescriptor.info.name;

    if (isStateInputName(zeDescriptorName)) {
        zeDescriptorName = zeDescriptorName.substr(READVALUE_PREFIX.length());
    } else if (isStateOutputName(zeDescriptorName)) {
        zeDescriptorName = zeDescriptorName.substr(ASSIGN_PREFIX.length());
    } else if (isShapeTensorName(zeDescriptorName)) {
        zeDescriptorName = zeDescriptorName.substr(SHAPE_TENSOR_PREFIX.length());
    }

    OPENVINO_ASSERT(ioDescriptor.nameFromCompiler == zeDescriptorName,
                    "Name mismatch between the I/O structure used internally and its Level Zero correspondent: ",
                    ioDescriptor.nameFromCompiler,
                    " vs. ",
                    zeDescriptorName,
                    ". The I/O order may have been altered, which could lead to an erroneous behavior.");
    OPENVINO_ASSERT(zeroUtils::getZePrecision(ioDescriptor.precision) == zeDescriptor.info.devicePrecision,
                    "Precision mismatch for input/output named " + ioDescriptor.nameFromCompiler);

    const std::vector<size_t>& ovDimensions = ioDescriptor.shapeFromCompiler.get_max_shape();
    OPENVINO_ASSERT(ovDimensions.size() <= ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE,
                    "Maximum number of dimensions supported: " + std::to_string(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) +
                        '\n' + "Given: " + std::to_string(ovDimensions.size()));

    for (size_t index = 0; index < ovDimensions.size(); ++index) {
        OPENVINO_ASSERT(
            ioDescriptor.shapeFromCompiler.is_dynamic() || ovDimensions[index] == zeDescriptor.info.dims[index],
            "Shape mismatch for input/output named " + ioDescriptor.nameFromCompiler);
    }
    for (size_t index = ovDimensions.size(); index < ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE; ++index) {
        OPENVINO_ASSERT(zeDescriptor.info.dims[index] == 0 || zeDescriptor.info.dims[index] == 1,
                        "Shape mismatch for input/output named " + ioDescriptor.nameFromCompiler);
    }
}

template <typename Type>
Type extract_object(const ov::AnyMap& params, const ov::Property<Type>& p) {
    auto itrHandle = params.find(p.name());
    ov::Any res = nullptr;
    if (itrHandle == params.end()) {
        OPENVINO_THROW("No parameter ", p.name(), " found in parameters map");
    }
    res = itrHandle->second;
    return res.as<Type>();
}

bool memory_was_allocated_in_the_same_l0_context(ze_context_handle_t hContext, const void* ptr) {
    ze_memory_allocation_properties_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    auto res = intel_npu::zeMemGetAllocProperties(hContext, ptr, &desc, nullptr);
    if (res == ZE_RESULT_SUCCESS) {
        if (desc.id) {
            if ((desc.type & ZE_MEMORY_TYPE_HOST) || (desc.type & ZE_MEMORY_TYPE_DEVICE) ||
                (desc.type & ZE_MEMORY_TYPE_SHARED)) {
                return true;
            }
        }
    }

    return false;
}

}  // namespace

std::optional<size_t> ZeroInferRequest::get_batch_size(const NetworkMetadata& metadata) {
    if (!metadata.outputs.at(0).shapeFromIRModel.has_value()) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    const ov::PartialShape& firstOutputShape = *metadata.outputs.at(0).shapeFromIRModel;
    if (firstOutputShape.is_dynamic()) {
        _logger.warning("Networks using dynamic shapes are not supported when batching is handled by the plugin");
        return std::nullopt;
    }
    if (firstOutputShape.rank().get_length() == 0) {
        _logger.warning(
            "Networks using rank 0 shapes for inputs/outputs are not supported when batching is handled by the plugin");
        return std::nullopt;
    }

    const size_t candidateBatchSize = firstOutputShape[BATCH_AXIS].get_length();
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

    if (!checkDescriptorsUseCandidateBatchSize(metadata.inputs) ||
        !checkDescriptorsUseCandidateBatchSize(metadata.outputs)) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    _logger.debug("Batching is handled by the plugin");

    return candidateBatchSize;
}

//------------------------------------------------------------------------------
ZeroInferRequest::ZeroInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                                   const std::shared_ptr<const ICompiledModel>& compiledModel,
                                   const Config& config)
    : SyncInferRequest(compiledModel, config),
      _initStructs(initStructs),
      _graph(compiledModel->get_graph()),
      _config(config),
      _logger("ZeroInferRequest", config.get<LOG_LEVEL>()),
      _levelZeroInputTensors(_metadata.inputs.size(), std::vector<std::shared_ptr<ov::ITensor>>(1, nullptr)),
      _levelZeroOutputTensors(_metadata.outputs.size(), nullptr),
      _inputTensorsData(_metadata.inputs.size(), std::vector<std::optional<TensorData>>(1, std::nullopt)),
      _outputTensorsData(_metadata.outputs.size(), std::nullopt),
      _profilingPool(static_cast<ze_graph_handle_t>(_graph->get_handle()),
                     zeroProfiling::POOL_SIZE,
                     _initStructs->getProfilingDdiTable()),
      _profilingQuery(0, _initStructs->getDevice(), _initStructs->getProfilingDdiTable()) {
    _logger.debug("ZeroInferRequest::ZeroInferRequest - SyncInferRequest");
    const std::vector<ArgumentDescriptor>& executorInputDescriptors = _graph->get_input_descriptors();
    const std::vector<ArgumentDescriptor>& executorOutputDescriptors = _graph->get_output_descriptors();

    auto proftype = config.get<PROFILING_TYPE>();
    if (proftype == ov::intel_npu::ProfilingType::INFER) {
        _logger.debug("ZeroInferRequest::ZeroInferRequest - profiling type == ov::intel_npu::ProfilingType::INFER");
        _npuProfiling = std::make_shared<zeroProfiling::NpuInferProfiling>(_initStructs->getContext(),
                                                                           _initStructs->getDevice(),
                                                                           _config.get<LOG_LEVEL>());
    }

    _properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties",
                                zeDeviceGetProperties(_initStructs->getDevice(), &_properties));

    _outputAllocator = std::make_shared<const zeroMemory::HostMemAllocator>(_initStructs);
    _inputAllocator =
        std::make_shared<const zeroMemory::HostMemAllocator>(_initStructs, ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED);

    if (config.get<BATCH_MODE>() != ov::intel_npu::BatchMode::COMPILER) {
        _batchSize = get_batch_size(_metadata);
    }
    if (_batchSize.has_value()) {
        _numberOfCommandLists = *_batchSize;
    }

    _logger.debug("ZeroInferRequest::ZeroInferRequest - checking level zero attributes and allocating tensors");

    size_t ioIndex = 0;
    for (const IODescriptor& inputDescriptor : _metadata.inputs) {
        check_level_zero_attributes_match(inputDescriptor, executorInputDescriptors.at(ioIndex));

        if (!(inputDescriptor.isStateInput || inputDescriptor.isShapeTensor)) {
            ++ioIndex;
            continue;
        }

        get_level_zero_input(ioIndex) = allocate_tensor(inputDescriptor, ioIndex, INPUT, *_inputAllocator, _batchSize);
        get_input_tensor_data(ioIndex) =
            TensorData{get_level_zero_input(ioIndex)->data(), get_level_zero_input(ioIndex)->get_byte_size()};

        ++ioIndex;
    }

    ioIndex = 0;
    for (const IODescriptor& outputDescriptor : _metadata.outputs) {
        check_level_zero_attributes_match(outputDescriptor, executorOutputDescriptors.at(ioIndex));

        if (!(outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor)) {
            ++ioIndex;
            continue;
        }

        _levelZeroOutputTensors.at(ioIndex) =
            allocate_tensor(outputDescriptor, ioIndex, OUTPUT, *_outputAllocator, _batchSize);
        _outputTensorsData.at(ioIndex) =
            std::optional(TensorData{_levelZeroOutputTensors.at(ioIndex)->data(),
                                     _levelZeroOutputTensors.at(ioIndex)->get_byte_size()});

        ++ioIndex;
    }

    _logger.debug("ZeroInferRequest::ZeroInferRequest - SyncInferRequest completed");
}

void ZeroInferRequest::create_pipeline() {
    for (size_t inputIndex = 0; inputIndex < _metadata.inputs.size(); ++inputIndex) {
        if (is_batched_input(inputIndex)) {
            if (_batchSize.has_value()) {
                _logger.debug("ZeroInferRequest::create_pipeline - tensors %s were already allocated",
                              _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str());
                continue;
            }
        }

        if (get_level_zero_input(inputIndex)) {
            _logger.debug("ZeroInferRequest::create_pipeline - tensor %s was already allocated",
                          _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str());
            continue;
        }

        _logger.debug("ZeroInferRequest::create_pipeline - allocate new tensor");
        get_level_zero_input(inputIndex) =
            allocate_tensor(_metadata.inputs.at(inputIndex), inputIndex, INPUT, *_inputAllocator, _batchSize);
        get_input_tensor_data(inputIndex) = std::optional(
            TensorData{get_level_zero_input(inputIndex)->data(), get_level_zero_input(inputIndex)->get_byte_size()});
    }

    for (size_t outputIndex = 0; outputIndex < _metadata.outputs.size(); ++outputIndex) {
        if (_levelZeroOutputTensors.at(outputIndex)) {
            _logger.debug("ZeroInferRequest::create_pipeline - tensor %s was already allocated",
                          _metadata.outputs.at(outputIndex).nodeFriendlyName.c_str());
            continue;
        }
        _logger.debug("ZeroInferRequest::create_pipeline - allocate new tensor");
        _levelZeroOutputTensors.at(outputIndex) =
            allocate_tensor(_metadata.outputs.at(outputIndex), outputIndex, OUTPUT, *_outputAllocator, _batchSize);
        _outputTensorsData.at(outputIndex) =
            std::optional(TensorData{_levelZeroOutputTensors.at(outputIndex)->data(),
                                     _levelZeroOutputTensors.at(outputIndex)->get_byte_size()});
    }

    // Find the corresponding command queue group.
    _logger.debug("ZeroDevice::ZeroDevice - findGroupOrdinal");
    auto groupOrdinal = zeroUtils::findGroupOrdinal(_initStructs->getDevice(), _properties);
    _logger.debug("ZeroDevice::ZeroDevice - init completed");

    _logger.debug("ZeroInferRequest::create_pipeline - constructing pipeline");

    // Construct pipeline
    _pipeline = std::make_unique<Pipeline>(_config,
                                           _initStructs,
                                           _graph,
                                           _profilingPool,
                                           _profilingQuery,
                                           _npuProfiling,
                                           _inputTensorsData,
                                           _outputTensorsData,
                                           _numberOfCommandLists,
                                           groupOrdinal);

    _logger.debug("ZeroInferRequest::create_pipeline - SyncInferRequest completed");
}

void ZeroInferRequest::set_tensor_data(const std::shared_ptr<ov::ITensor> tensor,
                                       const size_t index,
                                       const bool isInput) {
    OV_ITT_TASK_CHAIN(ZERO_SET_TENSOR, itt::domains::LevelZeroBackend, "set_tensor", "set_tensor_data");
    auto& levelZeroTensors = isInput ? get_level_zero_input(index) : _levelZeroOutputTensors.at(index);
    auto& tensorsData = isInput ? get_input_tensor_data(index) : _outputTensorsData.at(index);

    bool setTensorData = false;
    bool levelZeroTensorCreatedLocally = true;

    OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "check_data_allocation");
    if (memory_was_allocated_in_the_same_l0_context(_initStructs->getContext(), tensor->data())) {
        _logger.debug("ZeroInferRequest::set_tensor_data - tensor was created in the same L0 context");
        levelZeroTensors = tensor;
        levelZeroTensorCreatedLocally = false;
        setTensorData = true;
    }

    if (!setTensorData) {
        // make sure that the L0 tensor was allocated locally and is not received from the user when receiving
        // random tensor
        if (tensorsData.has_value() && !tensorsData->levelZeroTensorCreatedLocally) {
            _logger.debug("ZeroInferRequest::set_tensor_data - create locally L0 tensor");
            OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "allocate tensor");

            levelZeroTensors = allocate_tensor(isInput ? _metadata.inputs.at(index) : _metadata.outputs.at(index),
                                               index,
                                               isInput,
                                               isInput ? *_inputAllocator : *_outputAllocator,
                                               _batchSize);

            setTensorData = true;
            levelZeroTensorCreatedLocally = true;
        }
    }

    if (setTensorData) {
        tensorsData = std::optional(
            TensorData{levelZeroTensors->data(), levelZeroTensors->get_byte_size(), levelZeroTensorCreatedLocally});

        if (_pipelineIsCreated) {
            _logger.debug("ZeroInferRequest::infer_async - update command list");

            OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "updateCommandList");
            _pipeline->updateCommandList(*tensorsData,
                                         isInput ? _graph->get_input_descriptors().at(index).idx
                                                 : _graph->get_output_descriptors().at(index).idx);
        }
    }
}

void ZeroInferRequest::set_remote_tensor_data(const std::shared_ptr<ZeroRemoteTensor> tensor,
                                              const size_t index,
                                              const bool isInput) {
    OV_ITT_TASK_CHAIN(ZERO_SET_REMOTE_TENSOR, itt::domains::LevelZeroBackend, "set_tensor", "set_remote_tensor_data");

    auto l0_context = reinterpret_cast<ze_context_handle_t>(
        extract_object(tensor->get_context()->get_property(), ov::intel_npu::l0_context));
    if (_initStructs->getContext() != l0_context) {
        OPENVINO_THROW("Using different context for creating the tensor is not supported");
    }

    auto data = extract_object(tensor->get_properties(), ov::intel_npu::mem_handle);
    if (data == nullptr) {
        OPENVINO_THROW("Empty buffer");
    }

    auto& levelZeroTensors = isInput ? get_level_zero_input(index) : _levelZeroOutputTensors.at(index);
    auto& tensorsData = isInput ? get_input_tensor_data(index) : _outputTensorsData.at(index);

    levelZeroTensors = tensor;
    tensorsData = std::optional(TensorData{data, tensor->get_byte_size(), false});

    if (_pipelineIsCreated) {
        _logger.debug("ZeroInferRequest::infer_async - update command list");

        OV_ITT_TASK_NEXT(ZERO_SET_REMOTE_TENSOR, "updateCommandList");
        _pipeline->updateCommandList(
            *tensorsData,
            isInput ? _graph->get_input_descriptors().at(index).idx : _graph->get_output_descriptors().at(index).idx);
    }
}

void ZeroInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "set_tensor");

    auto foundPort = find_port(port);
    OPENVINO_ASSERT(foundPort.found(), "Cannot find tensor for port ", port);
    try {
        check_tensor(port, tensor);
    } catch (const ov::Exception& ex) {
        OPENVINO_THROW("Failed to set tensor. ", ex.what());
    }

    if (foundPort.is_input()) {
        if (get_user_input(foundPort.idx)._ptr == tensor._ptr) {
            // Got set_tensor with the same object - do nothing
            _logger.debug("ZeroInferRequest::set_tensor - got the same tensor, do nothing");
            return;
        }
        if (is_batched_input(foundPort.idx)) {
            // resize vector size to 1 if set_tensor is called after set_tensors
            get_input_tensors_data(foundPort.idx).resize(1);
            get_input_tensors_data(foundPort.idx).shrink_to_fit();
            get_level_zero_inputs(foundPort.idx).resize(1);
            get_level_zero_inputs(foundPort.idx).shrink_to_fit();
            get_user_inputs(foundPort.idx).resize(1);
            get_user_inputs(foundPort.idx).shrink_to_fit();
        }

        get_user_input(foundPort.idx) = tensor;
    } else {
        if (_userOutputTensors.at(foundPort.idx)._ptr == tensor._ptr) {
            // Got set_tensor with the same object here too - do nothing
            return;
        }
        _userOutputTensors.at(foundPort.idx) = tensor;
    }

    if (_initStructs->getMutableCommandListVersion()) {
        auto remoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(tensor._ptr);

        if (remoteTensor == nullptr) {
            _logger.debug("ZeroInferRequest::set_tensor - set new tensor");
            set_tensor_data(tensor._ptr, foundPort.idx, foundPort.is_input());
        } else {
            _logger.debug("ZeroInferRequest::set_tensor - set new remote tensor");
            set_remote_tensor_data(std::move(remoteTensor), foundPort.idx, foundPort.is_input());
        }
    }
}

void ZeroInferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                   const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    OV_ITT_TASK_CHAIN(SET_TENSORS, itt::domains::LevelZeroBackend, "set_tensors", "set_tensors");
    if (tensors.size() == 1) {
        set_tensor(port, tensors[0]);
        return;
    }

    auto foundPort = find_port(port);
    OPENVINO_ASSERT(foundPort.found(), "Cannot find input tensor for port ", port);
    if (!foundPort.is_input()) {
        OPENVINO_THROW("set_input_tensors/set_tensors is not supported for output port.");
    }

    check_batched_tensors(port, tensors);

    get_user_inputs(foundPort.idx).resize(tensors.size());
    get_user_inputs(foundPort.idx) = tensors;

    if (_initStructs->getMutableCommandListVersion()) {
        if (_batchSize.has_value()) {
            for (size_t i = 0; i < tensors.size(); i++) {
                auto remoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(tensors[i]._ptr);

                get_level_zero_inputs(foundPort.idx).resize(tensors.size());
                get_input_tensors_data(foundPort.idx).resize(tensors.size());

                if (remoteTensor == nullptr) {
                    bool tensorHasSameL0Context = false;

                    OV_ITT_TASK_NEXT(SET_TENSORS, "check_data_allocation");
                    if (memory_was_allocated_in_the_same_l0_context(_initStructs->getContext(), tensors[i]->data())) {
                        _logger.debug("ZeroInferRequest::set_tensors - tensor was created in the same L0 context");

                        get_level_zero_input(foundPort.idx, i) = tensors.at(i)._ptr;
                        tensorHasSameL0Context = true;
                    }

                    if (!tensorHasSameL0Context) {
                        _logger.debug("ZeroInferRequest::set_tensors - tensor wasn't created in the same L0 context, "
                                      "create a L0 tensor");

                        get_level_zero_input(foundPort.idx, i) =
                            allocate_tensor(_metadata.inputs.at(foundPort.idx), foundPort.idx, true, *_inputAllocator);
                    }

                    get_input_tensor_data(foundPort.idx, i) =
                        std::optional(TensorData{get_level_zero_input(foundPort.idx, i)->data(),
                                                 get_level_zero_input(foundPort.idx, i)->get_byte_size(),
                                                 false});
                } else {
                    _logger.debug("ZeroInferRequest::set_tensors - remote tensor is used");

                    get_input_tensor_data(foundPort.idx, i) = std::optional(
                        TensorData{extract_object(remoteTensor->get_properties(), ov::intel_npu::mem_handle),
                                   remoteTensor->get_byte_size(),
                                   false});

                    get_level_zero_input(foundPort.idx, i) = tensors.at(i)._ptr;
                }

                if (_pipelineIsCreated) {
                    OV_ITT_TASK_NEXT(SET_TENSORS, "updateCommandList");
                    _pipeline->updateCommandList(*get_input_tensor_data(foundPort.idx, i),
                                                 _graph->get_input_descriptors().at(foundPort.idx).idx,
                                                 i);
                }
            }
        }
    }
}

ov::SoPtr<ov::ITensor> ZeroInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "get_tensor");

    auto foundPort = find_port(port);
    OPENVINO_ASSERT(foundPort.found(), "Cannot find tensor for port ", port);

    const size_t ioIndex = foundPort.idx;
    const bool isInput = foundPort.is_input();

    if (isInput && is_batched_input(ioIndex)) {
        OPENVINO_THROW("Cannot return tensors in a tensor.");
    }

    auto& userTensors = isInput ? get_user_input(ioIndex) : _userOutputTensors.at(ioIndex);

    if (userTensors) {
        _logger.debug("ZeroInferRequest::get_tensor - tensor allocated, get the tensor");
        return userTensors;
    }

    _logger.debug("ZeroInferRequest::get_tensor - tensor is not allocated, create the tensor");

    auto& levelZeroTensors = isInput ? get_level_zero_input(ioIndex) : _levelZeroOutputTensors.at(ioIndex);
    auto& tensorsData = isInput ? get_input_tensor_data(ioIndex) : _outputTensorsData.at(ioIndex);

    levelZeroTensors = allocate_tensor(isInput ? _metadata.inputs.at(ioIndex) : _metadata.outputs.at(ioIndex),
                                       ioIndex,
                                       isInput,
                                       isInput ? *_inputAllocator : *_outputAllocator,
                                       _batchSize);
    tensorsData = std::optional(TensorData{levelZeroTensors->data(), levelZeroTensors->get_byte_size()});

    return levelZeroTensors;
}

void ZeroInferRequest::infer() {
    infer_async();
    get_result();
}

void ZeroInferRequest::infer_async() {
    _logger.debug("InferRequest::infer_async started");
    OV_ITT_TASK_CHAIN(ZERO_INFER, itt::domains::LevelZeroBackend, "infer_async", "start");

    {
        std::lock_guard<std::mutex> lock(_graph->get_mutex());

        if (!_pipelineIsCreated) {
            OV_ITT_TASK_NEXT(ZERO_INFER, "create_pipeline");
            create_pipeline();

            _pipelineIsCreated = true;
        }
    }

    size_t inputIndex = 0;
    for (const auto& userTensor : _userInputTensors) {
        const IODescriptor inputDescriptor = _metadata.inputs.at(inputIndex);
        if (inputDescriptor.isShapeTensor) {
            OPENVINO_ASSERT(inputDescriptor.relatedDescriptorIndex.has_value(),
                            "The link between the dynamic tensor and its shape tensor is missing, entry name: ",
                            inputDescriptor.nameFromCompiler);
            const auto& inputDims = get_user_input(*inputDescriptor.relatedDescriptorIndex)->get_shape();

            for (size_t i = 0; i < userTensor.at(SINGLE_TENSOR)->get_size(); ++i) {
                const auto reverseIdx = inputDims.size() - 1 - i;
                userTensor.at(SINGLE_TENSOR)->data<uint32_t>()[i] = static_cast<uint32_t>(inputDims[reverseIdx]);
            }
        }

        if (is_batched_input(inputIndex)) {
            if (_batchSize.has_value()) {
                for (size_t i = 0; i < userTensor.size(); i++) {
                    auto levelZeroBatchRemoteTensor =
                        std::dynamic_pointer_cast<ZeroRemoteTensor>(get_level_zero_input(inputIndex, i));
                    if (levelZeroBatchRemoteTensor == nullptr) {
                        void* levelZeroBuffer = get_level_zero_input(inputIndex, i)->data();

                        auto userBatchRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userTensor.at(i)._ptr);

                        void* userBuffer =
                            !userBatchRemoteTensor
                                ? userTensor.at(i)->data()
                                : extract_object(userBatchRemoteTensor->get_properties(), ov::intel_npu::mem_handle);

                        if (userBuffer != levelZeroBuffer) {
                            if (userBuffer == nullptr || levelZeroBuffer == nullptr) {
                                OPENVINO_THROW("Empty buffer");
                            }

                            _logger.info("Batched Tensors - Tensor is not allocated in the current Level Zero context");
                            OV_ITT_TASK_NEXT(ZERO_INFER, "memcpy");
                            std::memcpy(levelZeroBuffer, userBuffer, userTensor.at(i)->get_byte_size());
                        }
                    }
                }
            } else {
                void* levelZeroBuffer = get_level_zero_input(inputIndex)->data();

                _logger.info("Batched Tensors - Tensor is not allocated in the current Level Zero context or must be "
                             "in a continued memory space");

                for (size_t i = 0; i < userTensor.size(); i++) {
                    auto userBatchRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userTensor.at(i)._ptr);

                    void* userBuffer = !userBatchRemoteTensor ? userTensor.at(i)->data()
                                                              : extract_object(userBatchRemoteTensor->get_properties(),
                                                                               ov::intel_npu::mem_handle);

                    std::memcpy(static_cast<unsigned char*>(levelZeroBuffer) + (i * userTensor.at(i)->get_byte_size()),
                                userBuffer,
                                userTensor.at(i)->get_byte_size());
                }
            }

            ++inputIndex;
            continue;
        }

        auto userRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userTensor.at(SINGLE_TENSOR)._ptr);
        void* userBuffer = !userRemoteTensor
                               ? userTensor.at(SINGLE_TENSOR)->data()
                               : extract_object(userRemoteTensor->get_properties(), ov::intel_npu::mem_handle);

        const std::shared_ptr<ov::ITensor>& levelZeroTensor = get_level_zero_input(inputIndex);
        auto levelZeroRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(levelZeroTensor);
        if (levelZeroRemoteTensor == nullptr) {
            void* levelZeroBuffer = levelZeroTensor->data();

            if (userBuffer != levelZeroBuffer) {
                if (userBuffer == nullptr || levelZeroBuffer == nullptr) {
                    OPENVINO_THROW("Empty buffer");
                }

                _logger.info("Tensor is not allocated in the current Level Zero context");
                OV_ITT_TASK_NEXT(ZERO_INFER, "memcpy");
                std::memcpy(levelZeroBuffer, userBuffer, userTensor.at(SINGLE_TENSOR)->get_byte_size());
            }
        }

        ++inputIndex;
    }

    OV_ITT_TASK_NEXT(ZERO_INFER, "push");
    _pipeline->push();
}

void ZeroInferRequest::get_result() {
    OV_ITT_TASK_CHAIN(ZERO_RESULT, itt::domains::LevelZeroBackend, "get_result", "pull");
    _logger.debug("InferRequest::get_result start");
    _pipeline->pull();

    size_t outputIndex = 0;
    for (const auto& userTensor : _userOutputTensors) {
        const IODescriptor outputDescriptor = _metadata.outputs.at(outputIndex);
        if (outputDescriptor.isShapeTensor) {
            OPENVINO_ASSERT(outputDescriptor.relatedDescriptorIndex.has_value(),
                            "The link between the dynamic tensor and its shape tensor is missing, entry name: ",
                            outputDescriptor.nameFromCompiler);

            ov::Shape actualDims;
            actualDims.reserve(userTensor->get_size());

            for (size_t i = 0; i < userTensor->get_size(); ++i) {
                const auto reverseIdx = userTensor->get_size() - 1 - i;
                actualDims.push_back(userTensor->data<uint32_t>()[reverseIdx]);
            }
            auto& tensorToBeReshaped = _userOutputTensors.at(*outputDescriptor.relatedDescriptorIndex);
            tensorToBeReshaped->set_shape(actualDims);
        }

        auto userRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userTensor._ptr);
        void* userBuffer = !userRemoteTensor
                               ? userTensor->data()
                               : extract_object(userRemoteTensor->get_properties(), ov::intel_npu::mem_handle);

        const std::shared_ptr<ov::ITensor>& levelZeroTensor = _levelZeroOutputTensors.at(outputIndex);
        auto levelZeroRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(levelZeroTensor);
        if (levelZeroRemoteTensor == nullptr) {
            void* levelZeroBuffer = levelZeroTensor->data();

            if (userBuffer != levelZeroBuffer) {
                if (userBuffer == nullptr || levelZeroBuffer == nullptr) {
                    OPENVINO_THROW("Empty buffer");
                }

                _logger.info("Tensor is not allocated in the current Level Zero context");
                OV_ITT_TASK_NEXT(ZERO_RESULT, "memcpy");
                std::memcpy(userBuffer, levelZeroBuffer, userTensor->get_byte_size());
            }
        }

        ++outputIndex;
    }

    OV_ITT_TASK_NEXT(ZERO_RESULT, "reset");
    _pipeline->reset();
    _logger.debug("InferRequest::get_result finished");
}

void ZeroInferRequest::check_network_precision(const ov::element::Type_t precision) const {
    switch (precision) {
    case ov::element::Type_t::f32:
        break;
    case ov::element::Type_t::f16:
        break;
    case ov::element::Type_t::bf16:
        break;
    case ov::element::Type_t::u4:
        break;
    case ov::element::Type_t::i4:
        break;
    case ov::element::Type_t::u8:
        break;
    case ov::element::Type_t::i8:
        break;
    case ov::element::Type_t::u16:
        break;
    case ov::element::Type_t::i16:
        break;
    case ov::element::Type_t::u32:
        break;
    case ov::element::Type_t::i32:
        break;
    case ov::element::Type_t::u64:
        break;
    case ov::element::Type_t::i64:
        break;
    case ov::element::Type_t::f64:
        break;
    default:
        OPENVINO_THROW("Unsupported tensor precision: " + ov::element::Type(precision).get_type_name() +
                       "! Supported precisions: FP32, FP16, BF16, U4, I4, U8, I8, U16, I16, U32, I32, U64, I64, FP64");
    }
}

std::vector<ov::ProfilingInfo> ZeroInferRequest::get_profiling_info() const {
    _logger.debug("InferRequest::get_profiling_info started");
    const auto& compiledModel = *std::dynamic_pointer_cast<const ICompiledModel>(_compiledModel);
    const auto& compilerConfig = compiledModel.get_config();
    if (!compilerConfig.get<PERF_COUNT>() || !_config.get<PERF_COUNT>()) {
        _logger.warning("InferRequest::get_profiling_info complete with empty {}.");
        return {};
    }

    auto compilerType = compilerConfig.get<COMPILER_TYPE>();
    if (compilerType == ov::intel_npu::CompilerType::MLIR) {
        // For plugin compiler retreive raw profiling data from backend and delegate
        // processing to the compiler
        auto profData = get_raw_profiling_data();
        _logger.debug("InferRequest::get_profiling_info complete with compiler->process_profiling_output().");
        return _graph->process_profiling_output(profData, compilerConfig);
    } else {
        auto proftype = _config.get<PROFILING_TYPE>();
        if (proftype == ov::intel_npu::ProfilingType::INFER) {
            _logger.debug("InferRequest::get_profiling_info complete with _npuProfiling->getNpuInferStatistics().");
            return _npuProfiling->getNpuInferStatistics();
        } else {  /// proftype = MODEL or undefined = fallback to model profiling
            _logger.debug("InferRequest::get_profiling_info complete with _profilingQuery.getLayerStatistics().");
            return _profilingQuery.getLayerStatistics();
        }
    }
}

std::vector<uint8_t> ZeroInferRequest::get_raw_profiling_data() const {
    return _profilingQuery.getData<uint8_t>();
}

std::shared_ptr<ov::ITensor>& ZeroInferRequest::get_level_zero_input(size_t index, size_t tensorNo) const {
    return _levelZeroInputTensors.at(index).at(tensorNo);
}

std::vector<std::shared_ptr<ov::ITensor>>& ZeroInferRequest::get_level_zero_inputs(size_t index) const {
    return _levelZeroInputTensors.at(index);
}

std::optional<TensorData>& ZeroInferRequest::get_input_tensor_data(size_t index, size_t tensorNo) const {
    return _inputTensorsData.at(index).at(tensorNo);
}
std::vector<std::optional<TensorData>>& ZeroInferRequest::get_input_tensors_data(size_t index) const {
    return _inputTensorsData.at(index);
}
