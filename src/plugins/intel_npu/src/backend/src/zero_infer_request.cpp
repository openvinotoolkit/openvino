// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_infer_request.hpp"

#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/config/compiler.hpp"
#include "intel_npu/al/config/runtime.hpp"
#include "intel_npu/al/itt.hpp"
#include "intel_npu/al/prefix.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "zero_memory.hpp"

using namespace intel_npu;

namespace {

constexpr std::size_t BATCH_AXIS = 0;
constexpr bool INPUT = true;
constexpr bool OUTPUT = false;

/**
 * @brief Checks that the metadata of the provided descriptor corresponds to the values registered in the Level Zero
 * structure.
 * @param ioDescriptor The OpenVINO API specific I/O descriptor which shall be compared.
 * @param zeDescriptor The Level Zero specific structure used for comparison.
 */
void checkLevelZeroAttributesMatch(const IODescriptor& ioDescriptor,
                                   const ZeroExecutor::ArgumentDescriptor& zeDescriptor) {
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

}  // namespace

size_t ZeroInferRequest::getBatchSize(const NetworkMetadata& metadata) {
    if (!metadata.outputs.at(0).shapeFromIRModel.has_value()) {
        _logger.info("Batching on the plugin is not used, batching is handled by the compiler");
        return DEFAULT_BATCH_SIZE;
    }

    const ov::PartialShape& firstOutputShape = *metadata.outputs.at(0).shapeFromIRModel;
    if (firstOutputShape.is_dynamic()) {
        _logger.info("Networks using dynamic shapes are not supported when batching is handled by the plugin");
        return DEFAULT_BATCH_SIZE;
    }
    if (firstOutputShape.rank().get_length() == 0) {
        _logger.info(
            "Networks using rank 0 shapes for inputs/outputs are not supported when batching is handled by the plugin");
        return DEFAULT_BATCH_SIZE;
    }

    const size_t candidateBatchSize = firstOutputShape[BATCH_AXIS].get_length();
    if (candidateBatchSize == 0 || candidateBatchSize == DEFAULT_BATCH_SIZE) {
        _logger.info("Batching on the plugin is not used, batching is handled by the compiler");
        return DEFAULT_BATCH_SIZE;
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
        _logger.info("Batching on the plugin is not used, batching is handled by the compiler");
        return DEFAULT_BATCH_SIZE;
    }

    _logger.info("Batching is handled by the plugin");

    return candidateBatchSize;
}

//------------------------------------------------------------------------------
ZeroInferRequest::ZeroInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                                   const std::shared_ptr<const ICompiledModel>& compiledModel,
                                   const std::shared_ptr<const IExecutor>& executor,
                                   const Config& config)
    : SyncInferRequest(compiledModel),
      _initStructs(initStructs),
      _executorPtr(executor),
      _executor(static_cast<const ZeroExecutor*>(_executorPtr.get())),
      _config(config),
      _logger("ZeroInferRequest", config.get<LOG_LEVEL>()),
      _levelZeroInputTensors(_metadata.inputs.size(), nullptr),
      _levelZeroOutputTensors(_metadata.outputs.size(), nullptr),
      _inputTensorsData(_metadata.inputs.size(), std::nullopt),
      _outputTensorsData(_metadata.outputs.size(), std::nullopt),
      _profilingPool(_executor->graph(), zeroProfiling::POOL_SIZE, _executor->getInitStructs()->getProfilingDdiTable()),
      _profilingQuery(0,
                      _executor->getInitStructs()->getDevice(),
                      _executor->getInitStructs()->getProfilingDdiTable()) {
    _logger.debug("ZeroInferRequest::ZeroInferRequest - SyncInferRequest");
    const std::vector<ZeroExecutor::ArgumentDescriptor>& executorInputDescriptors = _executor->get_input_descriptors();
    const std::vector<ZeroExecutor::ArgumentDescriptor>& executorOutputDescriptors =
        _executor->get_output_descriptors();

    auto proftype = config.get<PROFILING_TYPE>();
    if (proftype == ov::intel_npu::ProfilingType::INFER) {
        _logger.debug("ZeroInferRequest::ZeroInferRequest - profiling type == ov::intel_npu::ProfilingType::INFER");
        _npuProfiling = std::make_shared<zeroProfiling::NpuInferProfiling>(_executor->getInitStructs()->getContext(),
                                                                           _executor->getInitStructs()->getDevice(),
                                                                           _config.get<LOG_LEVEL>());
    }

    _properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeroUtils::throwOnFail("zeDeviceGetProperties",
                           zeDeviceGetProperties(_executor->getInitStructs()->getDevice(), &_properties));

    auto allocator = zeroMemory::HostMemAllocator(_initStructs);

    if (config.get<BATCH_MODE>() != ov::intel_npu::BatchMode::COMPILER) {
        _batchSize = getBatchSize(_metadata);
    }
    if (_batchSize != DEFAULT_BATCH_SIZE) {
        _batchSizeArgument = std::optional(_batchSize);  // TODO refactor
    }

    _logger.debug("ZeroInferRequest::ZeroInferRequest - checking level zero attributes and allocating tensors");

    size_t ioIndex = 0;
    for (const IODescriptor& inputDescriptor : _metadata.inputs) {
        checkLevelZeroAttributesMatch(inputDescriptor, executorInputDescriptors.at(ioIndex));

        if (!(inputDescriptor.isStateInput || inputDescriptor.isShapeTensor)) {
            continue;
        }

        ov::Allocator inputAllocator;
        if (_properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) {
            inputAllocator = zeroMemory::HostMemAllocator(_initStructs, ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED);
        } else {
            inputAllocator = zeroMemory::HostMemAllocator(_initStructs);
        };

        _levelZeroInputTensors.at(ioIndex) =
            allocate_tensor(inputDescriptor, ioIndex, INPUT, inputAllocator, _batchSizeArgument);
        _inputTensorsData.at(ioIndex) =
            TensorData{_levelZeroInputTensors.at(ioIndex)->data(), _levelZeroInputTensors.at(ioIndex)->get_byte_size()};

        ++ioIndex;
    }

    ioIndex = 0;
    for (const IODescriptor& outputDescriptor : _metadata.outputs) {
        checkLevelZeroAttributesMatch(outputDescriptor, executorOutputDescriptors.at(ioIndex));

        if (!(outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor)) {
            continue;
        }

        _levelZeroOutputTensors.at(ioIndex) =
            allocate_tensor(outputDescriptor, ioIndex, OUTPUT, allocator, _batchSizeArgument);
        _outputTensorsData.at(ioIndex) =
            std::optional(TensorData{_levelZeroOutputTensors.at(ioIndex)->data(),
                                     _levelZeroOutputTensors.at(ioIndex)->get_byte_size()});

        ++ioIndex;
    }

    _logger.debug("ZeroInferRequest::ZeroInferRequest - SyncInferRequest completed");
}

void ZeroInferRequest::create_pipeline() {
    for (size_t inputIndex = 0; inputIndex < _metadata.inputs.size(); ++inputIndex) {
        if (_levelZeroInputTensors.at(inputIndex)) {
            _logger.debug("ZeroInferRequest::create_pipeline - tensor %s was already allocated",
                          _metadata.inputs.at(inputIndex).nodeFriendlyName);
            continue;
        }

        ov::Allocator inputAllocator;
        if (_properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) {
            inputAllocator = zeroMemory::HostMemAllocator(_initStructs, ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED);
        } else {
            inputAllocator = zeroMemory::HostMemAllocator(_initStructs);
        };

        _logger.debug("ZeroInferRequest::create_pipeline - Allocate new tensor");
        _levelZeroInputTensors.at(inputIndex) =
            allocate_tensor(_metadata.inputs.at(inputIndex), inputIndex, INPUT, inputAllocator, _batchSizeArgument);
        _inputTensorsData.at(inputIndex) =
            std::optional(TensorData{_levelZeroInputTensors.at(inputIndex)->data(),
                                     _levelZeroInputTensors.at(inputIndex)->get_byte_size()});
    }

    for (size_t outputIndex = 0; outputIndex < _metadata.outputs.size(); ++outputIndex) {
        if (_levelZeroOutputTensors.at(outputIndex)) {
            _logger.debug("ZeroInferRequest::create_pipeline - tensor %s was already allocated",
                          _metadata.outputs.at(outputIndex).nodeFriendlyName);
            continue;
        }

        _logger.debug("ZeroInferRequest::create_pipeline - allocate new tensor");
        _levelZeroOutputTensors.at(outputIndex) = allocate_tensor(_metadata.outputs.at(outputIndex),
                                                                  OUTPUT,
                                                                  outputIndex,
                                                                  zeroMemory::HostMemAllocator(_initStructs),
                                                                  _batchSizeArgument);
        _outputTensorsData.at(outputIndex) =
            std::optional(TensorData{_levelZeroOutputTensors.at(outputIndex)->data(),
                                     _levelZeroOutputTensors.at(outputIndex)->get_byte_size()});
    }

    _logger.debug("ZeroInferRequest::create_pipeline - constructing pipeline");
    // Construct pipeline
    _pipeline = makePipeline(_executorPtr,
                             _config,
                             _profilingPool,
                             _profilingQuery,
                             _npuProfiling,
                             _inputTensorsData,
                             _outputTensorsData,
                             _batchSize);
    _logger.debug("ZeroInferRequest::create_pipeline - SyncInferRequest completed");
}

void ZeroInferRequest::set_tensor_data(const std::shared_ptr<ov::ITensor> tensor,
                                       const size_t index,
                                       const bool isInput) {
    auto& levelZeroTensors = isInput ? _levelZeroInputTensors : _levelZeroOutputTensors;
    auto& tensorsData = isInput ? _inputTensorsData : _outputTensorsData;

    bool setTensorData = false;
    bool levelZeroTensorCreatedLocally = true;

    ze_memory_allocation_properties_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    auto res = zeMemGetAllocProperties(_initStructs->getContext(), tensor->data(), &desc, nullptr);
    if (res == ZE_RESULT_SUCCESS) {
        if (desc.id) {
            switch (desc.type) {
            case ZE_MEMORY_TYPE_HOST:
            case ZE_MEMORY_TYPE_DEVICE:
            case ZE_MEMORY_TYPE_SHARED:
                _logger.debug("ZeroInferRequest::set_tensor_data - tensor was created in the same L0 context");
                levelZeroTensors.at(index) = tensor;
                levelZeroTensorCreatedLocally = false;
                setTensorData = true;
                break;
            case ZE_MEMORY_TYPE_UNKNOWN:
            case ZE_MEMORY_TYPE_FORCE_UINT32:
                break;
            }
        }
    }

    if (!setTensorData) {
        // make sure that the L0 tensor was allocated locally and is not received from the user when receiving
        // random tensor
        if (tensorsData.at(index).has_value() && !tensorsData.at(index)->levelZeroTensorCreatedLocally) {
            _logger.debug("ZeroInferRequest::set_tensor_data - create locally L0 tensor");
            ov::Allocator allocator;
            if (isInput && (_properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED)) {
                allocator = zeroMemory::HostMemAllocator(_initStructs, ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED);
            } else {
                allocator = zeroMemory::HostMemAllocator(_initStructs);
            };

            levelZeroTensors.at(index) =
                allocate_tensor(isInput ? _metadata.inputs.at(index) : _metadata.outputs.at(index),
                                index,
                                isInput,
                                allocator,
                                _batchSizeArgument);

            setTensorData = true;
            levelZeroTensorCreatedLocally = true;
        }
    }

    if (setTensorData) {
        tensorsData.at(index) = std::optional(TensorData{levelZeroTensors.at(index)->data(),
                                                         levelZeroTensors.at(index)->get_byte_size(),
                                                         levelZeroTensorCreatedLocally,
                                                         !_createPipeline});

        _updateCommandList = true;
    }
}

void ZeroInferRequest::set_remote_tensor_data(const std::shared_ptr<ZeroRemoteTensor> tensor,
                                              const size_t index,
                                              const bool isInput) {
    auto l0_context = reinterpret_cast<ze_context_handle_t>(
        extract_object(tensor->get_context()->get_property(), ov::intel_npu::l0_context));
    if (_initStructs->getContext() != l0_context) {
        OPENVINO_THROW("Using different context for creating the tensor is not supported");
    }

    auto data = extract_object(tensor->get_properties(), ov::intel_npu::mem_handle);
    if (data == nullptr) {
        OPENVINO_THROW("Empty buffer");
    }

    if (isInput) {
        _levelZeroInputTensors.at(index) = tensor;
        _inputTensorsData.at(index) = std::optional(TensorData{data, tensor->get_byte_size(), false, !_createPipeline});
    } else {
        _levelZeroOutputTensors.at(index) = tensor;
        _outputTensorsData.at(index) =
            std::optional(TensorData{data, tensor->get_byte_size(), false, !_createPipeline});
    }
    _updateCommandList = true;
}

void ZeroInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    auto foundPort = find_port(port);
    OPENVINO_ASSERT(foundPort.found(), "Cannot find tensor for port ", port);
    try {
        check_tensor(port, tensor);
    } catch (const ov::Exception& ex) {
        OPENVINO_THROW("Failed to set tensor. ", ex.what());
    }

    if (foundPort.is_input()) {
        _userInputTensors.at(foundPort.idx) = tensor._ptr;
    } else {
        _userOutputTensors.at(foundPort.idx) = tensor._ptr;
    }

    if (_initStructs->getMutableCommandListVersion()) {
        auto remoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(tensor._ptr);

        if (remoteTensor == nullptr) {
            _logger.debug("ZeroInferRequest::set_tensor - set new tensor");
            set_tensor_data(tensor._ptr, foundPort.idx, foundPort.is_input());
        } else {
            _logger.debug("ZeroInferRequest::set_tensor - set new remote tensor");
            set_remote_tensor_data(remoteTensor, foundPort.idx, foundPort.is_input());
        }
    }
}

ov::SoPtr<ov::ITensor> ZeroInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    auto foundPort = find_port(port);
    OPENVINO_ASSERT(foundPort.found(), "Cannot find tensor for port ", port);

    const size_t ioIndex = foundPort.idx;
    const bool isInput = foundPort.is_input();
    auto& userTensors = isInput ? _userInputTensors : _userOutputTensors;

    if (userTensors.at(ioIndex)) {
        _logger.debug("ZeroInferRequest::get_tensor - tensor allocated, get the tensor");
        return userTensors.at(ioIndex);
    }

    _logger.debug("ZeroInferRequest::get_tensor - tensor is not allocated, create the tensor");

    ov::Allocator allocator;
    if (foundPort.is_input() && (_properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED)) {
        allocator = zeroMemory::HostMemAllocator(_initStructs, ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED);
    } else {
        allocator = zeroMemory::HostMemAllocator(_initStructs);
    };

    auto& levelZeroTensors = isInput ? _levelZeroInputTensors : _levelZeroOutputTensors;
    auto& tensorsData = isInput ? _inputTensorsData : _outputTensorsData;

    levelZeroTensors.at(ioIndex) =
        allocate_tensor(isInput ? _metadata.inputs.at(ioIndex) : _metadata.outputs.at(ioIndex),
                        ioIndex,
                        isInput,
                        allocator,
                        _batchSizeArgument);
    tensorsData.at(ioIndex) =
        std::optional(TensorData{levelZeroTensors.at(ioIndex)->data(), levelZeroTensors.at(ioIndex)->get_byte_size()});

    return levelZeroTensors.at(ioIndex);
}

void ZeroInferRequest::infer() {
    infer_async();
    get_result();
}

void ZeroInferRequest::infer_async() {
    _logger.debug("InferRequest::infer_async started");
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "infer_async");

    _executor->mutexLock();

    if (_createPipeline) {
        create_pipeline();

        _createPipeline = false;
        _updateCommandList = false;
    }

    if (_initStructs->getMutableCommandListVersion()) {
        if (_updateCommandList) {
            _logger.debug("ZeroInferRequest::infer_async - update command list");
            _pipeline->updateCommandList(_inputTensorsData, _outputTensorsData, _batchSize);

            _updateCommandList = false;
        }
    }

    _executor->mutexUnlock();

    size_t inputIndex = 0;
    for (const std::shared_ptr<ov::ITensor>& userTensor : _userInputTensors) {
        const IODescriptor inputDescriptor = _metadata.inputs.at(inputIndex);
        if (inputDescriptor.isShapeTensor) {
            OPENVINO_ASSERT(inputDescriptor.relatedDescriptorIndex.has_value(),
                            "The link between the dynamic tensor and its shape tensor is missing, entry name: ",
                            inputDescriptor.nameFromCompiler);
            const auto& inputDims = _userInputTensors.at(*inputDescriptor.relatedDescriptorIndex)->get_shape();

            for (size_t i = 0; i < userTensor->get_size(); ++i) {
                const auto reverseIdx = inputDims.size() - 1 - i;
                userTensor->data<uint32_t>()[i] = static_cast<uint32_t>(inputDims[reverseIdx]);
            }
        }

        auto userRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userTensor);
        void* userBuffer = !userRemoteTensor
                               ? userTensor->data()
                               : extract_object(userRemoteTensor->get_properties(), ov::intel_npu::mem_handle);

        const std::shared_ptr<ov::ITensor>& levelZeroTensor = _levelZeroInputTensors.at(inputIndex);
        auto levelZeroRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(levelZeroTensor);
        if (levelZeroRemoteTensor == nullptr) {
            void* levelZeroBuffer = levelZeroTensor->data();

            if (userBuffer != levelZeroBuffer) {
                if (userBuffer == nullptr || levelZeroBuffer == nullptr) {
                    OPENVINO_THROW("Empty buffer");
                }

                _logger.info("Tensor is not allocated in the current Level Zero context");
                std::memcpy(levelZeroBuffer, userBuffer, userTensor->get_byte_size());
            }
        }

        ++inputIndex;
    }

    for (size_t i = 0; i < _batchSize; i++) {
        _pipeline->push(i);
    }
}

void ZeroInferRequest::get_result() {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "get_result");

    for (size_t i = 0; i < _batchSize; i++) {
        _pipeline->pull(i);
    }

    size_t outputIndex = 0;
    for (const std::shared_ptr<ov::ITensor>& userTensor : _userOutputTensors) {
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

        auto userRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userTensor);
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
                std::memcpy(userBuffer, levelZeroBuffer, userTensor->get_byte_size());
            }
        }

        ++outputIndex;
    }

    for (size_t i = 0; i < _batchSize; i++) {
        _pipeline->reset(i);
    }
    _logger.debug("InferRequest::get_result finished");
}

void ZeroInferRequest::check_network_precision(const ov::element::Type_t precision) const {
    switch (precision) {
    case ov::element::Type_t::f32:
        break;
    case ov::element::Type_t::f16:
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
                       "! Supported precisions: FP32, FP16, U4, I4, U8, I8, U16, I16, U32, I32, U64, I64, FP64");
    }
}

std::vector<ov::ProfilingInfo> ZeroInferRequest::get_profiling_info() const {
    _logger.debug("InferRequest::get_profiling_info started");
    const auto& compiledModel = *std::dynamic_pointer_cast<const ICompiledModel>(_compiledModel);
    const auto& compilerConfig = compiledModel.get_config();
    if (!compilerConfig.get<PERF_COUNT>() || !_config.get<PERF_COUNT>()) {
        _logger.debug("InferRequest::get_profiling_info complete with empty {}.");
        return {};
    }

    auto compilerType = compilerConfig.get<COMPILER_TYPE>();
    if (compilerType == ov::intel_npu::CompilerType::MLIR) {
        // For plugin compiler retreive raw profiling data from backend and delegate
        // processing to the compiler
        const auto& networkDesc = compiledModel.get_network_description();
        const auto& compiler = compiledModel.get_compiler();
        const auto& blob = networkDesc->compiledNetwork;
        auto profData = get_raw_profiling_data();
        _logger.debug("InferRequest::get_profiling_info complete with compiler->process_profiling_output().");
        return compiler->process_profiling_output(profData, blob, compilerConfig);
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
