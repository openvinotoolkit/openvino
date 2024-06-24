// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_infer_request.hpp"

#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/config/compiler.hpp"
#include "intel_npu/al/config/runtime.hpp"
#include "intel_npu/al/itt.hpp"
#include "intel_npu/al/prefix.hpp"
#include "zero_memory.hpp"

using namespace intel_npu;

namespace {

constexpr std::size_t BATCH_AXIS = 0;

/**
 * @brief Checks that the metadata of the provided descriptor corresponds to the values registered in the Level Zero
 * structure.
 * @param nodeDescriptor The OpenVINO API specific I/O descriptor which shall be compared.
 * @param zeDescriptor The Level Zero specific structure used for comparison.
 * @param name Tensor identifier used for error logging.
 */
void checkLevelZeroAttributesMatch(const IONodeDescriptor& nodeDescriptor,
                                   const ZeroExecutor::ArgumentDescriptor& zeDescriptor,
                                   const std::string& name) {
    const ov::element::Type_t ovPrecision = nodeDescriptor.precision;
    const ze_graph_argument_precision_t zePrecision = zeDescriptor.info.devicePrecision;

    if (zeroUtils::getZePrecision(ovPrecision) != zePrecision) {
        OPENVINO_THROW("Precision mismatch for parameter " + name);
    }

    const std::vector<size_t>& ovDimensions = nodeDescriptor.transposedShape.get_max_shape();

    if (ovDimensions.size() > ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) {
        OPENVINO_THROW(
            "Maximum number of dimensions supported: " + std::to_string(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) + '\n' +
            "Given: " + std::to_string(ovDimensions.size()));
    }

    for (size_t index = ovDimensions.size(); index < ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE; ++index) {
        if (zeDescriptor.info.dims[index] != 0 && zeDescriptor.info.dims[index] != 1) {
            OPENVINO_THROW("Shape mismatch for parameter " + name);
        }
    }

    for (size_t index = 1; index < ovDimensions.size(); ++index) {
        if (ovDimensions[index] != zeDescriptor.info.dims[index] && !nodeDescriptor.transposedShape.is_dynamic()) {
            OPENVINO_THROW("Shape mismatch for parameter " + name);
        }
    }
}

std::optional<size_t> getBatchSizeForNode(const IONodeDescriptor& nodeDescriptor,
                                          const ZeroExecutor::ArgumentDescriptor& zeDescriptor) {
    Logger logger("GetBatchSizeForNode", Logger::global().level());

    if (nodeDescriptor.originalShape.rank().get_length() == 0) {
        logger.info("Networks with empty shapes are not supported when batching is handled by the plugin");
        return std::nullopt;
    }

    if (nodeDescriptor.originalShape.is_dynamic()) {
        logger.info("Dynamic networks are not supported when batching is handled by the plugin");
        return std::nullopt;
    }

    const std::vector<size_t>& ovDimensions = nodeDescriptor.originalShape.get_shape();

    if (ovDimensions[BATCH_AXIS] == zeDescriptor.info.dims[BATCH_AXIS] &&
        ovDimensions[BATCH_AXIS] != DEFAULT_BATCH_SIZE) {
        logger.info("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    if (zeDescriptor.info.dims[BATCH_AXIS] == DEFAULT_BATCH_SIZE) {
        return ovDimensions[BATCH_AXIS];
    }

    return DEFAULT_BATCH_SIZE;
}

/**
 * @brief Get the batch size to be handled on the plugin.
 * @details Analyze the shape from the compiled model with the shape from the originalShape and get the originalShape if
 * it is different.
 * @param metadata A map to represent descriptions for inputs and outputs of a network.
 * @param executorInputDescriptors A map to represent Level zero inputs descriptors.
 * @param executorOutputDescriptors A map to represent Level zero outputs descriptors.
 */

std::optional<size_t> getBatchSize(
    const NetworkMetadata& metadata,
    const std::unordered_map<std::string, ZeroExecutor::ArgumentDescriptor>& executorInputDescriptors,
    const std::unordered_map<std::string, ZeroExecutor::ArgumentDescriptor>& executorOutputDescriptors) {
    std::set<size_t> batch_size;

    Logger logger("getBatchSize", Logger::global().level());

    for (const std::string& inputName : metadata.inputNames) {
        auto batchSizeForNode =
            getBatchSizeForNode(metadata.parameters.at(inputName), executorInputDescriptors.at(inputName));

        if (batchSizeForNode.has_value()) {
            batch_size.insert(*batchSizeForNode);
        } else {
            return std::nullopt;
        }
    }

    for (const std::string& outputName : metadata.outputNames) {
        if (!executorOutputDescriptors.count(outputName)) {
            OPENVINO_THROW("Invalid graph output descriptor key: " + outputName);
        }
        auto batchSizeForNode =
            getBatchSizeForNode(metadata.results.at(outputName), executorOutputDescriptors.at(outputName));

        if (batchSizeForNode.has_value()) {
            batch_size.insert(*batchSizeForNode);
        } else {
            return std::nullopt;
        }
    }

    if (batch_size.size() != 1) {
        logger.info("Batching works only when we have the same batch size for all tensors!");
        return std::nullopt;
    }

    auto it = batch_size.begin();
    if (*it) {
        return *it;
    }

    return std::nullopt;
}

}  // namespace

//------------------------------------------------------------------------------
ZeroInferRequest::ZeroInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& backendPtr,
                                   const std::shared_ptr<const ICompiledModel>& compiledModel,
                                   const std::shared_ptr<const IExecutor>& executor,
                                   const Config& config)
    : SyncInferRequest(compiledModel),
      _executorPtr(executor),
      _executor(static_cast<const ZeroExecutor*>(_executorPtr.get())),
      _config(config),
      _logger("ZeroInferRequest", config.get<LOG_LEVEL>()),
      _profilingPool(_executor->graph(), zeroProfiling::POOL_SIZE, _executor->getInitStructs()->getProfilingDdiTable()),
      _profilingQuery(0,
                      _executor->getInitStructs()->getDevice(),
                      _executor->getInitStructs()->getProfilingDdiTable()) {
    _logger.debug("ZeroInferRequest::ZeroInferRequest - SyncInferRequest");
    const std::unordered_map<std::string, ZeroExecutor::ArgumentDescriptor>& executorInputDescriptors =
        _executor->inputs_desc_map();
    const std::unordered_map<std::string, ZeroExecutor::ArgumentDescriptor>& executorOutputDescriptors =
        _executor->outputs_desc_map();

    auto proftype = config.get<PROFILING_TYPE>();
    if (proftype == ov::intel_npu::ProfilingType::INFER) {
        _logger.debug("ZeroInferRequest::ZeroInferRequest - profiling type == ov::intel_npu::ProfilingType::INFER");
        _npuProfiling = std::make_shared<zeroProfiling::NpuInferProfiling>(_executor->getInitStructs()->getContext(),
                                                                           _executor->getInitStructs()->getDevice(),
                                                                           _config.get<LOG_LEVEL>());
    }

    ze_device_properties_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeroUtils::throwOnFail("zeDeviceGetProperties",
                           zeDeviceGetProperties(_executor->getInitStructs()->getDevice(), &properties));

    const auto contains = [](const auto& container, const auto& value) {
        return std::find(container.begin(), container.end(), value) != container.end();
    };

    auto allocator = zeroMemory::HostMemAllocator(backendPtr);

    _logger.debug("ZeroInferRequest::ZeroInferRequest - performing I/O buffer allocation using Level Zero API");
    for (const std::string& inputName : _metadata.inputNames) {
        if (!executorInputDescriptors.count(inputName)) {
            OPENVINO_THROW("Invalid graph input descriptor key: " + inputName);
        }
    }

    for (const std::string& outputName : _metadata.outputNames) {
        if (!executorOutputDescriptors.count(outputName)) {
            OPENVINO_THROW("Invalid graph output descriptor key: " + outputName);
        }
    }

    if (config.get<BATCH_MODE>() != ov::intel_npu::BatchMode::COMPILER) {
        auto batchSize = getBatchSize(_metadata, executorInputDescriptors, executorOutputDescriptors);

        if (batchSize.has_value()) {
            _batchSize = *batchSize;
        }
    }

    for (const std::string& inputName : _metadata.inputNames) {
        IONodeDescriptor& parameterDescriptor = _metadata.parameters.at(inputName);
        checkLevelZeroAttributesMatch(parameterDescriptor, executorInputDescriptors.at(inputName), inputName);

        ov::Allocator inputAllocator;
        if (properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) {
            inputAllocator = zeroMemory::HostMemAllocator(backendPtr, ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED);
        } else {
            inputAllocator = zeroMemory::HostMemAllocator(backendPtr);
        };

        // When batching is handled by the plugin we need to modify transposed shape with the original batch size since
        // it will be forced to 1 at the compilation time
        if (_batchSize > DEFAULT_BATCH_SIZE) {
            parameterDescriptor.transposedShape[BATCH_AXIS] = _batchSize;
        }

        // The I/O buffers already allocated using the Level Zero API are being reused here
        allocate_tensor(inputName, parameterDescriptor, TensorType::InputOrOutput, inputAllocator);

        if (contains(_metadata.shapeNames, inputName)) {
            const std::string shapeBufferName = SHAPE_TENSOR_PREFIX + inputName;
            const IONodeDescriptor& shapeDescriptor = _metadata.shapes.at(inputName);

            checkLevelZeroAttributesMatch(shapeDescriptor,
                                          executorInputDescriptors.at(shapeBufferName),
                                          shapeBufferName);

            allocate_tensor(inputName, shapeDescriptor, TensorType::Shape, inputAllocator);
        }
    }

    _logger.debug("ZeroInferRequest::ZeroInferRequest - checking level zero attributes and allocate tensor");
    for (const std::string& outputName : _metadata.outputNames) {
        IONodeDescriptor& resultDescriptor = _metadata.results.at(outputName);
        checkLevelZeroAttributesMatch(resultDescriptor, executorOutputDescriptors.at(outputName), outputName);

        // When batching is handled by the plugin we need to modify transposed shape with the original batch size since
        // it will be forced to 1 at the compilation time
        if (_batchSize > DEFAULT_BATCH_SIZE) {
            resultDescriptor.transposedShape[BATCH_AXIS] = _batchSize;
        }

        allocate_tensor(outputName, resultDescriptor, TensorType::InputOrOutput, allocator);

        const auto& shapeNameMatch = _nodeNameToLegacyName.find(outputName);
        if (shapeNameMatch != _nodeNameToLegacyName.end()) {
            if (contains(_metadata.shapeNames, shapeNameMatch->second)) {
                const std::string shapeBufferName = SHAPE_TENSOR_PREFIX + shapeNameMatch->second;
                const IONodeDescriptor& shapeDescriptor = _metadata.shapes.at(shapeNameMatch->second);

                checkLevelZeroAttributesMatch(shapeDescriptor,
                                              executorOutputDescriptors.at(shapeBufferName),
                                              shapeBufferName);

                allocate_tensor(shapeNameMatch->second, shapeDescriptor, TensorType::Shape, allocator);
            }
        }
    }

    _logger.debug("ZeroInferRequest::ZeroInferRequest - capturing latest tensor value in output");
    for (const std::string& stateName : _metadata.stateNames) {
        const std::string& stateInputBufferName = READVALUE_PREFIX + stateName;
        const std::string& stateOutputBufferName = ASSIGN_PREFIX + stateName;

        if (!executorInputDescriptors.count(stateInputBufferName)) {
            OPENVINO_THROW("Invalid graph input descriptor key: " + stateInputBufferName);
        }
        if (!executorOutputDescriptors.count(stateOutputBufferName)) {
            OPENVINO_THROW("Invalid graph output descriptor key: " + stateOutputBufferName);
        }

        const IONodeDescriptor& stateDescriptor = _metadata.states.at(stateName);
        checkLevelZeroAttributesMatch(stateDescriptor,
                                      executorInputDescriptors.at(stateInputBufferName),
                                      stateInputBufferName);
        checkLevelZeroAttributesMatch(stateDescriptor,
                                      executorOutputDescriptors.at(stateOutputBufferName),
                                      stateOutputBufferName);

        // Only one buffer per state variable is required, we'll use the "output" one since this one captures the latest
        // tensor value
        allocate_tensor(stateName, stateDescriptor, TensorType::State, allocator);
    }

    _logger.debug("ZeroInferRequest::ZeroInferRequest - constructing pipeline");
    /// Construct pipepline
    _pipeline = makePipeline(_executorPtr,
                             _config,
                             _profilingPool,
                             _profilingQuery,
                             _npuProfiling,
                             _copyAllTensors,
                             _batchSize);
    _logger.debug("ZeroInferRequest::ZeroInferRequest - SyncInferRequest completed");
}

void ZeroInferRequest::infer() {
    infer_async();
    get_result();
}

void ZeroInferRequest::infer_async() {
    _logger.debug("InferRequest::infer_async started");
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "infer_async");

    for (const auto& name : _inputAndStateInputNames) {
        auto& inputTensor = _allTensors.at(name);
        const auto& wrapperInputTensor = _copyAllTensors.at(name);

        if (isShapeTensorName(name)) {
            const auto actualTensorName = name.substr(SHAPE_TENSOR_PREFIX.size());
            const auto& inputDims = _allTensors.at(actualTensorName)->get_shape();

            for (size_t i = 0; i < inputTensor->get_size(); ++i) {
                const auto reverseIdx = inputDims.size() - 1 - i;
                inputTensor->data<uint32_t>()[i] = static_cast<uint32_t>(inputDims[reverseIdx]);
            }
        }

        const uint8_t* tensorBuffer = reinterpret_cast<uint8_t*>(inputTensor->data());
        uint8_t* copyTensorBuffer = reinterpret_cast<uint8_t*>(wrapperInputTensor->data());

        if (tensorBuffer != copyTensorBuffer) {
            if (tensorBuffer == nullptr || copyTensorBuffer == nullptr) {
                OPENVINO_THROW("Empty buffer");
            }

            std::memcpy(copyTensorBuffer, tensorBuffer, inputTensor->get_byte_size());
        }
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

    for (const auto& name : _outputAndStateOutputNames) {
        const auto& outputTensor = _allTensors.at(name);
        const auto& wrapperOutputTensor = _copyAllTensors.at(name);

        if (isShapeTensorName(name)) {
            const auto actualTensorName = name.substr(SHAPE_TENSOR_PREFIX.size());
            const auto& shapeNameMatch = _legacyNameToNodeName.find(actualTensorName);
            if (shapeNameMatch != _legacyNameToNodeName.end()) {
                ov::Shape actualDims;
                actualDims.reserve(outputTensor->get_size());

                for (size_t i = 0; i < outputTensor->get_size(); ++i) {
                    const auto reverseIdx = outputTensor->get_size() - 1 - i;
                    actualDims.push_back(outputTensor->data<uint32_t>()[reverseIdx]);
                }
                auto& tensorToBeReshaped = _allTensors.at(shapeNameMatch->second);
                tensorToBeReshaped->set_shape(actualDims);
            }
        }

        uint8_t* tensorBuffer = reinterpret_cast<uint8_t*>(outputTensor->data());
        const uint8_t* copyTensorBuffer = reinterpret_cast<uint8_t*>(wrapperOutputTensor->data());

        if (tensorBuffer != copyTensorBuffer) {
            if (tensorBuffer == nullptr || copyTensorBuffer == nullptr) {
                OPENVINO_THROW("Empty buffer");
            }

            std::memcpy(tensorBuffer, copyTensorBuffer, outputTensor->get_byte_size());
        }
    }

    for (size_t i = 0; i < _batchSize; i++) {
        _pipeline->reset(i);
    }
    _logger.debug("InferRequest::get_result finished");
}

void ZeroInferRequest::check_network_precision(const ov::element::Type_t precision) {
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
