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
#include "zero_memory.hpp"

using namespace intel_npu;

namespace {

constexpr std::size_t BATCH_AXIS = 0;
constexpr bool IS_INPUT = true;
constexpr bool IS_OUTPUT = false;

/**
 * @brief Checks that the metadata of the provided descriptor corresponds to the values registered in the Level Zero
 * structure.
 * @param nodeDescriptor The OpenVINO API specific I/O descriptor which shall be compared.
 * @param zeDescriptor The Level Zero specific structure used for comparison.
 * @param name Tensor identifier used for error logging.
 */
void checkLevelZeroAttributesMatch(const IODescriptor& nodeDescriptor,
                                   const ZeroExecutor::ArgumentDescriptor& zeDescriptor,
                                   const std::string& name) {
    const ov::element::Type_t ovPrecision = nodeDescriptor.precision;
    const ze_graph_argument_precision_t zePrecision = zeDescriptor.info.devicePrecision;

    if (zeroUtils::getZePrecision(ovPrecision) != zePrecision) {
        OPENVINO_THROW("Precision mismatch for parameter " + name);
    }

    const std::vector<size_t>& ovDimensions = nodeDescriptor.shapeFromCompiler.get_max_shape();

    if (ovDimensions.size() > ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) {
        OPENVINO_THROW(
            "Maximum number of dimensions supported: " + std::to_string(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) + '\n' +
            "Given: " + std::to_string(ovDimensions.size()));
    }

    for (size_t index = 0; index < ovDimensions.size(); ++index) {
        if (!nodeDescriptor.shapeFromCompiler.is_dynamic() && ovDimensions[index] != zeDescriptor.info.dims[index]) {
            OPENVINO_THROW("Shape mismatch for parameter " + name);
        }
    }

    for (size_t index = ovDimensions.size(); index < ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE; ++index) {
        if (zeDescriptor.info.dims[index] != 0 && zeDescriptor.info.dims[index] != 1) {
            OPENVINO_THROW("Shape mismatch for parameter " + name);
        }
    }
}

/**
 * @brief Determines if batching can be addressed inside the plugin. In the positive case, the batch size used by the
 * model will also be deduced and stored within the network metadata.
 * @details Batching can be handled by the plugin only if:
 *  - The batch axis is the first axis.
 *  - The batch size viewed by the compiler takes the default value of 1.
 *  - The batch size found in the IR model corresponds for all inputs/outputs and takes a value different than the
 * default one. If any of the previous conditions is not fulfilled, the functon will not update the
 * "canUsePluginBatching" and "batchSize" attributes found inside the network metadata object and thus no custom
 * algorithm will be applied inside the plugin in order to address batching.
 *
 * @param metadata The metadata used by the compiled model. The parameter will be used for extracting the shape used
 * inside the compiler. If the function determines batching can be handled by the plugin, the flag as well as the batch
 * size attributes found inside this object will be updated.
 */
size_t getBatchSize(const NetworkMetadata& metadata) {
    Logger logger("getBatchSize", Logger::global().level());

    const ov::PartialShape& firstOutputShape = metadata.outputs.at(0).shapeFromIRModel;
    if (firstOutputShape.is_dynamic() || firstOutputShape.rank().get_length() == 0) {
        logger.warning("");  // TODO
        return DEFAULT_BATCH_SIZE;
    }

    const size_t candidateBatchSize = firstOutputShape[0].get_length();
    if (candidateBatchSize == 0 || candidateBatchSize == DEFAULT_BATCH_SIZE) {
        return DEFAULT_BATCH_SIZE;
    }

    auto checkDescriptorsUseCandidateBatchSize = [candidateBatchSize](const std::vector<IODescriptor>& descriptors) {
        for (const IODescriptor& descriptor : descriptors) {
            const ov::PartialShape& shapeFromCompiler = descriptor.shapeFromCompiler;
            const ov::PartialShape& shapeFromIRModel = descriptor.shapeFromCompiler;

            if (shapeFromCompiler.is_dynamic() || shapeFromCompiler.rank().get_length() == 0 ||
                *shapeFromCompiler.begin() != DEFAULT_BATCH_SIZE) {
                return false;
            }

            if (!descriptor.isStateInput && !descriptor.isShapeTensor) {
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
        return DEFAULT_BATCH_SIZE;
    }

    return candidateBatchSize;
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

    ze_device_properties_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeroUtils::throwOnFail("zeDeviceGetProperties",
                           zeDeviceGetProperties(_executor->getInitStructs()->getDevice(), &properties));

    const auto contains = [](const auto& container, const auto& value) {
        return std::find(container.begin(), container.end(), value) != container.end();
    };

    auto allocator = zeroMemory::HostMemAllocator(backendPtr);

    if (config.get<BATCH_MODE>() != ov::intel_npu::BatchMode::COMPILER) {
        _batchSize = getBatchSize(_metadata);
    }

    _logger.debug("ZeroInferRequest::ZeroInferRequest - checking level zero attributes and allocating tensors");

    size_t inputIndex = 0;
    for (const IODescriptor& inputDescriptor : _metadata.inputs) {
        checkLevelZeroAttributesMatch(inputDescriptor,
                                      executorInputDescriptors.at(inputIndex),
                                      inputDescriptor.nameFromCompiler);  // TODO move earlier

        ov::Allocator inputAllocator;
        if (properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) {
            inputAllocator = zeroMemory::HostMemAllocator(backendPtr, ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED);
        } else {
            inputAllocator = zeroMemory::HostMemAllocator(backendPtr);
        };

        // The I/O buffers already allocated using the Level Zero API are being reused here
        allocate_tensor(inputDescriptor, IS_INPUT, inputAllocator);

        ++inputIndex;
    }

    size_t outputIndex = 0;
    for (const IODescriptor& outputDescriptor : _metadata.outputs) {
        checkLevelZeroAttributesMatch(outputDescriptor,
                                      executorOutputDescriptors.at(outputIndex),
                                      outputDescriptor.nameFromCompiler);
        allocate_tensor(outputDescriptor, IS_OUTPUT, allocator);

        ++outputIndex;
    }

    _logger.debug("ZeroInferRequest::ZeroInferRequest - constructing pipeline");
    /// Construct pipepline
    _pipeline = makePipeline(_executorPtr,
                             _config,
                             _profilingPool,
                             _profilingQuery,
                             _npuProfiling,
                             _copyInputTensors,
                             _copyOutputTensors,
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

    size_t inputIndex = 0;
    for (const std::shared_ptr<ov::ITensor>& inputTensor : _inputTensors) {
        const std::shared_ptr<ov::ITensor>& wrapperInputTensor = _copyInputTensors.at(inputIndex);

        const IODescriptor inputDescriptor = _metadata.outputs.at(inputIndex);
        if (inputDescriptor.isShapeTensor) {
            OPENVINO_ASSERT(inputDescriptor.relatedDescriptorIndex.has_value());
            const auto& inputDims = _inputTensors.at(*inputDescriptor.relatedDescriptorIndex)->get_shape();

            // TODO optimize this
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
    for (const std::shared_ptr<ov::ITensor>& outputTensor : _outputTensors) {
        const std::shared_ptr<ov::ITensor>& wrapperOutputTensor = _copyOutputTensors.at(outputIndex);

        const IODescriptor outputDescriptor = _metadata.outputs.at(outputIndex);
        if (outputDescriptor.isShapeTensor) {
            OPENVINO_ASSERT(outputDescriptor.relatedDescriptorIndex.has_value());

            ov::Shape actualDims;
            actualDims.reserve(outputTensor->get_size());

            for (size_t i = 0; i < outputTensor->get_size(); ++i) {
                const auto reverseIdx = outputTensor->get_size() - 1 - i;
                actualDims.push_back(outputTensor->data<uint32_t>()[reverseIdx]);
            }
            auto& tensorToBeReshaped = _outputTensors.at(*outputDescriptor.relatedDescriptorIndex);
            tensorToBeReshaped->set_shape(actualDims);
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

    ++outputIndex;
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
