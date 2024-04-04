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

/**
 * @brief Checks that the metadata of the provided descriptor corresponds to the values registered in the Level Zero
 * structure.
 * @param nodeDescriptor The OpenVINO API specific I/O descriptor which shall be compared.
 * @param zeDescriptor The Level Zero specific structure used for comparison.
 * @param name Tensor identifier used for error logging.
 */
void check_level_zero_attributes_match(const IONodeDescriptor& nodeDescriptor,
                                       const ZeroExecutor::ArgumentDescriptor& zeDescriptor,
                                       const std::string& name) {
    const ov::element::Type_t ovPrecision = nodeDescriptor.precision;
    const ze_graph_argument_precision_t zePrecision = zeDescriptor.info.devicePrecision;

    if (zeroUtils::getZePrecision(ovPrecision) != zePrecision) {
        OPENVINO_THROW("Precision mismatch for parameter " + name);
    }

    const std::vector<size_t>& ovDimensions = nodeDescriptor.originalShape.get_max_shape();

    if (ovDimensions.size() > ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) {
        OPENVINO_THROW(
            "Maximum number of dimensions supported: " + std::to_string(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) + '\n' +
            "Given: " + std::to_string(ovDimensions.size()));
    }

    for (size_t index = 0; index < ovDimensions.size(); ++index) {
        if (ovDimensions[index] != zeDescriptor.info.dims[index] && !nodeDescriptor.originalShape.is_dynamic()) {
            OPENVINO_THROW("Shape mismatch for parameter " + name);
        }
    }
    for (size_t index = ovDimensions.size(); index < ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE; ++index) {
        if (zeDescriptor.info.dims[index] != 0 && zeDescriptor.info.dims[index] != 1) {
            OPENVINO_THROW("Shape mismatch for parameter " + name);
        }
    }
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
      _profiling_pool(_executor->graph(),
                      zeroProfiling::POOL_SIZE,
                      _executor->getInitStructs()->getProfilingDdiTable()),
      _profiling_query(0,
                       _executor->getInitStructs()->getDevice(),
                       _executor->getInitStructs()->getProfilingDdiTable()) {
    const std::unordered_map<std::string, ZeroExecutor::ArgumentDescriptor>& executorInputDescriptors =
        _executor->inputs_desc_map();
    const std::unordered_map<std::string, ZeroExecutor::ArgumentDescriptor>& executorOutputDescriptors =
        _executor->outputs_desc_map();

    auto proftype = config.get<PROFILING_TYPE>();
    if (proftype == ov::intel_npu::ProfilingType::INFER) {
        _npu_profiling = std::make_shared<zeroProfiling::NpuInferProfiling>(_executor->getInitStructs()->getContext(),
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

    for (const std::string& inputName : _metadata.inputNames) {
        if (!executorInputDescriptors.count(inputName)) {
            OPENVINO_THROW("Invalid graph input descriptor key: " + inputName);
        }

        const IONodeDescriptor& parameterDescriptor = _metadata.parameters.at(inputName);
        check_level_zero_attributes_match(parameterDescriptor, executorInputDescriptors.at(inputName), inputName);

        ov::Allocator allocator;
        if (properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) {
            allocator = zeroMemory::HostMemAllocator(backendPtr, ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED);
        } else {
            allocator = zeroMemory::HostMemAllocator(backendPtr);
        }

        // The I/O buffers already allocated using the Level Zero API are being reused here
        allocate_tensor(inputName, parameterDescriptor, TensorType::InputOrOutput, allocator);

        if (contains(_metadata.shapeNames, inputName)) {
            const std::string shapeBufferName = SHAPE_TENSOR_PREFIX + inputName;
            const IONodeDescriptor& shapeDescriptor = _metadata.shapes.at(inputName);

            check_level_zero_attributes_match(shapeDescriptor,
                                              executorInputDescriptors.at(shapeBufferName),
                                              shapeBufferName);

            auto allocator = zeroMemory::HostMemAllocator(backendPtr);
            allocate_tensor(inputName, shapeDescriptor, TensorType::Shape, allocator);
        }
    }

    for (const std::string& outputName : _metadata.outputNames) {
        if (!executorOutputDescriptors.count(outputName)) {
            OPENVINO_THROW("Invalid graph output descriptor key: " + outputName);
        }

        const IONodeDescriptor& resultDescriptor = _metadata.results.at(outputName);
        check_level_zero_attributes_match(resultDescriptor, executorOutputDescriptors.at(outputName), outputName);

        auto allocator = zeroMemory::HostMemAllocator(backendPtr);

        allocate_tensor(outputName, resultDescriptor, TensorType::InputOrOutput, allocator);

        if (contains(_metadata.shapeNames, outputName)) {
            const std::string shapeBufferName = SHAPE_TENSOR_PREFIX + outputName;
            const IONodeDescriptor& shapeDescriptor = _metadata.shapes.at(outputName);

            check_level_zero_attributes_match(shapeDescriptor,
                                              executorOutputDescriptors.at(shapeBufferName),
                                              shapeBufferName);

            auto allocator = zeroMemory::HostMemAllocator(backendPtr);
            allocate_tensor(outputName, shapeDescriptor, TensorType::Shape, allocator);
        }
    }

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
        check_level_zero_attributes_match(stateDescriptor,
                                          executorInputDescriptors.at(stateInputBufferName),
                                          stateInputBufferName);
        check_level_zero_attributes_match(stateDescriptor,
                                          executorOutputDescriptors.at(stateOutputBufferName),
                                          stateOutputBufferName);

        auto allocator = zeroMemory::HostMemAllocator(backendPtr);

        // Only one buffer per state variable is required, we'll use the "output" one since this one captures the latest
        // tensor value
        allocate_tensor(stateName, stateDescriptor, TensorType::State, allocator);
    }

    /// Construct pipepline
    _pipeline = makePipeline(_executorPtr, _config, _profiling_pool, _profiling_query, _npu_profiling, _copyAllTensors);
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

    _pipeline->push();
}

void ZeroInferRequest::get_result() {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "get_result");

    _pipeline->pull();

    for (const auto& name : _outputAndStateOutputNames) {
        const auto& outputTensor = _allTensors.at(name);
        const auto& wrapperOutputTensor = _copyAllTensors.at(name);

        if (isShapeTensorName(name)) {
            const auto actualTensorName = name.substr(SHAPE_TENSOR_PREFIX.size());
            ov::Shape actualDims;
            actualDims.reserve(outputTensor->get_size());

            for (size_t i = 0; i < outputTensor->get_size(); ++i) {
                const auto reverseIdx = outputTensor->get_size() - 1 - i;
                actualDims.push_back(outputTensor->data<uint32_t>()[reverseIdx]);
            }
            auto& tensorToBeReshaped = _allTensors.at(actualTensorName);
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

    _pipeline->reset();
    _logger.debug("InferRequest::get_result finished");
}

void ZeroInferRequest::check_network_precision(const ov::element::Type_t precision) {
    switch (precision) {
    case ov::element::Type_t::f32:
        break;
    case ov::element::Type_t::f16:
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
    default:
        OPENVINO_THROW("Unsupported tensor precision: " + ov::element::Type(precision).get_type_name() +
                       "! Supported precisions: FP32, FP16, U8, I8, U16, I16, U32, I32, U64, I64");
    }
}

std::vector<ov::ProfilingInfo> ZeroInferRequest::get_profiling_info() const {
    const auto& compiledModel = *std::dynamic_pointer_cast<const ICompiledModel>(_compiledModel);
    const auto& compilerConfig = compiledModel.get_config();
    if (!compilerConfig.get<PERF_COUNT>() || !_config.get<PERF_COUNT>()) {
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
        return compiler->process_profiling_output(profData, blob, compilerConfig);
    } else {
        auto proftype = _config.get<PROFILING_TYPE>();
        if (proftype == ov::intel_npu::ProfilingType::INFER) {
            return _npu_profiling->getNpuInferStatistics();
        } else {  /// proftype = MODEL or undefined = fallback to model profiling
            return _profiling_query.getLayerStatistics();
        }
    }
}

std::vector<uint8_t> ZeroInferRequest::get_raw_profiling_data() const {
    return _profiling_query.getData<uint8_t>();
}
