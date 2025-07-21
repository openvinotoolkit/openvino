// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_infer_request.hpp"

#include <ze_mem_import_system_memory_ext.h>

#include <cstdint>

#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "zero_memory.hpp"
#include "zero_variable_state.hpp"

using namespace intel_npu;

namespace {

constexpr std::size_t SINGLE_TENSOR = 0;
constexpr bool INPUT = true;
constexpr bool OUTPUT = false;

constexpr std::size_t DEFAULT_BATCH_SIZE = 1;

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
    } else if (isInitInputWeightsName(zeDescriptorName)) {
        zeDescriptorName = zeDescriptorName.substr(INIT_INPUT_WEIGHTS_PREFIX.length());
    } else if (isInitOutputWeightsName(zeDescriptorName)) {
        zeDescriptorName = zeDescriptorName.substr(INIT_OUTPUT_WEIGHTS_PREFIX.length());
    } else if (isMainInputWeightsName(zeDescriptorName)) {
        zeDescriptorName = zeDescriptorName.substr(MAIN_INPUT_WEIGHTS_PREFIX.length());
    }

    OPENVINO_ASSERT(ioDescriptor.nameFromCompiler == zeDescriptorName,
                    "Name mismatch between the I/O structure used internally and its Level Zero correspondent: ",
                    ioDescriptor.nameFromCompiler,
                    " vs. ",
                    zeDescriptorName,
                    ". The I/O order may have been altered, which could lead to an erroneous behavior.");
    OPENVINO_ASSERT(ioDescriptor.precision == zeroUtils::toOVElementType(zeDescriptor.info.devicePrecision),
                    "Precision mismatch for input/output named " + ioDescriptor.nameFromCompiler);

    const std::vector<size_t>& ovDimensions = ioDescriptor.shapeFromCompiler.get_max_shape();
    OPENVINO_ASSERT(ovDimensions.size() <= ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE,
                    "Maximum number of dimensions supported: " + std::to_string(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) +
                        '\n' + "Given: " + std::to_string(ovDimensions.size()));

    for (size_t index = 0; index < ovDimensions.size(); ++index) {
        OPENVINO_ASSERT(ovDimensions[index] == zeDescriptor.info.dims[index],
                        "Shape mismatch for input/output named \"" + ioDescriptor.nameFromCompiler +
                            "\" by dimension index: " + std::to_string(index) +
                            ". L0 has: " + std::to_string(zeDescriptor.info.dims[index]) +
                            " but meta has: " + std::to_string(ovDimensions[index]));
    }
    for (size_t index = ovDimensions.size(); index < ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE; ++index) {
        OPENVINO_ASSERT(zeDescriptor.info.dims[index] == 0 || zeDescriptor.info.dims[index] == 1,
                        "Shape mismatch for input/output named " + ioDescriptor.nameFromCompiler);
    }
}

bool memory_and_size_aligned_to_standard_page_size(void* addr, size_t size) {
    auto addr_int = reinterpret_cast<uintptr_t>(addr);

    // addr is aligned to standard page size
    return (addr_int % STANDARD_PAGE_SIZE == 0) && (size % STANDARD_PAGE_SIZE == 0);
}

}  // namespace

//------------------------------------------------------------------------------
ZeroInferRequest::ZeroInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                                   const std::shared_ptr<const ICompiledModel>& compiledModel,
                                   const Config& config)
    : SyncInferRequest(compiledModel, config),
      _initStructs(initStructs),
      _graph(compiledModel->get_graph()),
      _config(config),
      _logger("ZeroInferRequest", config.get<LOG_LEVEL>()),
      _graphInputDescriptors(_graph->get_input_descriptors()),
      _graphOutputDescriptors(_graph->get_output_descriptors()),
      _levelZeroInputTensors(_metadata.inputs.size(), std::vector<std::shared_ptr<ov::ITensor>>(1, nullptr)),
      _levelZeroOutputTensors(_metadata.outputs.size(), nullptr) {
    _logger.debug("ZeroInferRequest::ZeroInferRequest - SyncInferRequest");

    ze_device_external_memory_properties_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES;
    auto res = zeDeviceGetExternalMemoryProperties(_initStructs->getDevice(), &desc);
    if (res == ZE_RESULT_SUCCESS) {
        if (desc.memoryAllocationImportTypes & ZE_EXTERNAL_MEMORY_TYPE_FLAG_STANDARD_ALLOCATION) {
            _externalMemoryStandardAllocationSupported = true;
        }
    }

    _outputAllocator = std::make_shared<const zeroMemory::HostMemAllocator>(_initStructs);
    _inputAllocator =
        std::make_shared<const zeroMemory::HostMemAllocator>(_initStructs, ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED);

    _logger.debug("ZeroInferRequest::ZeroInferRequest - checking level zero attributes and allocating tensors");

    size_t ioIndex = 0;
    auto batchSize = _graph->get_batch_size(_metadata, {}, {});
    if (!_userInputTensors.empty() && !_graphInputDescriptors.empty()) {
        batchSize = _graph->get_batch_size(_metadata, _userInputTensors.at(0), _graphInputDescriptors[0]);
    }
    for (const IODescriptor& inputDescriptor : _metadata.inputs) {
        check_level_zero_attributes_match(inputDescriptor, _graphInputDescriptors.at(ioIndex));

        if (!(inputDescriptor.isStateInput || inputDescriptor.isShapeTensor)) {
            ++ioIndex;
            continue;
        }

        get_level_zero_input(ioIndex) = allocate_tensor(inputDescriptor, ioIndex, INPUT, *_inputAllocator, batchSize);

        ++ioIndex;
    }

    ioIndex = 0;
    for (const IODescriptor& outputDescriptor : _metadata.outputs) {
        check_level_zero_attributes_match(outputDescriptor, _graphOutputDescriptors.at(ioIndex));

        if (!(outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor)) {
            ++ioIndex;
            continue;
        }

        _levelZeroOutputTensors.at(ioIndex) =
            allocate_tensor(outputDescriptor, ioIndex, OUTPUT, *_outputAllocator, batchSize);

        ++ioIndex;
    }

    _logger.debug("ZeroInferRequest::ZeroInferRequest - SyncInferRequest completed");
}

void ZeroInferRequest::create_pipeline() {
    _logger.debug("ZeroInferRequest::create_pipeline");
    auto batchSize = _graph->get_batch_size(_metadata, {}, {});
    if (!_userInputTensors.empty() && !_graphInputDescriptors.empty()) {
        batchSize = _graph->get_batch_size(_metadata, _userInputTensors.at(0), _graphInputDescriptors[0]);
    }
    for (size_t inputIndex = 0; inputIndex < _metadata.inputs.size(); ++inputIndex) {
        if (_metadata.inputs.at(inputIndex).isMainInputWeights) {
            // These values were set while running the "WeightlessGraph::init" method
            continue;
        }

        if (is_batched_input(inputIndex) && batchSize.has_value()) {
            if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
                _logger.debug("ZeroInferRequest::create_pipeline - tensors %s were already allocated",
                              _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str());
            } else {
                for (size_t i = 0; i < get_user_inputs(inputIndex).size(); i++) {
                    get_level_zero_inputs(inputIndex).resize(get_user_inputs(inputIndex).size());

                    _logger.debug("ZeroInferRequest::create_pipeline - allocate new input tensor for batched input: "
                                  "%s, batch_size: %zu",
                                  _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str(),
                                  batchSize.value());

                    get_level_zero_input(inputIndex, i) =
                        allocate_tensor(_metadata.inputs.at(inputIndex), inputIndex, true, *_inputAllocator, batchSize);
                }
            }
            continue;
        }

        if (get_level_zero_input(inputIndex)) {
            _logger.debug("ZeroInferRequest::create_pipeline - tensor %s was already allocated and has size: %zu",
                          _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str(),
                          get_level_zero_input(inputIndex)->get_byte_size());

            continue;
        }

        get_level_zero_input(inputIndex) =
            allocate_tensor(_metadata.inputs.at(inputIndex), inputIndex, INPUT, *_inputAllocator, batchSize);
        _logger.debug("ZeroInferRequest::create_pipeline - new input tensor %s allocated, size: %zu",
                      _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str(),
                      get_level_zero_input(inputIndex)->get_byte_size());
    }

    for (size_t outputIndex = 0; outputIndex < _metadata.outputs.size(); ++outputIndex) {
        if (_levelZeroOutputTensors.at(outputIndex)) {
            _logger.debug("ZeroInferRequest::create_pipeline - tensor %s was already allocated",
                          _metadata.outputs.at(outputIndex).nodeFriendlyName.c_str());
            continue;
        }
        _logger.debug("ZeroInferRequest::create_pipeline - allocate new output tensor %s",
                      _metadata.outputs.at(outputIndex).nodeFriendlyName.c_str());
        _levelZeroOutputTensors.at(outputIndex) =
            allocate_tensor(_metadata.outputs.at(outputIndex), outputIndex, OUTPUT, *_outputAllocator, batchSize);
        _logger.debug("ZeroInferRequest::create_pipeline - new output tensor %s allocated, size: %zu",
                      _metadata.outputs.at(outputIndex).nodeFriendlyName.c_str(),
                      _levelZeroOutputTensors.at(outputIndex)->get_byte_size());
    }
    _logger.debug("ZeroInferRequest::create_pipeline - init completed");

    // Set new tensors and reset variable state flag if memory updated before creating the pipeline
    _logger.debug("ZeroInferRequest::create_pipeline - set new tensors and reset variable state flag if memory updated "
                  "before creating the pipeline");
    for (const auto& variableState : _variableStates) {
        auto zeroState = std::dynamic_pointer_cast<ZeroVariableState>(variableState._ptr);

        OPENVINO_ASSERT(zeroState != nullptr, "State is not compatible with NPU plugin");

        if (zeroState->tensor_was_updated()) {
            get_user_input(zeroState->get_tensor_index()) = zeroState->get_state();
            _userOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_state();

            zeroState->reset_tensor_updated_flag();

            if (zeroState->zero_tensor_should_be_updated()) {
                zeroState->reset_zero_tensor_updated_flag();

                get_level_zero_input(zeroState->get_tensor_index()) = zeroState->get_state()._ptr;
                _levelZeroOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_state()._ptr;
            }
        }
    }

    _logger.debug("ZeroInferRequest::create_pipeline - constructing pipeline");

    // Construct pipeline
    _pipeline = std::make_unique<Pipeline>(_config,
                                           _initStructs,
                                           _graph,
                                           _levelZeroInputTensors,
                                           _levelZeroOutputTensors,
                                           batchSize.has_value() ? batchSize.value() : DEFAULT_BATCH_SIZE);

    _logger.debug("ZeroInferRequest::create_pipeline - SyncInferRequest completed");
}

void ZeroInferRequest::set_tensor_data(const std::shared_ptr<ov::ITensor>& tensor,
                                       const size_t index,
                                       const bool isInput) {
    OV_ITT_TASK_CHAIN(ZERO_SET_TENSOR, itt::domains::LevelZeroBackend, "set_tensor", "set_tensor_data");
    auto& levelZeroTensors = isInput ? get_level_zero_input(index) : _levelZeroOutputTensors.at(index);

    bool updateCommandListArg = false;

    OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "check_data_allocation");
    if (zeroUtils::memory_was_allocated_in_the_same_l0_context(_initStructs->getContext(), tensor->data())) {
        _logger.debug("ZeroInferRequest::set_tensor_data - tensor was created in the same L0 context, size: %zu",
                      tensor->get_byte_size());
        levelZeroTensors = tensor;
        updateCommandListArg = true;
    } else {
        if (_externalMemoryStandardAllocationSupported &&
            memory_and_size_aligned_to_standard_page_size(tensor->data(), tensor->get_byte_size())) {
            _logger.debug("ZeroInferRequest::set_tensor_data - import memory from a system memory pointer");
            auto hostMemSharedAllocator =
                zeroMemory::HostMemSharedAllocator(_initStructs,
                                                   tensor,
                                                   isInput ? ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED : 0);
            levelZeroTensors = std::make_shared<ZeroTensor>(_initStructs,
                                                            _config,
                                                            tensor->get_element_type(),
                                                            tensor->get_shape(),
                                                            hostMemSharedAllocator);

            std::dynamic_pointer_cast<ZeroTensor>(levelZeroTensors)->set_tensor_shared_with_user();

            updateCommandListArg = true;
        } else {
            auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(levelZeroTensors);

            if (zeroTensor == nullptr || (zeroTensor != nullptr && zeroTensor->tensor_was_shared_with_user())) {
                _logger.debug("ZeroInferRequest::set_tensor_data - create locally L0 tensor");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "allocate tensor");

                ov::SoPtr<ov::ITensor> soPtrTensor(tensor);
                std::vector<ov::SoPtr<ov::ITensor>> tensorVector = {soPtrTensor};

                auto batch = _graph->get_batch_size(
                    _metadata,
                    tensorVector,
                    isInput ? _graphInputDescriptors.at(index) : _graphOutputDescriptors.at(index));

                levelZeroTensors = allocate_tensor(isInput ? _metadata.inputs.at(index) : _metadata.outputs.at(index),
                                                   index,
                                                   isInput,
                                                   isInput ? *_inputAllocator : *_outputAllocator,
                                                   batch);

                updateCommandListArg = true;
            }
        }
    }

    if (_pipelineIsCreated && updateCommandListArg) {
        _logger.debug("ZeroInferRequest::infer_async - update command list");

        OPENVINO_ASSERT(levelZeroTensors->data(), "Empty buffer");

        OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "update_graph_arguments");
        _pipeline->update_graph_arguments(
            isInput ? _graph->get_input_descriptors().at(index).idx : _graph->get_output_descriptors().at(index).idx,
            levelZeroTensors->data(),
            levelZeroTensors->get_byte_size());
    }
}

void ZeroInferRequest::set_remote_tensor_data(const std::shared_ptr<ZeroRemoteTensor>& tensor,
                                              const size_t index,
                                              const bool isInput) {
    OV_ITT_TASK_CHAIN(ZERO_SET_REMOTE_TENSOR, itt::domains::LevelZeroBackend, "set_tensor", "set_remote_tensor_data");

    auto l0_context = tensor->get_zero_context_handle();
    if (_initStructs->getContext() != l0_context) {
        OPENVINO_THROW("Using different context for creating the tensor is not supported");
    }

    auto& levelZeroTensors = isInput ? get_level_zero_input(index) : _levelZeroOutputTensors.at(index);
    levelZeroTensors = tensor;

    if (_pipelineIsCreated) {
        _logger.debug("ZeroInferRequest::infer_async - update command list");

        auto data = tensor->get_original_memory();
        OPENVINO_ASSERT(data, "Empty buffer");

        OV_ITT_TASK_NEXT(ZERO_SET_REMOTE_TENSOR, "update_graph_arguments");
        _pipeline->update_graph_arguments(
            isInput ? _graph->get_input_descriptors().at(index).idx : _graph->get_output_descriptors().at(index).idx,
            data,
            tensor->get_byte_size());
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
        // Check if batch has been changed
        if (get_user_input(foundPort.idx) != nullptr) {
            _logger.debug(
                "ZeroInferRequest::set_tensor - check if input tensors may have their sizes changed, existing: %zu and "
                "count: %zu, new: %zu - to determine whether we need for pipeline reallocation",
                get_user_input(foundPort.idx)->get_byte_size(),
                get_user_inputs(foundPort.idx).size(),
                tensor->get_byte_size());
            if (get_user_input(foundPort.idx)->get_byte_size() * get_user_inputs(foundPort.idx).size() !=
                tensor->get_byte_size()) {
                _pipelineNeedsReallocation = true;
            }
        }

        std::vector<ov::SoPtr<ov::ITensor>> tensorVector = {tensor};
        _graph->reset_last_batch_size();
        auto batchSizeCandidate =
            _graph->get_batch_size(_metadata, tensorVector, _graphInputDescriptors.at(foundPort.idx));
        if (is_batched_input(foundPort.idx) || batchSizeCandidate.has_value()) {
            // resize vector size to 1 if set_tensor is called after set_tensors
            get_level_zero_inputs(foundPort.idx).resize(1);
            get_level_zero_inputs(foundPort.idx).shrink_to_fit();
            get_user_inputs(foundPort.idx).resize(1);
            get_user_inputs(foundPort.idx).shrink_to_fit();

            _graph->set_batch_size(batchSizeCandidate.has_value() ? batchSizeCandidate.value() : DEFAULT_BATCH_SIZE);
        }

        get_user_input(foundPort.idx) = tensor;
    } else {
        if (_userOutputTensors.at(foundPort.idx)._ptr == tensor._ptr) {
            // Got set_tensor with the same object here too - do nothing
            _logger.debug("ZeroInferRequest::set_tensor - got the same tensor, do nothing");
            return;
        }
        // Check if batch has been changed
        if (_userOutputTensors.at(foundPort.idx) != nullptr) {
            _logger.debug("ZeroInferRequest::set_tensor - check if tensors have their sizes changed, existing: %zu, "
                          "new: %zu - to determien whether we need for pipeline reallocation",
                          _userOutputTensors.at(foundPort.idx)->get_byte_size(),
                          tensor->get_byte_size());
            if (_userOutputTensors.at(foundPort.idx)->get_byte_size() != tensor->get_byte_size()) {
                _pipelineNeedsReallocation = true;
            }
        }
        _userOutputTensors.at(foundPort.idx) = tensor;
    }

    if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
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

    _logger.debug("ZeroInferRequest::set_tensors: %zu", tensors.size());
    auto batch_size = _graph->get_batch_size(_metadata, tensors, _graphInputDescriptors.at(foundPort.idx));

    if (batch_size.has_value()) {
        _logger.debug("ZeroInferRequest::set_tensors: determined batch: %zu, preallocated L0 tensors: %zu ",
                      batch_size.value(),
                      _levelZeroInputTensors.at(foundPort.idx).size());
        if (tensors.size() != _levelZeroInputTensors.at(foundPort.idx).size() && batch_size.value() != tensors.size()) {
            batch_size = tensors.size();
            _logger.debug("ZeroInferRequest::set_tensors: batch sized has been changed to: %zu", batch_size.value());
            _graph->set_batch_size(tensors.size());
            _pipelineNeedsReallocation = true;
        }
    }

    if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
        if (batch_size) {
            for (size_t i = 0; i < tensors.size(); i++) {
                auto remoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(tensors[i]._ptr);

                get_level_zero_inputs(foundPort.idx).resize(tensors.size());
                void* data = nullptr;

                if (remoteTensor == nullptr) {
                    bool tensorHasSameL0Context = false;

                    OV_ITT_TASK_NEXT(SET_TENSORS, "check_data_allocation");
                    if (zeroUtils::memory_was_allocated_in_the_same_l0_context(_initStructs->getContext(),
                                                                               tensors[i]->data())) {
                        _logger.debug("ZeroInferRequest::set_tensors - tensor was created in the same L0 context");

                        get_level_zero_input(foundPort.idx, i) = tensors.at(i)._ptr;
                        tensorHasSameL0Context = true;
                    }

                    if (!tensorHasSameL0Context) {
                        _logger.debug("ZeroInferRequest::set_tensors - tensor wasn't created in the same L0 context, "
                                      "create a L0 tensor");

                        get_level_zero_input(foundPort.idx, i) = allocate_tensor(_metadata.inputs.at(foundPort.idx),
                                                                                 foundPort.idx,
                                                                                 true,
                                                                                 *_inputAllocator,
                                                                                 batch_size);
                    }

                    data = get_level_zero_input(foundPort.idx, i)->data();
                } else {
                    _logger.debug("ZeroInferRequest::set_tensors - remote tensor is used");

                    data = remoteTensor->get_original_memory();

                    get_level_zero_input(foundPort.idx, i) = tensors.at(i)._ptr;
                }

                if (_pipelineIsCreated && !_pipelineNeedsReallocation) {
                    OPENVINO_ASSERT(data, "Empty buffer");
                    OV_ITT_TASK_NEXT(SET_TENSORS, "updateCommandList");

                    _pipeline->update_graph_arguments_batching(_graph->get_input_descriptors().at(foundPort.idx).idx,
                                                               data,
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

    if (userTensors && !_pipelineNeedsReallocation) {
        auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(userTensors._ptr);
        if (zeroTensor != nullptr) {
            zeroTensor->set_tensor_shared_with_user();
        }

        _logger.debug("ZeroInferRequest::get_tensor - tensor allocated, get the tensor by index: %zu", ioIndex);
        return userTensors;
    }

    auto& metadata = isInput ? _metadata.inputs.at(ioIndex) : _metadata.outputs.at(ioIndex);
    _logger.debug("ZeroInferRequest::get_tensor - tensor by index: %zu is not allocated, or the existing pipeline "
                  "needs reallocation: %s. New tensor %s will be created",
                  ioIndex,
                  _pipelineNeedsReallocation ? "true" : "false",
                  metadata.nodeFriendlyName.c_str());

    auto& levelZeroTensors = isInput ? get_level_zero_input(ioIndex) : _levelZeroOutputTensors.at(ioIndex);

    // LIMITATION for the dynamic batch implementation:
    // We need to allocate output tensors having the same batch size as input tensors.
    // Which means that input tensor batch sizes must have been determined.
    // In other words, it means that someone MUST HAVE called set_tensor() BEFORE
    // asking get_tensor(). Otherwise we won't deduct the actual batch size
    // which we must return here.
    // If we may return wrong batch size here then we must have a mechanism notifying
    // user that that returned tensor is now obsolete, when someone had changed batch using set_tensor()
    // by holding already the old tensor from get_tensor() with the old batch size.
    // OR we must reallocate that tensor by callback
    std::vector<ov::SoPtr<ov::ITensor>> tensorVector;
    _logger.debug("ZeroInferRequest::get_tensor - try to get batch size from input tensors, if output is not created");
    if (isInput) {
        tensorVector.push_back(get_user_input(ioIndex));
    } else {
        tensorVector.push_back(get_user_input(0));
    }

    auto batch_size =
        _graph->get_batch_size(_metadata,
                               tensorVector,
                               isInput ? _graphInputDescriptors.at(ioIndex) : _graphOutputDescriptors.at(ioIndex));

    levelZeroTensors =
        allocate_tensor(metadata, ioIndex, isInput, isInput ? *_inputAllocator : *_outputAllocator, batch_size);

    _logger.debug("ZeroInferRequest::get_tensor - tensor by index: %zu is allocated: %s, size: %zu",
                  ioIndex,
                  metadata.nodeFriendlyName.c_str(),
                  levelZeroTensors->get_byte_size());

    if (!isInput && _pipelineNeedsReallocation) {
        _logger.debug(
            "ZeroInferRequest::get_tensor - set new output tensor as pipeline reallocated required, batch size: %zu",
            batch_size.has_value() ? batch_size.value() : DEFAULT_BATCH_SIZE);
        userTensors = levelZeroTensors;
    }

    auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(levelZeroTensors);
    if (zeroTensor != nullptr) {
        zeroTensor->set_tensor_shared_with_user();
    }

    return userTensors;
}

void ZeroInferRequest::update_pipeline_if_memory_changed() {
    size_t ioIndex = 0;

    for (const auto& levelZeroTensor : _levelZeroInputTensors) {
        const auto& inputDescriptor = _metadata.inputs.at(ioIndex);
        auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(levelZeroTensor.at(SINGLE_TENSOR));

        if (is_batched_input(ioIndex) || inputDescriptor.isShapeTensor ||
            is_remote_tensor(levelZeroTensor.at(SINGLE_TENSOR)) || zeroTensor == nullptr) {
            ++ioIndex;
            continue;
        }

        if (zeroTensor->memory_address_changed()) {
            _logger.debug("Update input graph descriptor with the new tensor");
            OPENVINO_ASSERT(zeroTensor->data(), "Empty buffer");

            _pipeline->update_graph_arguments(_graph->get_input_descriptors().at(ioIndex).idx,
                                              zeroTensor->data(),
                                              zeroTensor->get_byte_size());

            if (!inputDescriptor.isStateInput) {
                zeroTensor->reset_memory_flag();
            }
        }

        ++ioIndex;
    }

    ioIndex = 0;

    for (const auto& levelZeroTensor : _levelZeroOutputTensors) {
        const auto& outputDescriptor = _metadata.outputs.at(ioIndex);
        auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(levelZeroTensor);

        if (outputDescriptor.isShapeTensor || is_remote_tensor(levelZeroTensor) || zeroTensor == nullptr) {
            ++ioIndex;
            continue;
        }

        if (zeroTensor->memory_address_changed()) {
            _logger.debug("Update output graph descriptor with the new tensor");
            OPENVINO_ASSERT(zeroTensor->data(), "Empty buffer");

            _pipeline->update_graph_arguments(_graph->get_output_descriptors().at(ioIndex).idx,
                                              zeroTensor->data(),
                                              zeroTensor->get_byte_size());

            zeroTensor->reset_memory_flag();
        }

        ++ioIndex;
    }
}

void ZeroInferRequest::update_states_if_memory_changed() {
    for (const auto& variableState : _variableStates) {
        auto zeroState = std::dynamic_pointer_cast<ZeroVariableState>(variableState._ptr);

        OPENVINO_ASSERT(zeroState != nullptr, "State is not compatible with NPU plugin");

        if (zeroState->tensor_was_updated()) {
            get_user_input(zeroState->get_tensor_index()) = zeroState->get_state();
            _userOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_state();

            zeroState->reset_tensor_updated_flag();

            if (zeroState->zero_tensor_should_be_updated()) {
                auto remoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(zeroState->get_state()._ptr);

                void* userBuffer = !remoteTensor ? zeroState->get_state()->data() : remoteTensor->get_original_memory();

                _pipeline->update_graph_arguments(_graphInputDescriptors.at(zeroState->get_tensor_index()).idx,
                                                  userBuffer,
                                                  zeroState->get_state()->get_byte_size());

                _pipeline->update_graph_arguments(_graphOutputDescriptors.at(zeroState->get_related_tensor_index()).idx,
                                                  userBuffer,
                                                  zeroState->get_state()->get_byte_size());

                zeroState->reset_zero_tensor_updated_flag();

                get_level_zero_input(zeroState->get_tensor_index()) = zeroState->get_state()._ptr;
                _levelZeroOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_state()._ptr;
            } else {
                if (_externalMemoryStandardAllocationSupported &&
                    memory_and_size_aligned_to_standard_page_size(zeroState->get_state()->data(),
                                                                  zeroState->get_state()->get_byte_size())) {
                    auto hostMemSharedAllocator =
                        zeroMemory::HostMemSharedAllocator(_initStructs, zeroState->get_state()._ptr);

                    get_level_zero_input(zeroState->get_tensor_index()) =
                        std::make_shared<ZeroTensor>(_initStructs,
                                                     _config,
                                                     zeroState->get_state()->get_element_type(),
                                                     zeroState->get_state()->get_shape(),
                                                     hostMemSharedAllocator);

                    _levelZeroOutputTensors.at(zeroState->get_related_tensor_index()) =
                        get_level_zero_input(zeroState->get_tensor_index());

                    _pipeline->update_graph_arguments(
                        _graphInputDescriptors.at(zeroState->get_tensor_index()).idx,
                        _levelZeroOutputTensors.at(zeroState->get_related_tensor_index())->data(),
                        _levelZeroOutputTensors.at(zeroState->get_related_tensor_index())->get_byte_size());

                    _pipeline->update_graph_arguments(
                        _graphOutputDescriptors.at(zeroState->get_related_tensor_index()).idx,
                        _levelZeroOutputTensors.at(zeroState->get_related_tensor_index())->data(),
                        _levelZeroOutputTensors.at(zeroState->get_related_tensor_index())->get_byte_size());
                }
            }
        }
    }
}

void ZeroInferRequest::infer() {
    if (_config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        OPENVINO_THROW("Only start async is supported when RUN_INFERENCES_SEQUENTIALLY is enabled!");
    }

    infer_async();
    get_result();
}

void ZeroInferRequest::infer_async() {
    _logger.debug("InferRequest::infer_async started");
    OV_ITT_TASK_CHAIN(ZERO_INFER, itt::domains::LevelZeroBackend, "infer_async", "start");

    {
        std::lock_guard<std::mutex> lock(_graph->get_mutex());

        if (!_pipelineIsCreated || _pipelineNeedsReallocation) {
            OV_ITT_TASK_NEXT(ZERO_INFER, "create_pipeline");
            create_pipeline();  // Reallocate pipeline if necessary
            _pipelineIsCreated = true;
            _pipelineNeedsReallocation = false;  // Reset reallocation flag
        } else {
            if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
                update_pipeline_if_memory_changed();
                update_states_if_memory_changed();
            }
        }
    }

    size_t inputIndex = 0;
    for (const auto& userTensor : _userInputTensors) {
        const IODescriptor inputDescriptor = _metadata.inputs.at(inputIndex);

        OPENVINO_ASSERT(!inputDescriptor.isInitInputWeights,
                        "This path should not be used for running inferences for the \"init\" model");

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

        auto batch_size =
            _graph->get_batch_size(_metadata, _userInputTensors.at(inputIndex), _graphInputDescriptors.at(inputIndex));
        if (is_batched_input(inputIndex) || batch_size.has_value()) {
            if (batch_size.has_value()) {
                for (size_t i = 0; i < userTensor.size(); i++) {
                    if (!is_remote_tensor(get_level_zero_input(inputIndex, i))) {
                        void* levelZeroBuffer = get_level_zero_input(inputIndex, i)->data();

                        auto userBatchRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userTensor.at(i)._ptr);

                        void* userBuffer = !userBatchRemoteTensor ? userTensor.at(i)->data()
                                                                  : userBatchRemoteTensor->get_original_memory();

                        if (userBuffer != levelZeroBuffer) {
                            if (userBuffer == nullptr || levelZeroBuffer == nullptr) {
                                OPENVINO_THROW("Empty buffer");
                            }

                            _logger.info(
                                "Batched Tensors - Tensor by index: %zu is not allocated in the current Level Zero "
                                "context, copy bytes from user tensor: %zu, into L0 with expected size: %zu",
                                inputIndex,
                                userTensor.at(i)->get_byte_size(),
                                get_level_zero_input(inputIndex, i)->get_byte_size());
                            OV_ITT_TASK_NEXT(ZERO_INFER, "memcpy");
                            std::memcpy(levelZeroBuffer, userBuffer, userTensor.at(i)->get_byte_size());
                        }
                    }
                }
            } else {
                void* levelZeroBuffer = get_level_zero_input(inputIndex)->data();

                _logger.info("Batched Tensors - Tensor by index: %zu is not allocated in the current Level Zero "
                             "context or must be "
                             "in a continued memory space, copy into L0 with size: %zu",
                             inputIndex,
                             get_level_zero_input(inputIndex)->get_byte_size());
                size_t copied_bytes_from_user = 0;
                for (size_t i = 0; i < userTensor.size(); i++) {
                    auto userBatchRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userTensor.at(i)._ptr);

                    void* userBuffer = !userBatchRemoteTensor ? userTensor.at(i)->data()
                                                              : userBatchRemoteTensor->get_original_memory();

                    std::memcpy(static_cast<unsigned char*>(levelZeroBuffer) + (i * userTensor.at(i)->get_byte_size()),
                                userBuffer,
                                userTensor.at(i)->get_byte_size());
                    copied_bytes_from_user += userTensor.at(i)->get_byte_size();
                    _logger.debug("Batched Tensors - Tensor by index: %zu copied bytes: [%zu/%zu]",
                                  inputIndex,
                                  copied_bytes_from_user,
                                  get_level_zero_input(inputIndex)->get_byte_size());
                }
                OPENVINO_ASSERT(get_level_zero_input(inputIndex)->get_byte_size() == copied_bytes_from_user,
                                "Bytes copied must be equal");
            }

            ++inputIndex;
            continue;
        }

        if (inputDescriptor.isMainInputWeights) {
            // These values were set while running the "WeightlessGraph::init" method
            continue;
        }

        auto userRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userTensor.at(SINGLE_TENSOR)._ptr);
        void* userBuffer =
            !userRemoteTensor ? userTensor.at(SINGLE_TENSOR)->data() : userRemoteTensor->get_original_memory();

        const auto& levelZeroTensor = get_level_zero_input(inputIndex);
        if (!is_remote_tensor(levelZeroTensor)) {
            void* levelZeroBuffer = levelZeroTensor->data();
            if (userBuffer == nullptr || levelZeroBuffer == nullptr) {
                OPENVINO_THROW("Empty buffer");
            }

            if (userBuffer != levelZeroBuffer) {
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
        void* userBuffer = !userRemoteTensor ? userTensor->data() : userRemoteTensor->get_original_memory();

        const std::shared_ptr<ov::ITensor>& levelZeroTensor = _levelZeroOutputTensors.at(outputIndex);
        if (!is_remote_tensor(levelZeroTensor)) {
            void* levelZeroBuffer = levelZeroTensor->data();
            if (userBuffer == nullptr || levelZeroBuffer == nullptr) {
                OPENVINO_THROW("Empty buffer");
            }

            if (userBuffer != levelZeroBuffer) {
                _logger.info("Output tensor by index: %zu is not allocated in the current Level Zero context",
                             outputIndex);
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
    case ov::element::Type_t::f8e4m3:
        break;
    case ov::element::Type_t::f8e5m2:
        break;
    case ov::element::Type_t::f8e8m0:
        break;
    case ov::element::Type_t::nf4:
        break;
    case ov::element::Type_t::u2:
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
    case ov::element::Type_t::boolean:
        break;
    default:
        OPENVINO_THROW(
            "Unsupported tensor precision: " + ov::element::Type(precision).get_type_name() +
            "! Supported precisions: FP32, FP16, BF16, FP8, NF4, U2, U4, I4, U8, I8, U16, I16, U32, I32, U64, "
            "I64, FP64, BOOLEAN");
    }
}

std::vector<ov::ProfilingInfo> ZeroInferRequest::get_profiling_info() const {
    OPENVINO_ASSERT(_pipeline, "Profiling information isn't available before running an inference!");

    return _pipeline->get_profiling_info();
}

std::shared_ptr<ov::ITensor> ZeroInferRequest::create_tensor(ov::element::Type type,
                                                             const ov::Shape& shape,
                                                             const ov::Allocator& allocator) const {
    OPENVINO_ASSERT(allocator, "Allocator mush be provided when creating a zero tensor!");

    return std::make_shared<ZeroTensor>(_initStructs, _config, type, shape, allocator);
}

void ZeroInferRequest::add_state(const IODescriptor& descriptor, size_t tensorIndex) const {
    OPENVINO_ASSERT(descriptor.relatedDescriptorIndex.has_value(),
                    "The link between state descriptors is missing, state name: ",
                    descriptor.nameFromCompiler);

    _variableStates.push_back(std::make_shared<ZeroVariableState>(_initStructs,
                                                                  descriptor.nameFromCompiler,
                                                                  get_user_input(tensorIndex),
                                                                  tensorIndex,
                                                                  descriptor.relatedDescriptorIndex.value(),
                                                                  _config));
}

std::shared_ptr<ov::ITensor>& ZeroInferRequest::get_level_zero_input(size_t index, size_t tensorNo) const {
    return _levelZeroInputTensors.at(index).at(tensorNo);
}

std::vector<std::shared_ptr<ov::ITensor>>& ZeroInferRequest::get_level_zero_inputs(size_t index) const {
    return _levelZeroInputTensors.at(index);
}
