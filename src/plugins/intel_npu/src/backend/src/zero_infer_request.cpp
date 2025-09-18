// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_infer_request.hpp"

#include <ze_mem_import_system_memory_ext.h>

#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"

using namespace intel_npu;

namespace {

constexpr std::size_t SINGLE_TENSOR = 0;
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

std::optional<size_t> determine_dynamic_batch_size(const IODescriptor& desc,
                                                   const std::shared_ptr<ov::ITensor>& tensor,
                                                   const std::optional<size_t> batchSize) {
    if (tensor == nullptr && !batchSize.has_value()) {
        return std::nullopt;
    }

    if (!desc.shapeFromIRModel.has_value() || !desc.shapeFromIRModel.value().is_dynamic()) {
        return std::nullopt;
    }

    if (batchSize.has_value()) {
        return batchSize.value();
    }

    if (tensor->get_shape().empty() || *desc.shapeFromCompiler.begin() != intel_npu::utils::DEFAULT_BATCH_SIZE) {
        return std::nullopt;
    }

    if ((*desc.shapeFromIRModel)[intel_npu::utils::BATCH_AXIS].is_dynamic()) {
        return tensor->get_shape()[intel_npu::utils::BATCH_AXIS];
    }

    return std::nullopt;
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
      _levelZeroInputTensors(_metadata.inputs.size(), std::vector<std::shared_ptr<ZeroTensor>>(1, nullptr)),
      _levelZeroOutputTensors(_metadata.outputs.size(), nullptr) {
    _logger.debug("ZeroInferRequest::ZeroInferRequest - checking level zero attributes and allocating tensors");

    size_t ioIndex = 0;
    for (const IODescriptor& inputDescriptor : _metadata.inputs) {
        check_level_zero_attributes_match(inputDescriptor, _graphInputDescriptors.at(ioIndex));

        if (!(inputDescriptor.isStateInput || inputDescriptor.isShapeTensor)) {
            ++ioIndex;
            continue;
        }

        get_level_zero_input(ioIndex) = allocate_tensor(ioIndex, INPUT);

        ++ioIndex;
    }

    ioIndex = 0;
    for (const IODescriptor& outputDescriptor : _metadata.outputs) {
        check_level_zero_attributes_match(outputDescriptor, _graphOutputDescriptors.at(ioIndex));

        if (!(outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor)) {
            ++ioIndex;
            continue;
        }

        _levelZeroOutputTensors.at(ioIndex) = allocate_tensor(ioIndex, OUTPUT);

        ++ioIndex;
    }

    _logger.debug("ZeroInferRequest::ZeroInferRequest - SyncInferRequest completed");
}

void ZeroInferRequest::create_pipeline() {
    _logger.debug("ZeroInferRequest::create_pipeline");
    auto batchSize = _graph->get_batch_size();

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

                    get_level_zero_input(inputIndex, i) = allocate_tensor(inputIndex, INPUT, batchSize);
                }
            }
            continue;
        }

        if (get_level_zero_input(inputIndex)) {
            if (_dynamicBatchValueChanged && batchSize.has_value() &&
                get_level_zero_input(inputIndex)->get_shape()[utils::BATCH_AXIS] != batchSize.value()) {
                OPENVINO_THROW("Input tensor ",
                               _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str(),
                               " has different batch size than other tensors.");
            }

            _logger.debug("ZeroInferRequest::create_pipeline - tensor %s was already allocated and has size: %zu",
                          _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str(),
                          get_level_zero_input(inputIndex)->get_byte_size());

            continue;
        }

        get_level_zero_input(inputIndex) = allocate_tensor(inputIndex, INPUT, batchSize);
        _logger.debug("ZeroInferRequest::create_pipeline - new input tensor %s allocated, size: %zu",
                      _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str(),
                      get_level_zero_input(inputIndex)->get_byte_size());
    }

    for (size_t outputIndex = 0; outputIndex < _metadata.outputs.size(); ++outputIndex) {
        if (_levelZeroOutputTensors.at(outputIndex)) {
            if (_dynamicBatchValueChanged) {
                if (batchSize.has_value() &&
                    _levelZeroOutputTensors.at(outputIndex)->get_shape()[utils::BATCH_AXIS] == batchSize.value()) {
                    _logger.debug("ZeroInferRequest::create_pipeline - tensor %s was already allocated",
                                  _metadata.outputs.at(outputIndex).nodeFriendlyName.c_str());
                    continue;
                }
            } else {
                _logger.debug("ZeroInferRequest::create_pipeline - tensor %s was already allocated",
                              _metadata.outputs.at(outputIndex).nodeFriendlyName.c_str());
                continue;
            }
        }
        _logger.debug("ZeroInferRequest::create_pipeline - allocate new output tensor %s",
                      _metadata.outputs.at(outputIndex).nodeFriendlyName.c_str());

        _levelZeroOutputTensors.at(outputIndex) = allocate_tensor(outputIndex, OUTPUT, batchSize);

        if (_dynamicBatchValueChanged && !_userOutputTensors.at(outputIndex)->get_shape().empty() &&
            _userOutputTensors.at(outputIndex)->get_shape()[utils::BATCH_AXIS] !=
                _levelZeroOutputTensors.at(outputIndex)->get_shape()[utils::BATCH_AXIS]) {
            if (_userOutputTensors.at(outputIndex)._ptr == nullptr) {
                OPENVINO_THROW("Output tensor ",
                               _metadata.outputs.at(outputIndex).nodeFriendlyName.c_str(),
                               " has different batch size than other tensors.");
            }

            _userOutputTensors.at(outputIndex) = _levelZeroOutputTensors.at(outputIndex);
        }

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

        if (zeroState->state_update_pending()) {
            _logger.debug("ZeroInferRequest::create_pipeline - user state tensor should be updated");

            get_user_input(zeroState->get_tensor_index()) = zeroState->get_user_state();
            _userOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_user_state();
            zeroState->clear_state_update_pending();

            if (zeroState->zero_state_update_pending()) {
                _logger.debug("ZeroInferRequest::create_pipeline - level zero state tensor should be updated");

                get_level_zero_input(zeroState->get_tensor_index()) = zeroState->get_zero_state();
                _levelZeroOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_zero_state();
                zeroState->clear_zero_state_update_pending();
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
                                           batchSize.has_value() ? batchSize.value() : utils::DEFAULT_BATCH_SIZE);

    _logger.debug("ZeroInferRequest::create_pipeline - SyncInferRequest completed");
}

void ZeroInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OV_ITT_TASK_CHAIN(ZERO_SET_TENSOR, itt::domains::LevelZeroBackend, "set_tensor", "set_tensor");

    auto foundPort = find_port(port);
    OPENVINO_ASSERT(foundPort.found(), "Cannot find tensor for port ", port);
    try {
        check_tensor(port, tensor);
    } catch (const ov::Exception& ex) {
        OPENVINO_THROW("Failed to set tensor. ", ex.what());
    }

    if (foundPort.is_input()) {
        if (get_user_input(foundPort.idx)._ptr == tensor._ptr) {
            // set_tensor called with the same tensor object; no action needed
            _logger.debug("ZeroInferRequest::set_tensor - got the same tensor, do nothing");
            return;
        }

        auto batchSizeCandidate =
            determine_dynamic_batch_size(_metadata.inputs.at(foundPort.idx), tensor._ptr, std::nullopt);

        if (batchSizeCandidate.has_value()) {
            if (!_dynamicBatchValueChanged) {
                if (get_user_input(foundPort.idx)._ptr != nullptr &&
                    get_user_input(foundPort.idx)->get_byte_size() * get_user_inputs(foundPort.idx).size() !=
                        tensor->get_byte_size()) {
                    _dynamicBatchValueChanged = true;
                    _graph->set_batch_size(batchSizeCandidate.value());
                } else if (_graph->get_batch_size().has_value()) {
                    if (batchSizeCandidate.value() != _graph->get_batch_size().value()) {
                        _dynamicBatchValueChanged = true;
                        _graph->set_batch_size(batchSizeCandidate.value());
                    }
                } else {
                    _graph->set_batch_size(batchSizeCandidate.value());
                }
            } else if (batchSizeCandidate.value() != _graph->get_batch_size().value()) {
                OPENVINO_THROW("Batching size is not matching all the tensors.");
            }
        }

        if (is_batched_input(foundPort.idx)) {
            // Reset vector size to 1 if set_tensor is called after set_tensors
            get_level_zero_inputs(foundPort.idx).resize(1);
            get_level_zero_inputs(foundPort.idx).shrink_to_fit();
            get_level_zero_input(foundPort.idx) = {};
            get_user_inputs(foundPort.idx).resize(1);
            get_user_inputs(foundPort.idx).shrink_to_fit();
            get_user_input(foundPort.idx) = {};
        }

        get_user_input(foundPort.idx) = tensor;
    } else {
        if (_userOutputTensors.at(foundPort.idx)._ptr == tensor._ptr) {
            // set_tensor called with the same tensor object; no action needed
            _logger.debug("ZeroInferRequest::set_tensor - got the same tensor, do nothing");
            return;
        }

        auto batchSizeCandidate =
            determine_dynamic_batch_size(_metadata.outputs.at(foundPort.idx), tensor._ptr, std::nullopt);

        if (batchSizeCandidate.has_value()) {
            if (!_dynamicBatchValueChanged) {
                if (_userOutputTensors.at(foundPort.idx)._ptr != nullptr &&
                    _userOutputTensors.at(foundPort.idx)->get_byte_size() != tensor->get_byte_size()) {
                    _dynamicBatchValueChanged = true;
                    _graph->set_batch_size(batchSizeCandidate.value());
                } else if (_graph->get_batch_size().has_value()) {
                    if (batchSizeCandidate.value() != _graph->get_batch_size().value()) {
                        _dynamicBatchValueChanged = true;
                        _graph->set_batch_size(batchSizeCandidate.value());
                    }
                } else {
                    _graph->set_batch_size(batchSizeCandidate.value());
                }
            } else if (batchSizeCandidate.value() != _graph->get_batch_size().value()) {
                OPENVINO_THROW("Batching size is not matching all the tensors.");
            }
        }

        _userOutputTensors.at(foundPort.idx) = tensor;
    }

    if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
        auto& levelZeroTensor =
            foundPort.is_input() ? get_level_zero_input(foundPort.idx) : _levelZeroOutputTensors.at(foundPort.idx);

        bool updateCommandListArg = false;

        try {
            _logger.debug("ZeroInferRequest::set_tensor - create zero tensor");
            OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "create zero tensor");
            // Try to use the user tensor directly if its underlying data is already allocated in the same Level Zero
            // context.
            levelZeroTensor = std::make_shared<ZeroTensor>(_initStructs, tensor, _config);
            updateCommandListArg = true;
        } catch (const ZeroTensorException&) {
            // Check if the current Level Zero tensor was previously shared with the user. If so, it cannot be reused;
            // allocate a new tensor to back up the user tensor (which cannot be imported or used directly).
            if (_dynamicBatchValueChanged || levelZeroTensor == nullptr || !levelZeroTensor->can_be_reused()) {
                _logger.debug("ZeroInferRequest::set_tensor - allocate locally L0 tensor");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "allocate tensor");

                auto batch = _graph->get_batch_size();
                levelZeroTensor = allocate_tensor(foundPort.idx, foundPort.is_input(), batch);
                updateCommandListArg = true;
            } else {
                _logger.debug("ZeroInferRequest::set_tensor - reusing the level zero tensor since it is not shared "
                              "with the user");
            }
        }

        if (_pipelineIsCreated && updateCommandListArg && !_dynamicBatchValueChanged) {
            _logger.debug("ZeroInferRequest::infer_async - update command list");

            OPENVINO_ASSERT(levelZeroTensor->data(), "Empty buffer");

            OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "update_graph_arguments");
            _pipeline->update_graph_arguments(foundPort.is_input()
                                                  ? _graph->get_input_descriptors().at(foundPort.idx).idx
                                                  : _graph->get_output_descriptors().at(foundPort.idx).idx,
                                              levelZeroTensor->data(),
                                              levelZeroTensor->get_byte_size());
        }
    }
    // If command list updates are not supported, fallback to copying tensors every time.
}

void ZeroInferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                   const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    OV_ITT_TASK_CHAIN(ZERO_SET_TENSORS, itt::domains::LevelZeroBackend, "set_tensors", "set_tensors");
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

    _logger.debug("ZeroInferRequest::set_tensors: %zu", tensors.size());

    auto batchSizeCandidate = determine_dynamic_batch_size(_metadata.inputs.at(foundPort.idx), nullptr, tensors.size());

    // Check if batch has been changed
    if (batchSizeCandidate.has_value()) {
        if (!_dynamicBatchValueChanged) {
            if (get_user_inputs(foundPort.idx).size() != tensors.size()) {
                _dynamicBatchValueChanged = true;
                _graph->set_batch_size(batchSizeCandidate.value());
            } else if (_graph->get_batch_size().has_value()) {
                if (batchSizeCandidate.value() != _graph->get_batch_size().value()) {
                    _dynamicBatchValueChanged = true;
                    _graph->set_batch_size(batchSizeCandidate.value());
                }
            } else {
                _graph->set_batch_size(batchSizeCandidate.value());
            }
        } else if (batchSizeCandidate.value() != _graph->get_batch_size().value()) {
            OPENVINO_THROW("Batching size is not matching all the tensors.");
        }
    } else {
        batchSizeCandidate = _graph->get_batch_size();
    }

    get_user_inputs(foundPort.idx).resize(tensors.size());
    get_user_inputs(foundPort.idx) = tensors;

    if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0) && batchSizeCandidate.has_value()) {
        get_level_zero_inputs(foundPort.idx).resize(tensors.size());

        for (size_t i = 0; i < tensors.size(); i++) {
            try {
                _logger.debug("ZeroInferRequest::set_tensors - create zero tensor");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSORS, "create zero tensor");

                get_level_zero_input(foundPort.idx, i) =
                    std::make_shared<ZeroTensor>(_initStructs, tensors.at(i), _config);
            } catch (const ZeroTensorException&) {
                _logger.debug("ZeroInferRequest::set_tensors - allocate locally L0 tensor");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSORS, "allocate tensor");

                get_level_zero_input(foundPort.idx, i) = allocate_tensor(foundPort.idx, INPUT, batchSizeCandidate);
            }

            if (_pipelineIsCreated && !_dynamicBatchValueChanged) {
                OPENVINO_ASSERT(get_level_zero_input(foundPort.idx, i)->data(), "Empty buffer");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSORS, "updateCommandList");

                _pipeline->update_graph_arguments_batching(_graph->get_input_descriptors().at(foundPort.idx).idx,
                                                           get_level_zero_input(foundPort.idx, i)->data(),
                                                           i);
            }
        }
    }
    // If command list updates are not supported, fallback to copying tensors every time.
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

    auto& userTensor = isInput ? get_user_input(ioIndex) : _userOutputTensors.at(ioIndex);

    auto batchSize = _graph->get_batch_size();

    // LIMITATION for dynamic batch implementation:
    // Output tensors must have the same batch size as input tensors, so input batch sizes must be determined first.
    // This means set_tensor() MUST be called before get_tensor().
    // Otherwise, the batch size returned may be incorrect.
    // If the batch size changes after get_tensor(), the user must be notified that the returned tensor is obsolete,
    // or the tensor must be reallocated via a callback.

    if (userTensor) {
        if (!_dynamicBatchValueChanged) {
            _logger.debug("ZeroInferRequest::get_tensor - tensor allocated, get the tensor by index: %zu", ioIndex);

            auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(userTensor._ptr);
            if (zeroTensor != nullptr) {
                zeroTensor->prevent_reuse();
            }

            return userTensor;
        } else {
            if (batchSize.has_value() && userTensor->get_shape()[utils::BATCH_AXIS] == batchSize.value()) {
                _logger.debug("ZeroInferRequest::get_tensor - tensor by index: %zu is already allocated", ioIndex);

                auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(userTensor._ptr);
                if (zeroTensor != nullptr) {
                    zeroTensor->prevent_reuse();
                }

                return userTensor;
            }
            // If different shape was found, go further and allocate new zero tensor for it.
        }
    }

    auto& metadata = isInput ? _metadata.inputs.at(ioIndex) : _metadata.outputs.at(ioIndex);
    _logger.debug("ZeroInferRequest::get_tensor - tensor by index: %zu is not allocated.New tensor %s will be created",
                  ioIndex,
                  metadata.nodeFriendlyName.c_str());

    auto& levelZeroTensor = isInput ? get_level_zero_input(ioIndex) : _levelZeroOutputTensors.at(ioIndex);
    levelZeroTensor = allocate_tensor(ioIndex, isInput, batchSize);

    if (!_dynamicBatchValueChanged) {
        userTensor = levelZeroTensor;
    }

    levelZeroTensor->prevent_reuse();

    return levelZeroTensor;
}

std::shared_ptr<ZeroTensor> ZeroInferRequest::allocate_tensor(const size_t index,
                                                              const bool isInput,
                                                              const std::optional<std::size_t> batchSize) const {
    const auto& descriptor = isInput ? _metadata.inputs.at(index) : _metadata.outputs.at(index);
    check_network_precision(descriptor.precision);

    std::shared_ptr<ZeroTensor> tensor;
    ov::Shape allocatedTensorShape = descriptor.shapeFromCompiler.get_max_shape();

    if (batchSize.has_value()) {
        allocatedTensorShape[utils::BATCH_AXIS] = *batchSize;
    }

    if (descriptor.isStateOutput) {
        // Only one buffer is required for each (state input, state output) pair, acting as an input before running the
        // inference and as an output after performing it. Thus both the "state input" and "state output" entries shall
        // point to the same buffer.
        OPENVINO_ASSERT(descriptor.relatedDescriptorIndex.has_value(),
                        "The link between state descriptors is missing, state name: ",
                        descriptor.nameFromCompiler);
        tensor = get_level_zero_input(*descriptor.relatedDescriptorIndex);
    } else {
        tensor =
            std::make_shared<ZeroTensor>(_initStructs, _config, descriptor.precision, allocatedTensorShape, isInput);
    }

    if (isInput) {
        if (get_user_input(index) == nullptr) {
            get_user_input(index) = tensor;
        }

        if (descriptor.isStateInput) {
            add_state(descriptor, index, tensor);
        }
    } else if (_userOutputTensors.at(index) == nullptr) {
        _userOutputTensors.at(index) = tensor;
    }

    return tensor;
}

void ZeroInferRequest::update_pipeline_if_memory_changed() {
    size_t ioIndex = 0;

    for (const auto& levelZeroTensor : _levelZeroInputTensors) {
        const auto& inputDescriptor = _metadata.inputs.at(ioIndex);

        if (is_batched_input(ioIndex) || inputDescriptor.isShapeTensor ||
            levelZeroTensor.at(SINGLE_TENSOR) == nullptr) {
            ++ioIndex;
            continue;
        }

        if (levelZeroTensor.at(SINGLE_TENSOR)->memory_address_changed()) {
            if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
                OPENVINO_THROW("Reallocation of zero memory is not supported with this driver.");
            }

            _logger.debug("Update input graph descriptor with the new tensor");
            OPENVINO_ASSERT(levelZeroTensor.at(SINGLE_TENSOR)->data(), "Empty buffer");

            _pipeline->update_graph_arguments(_graph->get_input_descriptors().at(ioIndex).idx,
                                              levelZeroTensor.at(SINGLE_TENSOR)->data(),
                                              levelZeroTensor.at(SINGLE_TENSOR)->get_byte_size());

            if (!inputDescriptor.isStateInput) {
                levelZeroTensor.at(SINGLE_TENSOR)->reset_memory_flag();
            }
        }

        ++ioIndex;
    }

    ioIndex = 0;

    for (const auto& levelZeroTensor : _levelZeroOutputTensors) {
        const auto& outputDescriptor = _metadata.outputs.at(ioIndex);

        if (outputDescriptor.isShapeTensor || levelZeroTensor == nullptr) {
            ++ioIndex;
            continue;
        }

        if (levelZeroTensor->memory_address_changed()) {
            if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
                OPENVINO_THROW("Reallocation of zero memory is not supported with this driver.");
            }

            _logger.debug("Update output graph descriptor with the new tensor");
            OPENVINO_ASSERT(levelZeroTensor->data(), "Empty buffer");

            _pipeline->update_graph_arguments(_graph->get_output_descriptors().at(ioIndex).idx,
                                              levelZeroTensor->data(),
                                              levelZeroTensor->get_byte_size());

            levelZeroTensor->reset_memory_flag();
        }

        ++ioIndex;
    }
}

void ZeroInferRequest::update_states_if_memory_changed() {
    for (const auto& variableState : _variableStates) {
        auto zeroState = std::dynamic_pointer_cast<ZeroVariableState>(variableState._ptr);
        OPENVINO_ASSERT(zeroState != nullptr, "State is not compatible with NPU plugin");

        if (zeroState->state_update_pending()) {
            get_user_input(zeroState->get_tensor_index()) = zeroState->get_user_state();
            _userOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_user_state();
            zeroState->clear_state_update_pending();

            // If command list updates are not supported, fallback to copying tensors every time.
            if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0) &&
                zeroState->zero_state_update_pending()) {
                get_level_zero_input(zeroState->get_tensor_index()) = zeroState->get_zero_state();
                _levelZeroOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_zero_state();
                zeroState->clear_zero_state_update_pending();

                _pipeline->update_graph_arguments(_graphInputDescriptors.at(zeroState->get_tensor_index()).idx,
                                                  get_level_zero_input(zeroState->get_tensor_index())->data(),
                                                  get_level_zero_input(zeroState->get_tensor_index())->get_byte_size());

                _pipeline->update_graph_arguments(
                    _graphOutputDescriptors.at(zeroState->get_related_tensor_index()).idx,
                    _levelZeroOutputTensors.at(zeroState->get_related_tensor_index())->data(),
                    _levelZeroOutputTensors.at(zeroState->get_related_tensor_index())->get_byte_size());
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

        if (!_pipelineIsCreated || _dynamicBatchValueChanged) {
            OV_ITT_TASK_NEXT(ZERO_INFER, "create_pipeline");
            create_pipeline();  // Reallocate pipeline if necessary
            _pipelineIsCreated = true;
            _dynamicBatchValueChanged = false;  // Reset reallocation flag
        } else {
            update_pipeline_if_memory_changed();
            update_states_if_memory_changed();
        }
    }

    auto batch_size = _graph->get_batch_size();
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

        // This path is used only for batched input when set_tensors is called and userTensor vector size is greater
        // than 1.
        // There are two cases:
        // 1. Batch size is set and batching is handled by the plugin.
        // 2. Batch size is not set and batching is handled by the compiler.
        if (is_batched_input(inputIndex)) {
            if (batch_size.has_value()) {
                for (size_t i = 0; i < userTensor.size(); i++) {
                    void* levelZeroBuffer = get_level_zero_input(inputIndex, i)->data();

                    auto userBatchRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userTensor.at(i)._ptr);

                    void* userBuffer = !userBatchRemoteTensor ? userTensor.at(i)->data()
                                                              : userBatchRemoteTensor->get_original_memory();

                    if (userBuffer != levelZeroBuffer) {
                        if (userBuffer == nullptr || levelZeroBuffer == nullptr) {
                            OPENVINO_THROW("Empty buffer");
                        }

                        _logger.debug(
                            "Batched Tensors - Tensor by index: %zu is not allocated in the current Level Zero "
                            "context, copy bytes from user tensor: %zu, into L0 with expected size: %zu",
                            inputIndex,
                            userTensor.at(i)->get_byte_size(),
                            get_level_zero_input(inputIndex, i)->get_byte_size());
                        OV_ITT_TASK_NEXT(ZERO_INFER, "memcpy");
                        std::memcpy(levelZeroBuffer, userBuffer, userTensor.at(i)->get_byte_size());
                    }
                }
            } else {
                void* levelZeroBuffer = get_level_zero_input(inputIndex)->data();

                _logger.debug("Batched Tensors - Tensor by index: %zu is not allocated in the current Level Zero "
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
        void* levelZeroBuffer = get_level_zero_input(inputIndex)->data();

        if (userBuffer == nullptr || levelZeroBuffer == nullptr) {
            OPENVINO_THROW("Empty buffer");
        }

        if (userBuffer != levelZeroBuffer) {
            _logger.info("Tensor is not allocated in the current Level Zero context");
            OV_ITT_TASK_NEXT(ZERO_INFER, "memcpy");
            std::memcpy(levelZeroBuffer, userBuffer, userTensor.at(SINGLE_TENSOR)->get_byte_size());
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
        void* levelZeroBuffer = _levelZeroOutputTensors.at(outputIndex)->data();

        if (userBuffer == nullptr || levelZeroBuffer == nullptr) {
            OPENVINO_THROW("Empty buffer");
        }

        if (userBuffer != levelZeroBuffer) {
            _logger.info("Output tensor by index: %zu is not allocated in the current Level Zero context", outputIndex);
            OV_ITT_TASK_NEXT(ZERO_RESULT, "memcpy");
            std::memcpy(userBuffer, levelZeroBuffer, userTensor->get_byte_size());
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

void ZeroInferRequest::add_state(const IODescriptor& descriptor,
                                 size_t tensorIndex,
                                 const std::shared_ptr<ZeroTensor>& zeroTensor) const {
    OPENVINO_ASSERT(descriptor.relatedDescriptorIndex.has_value(),
                    "The link between state descriptors is missing, state name: ",
                    descriptor.nameFromCompiler);

    _variableStates.push_back(std::make_shared<ZeroVariableState>(_initStructs,
                                                                  descriptor.nameFromCompiler,
                                                                  zeroTensor,
                                                                  tensorIndex,
                                                                  descriptor.relatedDescriptorIndex.value(),
                                                                  _config));
}

std::shared_ptr<ZeroTensor>& ZeroInferRequest::get_level_zero_input(size_t index, size_t tensorNo) const {
    return _levelZeroInputTensors.at(index).at(tensorNo);
}

std::vector<std::shared_ptr<ZeroTensor>>& ZeroInferRequest::get_level_zero_inputs(size_t index) const {
    return _levelZeroInputTensors.at(index);
}
