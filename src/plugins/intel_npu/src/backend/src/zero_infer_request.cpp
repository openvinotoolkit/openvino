// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_infer_request.hpp"

#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/plugin_itt.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/utils/utils.hpp"
#include "zero_variable_state.hpp"

namespace intel_npu {

namespace {

std::shared_ptr<const ov::ICompiledModel> validate_compiled_model(
    const std::shared_ptr<const ov::ICompiledModel>& compiledModel) {
    OPENVINO_ASSERT(compiledModel, "Inference request creation: compiled model is not available");
    return compiledModel;
}

}  // namespace

std::optional<size_t> determine_dynamic_batch_size(const IODescriptor& desc,
                                                   const ov::PartialShape& ioShape,
                                                   const std::shared_ptr<ov::ITensor>& tensor,
                                                   const std::optional<size_t> batchSize) {
    if (tensor == nullptr && !batchSize.has_value()) {
        return std::nullopt;
    }

    if (!ioShape.size()) {
        return std::nullopt;
    }

    auto batchFromModel = ioShape[intel_npu::utils::BATCH_AXIS];
    auto batchModelFromIR =
        desc.shapeFromIRModel.has_value() && desc.shapeFromIRModel.value()[intel_npu::utils::BATCH_AXIS].is_dynamic();
    if (!batchFromModel.is_dynamic() && !batchModelFromIR) {
        return std::nullopt;
    }

    if (batchSize.has_value()) {
        return batchSize.value();
    }

    if (tensor->get_shape().empty() || *desc.shapeFromCompiler.begin() != intel_npu::utils::DEFAULT_BATCH_SIZE) {
        return std::nullopt;
    }

    return tensor->get_shape()[intel_npu::utils::BATCH_AXIS];
}

void* get_tensor_data_ptr(const std::shared_ptr<ov::ITensor>& tensor) {
    if (auto userRemoteTensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor)) {
        if (auto userZeroRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userRemoteTensor)) {
            return userZeroRemoteTensor->get_original_memory();
        } else {
            std::optional<void*> memHandleObject =
                zeroUtils::extract_object(userRemoteTensor->get_properties(), ov::intel_npu::mem_handle);
            OPENVINO_ASSERT(memHandleObject.has_value(),
                            "Remote tensor does not have parameter with key ",
                            ov::intel_npu::mem_handle.name());
            return static_cast<uint8_t*>(memHandleObject.value()) + ov::get_tensor_data_offset(*userRemoteTensor);
        }
    } else {
        return tensor->data();
    }
}

//------------------------------------------------------------------------------
ZeroInferRequest::ZeroInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                                   const std::shared_ptr<const ICompiledModel>& compiledModel,
                                   const Config& config)
    : _initStructs(initStructs),
      _compiledModel(validate_compiled_model(compiledModel)),
      _graph(compiledModel->get_graph()),
      _metadata(_graph->get_metadata()),
      _config(config),
      _userInputTensors(_metadata.inputs.size(), std::vector<ov::SoPtr<ov::ITensor>>(1, {nullptr})),
      _userOutputTensors(_metadata.outputs.size(), {nullptr}),
      _levelZeroInputTensors(_metadata.inputs.size(), std::vector<std::shared_ptr<ZeroTensor>>(1, nullptr)),
      _levelZeroOutputTensors(_metadata.outputs.size(), nullptr),
      _logger("ZeroInferRequest", _config.get<LOG_LEVEL>()) {
    OPENVINO_ASSERT(!get_outputs().empty(), "Inference request creation: no output found for network ", _metadata.name);

    // Create map of empty tensors and cache ports from the compiled model
    // See the ov::ISyncInferRequest constructor
    auto portType = ZeroInferRequest::FoundPort::Type::INPUT;
    for (const auto& ports : {get_inputs(), get_outputs()}) {
        for (size_t i = 0; i < ports.size(); i++) {
            const auto& port = ports[i];
            size_t portHash = ov::util::hash_combine(
                {std::hash<const ov::Node*>()(port.get_node()), std::hash<size_t>()(port.get_index())});
            _cachedPorts[portHash] = {i, portType};
        }
        portType = ZeroInferRequest::FoundPort::Type::OUTPUT;
    }

    _logger.debug("ZeroInferRequest - checking level zero attributes and allocating tensors");
    size_t ioIndex = 0;
    for (const IODescriptor& inputDescriptor : _metadata.inputs) {
        // Tensors for regular inputs will be allocated later, only for ports that were not set by the user.
        // Allocating only tensors for shapes and states.
        if (!(inputDescriptor.isStateInput || inputDescriptor.isShapeTensor)) {
            ++ioIndex;
            continue;
        }

        if (inputDescriptor.isShapeTensor) {
            _isShapeTensorPresent = true;
        }

        get_level_zero_input(ioIndex) = allocate_tensor(ioIndex, INPUT);

        if (inputDescriptor.isStateInput) {
            add_state(inputDescriptor, ioIndex);
        }

        ++ioIndex;
    }

    ioIndex = 0;
    for (const IODescriptor& outputDescriptor : _metadata.outputs) {
        // Tensors for regular outputs will be allocated later, only for ports that were not set by the user.
        // Allocating only tensors for shapes and states.
        if (!(outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor)) {
            ++ioIndex;
            continue;
        }

        if (outputDescriptor.isStateOutput) {
            // Only one buffer is required for each (state input, state output) pair, acting as an input before running
            // the inference and as an output after performing it. Thus both the "state input" and "state output"
            // entries shall point to the same buffer.
            OPENVINO_ASSERT(outputDescriptor.relatedDescriptorIndex.has_value(),
                            "The link between state descriptors is missing, state name: ",
                            outputDescriptor.nameFromCompiler);
            _levelZeroOutputTensors.at(ioIndex) = get_level_zero_input(*outputDescriptor.relatedDescriptorIndex);
            _userOutputTensors.at(ioIndex) = _levelZeroOutputTensors.at(ioIndex);

            ++ioIndex;
            continue;
        }

        if (outputDescriptor.isShapeTensor) {
            _isShapeTensorPresent = true;
        }

        _levelZeroOutputTensors.at(ioIndex) = allocate_tensor(ioIndex, OUTPUT);

        ++ioIndex;
    }

    initialize_states();

    _logger.debug("ZeroInferRequest - completed");
}

ZeroInferRequest::FoundPort ZeroInferRequest::find_port(const ov::Output<const ov::Node>& port) const {
    // check if the tensor names of target port is a subset of source port's tensor names
    auto check_tensor_names = [](const std::unordered_set<std::string>& source,
                                 const std::unordered_set<std::string>& target) {
        for (const auto& name : target) {
            if (source.find(name) == source.end()) {
                return false;
            }
        }
        return true;
    };

    // This function is hotspot, need optimization.
    auto check_nodes = [](const ov::Node* node1, const ov::Node* node2) {
        return node1 == node2 ||
               (node1->outputs().size() == node2->outputs().size() &&
                node1->inputs().size() == node2->inputs().size() && node1->get_type_info() == node2->get_type_info() &&
                node1->get_friendly_name() == node2->get_friendly_name());
    };
    // Find port without caching work slow because we need each time iterate over all ports and compare different
    // strings So use WA with caching in order to make 2+ calls for the same ports faster.
    // Calculate hash for the port
    size_t port_hash =
        ov::util::hash_combine({std::hash<const ov::Node*>()(port.get_node()), std::hash<size_t>()(port.get_index())});
    {
        std::lock_guard<std::mutex> lock(_cacheMutex);
        if (_cachedPorts.find(port_hash) != _cachedPorts.end()) {
            // Cached port for the hash was found
            return _cachedPorts[port_hash];
        }
    }
    ZeroInferRequest::FoundPort::Type type = ZeroInferRequest::FoundPort::Type::INPUT;
    for (const auto& ports : {get_inputs(), get_outputs()}) {
        for (size_t i = 0; i < ports.size(); i++) {
            // The order of the arguments might matter for the "check_tensor_names" call. If the "CompiledModel" object
            // was obtained via "import_model", then the number of tensor names could be cut to 32 due to limitations
            // inside the NPU stack. For this particular scenario, we are checking if all tensor names corresponding to
            // the "CompiledModel" are found in the provided port instead of doing the opposite.
            if (ports[i].get_index() == port.get_index() && check_nodes(ports[i].get_node(), port.get_node()) &&
                check_tensor_names(port.get_names(), ports[i].get_names())) {
                std::lock_guard<std::mutex> lock(_cacheMutex);
                _cachedPorts[port_hash] = {i, type};
                return _cachedPorts[port_hash];
            }
        }
        type = ZeroInferRequest::FoundPort::Type::OUTPUT;
    }
    return {0, ZeroInferRequest::FoundPort::Type::NOT_FOUND};
}

const std::vector<ov::Output<const ov::Node>>& ZeroInferRequest::get_inputs() const {
    return _compiledModel->inputs();
}

const std::vector<ov::Output<const ov::Node>>& ZeroInferRequest::get_outputs() const {
    return _compiledModel->outputs();
}

const std::shared_ptr<const ov::ICompiledModel>& ZeroInferRequest::get_compiled_model() const {
    return _compiledModel;
}

void ZeroInferRequest::initialize_states() {
    for (const ov::SoPtr<ov::IVariableState>& variableState : _variableStates) {
        variableState->reset();
    }
}

void ZeroInferRequest::add_state(const IODescriptor& descriptor, size_t tensorIndex) const {
    OPENVINO_ASSERT(descriptor.relatedDescriptorIndex.has_value(),
                    "The link between state descriptors is missing, state name: ",
                    descriptor.nameFromCompiler);

    _variableStates.push_back(std::make_shared<ZeroVariableState>(_initStructs,
                                                                  descriptor.nameFromCompiler,
                                                                  get_level_zero_input(tensorIndex),
                                                                  tensorIndex,
                                                                  descriptor.relatedDescriptorIndex.value()));
}

std::vector<ov::SoPtr<ov::IVariableState>> ZeroInferRequest::query_state() const {
    return _variableStates;
}

void ZeroInferRequest::setup_pipeline() {
    _logger.debug("setup_pipeline - started");
    auto batchSize = _graph->get_batch_size();

    for (size_t inputIndex = 0; inputIndex < _metadata.inputs.size(); ++inputIndex) {
        if (_metadata.inputs.at(inputIndex).isMainInputWeights) {
            // These values were set while running the "WeightlessGraph::init" method
            continue;
        }

        if (is_batched_input(inputIndex) && batchSize.has_value()) {
            if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
                _logger.debug("setup_pipeline - tensors %s were already allocated",
                              _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str());
            } else {
                if (get_level_zero_input(inputIndex) == nullptr) {
                    get_level_zero_input(inputIndex) =
                        allocate_tensor(inputIndex, INPUT, get_user_inputs(inputIndex).size());
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

            _logger.debug("setup_pipeline - tensor %s was already allocated and has size: %zu",
                          _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str(),
                          get_level_zero_input(inputIndex)->get_byte_size());

            continue;
        }

        get_level_zero_input(inputIndex) = allocate_tensor(inputIndex, INPUT, batchSize);
        _logger.debug("setup_pipeline - new input tensor %s allocated, size: %zu",
                      _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str(),
                      get_level_zero_input(inputIndex)->get_byte_size());
    }

    for (size_t outputIndex = 0; outputIndex < _metadata.outputs.size(); ++outputIndex) {
        if (_levelZeroOutputTensors.at(outputIndex)) {
            if (_dynamicBatchValueChanged) {
                if (batchSize.has_value() &&
                    _levelZeroOutputTensors.at(outputIndex)->get_shape()[utils::BATCH_AXIS] == batchSize.value()) {
                    _logger.debug("setup_pipeline - tensor %s was already allocated",
                                  _metadata.outputs.at(outputIndex).nodeFriendlyName.c_str());
                    continue;
                }
            } else {
                _logger.debug("setup_pipeline - tensor %s was already allocated",
                              _metadata.outputs.at(outputIndex).nodeFriendlyName.c_str());
                continue;
            }
        }
        _logger.debug("setup_pipeline - allocate new output tensor %s",
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

        _logger.debug("setup_pipeline - new output tensor %s allocated, size: %zu",
                      _metadata.outputs.at(outputIndex).nodeFriendlyName.c_str(),
                      _levelZeroOutputTensors.at(outputIndex)->get_byte_size());
    }
    _logger.debug("setup_pipeline - initialization completed");

    // Set new tensors and reset variable state flag if memory updated before creating the pipeline
    _logger.debug("setup_pipeline - set new tensors and reset variable state flag if memory updated "
                  "before creating the pipeline");
    for (const auto& variableState : _variableStates) {
        auto zeroState = std::dynamic_pointer_cast<ZeroVariableState>(variableState._ptr);
        OPENVINO_ASSERT(zeroState != nullptr, "State is not compatible with NPU plugin");

        if (zeroState->state_update_pending()) {
            _logger.debug("setup_pipeline - user state tensor should be updated");

            get_user_input(zeroState->get_tensor_index()) = zeroState->get_user_state();
            _userOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_user_state();
            zeroState->clear_state_update_pending();

            if (zeroState->zero_state_update_pending()) {
                _logger.debug("setup_pipeline - level zero state tensor should be updated");

                get_level_zero_input(zeroState->get_tensor_index()) = zeroState->get_zero_state();
                _levelZeroOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_zero_state();
                zeroState->clear_zero_state_update_pending();
            }
        }
    }

    create_pipeline_impl();
}

void ZeroInferRequest::create_pipeline_impl() {
    _logger.debug("create_pipeline_impl - constructing pipeline");
    auto batchSize = _graph->get_batch_size();
    // Construct pipeline
    _pipeline = std::make_unique<Pipeline>(_initStructs,
                                           _graph,
                                           _config,
                                           _levelZeroInputTensors,
                                           _levelZeroOutputTensors,
                                           batchSize.has_value() ? batchSize.value() : utils::DEFAULT_BATCH_SIZE);

    _logger.debug("create_pipeline_impl - completed");
}

void ZeroInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OV_ITT_TASK_CHAIN(ZERO_SET_TENSOR, itt::domains::LevelZeroBackend, "set_tensor", "set_tensor");

    auto foundPort = find_port(port);
    OPENVINO_ASSERT(foundPort.found(), "Cannot find tensor for port ", port);
    try {
        check_tensor(port,
                     tensor,
                     foundPort.is_input() ? _metadata.inputs.at(foundPort.idx).supportsStridedLayout
                                          : _metadata.outputs.at(foundPort.idx).supportsStridedLayout);
    } catch (const ov::Exception& ex) {
        OPENVINO_THROW("Failed to set tensor. ", ex.what());
    }

    if (foundPort.is_input()) {
        if (get_user_input(foundPort.idx)._ptr == tensor._ptr) {
            // set_tensor called with the same tensor object; no action needed
            _logger.debug("set_tensor - got the same input tensor, do nothing");
            return;
        }

        const auto& ioShape = _compiledModel->inputs()[foundPort.idx].get_partial_shape();
        auto batchSizeCandidate =
            determine_dynamic_batch_size(_metadata.inputs.at(foundPort.idx), ioShape, tensor._ptr, std::nullopt);

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

        if (get_level_zero_inputs(foundPort.idx).size() > 1) {
            // Reset vector size to 1 if set_tensor is called after set_tensors
            get_level_zero_inputs(foundPort.idx).resize(1);
            get_level_zero_inputs(foundPort.idx).shrink_to_fit();
            get_level_zero_input(foundPort.idx) = {};
        }

        if (get_user_inputs(foundPort.idx).size() > 1) {
            get_user_inputs(foundPort.idx).resize(1);
            get_user_inputs(foundPort.idx).shrink_to_fit();
            get_user_input(foundPort.idx) = {};
        }

        get_user_input(foundPort.idx) = tensor;
    } else {
        if (_userOutputTensors.at(foundPort.idx)._ptr == tensor._ptr) {
            // set_tensor called with the same tensor object; no action needed
            _logger.debug("set_tensor - got the same output tensor, do nothing");
            return;
        }

        const auto& ioShape = _compiledModel->outputs()[foundPort.idx].get_partial_shape();
        auto batchSizeCandidate =
            determine_dynamic_batch_size(_metadata.outputs.at(foundPort.idx), ioShape, tensor._ptr, std::nullopt);

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

    sync_zero_tensor_with_graph(foundPort, tensor);
}

void ZeroInferRequest::sync_zero_tensor_with_graph(const ZeroInferRequest::FoundPort& foundPort,
                                                   const ov::SoPtr<ov::ITensor>& tensor) {
    OV_ITT_TASK_CHAIN(ZERO_SET_TENSOR,
                      itt::domains::LevelZeroBackend,
                      "ZeroInferRequest",
                      "sync_zero_tensor_with_graph");
    auto& levelZeroTensor =
        foundPort.is_input() ? get_level_zero_input(foundPort.idx) : _levelZeroOutputTensors.at(foundPort.idx);

    if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0) && !_isShapeTensorPresent) {
        bool updateCommandListArg = false;
        try {
            _logger.debug("sync_zero_tensor_with_graph - create zero tensor");
            OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "create zero tensor");

            // Try to use the user tensor directly if its underlying data is already allocated in the same Level Zero
            // context.
            levelZeroTensor = std::make_shared<ZeroTensor>(_initStructs, tensor);
            updateCommandListArg = true;
        } catch (const ZeroMemException& exception) {
            _logger.debug("sync_zero_tensor_with_graph - exception caught while trying to create a "
                          "Level Zero tensor from the user tensor: %s",
                          exception.what());

            // Check if the current Level Zero tensor was previously shared with the user. If so, it cannot be reused;
            // allocate a new tensor to back up the user tensor (which cannot be imported or used directly).
            if (_dynamicBatchValueChanged || levelZeroTensor == nullptr || !levelZeroTensor->can_be_reused()) {
                _logger.debug("sync_zero_tensor_with_graph - allocate locally L0 tensor");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "allocate_tensor");

                levelZeroTensor = allocate_tensor(foundPort.idx, foundPort.is_input(), _graph->get_batch_size());
                updateCommandListArg = true;
            } else {
                _logger.debug("sync_zero_tensor_with_graph - reusing the level zero tensor since it "
                              "is not shared with the user");
            }

            const auto& userTensorElementType = tensor->get_element_type();
            if (userTensorElementType == ov::element::boolean &&
                levelZeroTensor->get_element_type() == ov::element::u8) {
                levelZeroTensor->set_element_type(userTensorElementType);
            }
        }

        if (_pipelineIsCreated && updateCommandListArg && !_dynamicBatchValueChanged) {
            _logger.debug("sync_zero_tensor_with_graph - update command list");

            OPENVINO_ASSERT(levelZeroTensor->data(), "Empty buffer");

            OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "update_graph_arguments");
            _pipeline->update_graph_arguments(foundPort.is_input()
                                                  ? _metadata.inputs.at(foundPort.idx).indexUsedByDriver
                                                  : _metadata.outputs.at(foundPort.idx).indexUsedByDriver,
                                              levelZeroTensor);
        }
    } else {
        // If command list updates are not supported, fallback to copying tensors every time.
        if (levelZeroTensor == nullptr) {
            levelZeroTensor = allocate_tensor(foundPort.idx, foundPort.is_input(), _graph->get_batch_size());
        }

        const auto& userTensorElementType = tensor->get_element_type();
        if (userTensorElementType == ov::element::boolean && levelZeroTensor->get_element_type() == ov::element::u8) {
            levelZeroTensor->set_element_type(userTensorElementType);
        }
    }
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

    check_batched_tensors(port, tensors, _metadata.inputs.at(foundPort.idx).supportsStridedLayout);

    _logger.debug("set_tensors - tensor count: %zu", tensors.size());

    const auto& ioShape = _compiledModel->inputs()[foundPort.idx].get_partial_shape();
    auto batchSizeCandidate =
        determine_dynamic_batch_size(_metadata.inputs.at(foundPort.idx), ioShape, nullptr, tensors.size());

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

    sync_zero_tensors_with_graph(foundPort, tensors, batchSizeCandidate);
}

void ZeroInferRequest::sync_zero_tensors_with_graph(const ZeroInferRequest::FoundPort& foundPort,
                                                    const std::vector<ov::SoPtr<ov::ITensor>>& tensors,
                                                    const std::optional<size_t>& batchSize) {
    OV_ITT_TASK_CHAIN(ZERO_SET_TENSORS,
                      itt::domains::LevelZeroBackend,
                      "ZeroInferRequest",
                      "sync_zero_tensors_with_graph");

    if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0) && !_isShapeTensorPresent) {
        if (batchSize.has_value()) {
            get_level_zero_inputs(foundPort.idx).resize(tensors.size());
            for (size_t i = 0; i < tensors.size(); i++) {
                try {
                    _logger.debug("sync_zero_tensors_with_graph - create zero tensor");
                    OV_ITT_TASK_NEXT(ZERO_SET_TENSORS, "create zero tensor");
                    get_level_zero_input(foundPort.idx, i) = std::make_shared<ZeroTensor>(_initStructs, tensors.at(i));
                } catch (const ZeroMemException& exception) {
                    _logger.debug(
                        "sync_zero_tensors_with_graph - exception caught while trying to create a Level Zero tensor "
                        "from the user tensor: %s",
                        exception.what());

                    _logger.debug("sync_zero_tensors_with_graph - allocate locally L0 tensor");
                    OV_ITT_TASK_NEXT(ZERO_SET_TENSORS, "allocate_tensor");
                    auto& levelZeroTensor = get_level_zero_input(foundPort.idx, i);
                    levelZeroTensor = allocate_tensor(foundPort.idx, INPUT);

                    // corner case for boolean inputs as compiler maps them to u8
                    const auto& userTensorElementType = tensors.at(i)->get_element_type();
                    if (userTensorElementType == ov::element::boolean &&
                        levelZeroTensor->get_element_type() == ov::element::u8) {
                        levelZeroTensor->set_element_type(userTensorElementType);
                    }
                }

                if (_pipelineIsCreated && !_dynamicBatchValueChanged) {
                    OPENVINO_ASSERT(get_level_zero_input(foundPort.idx, i)->data(), "Empty buffer");
                    OV_ITT_TASK_NEXT(ZERO_SET_TENSORS, "update_graph_arguments");
                    _pipeline->update_graph_arguments(_metadata.inputs.at(foundPort.idx).indexUsedByDriver,
                                                      get_level_zero_input(foundPort.idx, i),
                                                      i);
                }
            }
        } else {
            auto& levelZeroTensor = get_level_zero_input(foundPort.idx);
            // Check if the current Level Zero tensor was previously shared with the user. If so, it cannot be reused;
            // allocate a new tensor to back up the user tensor (which cannot be imported or used directly).
            if (levelZeroTensor == nullptr || !levelZeroTensor->can_be_reused()) {
                _logger.debug("sync_zero_tensors_with_graph - allocate locally L0 tensor");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSORS, "allocate_tensor");
                levelZeroTensor = allocate_tensor(foundPort.idx, INPUT, tensors.size());

                if (_pipelineIsCreated && !_dynamicBatchValueChanged) {
                    OPENVINO_ASSERT(levelZeroTensor->data(), "Empty buffer");
                    OV_ITT_TASK_NEXT(ZERO_SET_TENSORS, "update_graph_arguments");
                    _pipeline->update_graph_arguments(_metadata.inputs.at(foundPort.idx).indexUsedByDriver,
                                                      levelZeroTensor);
                }
            } else {
                _logger.debug("sync_zero_tensors_with_graph - reusing the level zero tensor since it is "
                              "not shared with the user");
            }

            const auto& userTensorElementType = tensors.at(SINGLE_TENSOR)->get_element_type();
            if (userTensorElementType == ov::element::boolean &&
                levelZeroTensor->get_element_type() == ov::element::u8) {
                levelZeroTensor->set_element_type(userTensorElementType);
            }
        }
    } else {
        // If command list updates are not supported, fallback to copying tensors every time.
        auto& levelZeroTensor = get_level_zero_input(foundPort.idx);
        if (levelZeroTensor == nullptr) {
            levelZeroTensor = allocate_tensor(foundPort.idx, foundPort.is_input(), tensors.size());
        }

        const auto& userTensorElementType = tensors.at(SINGLE_TENSOR)->get_element_type();
        if (userTensorElementType == ov::element::boolean && levelZeroTensor->get_element_type() == ov::element::u8) {
            levelZeroTensor->set_element_type(userTensorElementType);
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
            _logger.debug("get_tensor - tensor allocated, get tensor by index: %zu", ioIndex);

            auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(userTensor._ptr);
            if (zeroTensor != nullptr) {
                zeroTensor->prevent_reuse();
            }

            return userTensor;
        } else {
            if (batchSize.has_value() && userTensor->get_shape()[utils::BATCH_AXIS] == batchSize.value()) {
                _logger.debug("get_tensor - tensor by index: %zu is already allocated", ioIndex);

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
    _logger.debug("get_tensor - tensor by index: %zu is not allocated, new tensor %s will be created",
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

std::vector<ov::SoPtr<ov::ITensor>> ZeroInferRequest::get_tensors(const ov::Output<const ov::Node>& port) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::Plugin, "get_tensors");

    auto foundPort = find_port(port);
    OPENVINO_ASSERT(foundPort.found(), "Cannot find input tensors for port ", port);

    if (foundPort.is_input() && is_batched_input(foundPort.idx)) {
        return get_user_inputs(foundPort.idx);
    }

    return {};
}

std::shared_ptr<ZeroTensor> ZeroInferRequest::allocate_tensor(const size_t index,
                                                              const bool isInput,
                                                              const std::optional<std::size_t>& batchSize) const {
    const auto& descriptor = isInput ? _metadata.inputs.at(index) : _metadata.outputs.at(index);
    check_network_precision(descriptor.precision);

    ov::Shape allocatedTensorShape = descriptor.shapeFromCompiler.get_max_shape();

    if (batchSize.has_value()) {
        allocatedTensorShape[utils::BATCH_AXIS] = *batchSize;
    }

    auto tensor = std::make_shared<ZeroTensor>(_initStructs, descriptor.precision, allocatedTensorShape, isInput);

    if (isInput) {
        if (get_user_input(index) == nullptr) {
            get_user_input(index) = tensor;
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
            // Memory address can change only a through tensor reshape. Tensor reallocation for a larger shape is
            // allowed only when mutable command list version >= 1.0. This point should not be reached otherwise.
            if (_initStructs->getMutableCommandListExtVersion() < ZE_MAKE_VERSION(1, 0)) {
                OPENVINO_THROW("Reallocation of zero memory is not supported with this driver.");
            }

            _logger.debug("update_pipeline_if_memory_changed - update input graph descriptor with the new tensor");
            OPENVINO_ASSERT(levelZeroTensor.at(SINGLE_TENSOR)->data(), "Empty buffer");
            _pipeline->update_graph_arguments(_metadata.inputs.at(ioIndex).indexUsedByDriver,
                                              levelZeroTensor.at(SINGLE_TENSOR));

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
            // Memory address can change only a through tensor reshape. Tensor reallocation for a larger shape is
            // allowed only when mutable command list version >= 1.0. This point should not be reached otherwise.
            if (_initStructs->getMutableCommandListExtVersion() < ZE_MAKE_VERSION(1, 0)) {
                OPENVINO_THROW("Reallocation of zero memory is not supported with this driver.");
            }

            _logger.debug("update_pipeline_if_memory_changed - update output graph descriptor with the new tensor");
            OPENVINO_ASSERT(levelZeroTensor->data(), "Empty buffer");
            _pipeline->update_graph_arguments(_metadata.outputs.at(ioIndex).indexUsedByDriver, levelZeroTensor);

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

            // State's tensor was previously updated. This change needs to be reflected in the inference request since
            // states tensors are not visible inside the pipeline.
            // Update input and output arguments that correspond to the state only if command lists are supported.
            // Push/pull methods would later perform memory copies otherwise.
            if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0) &&
                zeroState->zero_state_update_pending()) {
                get_level_zero_input(zeroState->get_tensor_index()) = zeroState->get_zero_state();
                _levelZeroOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_zero_state();
                zeroState->clear_zero_state_update_pending();

                _pipeline->update_graph_arguments(_metadata.inputs.at(zeroState->get_tensor_index()).indexUsedByDriver,
                                                  get_level_zero_input(zeroState->get_tensor_index()));

                _pipeline->update_graph_arguments(
                    _metadata.outputs.at(zeroState->get_related_tensor_index()).indexUsedByDriver,
                    _levelZeroOutputTensors.at(zeroState->get_related_tensor_index()));
            }
        }
    }
}

void ZeroInferRequest::infer() {
    OV_ITT_SCOPED_TASK_BASE(itt::domains::InferenceNPU, "SyncInferenceNPU");
    if (_config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        OPENVINO_THROW("Only start async is supported when RUN_INFERENCES_SEQUENTIALLY is enabled!");
    }

    infer_async();
    get_result();
}

void ZeroInferRequest::infer_async() {
    _logger.debug("infer_async - started");
    // TASK_BASE marker is always on by default
    OV_ITT_SCOPED_TASK_BASE(itt::domains::InferenceNPU, "Inference::start");
    // This task chain marker will only be available when ENABLE_PROFILING_ITT=FULL
    OV_ITT_TASK_CHAIN(ZERO_INFER, itt::domains::LevelZeroBackend, "infer_async", "start");
    prepare_inputs();
    prepare_outputs();

    OV_ITT_TASK_NEXT(ZERO_INFER, "push");
    _pipeline->push();
}

void ZeroInferRequest::prepare_inputs() {
    OV_ITT_TASK_CHAIN(ZERO_INFER, itt::domains::LevelZeroBackend, "infer_async", "prepare_inputs");
    {
        std::lock_guard<std::mutex> lock(_graph->get_mutex());

        if (!_pipelineIsCreated || _dynamicBatchValueChanged) {
            OV_ITT_TASK_NEXT(ZERO_INFER, "create_pipeline");
            setup_pipeline();  // Reallocate pipeline if necessary
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
        const IODescriptor& inputDescriptor = _metadata.inputs.at(inputIndex);

        OPENVINO_ASSERT(!inputDescriptor.isInitInputWeights,
                        "This path should not be used for running inferences for the \"init\" model");

        if (inputDescriptor.isMainInputWeights) {
            // These values were set while running the "WeightlessGraph::init" method
            continue;
        }

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
            if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0) && batch_size.has_value()) {
                for (size_t i = 0; i < userTensor.size(); i++) {
                    const auto& levelZeroTensor = get_level_zero_input(inputIndex, i);
                    OPENVINO_ASSERT(levelZeroTensor, "Input zero tensor is not allocated.");

                    const auto& userTensorPtr = userTensor.at(i);
                    void* userBuffer = get_tensor_data_ptr(userTensorPtr._ptr);
                    void* levelZeroBuffer = levelZeroTensor->data();

                    if (levelZeroBuffer == nullptr) {
                        levelZeroTensor->allocate_data();
                        levelZeroBuffer = levelZeroTensor->data();
                        _pipeline->update_graph_arguments(_metadata.inputs.at(inputIndex).indexUsedByDriver,
                                                          levelZeroTensor,
                                                          i);
                    }

                    if (userBuffer != levelZeroBuffer) {
                        OPENVINO_ASSERT(userBuffer != nullptr && levelZeroBuffer != nullptr, "Empty buffer");

                        _logger.debug(
                            "prepare_inputs - batched tensor by index: %zu is not allocated in the current Level Zero "
                            "context, copy bytes from user tensor: %zu, into L0 with expected size: %zu",
                            inputIndex,
                            userTensorPtr->get_byte_size(),
                            levelZeroTensor->get_byte_size());
                        OV_ITT_TASK_NEXT(ZERO_INFER, "memcpy");
                        userTensorPtr->copy_to(levelZeroTensor);
                    }
                }
            } else {
                const auto& levelZeroTensor = get_level_zero_input(inputIndex);
                OPENVINO_ASSERT(levelZeroTensor, "Input zero tensor is not allocated.");

                _logger.debug(
                    "prepare_inputs - batched tensor by index: %zu is not allocated in the current Level Zero "
                    "context or must be in a continued memory space, copy into L0 with size: %zu",
                    inputIndex,
                    levelZeroTensor->get_byte_size());
                size_t copied_bytes_from_user = 0;
                for (size_t i = 0; i < userTensor.size(); i++) {
                    auto viewTensor = ov::make_tensor(
                        levelZeroTensor->get_element_type(),
                        levelZeroTensor->get_shape(),
                        static_cast<unsigned char*>(levelZeroTensor->data()) + (i * userTensor.at(i)->get_byte_size()));

                    userTensor.at(i)->copy_to(viewTensor);
                    copied_bytes_from_user += userTensor.at(i)->get_byte_size();
                }
                OPENVINO_ASSERT(levelZeroTensor->get_byte_size() == copied_bytes_from_user,
                                "Bytes copied must be equal");
            }

            ++inputIndex;
            continue;
        }

        const auto& levelZeroTensor = get_level_zero_input(inputIndex);
        OPENVINO_ASSERT(levelZeroTensor, "Input zero tensor is not allocated.");

        void* userBuffer = get_tensor_data_ptr(userTensor.at(SINGLE_TENSOR)._ptr);
        void* levelZeroBuffer = levelZeroTensor->data();

        if (levelZeroBuffer == nullptr) {
            levelZeroTensor->allocate_data();
            levelZeroBuffer = levelZeroTensor->data();
            _pipeline->update_graph_arguments(_metadata.inputs.at(inputIndex).indexUsedByDriver, levelZeroTensor);
        }

        if (userBuffer != levelZeroBuffer) {
            OPENVINO_ASSERT(userBuffer != nullptr && levelZeroBuffer != nullptr, "Empty buffer");

            _logger.info("prepare_inputs - tensor is not allocated in the current Level Zero context");
            OV_ITT_TASK_NEXT(ZERO_INFER, "memcpy");
            if (_isShapeTensorPresent && userTensor.at(SINGLE_TENSOR)->get_shape() != levelZeroTensor->get_shape()) {
                auto viewTensor = ov::make_tensor(levelZeroTensor->get_element_type(),
                                                  userTensor.at(SINGLE_TENSOR)->get_shape(),
                                                  static_cast<unsigned char*>(levelZeroTensor->data()));
                userTensor.at(SINGLE_TENSOR)->copy_to(viewTensor);
            } else {
                userTensor.at(SINGLE_TENSOR)->copy_to(levelZeroTensor);
            }
        }

        ++inputIndex;
    }
}

void ZeroInferRequest::prepare_outputs() {
    OV_ITT_TASK_CHAIN(ZERO_INFER, itt::domains::LevelZeroBackend, "infer_async", "prepare_outputs");
    for (size_t outputIndex = 0; outputIndex < _userOutputTensors.size(); ++outputIndex) {
        const auto& levelZeroTensor = _levelZeroOutputTensors.at(outputIndex);
        OPENVINO_ASSERT(levelZeroTensor, "Output zero tensor is not allocated.");

        if (levelZeroTensor->data() == nullptr) {
            levelZeroTensor->allocate_data();
            _pipeline->update_graph_arguments(_metadata.outputs.at(outputIndex).indexUsedByDriver, levelZeroTensor);
        }
    }
}

void ZeroInferRequest::get_result() {
    OV_ITT_SCOPED_TASK_BASE(itt::domains::InferenceNPU, "Inference::get_result");
    OV_ITT_TASK_CHAIN(ZERO_RESULT, itt::domains::LevelZeroBackend, "get_result", "pull");
    _logger.debug("get_result - started");
    _pipeline->pull();

    size_t outputIndex = 0;
    for (const auto& userTensor : _userOutputTensors) {
        const IODescriptor& outputDescriptor = _metadata.outputs.at(outputIndex);
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

        const auto& levelZeroTensor = _levelZeroOutputTensors.at(outputIndex);
        void* userBuffer = get_tensor_data_ptr(userTensor._ptr);
        void* levelZeroBuffer = levelZeroTensor->data();

        if (userBuffer == nullptr || levelZeroBuffer == nullptr) {
            OPENVINO_THROW("Empty buffer");
        }

        if (userBuffer != levelZeroBuffer) {
            _logger.info("get_result - output tensor by index: %zu is not allocated in the current Level Zero context",
                         outputIndex);
            OV_ITT_TASK_NEXT(ZERO_RESULT, "memcpy");
            if (_isShapeTensorPresent && userTensor->get_shape() != levelZeroTensor->get_shape()) {
                auto viewTensor = ov::make_tensor(levelZeroTensor->get_element_type(),
                                                  userTensor->get_shape(),
                                                  static_cast<unsigned char*>(levelZeroTensor->data()));
                viewTensor->copy_to(userTensor._ptr);
            } else {
                levelZeroTensor->copy_to(userTensor._ptr);
            }
        }

        levelZeroTensor->detach_imported_allocation_for_custom_tensor();

        ++outputIndex;
    }

    for (size_t inputIndex = 0; inputIndex < _levelZeroInputTensors.size(); ++inputIndex) {
        for (const auto& levelZeroTensor : get_level_zero_inputs(inputIndex)) {
            if (levelZeroTensor != nullptr) {
                levelZeroTensor->detach_imported_allocation_for_custom_tensor();
            }
        }
    }

    OV_ITT_TASK_NEXT(ZERO_RESULT, "reset");
    _pipeline->reset();
    _logger.debug("get_result - finished");
}

void ZeroInferRequest::check_tensor(const ov::Output<const ov::Node>& port,
                                    const ov::SoPtr<ov::ITensor>& tensor,
                                    const bool supportStrides) const {
    if (tensor == nullptr)
        OPENVINO_THROW("The tensor is not initialized!");

    bool is_input = ov::op::util::is_parameter(port.get_node());
    const std::string_view tensor_type = is_input ? "input" : "output";

    if (!supportStrides) {
        OPENVINO_ASSERT(
            tensor->is_continuous(),
            "The tensor has a non-contiguous memory layout (custom strides), which is not supported by the "
            "current driver/compiler version. To use strided tensors, either:\n"
            "  1. Upgrade to a driver version that supports strides, or\n"
            "  2. Enable stride support using the 'enable_strides_for' configuration property if this is supported.");
    }

    const auto& port_element_type = port.get_element_type();
    const auto& tensor_element_type = tensor->get_element_type();

    if ((port_element_type == ov::element::Type_t::boolean || tensor_element_type == ov::element::Type_t::boolean) &&
        port_element_type != tensor_element_type) {
        // Exception case for boolean treated as u8 in the NPU driver
        OPENVINO_ASSERT(port_element_type == ov::element::Type_t::u8 || tensor_element_type == ov::element::Type_t::u8,
                        "The tensor element type is not corresponding with output element type (",
                        tensor_element_type,
                        " != ",
                        port_element_type);
    } else {
        OPENVINO_ASSERT(port_element_type == tensor_element_type,
                        "The tensor element type is not corresponding with output element type (",
                        tensor_element_type,
                        " != ",
                        port_element_type);
    }

    const auto& port_partial_shape = port.get_partial_shape();
    const auto& tensor_shape = tensor->get_shape();

    bool is_dynamic = port_partial_shape.is_dynamic();

    if (is_dynamic) {
        auto port_length = port_partial_shape.rank().get_length();
        OPENVINO_ASSERT(ov::PartialShape(tensor_shape).rank().get_length() == port_length,
                        "The tensor shape size is not equal to the model input/output rank: got ",
                        tensor_shape.size(),
                        " expecting ",
                        port_length);

        if (port_length > 0) {
            const auto& port_max_shape = port_partial_shape.get_max_shape();
            const auto& port_min_shape = port_partial_shape.get_min_shape();
            for (auto i = 0; i < port_length; ++i) {
                if (port_min_shape[i] != port_max_shape[i] && tensor_shape[i] > port_max_shape[i]) {
                    OPENVINO_THROW("The tensor shape is not compatible with the model input/output max shape: got ",
                                   tensor_shape,
                                   " expecting max shape ",
                                   port_max_shape);
                }

                if (port_min_shape[i] == port_max_shape[i] && tensor_shape[i] != port_min_shape[i]) {
                    OPENVINO_THROW("The tensor shape is not compatible with the model input/output shape: got ",
                                   tensor_shape,
                                   " expecting shape ",
                                   port_min_shape);
                }
            }
        }
    }

    OPENVINO_ASSERT(is_dynamic || port_partial_shape == tensor_shape,
                    "The ",
                    tensor_type,
                    " tensor size is not equal to the model ",
                    tensor_type,
                    " type: got ",
                    tensor_shape,
                    " expecting ",
                    port_partial_shape,
                    ".");
    OPENVINO_ASSERT(
        std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr) || tensor->data() != nullptr || is_dynamic,
        "Tensor data equal nullptr!");
}

void ZeroInferRequest::check_batched_tensors(const ov::Output<const ov::Node>& port,
                                             const std::vector<ov::SoPtr<ov::ITensor>>& tensors,
                                             const bool supportStrides) const {
    OPENVINO_ASSERT(!tensors.empty(), "set_input_tensors/set_tensors can't be called with empty tensors");
    OPENVINO_ASSERT(
        tensors.size() != 1,
        "Internal error (plugin): check_batched_tensors is not allowed to have only one tensor inside batch");

    auto layout = ov::layout::get_layout(port);

    int64_t batch_idx;

    if (layout.empty()) {
        _logger.warning("check_batched_tensors - set_input_tensors/set_tensors layout is not set, assuming batch "
                        "dimension is found on 0 axis");
        batch_idx = utils::BATCH_AXIS;
    } else {
        OPENVINO_ASSERT(ov::layout::has_batch(layout),
                        "set_input_tensors/set_tensors can be used only for inputs with N(batch) dimension"
                        " 'layout' defined. Current layout is ",
                        layout.to_string());
        batch_idx = ov::layout::batch_idx(layout);
    }

    if (batch_idx < 0) {
        batch_idx += static_cast<int64_t>(tensors[utils::BATCH_AXIS]->get_shape().size());
    }
    OPENVINO_ASSERT(batch_idx == utils::BATCH_AXIS,
                    "set_input_tensors/set_tensors is not currently supported for batch dimension index ",
                    batch_idx,
                    " != 0");
    std::for_each(tensors.begin(), tensors.end(), [&batch_idx](const ov::SoPtr<ov::ITensor>& item) {
        OPENVINO_ASSERT(item, "Uninitialized tensor is provided!");
        OPENVINO_ASSERT(item->get_shape()[batch_idx] == 1,
                        "set_input_tensors/set_tensors. Tensors shall represent one item in a batch, ",
                        item->get_shape()[batch_idx],
                        " provided");
    });
    auto tensors_size = static_cast<int>(tensors.size());
    if (port.get_partial_shape().rank().is_static()) {
        OPENVINO_ASSERT(batch_idx >= 0 && batch_idx < port.get_partial_shape().rank().get_length(),
                        "set_input_tensors/set_tensors error. Layout ",
                        layout.to_string(),
                        " is incorrect for operation with shape ",
                        port.get_partial_shape());
        auto batch = port.get_partial_shape()[batch_idx];

        OPENVINO_ASSERT(batch.is_dynamic() || batch.get_length() == tensors_size,
                        "set_input_tensors/set_tensors error. Input shape ",
                        port.get_partial_shape(),
                        "batch ",
                        batch,
                        "doesn't match with total blobs count: ",
                        tensors_size);
    }

    auto batched_shape = tensors[utils::BATCH_AXIS]->get_shape();
    auto element_type = tensors[utils::BATCH_AXIS]->get_element_type();
    batched_shape[batch_idx] = tensors_size;
    for (const auto& item : tensors) {
        OPENVINO_ASSERT(item, "Uninitialized tensor is provided!");
        auto item_shape = item->get_shape();
        item_shape[batch_idx] = batched_shape[batch_idx];
        OPENVINO_ASSERT(item_shape == batched_shape && item->get_element_type() == element_type,
                        "set_input_tensors/set_tensors error. Tensor with element type ",
                        item->get_element_type(),
                        " and shape ",
                        item_shape,
                        " is not compatible with batched tensor with element type ",
                        element_type,
                        " and shape ",
                        batched_shape);

        if (!supportStrides) {
            OPENVINO_ASSERT(
                item->is_continuous(),
                "The tensor has a non-contiguous memory layout (custom strides), which is not supported by the "
                "current driver/compiler version. To use strided tensors, either:\n"
                "  1. Upgrade to a driver version that supports strides, or\n"
                "  2. Enable stride support using the 'NPU_ENABLE_STRIDES_FOR' configuration property if this is "
                "supported.");
        }
    }
}

void ZeroInferRequest::check_tensors() const {
    const auto& inputs = _compiledModel->inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        if (is_batched_input(i)) {
            check_batched_tensors(inputs[i], get_user_inputs(i), _metadata.inputs.at(i).supportsStridedLayout);
            continue;
        }
        if (get_user_input(i)) {
            check_tensor(inputs[i], get_user_input(i), _metadata.inputs.at(i).supportsStridedLayout);
        }
    }

    const auto& outputs = _compiledModel->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        if (_userOutputTensors.at(i)) {
            check_tensor(outputs[i], _userOutputTensors.at(i), _metadata.outputs.at(i).supportsStridedLayout);
        }
    }
}

bool ZeroInferRequest::is_batched_input(size_t idx) const {
    return _userInputTensors.at(idx).size() > 1;
}

ov::SoPtr<ov::ITensor>& ZeroInferRequest::get_user_input(size_t index) const {
    return _userInputTensors.at(index).at(0);
}

std::vector<ov::SoPtr<ov::ITensor>>& ZeroInferRequest::get_user_inputs(size_t index) const {
    return _userInputTensors.at(index);
}

std::shared_ptr<ZeroTensor>& ZeroInferRequest::get_level_zero_input(size_t index, size_t tensorNo) const {
    return _levelZeroInputTensors.at(index).at(tensorNo);
}

std::vector<std::shared_ptr<ZeroTensor>>& ZeroInferRequest::get_level_zero_inputs(size_t index) const {
    return _levelZeroInputTensors.at(index);
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

}  // namespace intel_npu
