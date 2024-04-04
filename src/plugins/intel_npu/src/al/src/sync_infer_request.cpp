// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sync_infer_request.hpp"

#include "intel_npu/al/prefix.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/plugin_itt.hpp"
#include "transformations/utils/utils.hpp"

namespace intel_npu {

SyncInferRequest::SyncInferRequest(const std::shared_ptr<const ICompiledModel>& compiledModel)
    : _compiledModel(compiledModel),
      _metadata(compiledModel->get_network_metadata()) {
    OPENVINO_ASSERT(_compiledModel);

    const std::vector<ov::Output<const ov::Node>>& inputs = get_inputs();
    const std::vector<ov::Output<const ov::Node>>& outputs = get_outputs();

    if (inputs.empty()) {
        OPENVINO_THROW("Inference request creation: no input found for network " + _metadata.name);
    }
    if (outputs.empty()) {
        OPENVINO_THROW("Inference request creation: no output found for network " + _metadata.name);
    }

    // Map the node names to the legacy ones used by the I/O tensors in order to allow an easier access to the tensors'
    // contents
    for (const auto& [legacyName, parameterDescriptor] : _metadata.parameters) {
        _nodeNameToLegacyName[parameterDescriptor.currentNodeName] = legacyName;
    }
    for (const auto& [legacyName, resultDescriptor] : _metadata.results) {
        _nodeNameToLegacyName[resultDescriptor.currentNodeName] = legacyName;
    }

    _inputAndStateInputNames = _metadata.inputNames;
    _outputAndStateOutputNames = _metadata.outputNames;

    for (const std::string& stateName : _metadata.stateNames) {
        // State variables shall be identified by specific prefixes in order to avoid a potential tensor name collision
        _inputAndStateInputNames.push_back(READVALUE_PREFIX + stateName);
        _outputAndStateOutputNames.push_back(ASSIGN_PREFIX + stateName);
    }

    const auto contains = [](const auto& container, const auto& value) {
        return std::find(container.begin(), container.end(), value) != container.end();
    };

    for (const auto& shapeName : _metadata.shapeNames) {
        if (contains(_inputAndStateInputNames, shapeName)) {
            _inputAndStateInputNames.push_back(SHAPE_TENSOR_PREFIX + shapeName);
        }
        if (contains(_outputAndStateOutputNames, shapeName)) {
            _outputAndStateOutputNames.push_back(SHAPE_TENSOR_PREFIX + shapeName);
        }
    }
}

const std::vector<ov::Output<const ov::Node>>& SyncInferRequest::get_inputs() const {
    return _compiledModel->inputs();
}

const std::vector<ov::Output<const ov::Node>>& SyncInferRequest::get_outputs() const {
    return _compiledModel->outputs();
}

const std::shared_ptr<const ov::ICompiledModel>& SyncInferRequest::get_compiled_model() const {
    return _compiledModel;
}

void SyncInferRequest::initialize_states() {
    for (const std::string& stateName : _metadata.stateNames) {
        _variableStates.at(stateName)->reset();
    }
}

std::vector<ov::SoPtr<ov::IVariableState>> SyncInferRequest::query_state() const {
    std::vector<ov::SoPtr<ov::IVariableState>> queryResult;

    for (const std::string& stateName : _metadata.stateNames) {
        queryResult.push_back(_variableStates.at(stateName));
    }

    return queryResult;
}

ov::SoPtr<ov::ITensor> SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    const auto& nodeNameMatch = _nodeNameToLegacyName.find(port.get_node()->get_friendly_name());
    OPENVINO_ASSERT(nodeNameMatch != _nodeNameToLegacyName.end(), "Cannot find tensor for port ", port);

    return _allTensors.at(nodeNameMatch->second);
}

void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::Plugin, "set_tensor");
    try {
        check_tensor(port, tensor);
    } catch (const ov::Exception& ex) {
        OPENVINO_THROW("Failed to set tensor. ", ex.what());
    }

    const std::string& legacyName = _nodeNameToLegacyName.at(port.get_node()->get_friendly_name());
    _allTensors[legacyName] = tensor._ptr;
}

std::vector<ov::SoPtr<ov::ITensor>> SyncInferRequest::get_tensors(const ov::Output<const ov::Node>& /*port*/) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::Plugin, "get_tensors");
    // Using batches of tensors is currently not supported by the NPU plugin. In this scenario, the OpenVINO API demands
    // returning an empty vector.
    return {};
}

void SyncInferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                   const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::Plugin, "set_tensors");
    if (tensors.size() == 1) {
        set_tensor(port, tensors[0]);
        return;
    }

    OPENVINO_THROW_NOT_IMPLEMENTED("set_input_tensors/set_tensors are not supported by this plugin");
}

void SyncInferRequest::check_tensor(const ov::Output<const ov::Node>& port,
                                    const ov::SoPtr<ov::ITensor>& tensor) const {
    if (tensor == nullptr)
        OPENVINO_THROW("The tensor is not initialized!");

    bool is_input = ov::op::util::is_parameter(port.get_node());
    std::string tensor_type = is_input ? "input" : "output";

    OPENVINO_ASSERT(port.get_element_type() == tensor->get_element_type(),
                    "The tensor element type is not corresponding with output element type (",
                    tensor->get_element_type(),
                    " != ",
                    port.get_element_type());
    bool is_dynamic = port.get_partial_shape().is_dynamic();
    OPENVINO_ASSERT(is_dynamic || port.get_shape() == tensor->get_shape(),
                    "The ",
                    tensor_type,
                    " tensor size is not equal to the model ",
                    tensor_type,
                    " type: got ",
                    tensor->get_shape(),
                    " expecting ",
                    port.get_shape(),
                    ".");
    OPENVINO_ASSERT(
        std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr) || tensor->data() != nullptr || is_dynamic,
        "Tensor data equal nullptr!");
}

void SyncInferRequest::check_tensors() const {
    const auto& inputs = _compiledModel->inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        const std::string& legacyName = _nodeNameToLegacyName.at(inputs[i].get_node()->get_friendly_name());
        check_tensor(inputs[i], _allTensors.at(legacyName));
    }

    const auto& outputs = _compiledModel->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        const std::string& legacyName = _nodeNameToLegacyName.at(outputs[i].get_node()->get_friendly_name());
        check_tensor(outputs[i], _allTensors.at(legacyName));
    }
}

void SyncInferRequest::allocate_tensor(std::string tensorName,
                                       const IONodeDescriptor& descriptor,
                                       TensorType tensorType,
                                       const ov::Allocator& allocator) {
    std::shared_ptr<ov::ITensor> tensor;

    check_network_precision(descriptor.precision);

    if (allocator) {
        tensor = ov::make_tensor(descriptor.precision, descriptor.transposedShape.get_max_shape(), allocator);
    } else {
        tensor = ov::make_tensor(descriptor.precision, descriptor.transposedShape.get_max_shape());
    }

    if (tensorType == TensorType::Shape) {
        _shapesTensors[tensorName] = tensor;
        tensorName = SHAPE_TENSOR_PREFIX + tensorName;
    }
    if (tensorType == TensorType::State) {
        _variableStates[tensorName] = std::make_shared<VariableState>(tensorName, tensor);

        // State variables shall be identified by specific prefixes in order to avoid a potential tensor name collision.
        // Additionally, only one buffer is required in the whole flow, acting as an input before running the inference
        // and as an output after performing it. Thus both the "state input" and "state output" entries shall point to
        // the same buffer.
        _copyAllTensors[READVALUE_PREFIX + tensorName] = std::move(tensor);
        _copyAllTensors[ASSIGN_PREFIX + tensorName] = _copyAllTensors[READVALUE_PREFIX + tensorName];
        _allTensors[READVALUE_PREFIX + tensorName] = _copyAllTensors[READVALUE_PREFIX + tensorName];
        _allTensors[ASSIGN_PREFIX + tensorName] = _copyAllTensors[READVALUE_PREFIX + tensorName];
    } else {
        _copyAllTensors[tensorName] = std::move(tensor);
        _allTensors[tensorName] = _copyAllTensors[tensorName];
    }
}
}  // namespace intel_npu
