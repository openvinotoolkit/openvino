// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sync_infer_request.hpp"

#include "intel_npu/al/prefix.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/plugin_itt.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/utils/utils.hpp"

namespace intel_npu {

SyncInferRequest::SyncInferRequest(const std::shared_ptr<const ICompiledModel>& compiledModel)
    : _compiledModel(compiledModel),
      _metadata(compiledModel->get_network_metadata()) {
    OPENVINO_ASSERT(_compiledModel);

    const std::vector<ov::Output<const ov::Node>>& outputs = get_outputs();
    if (outputs.empty()) {
        OPENVINO_THROW("Inference request creation: no output found for network " + _metadata.name);
    }

    // Create map of empty tensors and cache ports from the compiled model
    auto portType = SyncInferRequest::FoundPort::Type::INPUT;
    for (const auto& ports : {get_inputs(), get_outputs()}) {
        for (size_t i = 0; i < ports.size(); i++) {
            const auto& port = ports[i];
            size_t portHash = ov::util::hash_combine(std::vector<size_t>{std::hash<const ov::Node*>()(port.get_node()),
                                                                         std::hash<size_t>()(port.get_index())});
            _cachedPorts[portHash] = {i, portType};
        }
        portType = SyncInferRequest::FoundPort::Type::OUTPUT;
    }

    for (const IODescriptor& inputDescriptor : _metadata.inputs) {
        if (inputDescriptor.isStateInput) {
            _prefixedInputNames.push_back(READVALUE_PREFIX + inputDescriptor.nameFromCompiler);
        } else if (inputDescriptor.isShapeTensor) {
            _prefixedInputNames.push_back(SHAPE_TENSOR_PREFIX + inputDescriptor.nameFromCompiler);
        } else {
            _prefixedInputNames.push_back(inputDescriptor.nameFromCompiler);
        }
    }

    for (const IODescriptor& outputDescriptor : _metadata.outputs) {
        if (outputDescriptor.isStateOutput) {
            _prefixedOutputNames.push_back(ASSIGN_PREFIX + outputDescriptor.nameFromCompiler);
        } else if (outputDescriptor.isShapeTensor) {
            _prefixedOutputNames.push_back(SHAPE_TENSOR_PREFIX + outputDescriptor.nameFromCompiler);
        } else {
            _prefixedOutputNames.push_back(outputDescriptor.nameFromCompiler);
        }

        // Map the node names to the legacy ones used by the I/O tensors in order to allow an easier access to the
        // tensors' contents
        _nodeFriendlyNameToNameFromCompiler[outputDescriptor.nodeFriendlyName] = outputDescriptor.nameFromCompiler;
        _nameFromCompilerToNodeFriendlyName[outputDescriptor.nameFromCompiler] = outputDescriptor.nodeFriendlyName;
    }
}

SyncInferRequest::FoundPort SyncInferRequest::find_port(const ov::Output<const ov::Node>& port) const {
    // check if the tensor names of target port is a subset of source port's tensor names
    auto check_tensor_names = [](const std::unordered_set<std::string>& source,
                                 const std::unordered_set<std::string>& target) {
        for (auto const& name : target) {
            if (source.find(name) == source.end())
                return false;
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
    size_t port_hash = ov::util::hash_combine(
        std::vector<size_t>{std::hash<const ov::Node*>()(port.get_node()), std::hash<size_t>()(port.get_index())});
    {
        std::lock_guard<std::mutex> lock(_cacheMutex);
        if (_cachedPorts.find(port_hash) != _cachedPorts.end()) {
            // Cached port for the hash was found
            return _cachedPorts[port_hash];
        }
    }
    SyncInferRequest::FoundPort::Type type = SyncInferRequest::FoundPort::Type::INPUT;
    for (const auto& ports : {get_inputs(), get_outputs()}) {
        for (size_t i = 0; i < ports.size(); i++) {
            if (ports[i].get_index() == port.get_index() && check_nodes(ports[i].get_node(), port.get_node()) &&
                check_tensor_names(ports[i].get_names(), port.get_names())) {
                std::lock_guard<std::mutex> lock(_cacheMutex);
                _cachedPorts[port_hash] = {i, type};
                return _cachedPorts[port_hash];
            }
        }
        type = SyncInferRequest::FoundPort::Type::OUTPUT;
    }
    return {0, SyncInferRequest::FoundPort::Type::NOT_FOUND};
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
    for (const ov::SoPtr<ov::IVariableState>& variableState : _variableStates) {
        variableState->reset();
    }
}

std::vector<ov::SoPtr<ov::IVariableState>> SyncInferRequest::query_state() const {
    return _variableStates;
}

ov::SoPtr<ov::ITensor> SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    auto foundPort = find_port(port);
    OPENVINO_ASSERT(foundPort.found(), "Cannot find tensor for port ", port);

    if (foundPort.is_input()) {
        return _inputTensors.at(foundPort.idx);
    }
    return _outputTensors.at(foundPort.idx);
}

void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::Plugin, "set_tensor");

    auto foundPort = find_port(port);
    OPENVINO_ASSERT(foundPort.found(), "Cannot find tensor for port ", port);
    try {
        check_tensor(port, tensor);
    } catch (const ov::Exception& ex) {
        OPENVINO_THROW("Failed to set tensor. ", ex.what());
    }

    if (foundPort.is_input()) {
        _inputTensors.at(foundPort.idx) = tensor._ptr;
    } else {
        _outputTensors.at(foundPort.idx) = tensor._ptr;
    }
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
        check_tensor(inputs[i], _inputTensors.at(i));
    }

    const auto& outputs = _compiledModel->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        check_tensor(outputs[i], _outputTensors.at(i));
    }
}

void SyncInferRequest::allocate_tensor(const IODescriptor& descriptor,
                                       const bool isInput,
                                       const ov::Allocator& allocator) {
    std::shared_ptr<ov::ITensor> tensor;

    check_network_precision(descriptor.precision);

    if (descriptor.isStateOutput) {
        OPENVINO_ASSERT(descriptor.relatedDescriptorIndex.has_value());
        tensor = _inputTensors.at(*descriptor.relatedDescriptorIndex);
    } else if (allocator) {
        tensor = ov::make_tensor(descriptor.precision, descriptor.shapeFromCompiler.get_max_shape(), allocator);
    } else {
        tensor = ov::make_tensor(descriptor.precision, descriptor.shapeFromCompiler.get_max_shape());
    }

    if (isInput) {
        _inputTensors.push_back(tensor);
        _copyInputTensors.push_back(tensor);

        if (descriptor.isStateInput) {
            _variableStates.push_back(std::make_shared<VariableState>(descriptor.nameFromCompiler, tensor));
        } else if (descriptor.isShapeTensor) {
            _shapesInputTensors.push_back(tensor);
        }
    } else {
        _outputTensors.push_back(tensor);
        _copyOutputTensors.push_back(tensor);

        if (descriptor.isShapeTensor) {
            _shapesOutputTensors.push_back(tensor);
        }
    }
}
}  // namespace intel_npu
