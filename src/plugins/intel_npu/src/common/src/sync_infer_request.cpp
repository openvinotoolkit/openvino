// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/sync_infer_request.hpp"

#include "intel_npu/prefix.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/plugin_itt.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/utils/utils.hpp"

namespace {

constexpr size_t BATCH_AXIS = 0;

}

namespace intel_npu {

SyncInferRequest::SyncInferRequest(const std::shared_ptr<const ICompiledModel>& compiledModel, const Config& config)
    : _compiledModel(compiledModel),
      _metadata(compiledModel->get_graph()->get_metadata()),
      _logger("SyncInferRequest", config.get<LOG_LEVEL>()),
      _userInputTensors(_metadata.inputs.size(), std::vector<ov::SoPtr<ov::ITensor>>(1, {nullptr})),
      _userOutputTensors(_metadata.outputs.size(), {nullptr}) {
    OPENVINO_ASSERT(_compiledModel);

    if (get_outputs().empty()) {
        OPENVINO_THROW("Inference request creation: no output found for network " + _metadata.name);
    }

    // Create map of empty tensors and cache ports from the compiled model
    // See the ov::ISyncInferRequest constructor
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
}

SyncInferRequest::FoundPort SyncInferRequest::find_port(const ov::Output<const ov::Node>& port) const {
    // check if the tensor names of target port is a subset of source port's tensor names
    auto check_tensor_names = [](const std::unordered_set<std::string>& source,
                                 const std::unordered_set<std::string>& target) {
        for (auto const& name : target) {
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
        return get_user_input(foundPort.idx);
    }
    return _userOutputTensors.at(foundPort.idx);
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
        get_user_input(foundPort.idx) = tensor;
    } else {
        _userOutputTensors.at(foundPort.idx) = tensor;
    }
}

std::vector<ov::SoPtr<ov::ITensor>> SyncInferRequest::get_tensors(const ov::Output<const ov::Node>& port) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::Plugin, "get_tensors");

    auto foundPort = find_port(port);
    OPENVINO_ASSERT(foundPort.found(), "Cannot find input tensors for port ", port);

    if (foundPort.is_input() && is_batched_input(foundPort.idx)) {
        return get_user_inputs(foundPort.idx);
    }

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

void SyncInferRequest::check_batched_tensors(const ov::Output<const ov::Node>& port,
                                             const std::vector<ov::SoPtr<ov::ITensor>>& tensors) const {
    OPENVINO_ASSERT(!tensors.empty(), "set_input_tensors/set_tensors can't be called with empty tensors");
    OPENVINO_ASSERT(
        tensors.size() != 1,
        "Internal error (plugin): check_batched_tensors is not allowed to have only one tensor inside batch");

    auto layout = ov::layout::get_layout(port);

    int64_t batch_idx;

    if (layout.empty()) {
        _logger.warning("set_input_tensors/set_tensors layout is not set, assuming batch dimension is found on 0 axis");
        batch_idx = BATCH_AXIS;
    } else {
        OPENVINO_ASSERT(ov::layout::has_batch(layout),
                        "set_input_tensors/set_tensors can be used only for inputs with N(batch) dimension"
                        " 'layout' defined. Current layout is ",
                        layout.to_string());
        batch_idx = ov::layout::batch_idx(layout);
    }

    if (batch_idx < 0) {
        batch_idx += static_cast<int64_t>(tensors[BATCH_AXIS]->get_shape().size());
    }
    OPENVINO_ASSERT(batch_idx == BATCH_AXIS,
                    "set_input_tensors/set_tensors is not currently supported for batch dimension index ",
                    batch_idx,
                    " != 0");
    std::for_each(tensors.begin(), tensors.end(), [&batch_idx](const ov::SoPtr<ov::ITensor>& item) {
        OPENVINO_ASSERT(item, "Unintialized tensor is provided!");
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

    auto batched_shape = tensors[BATCH_AXIS]->get_shape();
    auto element_type = tensors[BATCH_AXIS]->get_element_type();
    batched_shape[batch_idx] = tensors_size;
    for (const auto& item : tensors) {
        OPENVINO_ASSERT(item, "Unintialized tensor is provided!");
        auto item_shape = item->get_shape();
        item_shape[batch_idx] = batched_shape[batch_idx];
        OPENVINO_ASSERT(item_shape == batched_shape && item->get_element_type() == element_type &&
                            "set_input_tensors/set_tensors error. Tensor with element type ",
                        item->get_element_type(),
                        " and shape ",
                        item_shape,
                        " is not compatible with batched tensor with element type ",
                        element_type,
                        " and shape ",
                        batched_shape);
        OPENVINO_ASSERT(item->is_continuous(), "Strides for batched tensors should be default.");
    }
}

void SyncInferRequest::check_tensors() const {
    const auto& inputs = _compiledModel->inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        if (is_batched_input(i)) {
            check_batched_tensors(inputs[i], get_user_inputs(i));
            continue;
        }
        if (get_user_input(i)) {
            check_tensor(inputs[i], get_user_input(i));
        }
    }

    const auto& outputs = _compiledModel->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        if (_userOutputTensors.at(i)) {
            check_tensor(outputs[i], _userOutputTensors.at(i));
        }
    }
}

std::shared_ptr<ov::ITensor> SyncInferRequest::allocate_tensor(const IODescriptor& descriptor,
                                                               const size_t index,
                                                               const bool isInput,
                                                               const ov::Allocator& allocator,
                                                               const std::optional<std::size_t> batchSize) const {
    check_network_precision(descriptor.precision);

    std::shared_ptr<ov::ITensor> tensor;
    ov::Shape allocatedTensorShape = descriptor.shapeFromCompiler.get_max_shape();

    if (batchSize.has_value()) {
        allocatedTensorShape[BATCH_AXIS] = *batchSize;
    }

    if (descriptor.isStateOutput) {
        // Only one buffer is required for each (state input, state output) pair, acting as an input before running the
        // inference and as an output after performing it. Thus both the "state input" and "state output" entries shall
        // point to the same buffer.
        OPENVINO_ASSERT(descriptor.relatedDescriptorIndex.has_value(),
                        "The link between state descriptors is missing, state name: ",
                        descriptor.nameFromCompiler);
        tensor = get_user_input(*descriptor.relatedDescriptorIndex)._ptr;
    } else if (allocator) {
        tensor = ov::make_tensor(descriptor.precision, allocatedTensorShape, allocator);
    } else {
        tensor = ov::make_tensor(descriptor.precision, allocatedTensorShape);
    }

    if (isInput) {
        if (get_user_input(index) == nullptr) {
            get_user_input(index) = tensor;
        }

        if (descriptor.isStateInput) {
            _variableStates.push_back(std::make_shared<VariableState>(descriptor.nameFromCompiler, tensor));
        }
    } else if (_userOutputTensors.at(index) == nullptr) {
        _userOutputTensors.at(index) = tensor;
    }

    return tensor;
}

bool SyncInferRequest::is_batched_input(size_t idx) const {
    return _userInputTensors.at(idx).size() > 1;
}

ov::SoPtr<ov::ITensor>& SyncInferRequest::get_user_input(size_t index) const {
    return _userInputTensors.at(index).at(0);
}

std::vector<ov::SoPtr<ov::ITensor>>& SyncInferRequest::get_user_inputs(size_t index) const {
    return _userInputTensors.at(index);
}

}  // namespace intel_npu
