// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/isync_infer_request.hpp"

#include <functional>
#include <memory>
#include <unordered_map>

#include "openvino/core/except.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/plugin_itt.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/common_util.hpp"

namespace {
void check_batched_tensors(const ov::Output<const ov::Node>& input,
                           const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    OPENVINO_ASSERT(!tensors.empty(), "set_input_tensors/set_tensors can't be called with empty tensors");
    OPENVINO_ASSERT(
        tensors.size() != 1,
        "Internal error (plugin): check_batched_tensors is not allowed to have only one tensor inside batch");

    auto layout = ov::layout::get_layout(input);
    OPENVINO_ASSERT(ov::layout::has_batch(layout),
                    "set_input_tensors/set_tensors can be used only for inputs with N(batch) dimension"
                    " 'layout' defined. Current layout is ",
                    layout.to_string());
    auto batch_idx = ov::layout::batch_idx(layout);
    if (batch_idx < 0) {
        // TODO: Do we need this logic?
        batch_idx += static_cast<int64_t>(tensors[0]->get_shape().size());
    }
    OPENVINO_ASSERT(batch_idx == 0,
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
    if (input.get_partial_shape().rank().is_static()) {
        OPENVINO_ASSERT(batch_idx >= 0 && batch_idx < input.get_partial_shape().rank().get_length(),
                        "set_input_tensors/set_tensors error. Layout ",
                        layout.to_string(),
                        " is incorrect for operation with shape ",
                        input.get_partial_shape());
        auto batch = input.get_partial_shape()[batch_idx];

        OPENVINO_ASSERT(batch.is_dynamic() || batch.get_length() == tensors_size,
                        "set_input_tensors/set_tensors error. Input shape ",
                        input.get_partial_shape(),
                        "batch ",
                        batch,
                        "doesn't match with total blobs count: ",
                        tensors_size);
    }

    // In future consider checking if blobs point to contiguous range of memory and use single 'SetBlob' instead
    auto batched_shape = tensors[0]->get_shape();
    auto element_type = tensors[0]->get_element_type();
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

}  // namespace

ov::IInferRequest::~IInferRequest() = default;

ov::ISyncInferRequest::ISyncInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model)
    : m_compiled_model(compiled_model) {
    OPENVINO_ASSERT(m_compiled_model);
    // Create map of empty tensors and cache ports from the compiled model
    auto port_type = ov::ISyncInferRequest::FoundPort::Type::INPUT;
    for (const auto& ports : {get_inputs(), get_outputs()}) {
        for (size_t i = 0; i < ports.size(); i++) {
            const auto& port = ports[i];
            if (m_tensors.find(port.get_tensor_ptr()) == m_tensors.end())
                m_tensors[port.get_tensor_ptr()] = ov::SoPtr<ov::ITensor>();
            size_t port_hash = ov::util::hash_combine(std::vector<size_t>{std::hash<const ov::Node*>()(port.get_node()),
                                                                          std::hash<size_t>()(port.get_index())});
            m_cached_ports[port_hash] = {i, port_type};
        }
        port_type = ov::ISyncInferRequest::FoundPort::Type::OUTPUT;
    }
}

const std::vector<ov::Output<const ov::Node>>& ov::ISyncInferRequest::get_inputs() const {
    return m_compiled_model->inputs();
}
const std::vector<ov::Output<const ov::Node>>& ov::ISyncInferRequest::get_outputs() const {
    return m_compiled_model->outputs();
}
const std::shared_ptr<const ov::ICompiledModel>& ov::ISyncInferRequest::get_compiled_model() const {
    return m_compiled_model;
}

ov::ISyncInferRequest::FoundPort ov::ISyncInferRequest::find_port(const ov::Output<const ov::Node>& port) const {
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
        std::lock_guard<std::mutex> lock(m_cache_mutex);
        if (m_cached_ports.find(port_hash) != m_cached_ports.end()) {
            // Cached port for the hash was found
            return m_cached_ports[port_hash];
        }
    }
    ov::ISyncInferRequest::FoundPort::Type type = ov::ISyncInferRequest::FoundPort::Type::INPUT;
    for (const auto& ports : {get_inputs(), get_outputs()}) {
        for (size_t i = 0; i < ports.size(); i++) {
            if (ports[i].get_index() == port.get_index() && check_nodes(ports[i].get_node(), port.get_node()) &&
                check_tensor_names(ports[i].get_names(), port.get_names())) {
                std::lock_guard<std::mutex> lock(m_cache_mutex);
                m_cached_ports[port_hash] = {i, type};
                return m_cached_ports[port_hash];
            }
        }
        type = ov::ISyncInferRequest::FoundPort::Type::OUTPUT;
    }
    return {0, ov::ISyncInferRequest::FoundPort::Type::NOT_FOUND};
}

void ov::ISyncInferRequest::convert_batched_tensors() {
    std::unordered_map<std::shared_ptr<ov::descriptor::Tensor>, ov::SoPtr<ov::ITensor>> prepared_tensors;
    for (const auto& item : m_batched_tensors) {
        OPENVINO_ASSERT(item.second.at(0), "Unintialized tensor is provided!");
        auto tmp_shape = item.second.at(0)->get_shape();
        auto tmp_et = item.second.at(0)->get_element_type();
        tmp_shape[0] = item.second.size();
        ov::SoPtr<ov::IRemoteContext> remote_context;
        ov::SoPtr<ov::ITensor> input_tensor;
        try {
            auto net = get_compiled_model();
            if (net) {
                remote_context = net->get_context();
            }
        } catch (const ov::NotImplemented&) {
        }
        if (remote_context) {
            input_tensor = remote_context->create_host_tensor(tmp_et, tmp_shape);
        } else {
            input_tensor = {ov::make_tensor(tmp_et, tmp_shape), nullptr};
        }
        auto ptr = static_cast<uint8_t*>(input_tensor->data());

        // Perform memory copy
        ov::parallel_for(item.second.size(), [&](size_t i) {
            const auto& tensor = item.second.at(i);
            memcpy(ptr + i * tensor->get_byte_size(), static_cast<uint8_t*>(tensor->data()), tensor->get_byte_size());
        });
        prepared_tensors[item.first] = input_tensor;
    }

    for (const auto& item : prepared_tensors) {
        if (m_tensors.count(item.first))
            m_tensors[item.first] = item.second;
    }
}

ov::SoPtr<ov::ITensor>& ov::ISyncInferRequest::get_tensor_ptr(const ov::Output<const ov::Node>& port) const {
    auto found_port = find_port(port);
    OPENVINO_ASSERT(found_port.found(), "Cannot find tensor for port ", port);
    auto ports = found_port.is_input() ? get_inputs() : get_outputs();
    auto it = m_tensors.find(ports.at(found_port.idx).get_tensor_ptr());
    OPENVINO_ASSERT(it != m_tensors.end(), "Cannot find tensor for port: ", port);

    return it->second;
}

ov::SoPtr<ov::ITensor> ov::ISyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::Plugin, "get_tensor");
    return get_tensor_ptr(port);
}

void ov::ISyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::Plugin, "set_tensor");
    auto found_port = find_port(port);
    OPENVINO_ASSERT(found_port.found(), "Cannot find tensor for port ", port);
    try {
        check_tensor(port, tensor);
    } catch (const ov::Exception& ex) {
        OPENVINO_THROW("Failed to set tensor. ", ex.what());
    }
    if (found_port.is_input()) {
        m_tensors.at(get_inputs().at(found_port.idx).get_tensor_ptr()) = tensor;
        m_batched_tensors.erase(get_inputs().at(found_port.idx).get_tensor_ptr());
    } else {
        m_tensors.at(get_outputs().at(found_port.idx).get_tensor_ptr()) = tensor;
    }
}

std::vector<ov::SoPtr<ov::ITensor>> ov::ISyncInferRequest::get_tensors(const ov::Output<const ov::Node>& port) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::Plugin, "get_tensors");
    auto found_port = find_port(port);
    OPENVINO_ASSERT(found_port.found(), "Cannot find input tensors for port ", port);
    if (found_port.is_input() && m_batched_tensors.count(get_inputs().at(found_port.idx).get_tensor_ptr()))
        return m_batched_tensors.at(get_inputs().at(found_port.idx).get_tensor_ptr());
    return {};
}

void ov::ISyncInferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                        const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::Plugin, "set_tensors");
    auto found_port = find_port(port);
    OPENVINO_ASSERT(found_port.found() && found_port.is_input(), "Cannot find input tensors for port ", port);
    if (tensors.size() == 1) {
        set_tensor(port, tensors[0]);
        return;
    }

    check_batched_tensors(port, tensors);
    set_tensors_impl(port, tensors);
}

void ov::ISyncInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port,
                                             const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    OPENVINO_THROW_NOT_IMPLEMENTED("Not Implemented set_input_tensors/set_tensors are not supported by this plugin");
}

void ov::ISyncInferRequest::check_tensor(const ov::Output<const ov::Node>& port,
                                         const ov::SoPtr<ov::ITensor>& tensor) const {
    OPENVINO_ASSERT(tensor);
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

void ov::ISyncInferRequest::allocate_tensor(
    const ov::Output<const ov::Node>& port,
    const std::function<void(ov::SoPtr<ov::ITensor>& tensor)>& allocate_callback) {
    auto& tensor = get_tensor_ptr(port);
    allocate_callback(tensor);
}

void ov::ISyncInferRequest::check_tensors() const {
    const auto& inputs = m_compiled_model->inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        check_tensor(inputs[i], m_tensors.at(inputs[i].get_tensor_ptr()));
    }
    const auto& outputs = m_compiled_model->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        check_tensor(outputs[i], m_tensors.at(outputs[i].get_tensor_ptr()));
    }
}
