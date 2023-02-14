// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/isync_infer_request.hpp"

#include "cpp_interfaces/plugin_itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

void check_batched_tensors(const ov::Output<const ov::Node>& input, const std::vector<ov::Tensor>& tensors) {
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
        batch_idx += static_cast<int64_t>(tensors[0].get_shape().size());
    }
    OPENVINO_ASSERT(batch_idx == 0,
                    "set_input_tensors/set_tensors is not currently supported for batch dimension index ",
                    batch_idx,
                    " != 0");
    std::for_each(tensors.begin(), tensors.end(), [&batch_idx](const ov::Tensor& item) {
        OPENVINO_ASSERT(item.get_shape()[batch_idx] == 1,
                        "set_input_tensors/set_tensors. Tensors shall represent one item in a batch, ",
                        item.get_shape()[batch_idx],
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
    auto batched_shape = tensors[0].get_shape();
    auto element_type = tensors[0].get_element_type();
    batched_shape[batch_idx] = tensors_size;
    for (const auto& item : tensors) {
        auto item_shape = item.get_shape();
        item_shape[batch_idx] = batched_shape[batch_idx];
        OPENVINO_ASSERT(item_shape == batched_shape && item.get_element_type() == element_type &&
                            "set_input_tensors/set_tensors error. Tensor with element type ",
                        item.get_element_type(),
                        " and shape ",
                        item_shape,
                        " is not compatible with batched tensor with element type ",
                        element_type,
                        " and shape ",
                        batched_shape);
    }
}

}  // namespace

ov::IInferRequest::~IInferRequest() = default;

ov::ISyncInferRequest::ISyncInferRequest(const std::shared_ptr<ov::ICompiledModel>& compiled_model)
    : m_compiled_model(compiled_model) {}

const std::vector<ov::Output<const ov::Node>>& ov::ISyncInferRequest::get_inputs() const {
    return m_compiled_model->inputs();
}
const std::vector<ov::Output<const ov::Node>>& ov::ISyncInferRequest::get_outputs() const {
    return m_compiled_model->outputs();
}
const std::shared_ptr<ov::ICompiledModel>& ov::ISyncInferRequest::get_compiled_model() const {
    return m_compiled_model;
}

ov::ISyncInferRequest::FoundPort ov::ISyncInferRequest::find_port(const ov::Output<const ov::Node>& port) const {
    ov::ISyncInferRequest::FoundPort::Type type = ov::ISyncInferRequest::FoundPort::Type::INPUT;
    for (const auto& ports : {get_inputs(), get_outputs()}) {
        for (size_t i = 0; i < ports.size(); i++) {
            if (ports[i] == port) {
                return {i, type};
            }
        }
        type = ov::ISyncInferRequest::FoundPort::Type::OUTPUT;
    }
    return {0, ov::ISyncInferRequest::FoundPort::Type::NOT_FOUND};
}

void ov::ISyncInferRequest::convert_batched_tensors() {
    for (const auto& item : m_batched_tensors) {
        auto tmp_shape = item.second.at(0).get_shape();
        auto tmp_et = item.second.at(0).get_element_type();
        tmp_shape[0] = item.second.size();
        ov::RemoteContext remote_context;
        ov::Tensor input_tensor;
        try {
            auto net = get_compiled_model();
            if (net) {
                remote_context = net->get_context();
            }
        } catch (const ov::NotImplemented&) {
        }
        if (remote_context._impl) {
            input_tensor = remote_context.create_host_tensor(tmp_et, tmp_shape);
        } else {
            input_tensor = ov::Tensor(tmp_et, tmp_shape);
        }
        auto ptr = input_tensor.data<uint8_t>();

        // Perform memory copy
        ov::parallel_for(input_tensor.get_size(), [&](size_t i) {
            const auto& tensor = item.second.at(i);
            memcpy(ptr + i * tensor.get_byte_size(), tensor.data<uint8_t>(), tensor.get_byte_size());
        });
        set_tensor(get_inputs()[item.first], input_tensor);
    }
}

ov::Tensor ov::ISyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    OV_ITT_SCOPED_TASK(InferenceEngine::itt::domains::Plugin, "get_tensor");
    auto found_port = find_port(port);
    OPENVINO_ASSERT(!found_port.found(), "Cannot find tensor for port ", port);
    if (found_port.is_input()) {
        auto input = m_compiled_model->inputs().at(found_port.idx);
        // TODO: Support dynamic inputs
        // if (input.get_partial_shape().is_dynamic())
        return m_input_tensors.at(found_port.idx);
    }

    auto output = m_compiled_model->outputs().at(found_port.idx);
    // TODO: Support dynamic inputs
    // if (output.get_partial_shape().is_dynamic())
    return m_output_tensors.at(found_port.idx);
}

void ov::ISyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) {
    OV_ITT_SCOPED_TASK(InferenceEngine::itt::domains::Plugin, "set_tensor");
    auto found_port = find_port(port);
    OPENVINO_ASSERT(!found_port.found(), "Cannot find tensor for port ", port);
    OPENVINO_ASSERT(
        port.get_element_type() == tensor.get_element_type(),
        "Failed to set output tensor, the tensor element type is not corresponding with output element type");
    OPENVINO_ASSERT(port.get_partial_shape().is_dynamic() || tensor.get_shape() == port.get_shape(),
                    "Input tensor size is not equal with model input size (",
                    tensor.get_shape(),
                    " != ",
                    port.get_shape(),
                    ").");
    if (found_port.is_input()) {
        m_input_tensors.at(found_port.idx) = tensor;
        m_batched_tensors.erase(found_port.idx);
    } else {
        m_output_tensors.at(found_port.idx) = tensor;
    }
}

std::vector<ov::Tensor> ov::ISyncInferRequest::get_tensors(const ov::Output<const ov::Node>& port) const {
    OV_ITT_SCOPED_TASK(InferenceEngine::itt::domains::Plugin, "get_tensors");
    auto found_port = find_port(port);
    OPENVINO_ASSERT(!found_port.found() && found_port.is_input(), "Cannot find input tensors for port ", port);
    if (m_batched_tensors.count(found_port.idx))
        return m_batched_tensors.at(found_port.idx);
    return {};
}

void ov::ISyncInferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                        const std::vector<ov::Tensor>& tensors) {
    OV_ITT_SCOPED_TASK(InferenceEngine::itt::domains::Plugin, "set_tensors");
    auto found_port = find_port(port);
    OPENVINO_ASSERT(!found_port.found() && found_port.is_input(), "Cannot find input tensors for port ", port);
    if (tensors.size() == 1) {
        set_tensor(port, tensors[0]);
        return;
    }

    check_batched_tensors(port, tensors);
    set_tensors_impl(port, tensors);
}

void ov::ISyncInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port,
                                             const std::vector<ov::Tensor>& tensors) {
    OPENVINO_ASSERT_HELPER(::ov::NotImplemented,
                           "",
                           false,
                           "Not Implemented",
                           "set_input_tensors/set_tensors are not supported by this plugin");
}

void ov::ISyncInferRequest::check_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) const {
    bool is_input = ov::op::util::is_parameter(port.get_node());
    std::string tensor_type = is_input ? "input" : "output";

    bool is_dynamic = port.get_partial_shape().is_dynamic();
    OPENVINO_ASSERT(is_dynamic || port.get_shape() == tensor.get_shape(),
                    "The ",
                    tensor_type,
                    " tensor size is not equal to the model ",
                    tensor_type,
                    " type: got ",
                    tensor.get_size(),
                    " expecting ",
                    port.get_shape(),
                    ".");
}

void ov::ISyncInferRequest::check_tensors() const {
    const auto& inputs = m_compiled_model->inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        check_tensor(inputs[i], m_input_tensors[i]);
    }
    const auto& outputs = m_compiled_model->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        check_tensor(outputs[i], m_output_tensors[i]);
    }
}
