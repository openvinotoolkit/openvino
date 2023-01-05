// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iinfer_request.hpp"

#include <openvino/core/except.hpp>
#include <openvino/core/layout.hpp>
#include <openvino/op/util/op_types.hpp>

#include "cpp_interfaces/plugin_itt.hpp"

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

void check_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) {
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

}  // namespace

ov::IInferRequest::IInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model)
    : m_compiled_model(compiled_model) {}

void ov::IInferRequest::start_async() {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::IInferRequest::wait() {
    OPENVINO_NOT_IMPLEMENTED;
}
bool ov::IInferRequest::wait_for(const std::chrono::milliseconds timeout) {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::IInferRequest::cancel() {
    OPENVINO_NOT_IMPLEMENTED;
}

std::vector<ov::ProfilingInfo> ov::IInferRequest::get_profiling_info() const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Tensor ov::IInferRequest::get_input_tensor(size_t idx) const {
    OV_ITT_SCOPED_TASK(InferenceEngine::itt::domains::Plugin, "get_input_tensor");
    OPENVINO_ASSERT(idx < m_compiled_model->inputs().size(),
                    "Cannot find input tensor for index ",
                    idx,
                    " number of inputs is ",
                    m_compiled_model->inputs().size());
    auto input = m_compiled_model->inputs().at(idx);
    // TODO: Support dynamic inputs
    // if (input.get_partial_shape().is_dynamic())
    return m_input_tensors.at(idx);
}

void ov::IInferRequest::set_input_tensor(size_t idx, const ov::Tensor& tensor) {
    OV_ITT_SCOPED_TASK(InferenceEngine::itt::domains::Plugin, "set_input_tensor");
    OPENVINO_ASSERT(idx < m_compiled_model->inputs().size(),
                    "Cannot find input tensor for index ",
                    idx,
                    " number of inputs is ",
                    m_compiled_model->inputs().size());
    auto input = m_compiled_model->inputs().at(idx);
    OPENVINO_ASSERT(
        input.get_element_type() == tensor.get_element_type(),
        "Failed to set output tensor, the tensor element type is not corresponding with output element type");
    OPENVINO_ASSERT(input.get_partial_shape().is_dynamic() || tensor.get_shape() == input.get_shape(),
                    "Input tensor size is not equal with model input size (",
                    tensor.get_shape(),
                    " != ",
                    input.get_shape(),
                    ").");
    m_input_tensors.at(idx) = tensor;
    m_batched_tensors.erase(idx);
}

std::vector<ov::Tensor> ov::IInferRequest::get_input_tensors(size_t idx) const {
    if (m_batched_tensors.count(idx))
        return m_batched_tensors.at(idx);
    return {};
}

void ov::IInferRequest::set_input_tensors(size_t idx, const std::vector<ov::Tensor>& tensors) {
    OPENVINO_ASSERT(idx < m_compiled_model->inputs().size(),
                    "Cannot find input tensor for index ",
                    idx,
                    " number of inputs is ",
                    m_compiled_model->inputs().size());
    if (tensors.size() == 1) {
        set_input_tensor(idx, tensors[0]);
        return;
    }

    check_batched_tensors(m_compiled_model->inputs().at(idx), tensors);
    set_input_tensors_imp(idx, tensors);
}
void ov::IInferRequest::set_input_tensors_imp(size_t idx, const std::vector<ov::Tensor>& tensors) {
    OPENVINO_ASSERT_HELPER(::ov::NotImplemented,
                           "",
                           false,
                           "Not Implemented",
                           "set_input_tensors/set_tensors are not supported by this plugin");
}

ov::Tensor ov::IInferRequest::get_output_tensor(size_t idx) const {
    OV_ITT_SCOPED_TASK(InferenceEngine::itt::domains::Plugin, "get_output_tensor");
    OPENVINO_ASSERT(idx < m_compiled_model->outputs().size(),
                    "Cannot find output tensor for index ",
                    idx,
                    " number of outputs is ",
                    m_compiled_model->outputs().size());
    auto output = m_compiled_model->outputs().at(idx);
    // TODO: Support dynamic inputs
    // if (output.get_partial_shape().is_dynamic())
    return m_output_tensors.at(idx);
}

void ov::IInferRequest::set_output_tensor(size_t idx, const ov::Tensor& tensor) {
    OV_ITT_SCOPED_TASK(InferenceEngine::itt::domains::Plugin, "set_output_tensor");
    OPENVINO_ASSERT(idx < m_compiled_model->outputs().size(),
                    "Cannot find output tensor for index ",
                    idx,
                    " number of outputs is ",
                    m_compiled_model->outputs().size());
    auto output = m_compiled_model->outputs().at(idx);
    OPENVINO_ASSERT(
        output.get_element_type() == tensor.get_element_type(),
        "Failed to set output tensor, the tensor element type is not corresponding with output element type");
    OPENVINO_ASSERT(output.get_partial_shape().is_dynamic() || tensor.get_shape() == output.get_shape(),
                    "Output tensor size is not equal with model output size (",
                    tensor.get_shape(),
                    " != ",
                    output.get_shape(),
                    ").");
    m_output_tensors.at(idx) = tensor;
}

std::vector<ov::VariableState> ov::IInferRequest::query_state() const {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::IInferRequest::set_callback(std::function<void(std::exception_ptr)> callback) {
    m_callback = std::move(callback);
}

void ov::IInferRequest::check_tensors() {
    const auto& inputs = m_compiled_model->inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        check_tensor(inputs[i], m_input_tensors[i]);
    }
    const auto& outputs = m_compiled_model->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        check_tensor(outputs[i], m_output_tensors[i]);
    }
}
