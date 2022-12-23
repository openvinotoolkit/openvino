// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_flatbuffer.h"
#include "schema_generated.h"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/special_types.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

ov::Any DecoderFlatBuffer::get_attribute(const std::string& name) const {
    auto bopts = m_node_def->builtin_options();
    auto copts = m_node_def->custom_options();
    if (bopts == NULL) {
        // TODO: try custom
        return {};
    }
    if (name == "Conv2DOptions")
        return m_node_def->builtin_options_as_Conv2DOptions();
    if (name == "DepthwiseConv2DOptions")
        return m_node_def->builtin_options_as_DepthwiseConv2DOptions();
    if (name == "ConcatenationOptions")
        return m_node_def->builtin_options_as_ConcatenationOptions();
    if (name == "ReshapeOptions")
        return m_node_def->builtin_options_as_ReshapeOptions();
    return {};
}

size_t DecoderFlatBuffer::get_input_size() const {
    return m_node_def->inputs()->size();
}

void DecoderFlatBuffer::get_input_node(size_t input_port_idx,
                                      std::string& producer_name,
                                      size_t& producer_output_port_index) const {
    const auto inputs = m_node_def->inputs();
    FRONT_END_GENERAL_CHECK(inputs->size() > input_port_idx, "Input port index is out of range for node ", get_op_name(), ". Requested input index: ", input_port_idx, ". Number of inputs: ", inputs->size());
    auto input_tensor_idx = (*inputs)[input_port_idx];
    auto tensor = m_input_info.at(input_port_idx).tensor;
    std::string name = (*tensor).name()->str();
    producer_name = name;
    producer_output_port_index = input_tensor_idx;
}

const std::string& DecoderFlatBuffer::get_op_type() const {
    return m_type;
}

const std::string& DecoderFlatBuffer::get_op_name() const {
    return m_name;
}

std::vector<size_t> DecoderFlatBuffer::get_output_tensor_indices() const {
    const auto outputs = m_node_def->outputs();
    return {outputs->begin(), outputs->end()};
}

size_t DecoderFlatBuffer::get_output_size() const {
    return m_node_def->outputs()->size();
}

std::string DecoderFlatBuffer::get_output_tensor_name(size_t idx) const {
    // FIXME add checks
    return m_output_info.at(idx).tensor->name()->str();
}

std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> DecoderFlatBuffer::decode_input_tensor(size_t idx, const InputModel& model) const {
    FRONT_END_GENERAL_CHECK(idx < get_input_size(), "Requested input is out-of-range");
    return decode_tensor(m_input_info.at(idx), model);
}

std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> DecoderFlatBuffer::decode_output_tensor(size_t idx, const InputModel& model) const {
    FRONT_END_GENERAL_CHECK(idx < get_output_size(), "Requested output is out-of-range");
    return decode_tensor(m_output_info.at(idx), model);
}

std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> DecoderFlatBuffer::decode_tensor(const ov::frontend::tensorflow_lite::TensorInfo& tensor_info, const InputModel& model) const {
    const auto tensor = tensor_info.tensor;
    std::vector<std::string> names = {tensor->name()->str()};

    return std::make_shared<ov::frontend::tensorflow_lite::TensorLitePlace>(
            model,
            ov::frontend::tensorflow_lite::get_ov_shape(tensor->shape()),
            ov::frontend::tensorflow_lite::get_ov_type(tensor->type()),
            names,
            ov::frontend::tensorflow_lite::get_quantization(tensor->quantization()),
            tensor_info.input_idx,
            tensor_info.output_idx,
            (tensor_info.buffer->data() ? tensor_info.buffer->data()->data() : nullptr));
}

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
