// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_flatbuffer.h"

#include "schema_generated.h"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

size_t DecoderFlatBuffer::get_input_size() const {
    return m_input_info.size();
}

void DecoderFlatBuffer::get_input_node(size_t input_port_idx,
                                       std::string& producer_name,
                                       size_t& producer_output_port_index) const {
    const auto inputs = m_node_def->inputs();
    FRONT_END_GENERAL_CHECK(inputs->size() > input_port_idx,
                            "Input port index is out of range for node ",
                            get_op_name(),
                            ". Requested input index: ",
                            input_port_idx,
                            ". Number of inputs: ",
                            inputs->size());
    auto input_tensor_idx = (*inputs)[static_cast<flatbuffers::uoffset_t>(input_port_idx)];
    auto tensor = m_input_info.at(input_port_idx).tensor;
    std::string name = (*tensor).name()->str();
    producer_name = name;
    producer_output_port_index = input_tensor_idx;
}

void DecoderFlatBuffer::get_input_node(size_t input_port_idx,
                                       std::string& producer_name,
                                       size_t& producer_output_port_index,
                                       const OpTypeByName& op_type_by_name) const {
    FRONT_END_NOT_IMPLEMENTED("get_input_node method with op_type_by_name map is not implemented for TFL FE.");
}

const std::string& DecoderFlatBuffer::get_op_type() const {
    return m_type;
}

const std::string& DecoderFlatBuffer::get_op_name() const {
    return m_name;
}

size_t DecoderFlatBuffer::get_output_size() const {
    return m_node_def->outputs()->size();
}

std::string DecoderFlatBuffer::get_input_tensor_name(size_t idx) const {
    FRONT_END_GENERAL_CHECK(idx < get_input_size(), "Requested input is out-of-range");
    return m_input_info.at(idx).tensor->name()->str();
}

std::string DecoderFlatBuffer::get_output_tensor_name(size_t idx) const {
    FRONT_END_GENERAL_CHECK(idx < get_output_size(), "Requested output is out-of-range");
    return m_output_info.at(idx).tensor->name()->str();
}

ov::element::Type DecoderFlatBuffer::get_output_tensor_type(size_t idx) const {
    FRONT_END_GENERAL_CHECK(idx < get_output_size(), "Requested output is out-of-range");
    return get_ov_type(m_output_info.at(idx).tensor->type());
}

std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> DecoderFlatBuffer::decode_input_tensor(
    size_t idx,
    const InputModel& model) const {
    FRONT_END_GENERAL_CHECK(idx < get_input_size(), "Requested input is out-of-range");
    return decode_tensor(m_input_info.at(idx), model);
}

std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> DecoderFlatBuffer::decode_output_tensor(
    size_t idx,
    const InputModel& model) const {
    FRONT_END_GENERAL_CHECK(idx < get_output_size(), "Requested output is out-of-range");
    return decode_tensor(m_output_info.at(idx), model);
}

std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> DecoderFlatBuffer::decode_tensor(
    const ov::frontend::tensorflow_lite::TensorInfo& tensor_info,
    const InputModel& model) const {
    const auto tensor = tensor_info.tensor;
    std::vector<std::string> names = {tensor->name()->str()};

    return std::make_shared<ov::frontend::tensorflow_lite::TensorLitePlace>(
        model,
        ov::frontend::tensorflow_lite::get_ov_shape(tensor->shape(), tensor->shape_signature()),
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
