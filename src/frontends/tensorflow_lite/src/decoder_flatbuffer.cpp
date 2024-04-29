// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_flatbuffer.h"

#ifdef FLATBUFFERS_LOCALE_INDEPENDENT
#    undef FLATBUFFERS_LOCALE_INDEPENDENT
#endif
#define FLATBUFFERS_LOCALE_INDEPENDENT 0
#include "flatbuffers/flexbuffers.h"
#include "schema_generated.h"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

size_t DecoderFlatBuffer::get_input_size() const {
    return m_input_info.size();
}

void DecoderFlatBuffer::get_input_node(size_t input_port_idx,
                                       std::string& producer_name,
                                       std::string& producer_output_port_name,
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
    const ov::frontend::InputModel& model) const {
    FRONT_END_GENERAL_CHECK(idx < get_input_size(), "Requested input is out-of-range");
    return decode_tensor(m_input_info.at(idx), model);
}

std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> DecoderFlatBuffer::decode_output_tensor(
    size_t idx,
    const ov::frontend::InputModel& model) const {
    FRONT_END_GENERAL_CHECK(idx < get_output_size(), "Requested output is out-of-range");
    return decode_tensor(m_output_info.at(idx), model);
}

std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> DecoderFlatBuffer::decode_tensor(
    const ov::frontend::tensorflow_lite::TensorInfo& tensor_info,
    const ov::frontend::InputModel& model) const {
    const auto tensor = tensor_info.tensor;
    std::vector<std::string> names = {tensor->name()->str()};

    return std::make_shared<ov::frontend::tensorflow_lite::TensorLitePlace>(
        model,
        ov::frontend::tensorflow_lite::get_ov_shape(tensor->shape(), tensor->shape_signature()),
        ov::frontend::tensorflow_lite::get_ov_type(tensor->type()),
        names,
        ov::frontend::tensorflow_lite::get_quantization(tensor->quantization()),
        ov::frontend::tensorflow_lite::get_sparsity(tensor->shape(), tensor->sparsity()),
        (tensor_info.buffer && tensor_info.buffer->data() ? tensor_info.buffer->data()->data() : nullptr));
}

ov::Any get_value_as_ov_any(const flexbuffers::Reference& value) {
#define CASE_MACRO(fbt, as_stmt) \
    case flexbuffers::fbt:       \
        return {value.as_stmt()};
    switch (value.GetType()) {
        CASE_MACRO(FBT_INT, AsInt32)
        CASE_MACRO(FBT_INDIRECT_INT, AsInt32)
        CASE_MACRO(FBT_UINT, AsUInt32)
        CASE_MACRO(FBT_INDIRECT_UINT, AsUInt32)
        CASE_MACRO(FBT_FLOAT, AsFloat)
        CASE_MACRO(FBT_INDIRECT_FLOAT, AsFloat)
        CASE_MACRO(FBT_STRING, AsString)
        CASE_MACRO(FBT_BOOL, AsBool)
    default:
        return {};
    }
    return {};
}

ov::Any DecoderFlatBuffer::get_attribute(const std::string& name) const {
    const auto opts = m_node_def->custom_options();
    if (opts == nullptr)
        return {};
    const flexbuffers::Map& m = flexbuffers::GetRoot(opts->Data(), opts->size()).AsMap();
    return get_value_as_ov_any(m[name]);
}

TensorMetaInfo DecoderFlatBuffer::create_tensor_meta_info(
    const ov::frontend::tensorflow_lite::TensorInfo& tensor_info) const {
    TensorMetaInfo tensor_meta_info;
    const auto tensor = tensor_info.tensor;

    tensor_meta_info.m_partial_shape =
        ov::frontend::tensorflow_lite::get_ov_shape(tensor->shape(), tensor->shape_signature());
    tensor_meta_info.m_element_type = ov::frontend::tensorflow_lite::get_ov_type(tensor->type());
    tensor_meta_info.m_quantization_info = ov::frontend::tensorflow_lite::get_quantization(tensor->quantization());
    tensor_meta_info.m_sparsity_info = ov::frontend::tensorflow_lite::get_sparsity(tensor->shape(), tensor->sparsity());
    tensor_meta_info.m_tensor_data =
        (tensor_info.buffer && tensor_info.buffer->data() ? tensor_info.buffer->data()->data() : nullptr);
    tensor_meta_info.m_tensor_names = {tensor->name()->str()};

    return tensor_meta_info;
}

TensorMetaInfo DecoderFlatBuffer::get_input_tensor_info(size_t idx) const {
    FRONT_END_GENERAL_CHECK(idx < get_input_size(), "Requested input is out-of-range");
    const auto& tensor_info = m_input_info.at(idx);
    return create_tensor_meta_info(tensor_info);
}

TensorMetaInfo DecoderFlatBuffer::get_output_tensor_info(size_t idx) const {
    FRONT_END_GENERAL_CHECK(idx < get_output_size(), "Requested output is out-of-range");
    const auto& tensor_info = m_output_info.at(idx);
    return create_tensor_meta_info(tensor_info);
}

DecoderFlatBufferTensors::DecoderFlatBufferTensors(const TensorInfo& info, int64_t input_idx, int64_t output_idx)
    : m_input_idx(input_idx),
      m_output_idx(output_idx) {
    const auto tensor = info.tensor;

    m_tensor_meta_info.m_partial_shape =
        ov::frontend::tensorflow_lite::get_ov_shape(tensor->shape(), tensor->shape_signature());
    m_tensor_meta_info.m_element_type = ov::frontend::tensorflow_lite::get_ov_type(tensor->type());
    m_tensor_meta_info.m_quantization_info = ov::frontend::tensorflow_lite::get_quantization(tensor->quantization());
    m_tensor_meta_info.m_sparsity_info =
        ov::frontend::tensorflow_lite::get_sparsity(tensor->shape(), tensor->sparsity());
    m_tensor_meta_info.m_tensor_data = (info.buffer && info.buffer->data() ? info.buffer->data()->data() : nullptr);
    m_tensor_meta_info.m_tensor_names = {tensor->name()->str()};
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
