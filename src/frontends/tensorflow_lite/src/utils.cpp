// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <openvino/frontend/tensorflow_lite/node_context.hpp>
#include <openvino/opsets/opset10.hpp>

#include "tflite_ops/tflite_quantize.hpp"

using namespace ov;

std::shared_ptr<ov::frontend::tensorflow_lite::QuantizationInfo> ov::frontend::tensorflow_lite::get_quantization(
    const tflite::QuantizationParameters* tf_quantization) {
    if (tf_quantization == nullptr)
        return {};
    auto quantization = std::make_shared<ov::frontend::tensorflow_lite::QuantizationInfo>();
    auto tf_zp = tf_quantization->zero_point();
    auto tf_scale = tf_quantization->scale();
    if (tf_zp != nullptr)
        quantization->set_zero_point({(*tf_zp).begin(), (*tf_zp).end()});
    if (tf_scale != nullptr)
        quantization->set_scale({(*tf_scale).begin(), (*tf_scale).end()});
    if (quantization->get_zero_point().empty() && quantization->get_scale().empty())
        return {};
    quantization->set_axis(tf_quantization->quantized_dimension());
    return quantization;
}

std::shared_ptr<ov::frontend::tensorflow_lite::SparsityInfo> ov::frontend::tensorflow_lite::get_sparsity(
    const flatbuffers::Vector<int32_t>* tf_shape,
    const tflite::SparsityParameters* tf_sparsity,
    const ov::element::Type target_type,
    const uint8_t* buffer) {
    if (tf_shape == nullptr)
        return {};
    if (tf_sparsity == nullptr)
        return {};
    auto sparsity = std::make_shared<ov::frontend::tensorflow_lite::SparsityInfo>();
    sparsity->set_shape({tf_shape->begin(), tf_shape->end()});
    sparsity->set_values(buffer);
    sparsity->set_target_type(target_type);
    if (tf_sparsity->traversal_order() != nullptr)
        sparsity->set_traversal_order({tf_sparsity->traversal_order()->begin(), tf_sparsity->traversal_order()->end()});
    if (tf_sparsity->block_map() != nullptr)
        sparsity->set_block_map({tf_sparsity->block_map()->begin(), tf_sparsity->block_map()->end()});
    if (tf_sparsity->dim_metadata() != nullptr) {
        std::vector<uint16_t> dim_format = {};
        std::vector<ov::frontend::tensorflow_lite::SparsityInfo::SparsityDataDesc> data_desc = {};
        dim_format.reserve(tf_sparsity->dim_metadata()->size());
        data_desc.reserve(tf_sparsity->dim_metadata()->size());
        for (auto it = tf_sparsity->dim_metadata()->begin(); it != tf_sparsity->dim_metadata()->end(); ++it) {
            dim_format.push_back(it->format());
            data_desc.push_back({static_cast<uint8_t>(it->array_segments_type()),
                                 it->array_segments(),
                                 static_cast<uint8_t>(it->array_indices_type()),
                                 it->array_indices()});
        }
        sparsity->set_dim_format(dim_format);
        sparsity->set_data_desc(data_desc);
    }
    return sparsity;
}

ov::element::Type ov::frontend::tensorflow_lite::get_ov_type(const tflite::TensorType& tf_type) {
    static const std::map<tflite::TensorType, ov::element::Type> type_map{
        {tflite::TensorType_FLOAT32, element::f32},
        {tflite::TensorType_FLOAT16, element::f16},
        {tflite::TensorType_INT32, element::i32},
        {tflite::TensorType_UINT8, element::u8},
        {tflite::TensorType_INT64, element::i64},
        {tflite::TensorType_BOOL, element::boolean},
        {tflite::TensorType_INT16, element::i16},
        {tflite::TensorType_INT8, element::i8},
        {tflite::TensorType_FLOAT64, element::f64},
        {tflite::TensorType_UINT64, element::u64},
        {tflite::TensorType_UINT32, element::u32},
        {tflite::TensorType_UINT16, element::u16},
        {tflite::TensorType_INT4, element::i4},
        {tflite::TensorType_COMPLEX64, element::dynamic},
        {tflite::TensorType_STRING, element::dynamic},
        // TODO: support the following types
        //          {TensorType_STRING,         element::string},
        //          {TensorType_COMPLEX128,     element::complex128},
        //          {TensorType_RESOURCE,       element::resource},
        //          {TensorType_VARIANT,        element::variant},
    };
    auto it = type_map.find(tf_type);
    FRONT_END_GENERAL_CHECK(it != type_map.end(), "Unexpected type: ", tflite::EnumNameTensorType(tf_type));
    return it->second;
}

ov::PartialShape ov::frontend::tensorflow_lite::get_ov_shape(const flatbuffers::Vector<int32_t>* tf_shape,
                                                             const flatbuffers::Vector<int32_t>* tf_shape_sig) {
    if (tf_shape_sig) {
        std::vector<int64_t> signature_vec{tf_shape_sig->begin(), tf_shape_sig->end()};
        return ov::PartialShape{signature_vec};
    }
    if (!tf_shape)
        return {};
    return ov::Shape{tf_shape->begin(), tf_shape->end()};
}

void ov::frontend::tensorflow_lite::apply_quantization(ov::Output<ov::Node>& output, element::Type type) {
    auto rt_info = output.get_rt_info();
    auto input_type = output.get_element_type();
    if (!rt_info.count(QuantizationInfo::get_type_info_static())) {  // no quantization
        FRONT_END_GENERAL_CHECK(input_type.compatible(type),
                                "Inconsistent type inference: tflite ",
                                type,
                                ", ov ",
                                input_type);
        return;
    }
    auto quantization = rt_info[QuantizationInfo::get_type_info_static()].as<std::shared_ptr<QuantizationInfo>>();
    if (!quantization || quantization->is_disabled()) {
        FRONT_END_GENERAL_CHECK(input_type.compatible(type),
                                "Inconsistent type inference: tflite ",
                                type,
                                ", ov ",
                                input_type);
        return;
    }

    output = std::make_shared<TFLQuantize>(output, quantization, type);
    return;
}

void ov::frontend::tensorflow_lite::dequantize_inputs(OutputVector& deq_inputs) {
    for (auto& deq_input : deq_inputs) {
        auto input = deq_input.get_node_shared_ptr();
        if (!ov::is_type<ov::frontend::tensorflow_lite::TFLQuantize>(input))
            continue;
        deq_input = std::make_shared<opset10::Convert>(deq_input, element::f32);
    }
}

namespace ov {
namespace frontend {
namespace tensorflow_lite {
// namespace required by arm compiler to specify template
template <>
OutputVector get_indexed_outputs(const OutputVector& outputs) {
    return outputs;
};

template <>
OutputVector get_indexed_outputs(const frontend::NamedOutputVector& outputs) {
    return indexed_from_named(outputs);
};
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
