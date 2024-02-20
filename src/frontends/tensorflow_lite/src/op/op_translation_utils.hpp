// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "common_op_table.hpp"
#include "decoder_map.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/tensorflow_lite/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

std::shared_ptr<DecoderFlatBuffer> get_decoder(const ov::frontend::tensorflow_lite::NodeContext& node);
void set_output_names(const ov::frontend::tensorflow_lite::NodeContext& node, OutputVector& outputs);
void del_output_names(OutputVector& outputs);

// convolutions
template <class T>
std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap> get_conv_decoder_map(
    const std::string& new_type_name,
    const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = get_decoder(node);
    const std::map<std::string, ov::Any> attrs{
        {"strides",
         std::vector<int64_t>{1, decoder->get_attribute(&T::stride_h), decoder->get_attribute(&T::stride_w), 1}},
        {"padding", std::string(EnumNamePadding(decoder->get_attribute(&T::padding)))},
        {"dilations",
         std::vector<int64_t>{1,
                              decoder->get_attribute(&T::dilation_h_factor),
                              decoder->get_attribute(&T::dilation_w_factor),
                              1}},
        {"data_format", "NHWC"},
        {"activation", EnumNameActivationFunctionType(decoder->get_attribute(&T::fused_activation_function))},
    };
    return std::make_shared<ov::frontend::tensorflow_lite::DecoderMap>(node.get_decoder(), attrs, new_type_name, true);
}
void get_conv(ov::OutputVector& output,
              const ov::frontend::NodeContext& node,
              const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder,
              ov::OutputVector (*converter)(const ov::frontend::NodeContext&),
              ov::AxisVector transpose_axes = {1, 2, 3, 0});
void get_bias(ov::OutputVector& output,
              const ov::frontend::NodeContext& node,
              const std::shared_ptr<ov::frontend::DecoderBase>& decoder);
void get_activation(ov::OutputVector& output,
                    const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder);
void get_activation(ov::OutputVector& output,
                    const ov::frontend::tensorflow_lite::NodeContext& node,
                    const std::string& activation);

std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap> get_pool_decoder_map(
    const std::string& new_type_name,
    const ov::frontend::tensorflow_lite::NodeContext& node);
void get_pool(ov::OutputVector& output,
              const ov::frontend::NodeContext& node,
              const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder,
              ov::OutputVector (*converter)(const ov::frontend::NodeContext&));

template <typename OV_TYPE, typename TF_TYPE>
OutputVector translate_binary_op_with_activation(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto inputs = node.get_inputs();
    ov::frontend::tensorflow_lite::dequantize_inputs(inputs);
    auto context = ov::frontend::tensorflow_lite::NodeContext(node.get_decoder(), inputs);
    auto output = ov::frontend::tensorflow::op::translate_binary_op<OV_TYPE>(context);
    output[0].get_node()->set_friendly_name("");
    output[0].set_names({});
    const auto& decoder = get_decoder(context);
    get_activation(output,
                   context,
                   EnumNameActivationFunctionType(decoder->get_attribute(&TF_TYPE::fused_activation_function)));
    output[0].get_node()->set_friendly_name(node.get_name());
    return output;
}

template OutputVector translate_binary_op_with_activation<opset10::Add, tflite::AddOptions>(
    const ov::frontend::tensorflow_lite::NodeContext& node);
template OutputVector translate_binary_op_with_activation<opset10::Subtract, tflite::SubOptions>(
    const ov::frontend::tensorflow_lite::NodeContext& node);
template OutputVector translate_binary_op_with_activation<opset10::Multiply, tflite::MulOptions>(
    const ov::frontend::tensorflow_lite::NodeContext& node);
template OutputVector translate_binary_op_with_activation<opset10::Divide, tflite::DivOptions>(
    const ov::frontend::tensorflow_lite::NodeContext& node);

OutputVector attribute_helper(const ov::frontend::tensorflow_lite::NodeContext& node,
                              const std::map<std::string, ov::Any>& attrs,
                              ov::frontend::CreatorFunction converter,
                              std::string new_op_type = "",
                              bool empty_name = false,
                              ov::OutputVector inputs = {});

OutputVector attribute_helper(const ov::frontend::tensorflow_lite::NodeContext& node,
                              const std::map<std::string, ov::Any>& attrs,
                              ov::frontend::CreatorFunctionNamedAndIndexed converter,
                              std::string new_op_type = "",
                              bool empty_name = false,
                              ov::OutputVector inputs = {});

void transform_reduce_name(std::string& op_type);

template <typename OV_TYPE>
OutputVector translate_reduce_op(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto inputs = node.get_inputs();
    ov::frontend::tensorflow_lite::dequantize_inputs(inputs);
    auto context = ov::frontend::tensorflow_lite::NodeContext(node.get_decoder(), inputs);
    const auto& original_decoder = std::dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(original_decoder != nullptr,
                            "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");
    auto op_type = original_decoder->get_op_type();
    transform_reduce_name(op_type);
    const std::map<std::string, ov::Any> attrs{
        {"keep_dims", original_decoder->get_attribute(&tflite::ReducerOptions::keep_dims)}};
    return attribute_helper(context, attrs, ov::frontend::tensorflow::op::translate_direct_reduce_op<OV_TYPE>, op_type);
}

template OutputVector translate_reduce_op<opset8::ReduceMean>(const ov::frontend::tensorflow_lite::NodeContext& node);
template OutputVector translate_reduce_op<opset8::ReduceLogicalAnd>(
    const ov::frontend::tensorflow_lite::NodeContext& node);
template OutputVector translate_reduce_op<opset8::ReduceLogicalOr>(
    const ov::frontend::tensorflow_lite::NodeContext& node);
template OutputVector translate_reduce_op<opset8::ReduceMax>(const ov::frontend::tensorflow_lite::NodeContext& node);
template OutputVector translate_reduce_op<opset8::ReduceMin>(const ov::frontend::tensorflow_lite::NodeContext& node);
template OutputVector translate_reduce_op<opset8::ReduceProd>(const ov::frontend::tensorflow_lite::NodeContext& node);
template OutputVector translate_reduce_op<opset8::ReduceSum>(const ov::frontend::tensorflow_lite::NodeContext& node);

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
