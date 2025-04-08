// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "common_op_table.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/tensorflow_lite/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

void set_output_names(const ov::frontend::tensorflow_lite::NodeContext& node, OutputVector& outputs);

void del_output_names(OutputVector& outputs);

void get_conv(ov::OutputVector& output,
              const ov::frontend::NodeContext& node,
              const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase>& decoder,
              ov::OutputVector (*converter)(const ov::frontend::NodeContext&),
              ov::AxisVector transpose_axes = {1, 2, 3, 0});

void get_bias(ov::OutputVector& output,
              const ov::frontend::NodeContext& node,
              const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase>& decoder);

void get_activation(ov::OutputVector& output,
                    const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase>& decoder);

void get_activation(ov::OutputVector& output,
                    const ov::frontend::tensorflow_lite::NodeContext& node,
                    const std::string& activation);

void get_pool(ov::OutputVector& output,
              const ov::frontend::NodeContext& node,
              const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase>& decoder,
              ov::OutputVector (*converter)(const ov::frontend::NodeContext&));

template <typename OV_TYPE>
OutputVector translate_binary_op_with_activation(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto fused_activation_function = node.get_attribute<std::string>("fused_activation_function");
    auto inputs = node.get_inputs();
    ov::frontend::tensorflow_lite::dequantize_inputs(inputs);
    auto context = ov::frontend::tensorflow_lite::NodeContext(node.get_decoder(), inputs);
    auto output = ov::frontend::tensorflow::op::translate_binary_op<OV_TYPE>(context);
    output[0].get_node()->set_friendly_name("");
    output[0].set_names({});
    get_activation(output, context, fused_activation_function);
    output[0].get_node()->set_friendly_name(node.get_name());
    return output;
}

template OutputVector translate_binary_op_with_activation<opset10::Add>(
    const ov::frontend::tensorflow_lite::NodeContext& node);
template OutputVector translate_binary_op_with_activation<opset10::Subtract>(
    const ov::frontend::tensorflow_lite::NodeContext& node);
template OutputVector translate_binary_op_with_activation<opset10::Multiply>(
    const ov::frontend::tensorflow_lite::NodeContext& node);
template OutputVector translate_binary_op_with_activation<opset10::Divide>(
    const ov::frontend::tensorflow_lite::NodeContext& node);

template <typename OV_TYPE>
OutputVector translate_reduce_op(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto inputs = node.get_inputs();
    ov::frontend::tensorflow_lite::dequantize_inputs(inputs);
    auto context = ov::frontend::tensorflow_lite::NodeContext(node.get_decoder(), inputs);
    auto outputs = ov::frontend::tensorflow::op::translate_direct_reduce_op<OV_TYPE>(context);
    del_output_names(outputs);
    return outputs;
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
