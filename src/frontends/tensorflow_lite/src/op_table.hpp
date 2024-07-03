// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "common_op_table.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/tensorflow_lite/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {
using CreatorFunction = std::function<OutputVector(const ov::frontend::tensorflow_lite::NodeContext&)>;

std::map<std::string, CreatorFunction> get_supported_ops();

#define TFL_OP_CONVERTER(op) OutputVector op(const ov::frontend::tensorflow_lite::NodeContext& node)

/// built-in ops
TFL_OP_CONVERTER(avg_pool_2d);
TFL_OP_CONVERTER(complex_abs);
TFL_OP_CONVERTER(concatenation);
TFL_OP_CONVERTER(conv2d);
TFL_OP_CONVERTER(depthwise_conv2d);
TFL_OP_CONVERTER(dequantize);
TFL_OP_CONVERTER(fully_connected);
TFL_OP_CONVERTER(gather);
TFL_OP_CONVERTER(l2_normalization);
TFL_OP_CONVERTER(max_pool_2d);
TFL_OP_CONVERTER(quantize);
TFL_OP_CONVERTER(reshape);
TFL_OP_CONVERTER(rfft2d);
TFL_OP_CONVERTER(softmax);
TFL_OP_CONVERTER(transpose_conv);
TFL_OP_CONVERTER(unique);
TFL_OP_CONVERTER(while_op);

template <typename OV_TYPE>
OutputVector translate_binary_op_with_activation(const ov::frontend::tensorflow_lite::NodeContext& node);

template <typename OV_TYPE>
OutputVector translate_reduce_op(const ov::frontend::tensorflow_lite::NodeContext& node);

template <typename OV_TYPE>
OutputVector translate_unary(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto inputs = node.get_inputs();
    ov::frontend::tensorflow_lite::dequantize_inputs(inputs);
    auto context = ov::frontend::tensorflow_lite::NodeContext(node.get_decoder(), inputs);
    return ov::frontend::tensorflow::op::translate_unary_op<OV_TYPE>(context);
}

template <typename OV_TYPE>
OutputVector translate_binary(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto inputs = node.get_inputs();
    ov::frontend::tensorflow_lite::dequantize_inputs(inputs);
    auto context = ov::frontend::tensorflow_lite::NodeContext(node.get_decoder(), inputs);
    return ov::frontend::tensorflow::op::translate_binary_op<OV_TYPE>(context);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
