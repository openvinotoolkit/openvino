// Copyright (C) 2018-2023 Intel Corporation
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
#include "openvino_conversions.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {
using CreatorFunction = std::function<OutputVector(const ov::frontend::tensorflow_lite::NodeContext&)>;

std::map<std::string, CreatorFunction> get_supported_ops();

OutputVector batch_matmul(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector cast(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector conv2d(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector depthwise_conv2d(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector fully_connected(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector max_pool_2d(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector avg_pool_2d(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector concatenation(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector reshape(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector pack(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector softmax(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector resize_nearest_neightbor(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector resize_bilinear(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector squeeze(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector split(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector shape(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector range(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector strided_slice(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector gather(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector space_to_depth(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector depth_to_space(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector leaky_relu(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector mirror_pad(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector one_hot(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector reverse_sequence(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector unique(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector unpack(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector l2_normalization(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector arg_min(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector arg_max(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector transpose_conv(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector complex_abs(const ov::frontend::tensorflow_lite::NodeContext& node);
OutputVector rfft2d(const ov::frontend::tensorflow_lite::NodeContext& node);

template <typename OV_TYPE, typename TF_TYPE>
OutputVector translate_binary_op_with_activation(const ov::frontend::tensorflow_lite::NodeContext& node);

template <typename OV_TYPE>
OutputVector translate_reduce_op(const ov::frontend::tensorflow_lite::NodeContext& node);
}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
