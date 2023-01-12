// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "decoder_map.hpp"
#include "op_table.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino_conversions.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {
using CreatorFunction = std::function<OutputVector(const ov::frontend::tensorflow::NodeContext&)>;

std::map<std::string, CreatorFunction> get_supported_ops();

OutputVector conv2d(const ov::frontend::tensorflow::NodeContext& node);
OutputVector depthwise_conv2d(const ov::frontend::tensorflow::NodeContext& node);
OutputVector fully_connected(const ov::frontend::tensorflow::NodeContext& node);
OutputVector max_pool_2d(const ov::frontend::tensorflow::NodeContext& node);
OutputVector avg_pool_2d(const ov::frontend::tensorflow::NodeContext& node);
OutputVector concatenation(const ov::frontend::tensorflow::NodeContext& node);
OutputVector reshape(const ov::frontend::tensorflow::NodeContext& node);
OutputVector pack(const ov::frontend::tensorflow::NodeContext& node);
OutputVector softmax(const ov::frontend::tensorflow::NodeContext& node);
OutputVector resize_nearest_neightbor(const ov::frontend::tensorflow::NodeContext& node);
OutputVector resize_bilinear(const ov::frontend::tensorflow::NodeContext& node);
OutputVector squeeze(const ov::frontend::tensorflow::NodeContext& node);
OutputVector split(const ov::frontend::tensorflow::NodeContext& node);
OutputVector shape(const ov::frontend::tensorflow::NodeContext& node);
OutputVector range(const ov::frontend::tensorflow::NodeContext& node);
OutputVector strided_slice(const ov::frontend::tensorflow::NodeContext& node);
OutputVector gather(const ov::frontend::tensorflow::NodeContext& node);

template <typename OV_TYPE, typename TF_TYPE>
OutputVector translate_binary_op_with_activation(const ov::frontend::tensorflow::NodeContext& node);

template <typename OV_TYPE>
OutputVector translate_reduce_op(const ov::frontend::tensorflow::NodeContext& node);
}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
