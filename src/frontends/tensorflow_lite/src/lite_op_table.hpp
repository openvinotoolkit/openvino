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
OutputVector concatenation(const ov::frontend::tensorflow::NodeContext& node);
OutputVector reshape(const ov::frontend::tensorflow::NodeContext& node);
OutputVector pad(const ov::frontend::tensorflow::NodeContext& node);

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov