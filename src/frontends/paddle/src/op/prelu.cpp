// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs cross(const NodeContext &node) {
  auto x = node.get_input("X");
  auto weight = node.get_input("Alpha");

  auto data_format = node.get_attribute<std::string>("data_format", 'NCHW');

  auto node_prelu = std::make_shared<default_opset::PRelu>(x, weight);

  return node_prelu;
}

} // namespace op
} // namespace paddle
} // namespace frontend
} // namespace ov
