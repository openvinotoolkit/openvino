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

  auto data_format = node.get_attribute<std::string>("data_format", "NCHW");

  return node.default_single_output_mapping(
      {std::make_shared<ov::opset6::PRelu>(x, weight)}, {"Out"});
}

} // namespace op
} // namespace paddle
} // namespace frontend
} // namespace ov
