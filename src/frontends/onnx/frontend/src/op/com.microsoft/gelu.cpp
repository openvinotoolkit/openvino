// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/com.microsoft/gelu.hpp"
#include "default_opset.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector gelu(const Node &node) {
  auto nodes = node.get_ng_inputs();
  NGRAPH_CHECK(nodes.size() == 1,
               "Gelu takes 1 input. Provided:" + std::to_string(nodes.size()));
  return {std::make_shared<default_opset::Gelu>(nodes.at(0))};
}
} // namespace set_1
} // namespace op
} // namespace onnx_import
} // namespace ngraph
