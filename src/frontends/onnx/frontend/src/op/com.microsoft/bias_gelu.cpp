// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/com.microsoft/bias_gelu.hpp"

#include "default_opset.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector bias_gelu(const Node& node) {
    auto nodes = node.get_ng_inputs();
    NGRAPH_CHECK(nodes.size() == 2, "BiasGelu takes 2 inputs. Provided " + std::to_string(nodes.size()));
    return {std::make_shared<default_opset::Gelu>(std::make_shared<default_opset::Add>(nodes.at(0), nodes.at(1)))};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
