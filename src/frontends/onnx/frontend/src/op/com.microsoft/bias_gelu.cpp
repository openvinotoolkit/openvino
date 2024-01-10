// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/com.microsoft/bias_gelu.hpp"

#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/gelu.hpp"

using namespace ov::op;

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector bias_gelu(const Node& node) {
    auto nodes = node.get_ng_inputs();
    FRONT_END_GENERAL_CHECK(nodes.size() == 2, "BiasGelu takes 2 inputs. Provided " + std::to_string(nodes.size()));
    return {std::make_shared<v7::Gelu>(std::make_shared<v1::Add>(nodes.at(0), nodes.at(1)))};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
