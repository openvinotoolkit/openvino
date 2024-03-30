// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/com.microsoft/bias_gelu.hpp"

#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/gelu.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector bias_gelu(const ov::frontend::onnx::Node& node) {
    auto nodes = node.get_ov_inputs();
    FRONT_END_GENERAL_CHECK(nodes.size() == 2, "BiasGelu takes 2 inputs. Provided " + std::to_string(nodes.size()));
    return {std::make_shared<v7::Gelu>(std::make_shared<v1::Add>(nodes.at(0), nodes.at(1)))};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
