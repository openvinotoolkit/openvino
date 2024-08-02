// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/unsqueeze.hpp"

#include "exceptions.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector unsqueeze(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    auto axes_node = node.get_attribute_as_constant<std::vector<std::int64_t>>("axes", {});
    return {std::make_shared<v0::Unsqueeze>(data, axes_node)};
}

}  // namespace set_1

namespace set_13 {
ov::OutputVector unsqueeze(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    return {std::make_shared<v0::Unsqueeze>(inputs.at(0), inputs.at(1))};
}

}  // namespace set_13
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
