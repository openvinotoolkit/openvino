// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/multiply.hpp"
#include "utils/common.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector celu(const ov::frontend::onnx::Node& node) {
    auto alpha_node = node.get_attribute_as_constant<float>("alpha", 1.0f);
    auto x_celu = node.get_ov_inputs().at(0);

    auto divide_node = std::make_shared<v1::Divide>(x_celu, alpha_node);
    auto elu_node = std::make_shared<v0::Elu>(divide_node, 1.0);

    return {std::make_shared<v1::Multiply>(alpha_node, elu_node)};
}
ONNX_OP("Celu", OPSET_SINCE(1), ai_onnx::opset_1::celu);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
