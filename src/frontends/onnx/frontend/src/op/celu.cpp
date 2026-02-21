// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector celu(const ov::frontend::onnx::Node& node) {
    auto alpha_node = node.get_attribute_as_constant<float>("alpha", 1.0f);
    auto zero = v0::Constant::create(alpha_node->get_element_type(), ov::Shape{}, {0.0f});
    auto x_celu = node.get_ov_inputs().at(0);

    auto x_max = std::make_shared<v1::Maximum>(x_celu, zero);

    auto divide_node = std::make_shared<v1::Divide>(x_celu, alpha_node);
    auto exp_node = std::make_shared<v0::Exp>(divide_node);
    auto exp_minus_one =
        std::make_shared<v1::Subtract>(exp_node,
                                       v0::Constant::create(exp_node->get_element_type(), ov::Shape{}, {1.0f}));
    auto celu_temp = std::make_shared<v1::Multiply>(alpha_node, exp_minus_one);
    auto min_node = std::make_shared<v1::Minimum>(celu_temp, zero);

    return {std::make_shared<v1::Add>(x_max, min_node)};
}
ONNX_OP("Celu", OPSET_SINCE(1), ai_onnx::opset_1::celu);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
