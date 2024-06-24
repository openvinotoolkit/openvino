// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hard_sigmoid.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector hard_sigmoid(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);

    const auto alpha =
        v0::Constant::create<double>(data.get_element_type(),
                                     ov::Shape{},
                                     std::vector<double>{node.get_attribute_value<double>("alpha", 0.2)});

    const auto beta = v0::Constant::create<double>(data.get_element_type(),
                                                   ov::Shape{},
                                                   std::vector<double>{node.get_attribute_value<double>("beta", 0.5)});

    return {std::make_shared<v0::HardSigmoid>(data, alpha, beta)};
}

ONNX_OP("HardSigmoid", OPSET_SINCE(1), ai_onnx::opset_1::hard_sigmoid);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
