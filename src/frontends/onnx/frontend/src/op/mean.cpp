// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "utils/variadic.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector mean(const ov::frontend::onnx::Node& node) {
    auto sum = variadic::make_ng_variadic_op<v1::Add>(node).front();
    auto count = v0::Constant::create(sum.get_element_type(), ov::Shape{}, {node.get_ov_inputs().size()});

    return {std::make_shared<v1::Divide>(sum, count)};
}

ONNX_OP("Mean", OPSET_SINCE(1), ai_onnx::opset_1::mean);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
