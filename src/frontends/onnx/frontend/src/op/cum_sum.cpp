// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cum_sum.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector cum_sum(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    auto data = inputs.at(0);
    bool exclusive = node.get_attribute_value<std::int64_t>("exclusive", 0);
    bool reverse = node.get_attribute_value<std::int64_t>("reverse", 0);
    ov::Output<ov::Node> axis;

    if (inputs.size() > 1) {
        // optional input, 0-D or 1-D tensor
        const auto& axis_shape = inputs.at(1).get_partial_shape();
        axis = axis_shape.is_dynamic() ? inputs.at(1) : ov::frontend::onnx::reshape::interpret_as_scalar(inputs.at(1));
    } else {
        axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});  // default
    }
    return ov::OutputVector{std::make_shared<v0::CumSum>(data, axis, exclusive, reverse)};
}

ONNX_OP("CumSum", OPSET_SINCE(1), ai_onnx::opset_1::cum_sum);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
