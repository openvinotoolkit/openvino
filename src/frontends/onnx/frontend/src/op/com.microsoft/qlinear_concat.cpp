// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/op_types.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

ov::OutputVector qlinear_concat(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    FRONT_END_OP_CONVERSION_CHECK(inputs.size() >= 5 && (inputs.size() - 2) % 3 == 0,
                                  "QLinearConcat: expected 2 + 3*N inputs (Y_scale, Y_zero_point, and N groups of "
                                  "(X, X_scale, X_zero_point)), got: ",
                                  inputs.size());

    auto Y_scale = inputs[0];
    auto Y_zero_point =
        ov::op::util::is_null(inputs[1]) ? v0::Constant::create(Y_scale.get_element_type(), {}, {0}) : inputs[1];

    ov::pass::NodeRegistry reg;
    ov::OutputVector dequantized_inputs;
    for (size_t i = 2; i < inputs.size(); i += 3) {
        auto X = inputs[i];
        auto X_scale = inputs[i + 1];
        auto X_zero_point =
            ov::op::util::is_null(inputs[i + 2]) ? v0::Constant::create(X.get_element_type(), {}, {0}) : inputs[i + 2];

        dequantized_inputs.push_back(ov::decomposition::low_precision_dequantize(reg, X, X_scale, X_zero_point));
    }

    auto axis = node.get_attribute_value<int64_t>("axis");
    auto concatenated = std::make_shared<v0::Concat>(dequantized_inputs, axis);

    auto requantized = std::make_shared<v1::Divide>(concatenated, Y_scale);
    auto Y_zero_point_float = std::make_shared<v0::Convert>(Y_zero_point, Y_scale.get_element_type());
    auto Y_float = std::make_shared<v1::Add>(requantized, Y_zero_point_float);
    auto Y = std::make_shared<v0::Convert>(Y_float, inputs[2].get_element_type());

    return {Y->output(0)};
}

ONNX_OP("QLinearConcat", OPSET_SINCE(1), com_microsoft::opset_1::qlinear_concat, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
