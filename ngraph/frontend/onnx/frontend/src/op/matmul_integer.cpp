// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/matmul_integer.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "default_opset.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector matmul_integer(const Node& node) {
    const OutputVector& inputs = node.get_ng_inputs();

    const auto& A = inputs.at(0);
    const auto& B = inputs.at(1);
    const auto& A_zero_point =
        (inputs.size() > 2) ? inputs.at(2) : ngraph::op::Constant::create(ngraph::element::i32, {1}, {0});
    const auto& B_zero_point =
        (inputs.size() > 3) ? inputs.at(3) : ngraph::op::Constant::create(ngraph::element::i32, {1}, {0});

    const auto& converted_A = std::make_shared<default_opset::Convert>(A, element::i32);
    const auto& converted_B = std::make_shared<default_opset::Convert>(B, element::i32);

    const auto& converted_A_zero_point = std::make_shared<default_opset::Convert>(A_zero_point, element::i32);
    const auto& converted_B_zero_point = std::make_shared<default_opset::Convert>(B_zero_point, element::i32);

    const auto& one_node = ngraph::op::Constant::create(ngraph::element::i32, {1}, {1});
    const auto& reshaped_A_zero_point = std::make_shared<default_opset::Unsqueeze>(converted_A_zero_point, one_node);

    const auto& shifted_A = std::make_shared<default_opset::Subtract>(converted_A, reshaped_A_zero_point);
    const auto& shifted_B = std::make_shared<default_opset::Subtract>(converted_B, converted_B_zero_point);

    const auto& result = std::make_shared<default_opset::MatMul>(shifted_A, shifted_B);

    return {result};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
