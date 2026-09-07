// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sigmoid.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_unary_gelu(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    // ggml GELU is the tanh approximation; v7::Gelu defaults to ERF.
    auto res = std::make_shared<ov::op::v7::Gelu>(input, ov::op::GeluApproximationMode::TANH);

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

OutputVector translate_unary_gelu_quick(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    // ggml_gelu_quick_f32: x * (1 / (1 + exp(-1.702 * x))) == x * sigmoid(1.702 * x).
    // A different approximation from GGML_UNARY_OP_GELU; the two are not interchangeable.
    auto coef = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.702f});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(input, coef);
    auto res = std::make_shared<ov::op::v1::Multiply>(input, std::make_shared<ov::op::v0::Sigmoid>(scaled));

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
