// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_mulmat(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    ov::Output<ov::Node> B = context.get_input(0);
    ov::Output<ov::Node> A = context.get_input(1);
    A = apply_hadamard(context, A);

    // GGML MUL_MAT produces F32, including when both operands are F16.
    if (A.get_element_type() != ov::element::f32) {
        A = std::make_shared<ov::op::v0::Convert>(A, ov::element::f32);
    }
    if (B.get_element_type() != ov::element::f32) {
        B = std::make_shared<ov::op::v0::Convert>(B, ov::element::f32);
    }

    const auto B_shape = B.get_partial_shape();
    const auto A_shape = A.get_partial_shape();
    for (const auto& shape : {A_shape, B_shape}) {
        FRONT_END_OP_CONVERSION_CHECK(shape.rank().is_static() && shape.rank().get_length() >= 2 &&
                                          shape.rank().get_length() <= 4,
                                      "MUL_MAT ", context.get_name(),
                                      " requires rank-2 to rank-4 operands; activation shape: ",
                                      A_shape, ", weight shape: ", B_shape);
    }

    // Missing and singleton batch axes use MatMul's ordinary broadcasting. GGML additionally
    // repeats groups on non-singleton batch axes (e.g. KV heads [0,0,1,1], not [0,1,0,1]).
    for (int trailing_axis : {3, 4}) {
        const int a_axis = static_cast<int>(A_shape.rank().get_length()) - trailing_axis;
        const int b_axis = static_cast<int>(B_shape.rank().get_length()) - trailing_axis;
        if (a_axis < 0 || b_axis < 0) {
            continue;
        }
        const auto& a_dim = A_shape[a_axis];
        const auto& b_dim = B_shape[b_axis];
        if (a_dim == 1 || b_dim == 1 || (a_dim.is_static() && b_dim.is_static() && a_dim == b_dim)) {
            continue;
        }
        if (a_dim.is_static() && b_dim.is_static()) {
            const auto small = std::min(a_dim.get_length(), b_dim.get_length());
            const auto large = std::max(a_dim.get_length(), b_dim.get_length());
            FRONT_END_OP_CONVERSION_CHECK(small > 0 && large % small == 0,
                                          "MUL_MAT batch dimensions must be integer multiples");
        }
        auto a_count = get_dimensions(A, {a_axis});
        auto b_count = get_dimensions(B, {b_axis});
        auto target = std::make_shared<ov::op::v1::Maximum>(a_count, b_count);
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
        auto scalar_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto end = std::make_shared<ov::op::v0::Squeeze>(target, scalar_axis);
        auto positions = std::make_shared<ov::op::v4::Range>(zero, end, one, ov::element::i64);
        const auto repeat_groups = [&](const Output<Node>& value, const Output<Node>& count, int axis) {
            auto factor = std::make_shared<ov::op::v1::Divide>(target, count);
            auto indices = std::make_shared<ov::op::v1::Divide>(positions, factor);
            return std::make_shared<ov::op::v8::Gather>(
                value, indices, ov::op::v0::Constant::create(ov::element::i64, {}, {axis}));
        };
        if (!a_dim.is_static() || !b_dim.is_static() || a_dim.get_length() < b_dim.get_length()) {
            A = repeat_groups(A, a_count, a_axis);
        }
        if (!a_dim.is_static() || !b_dim.is_static() || b_dim.get_length() < a_dim.get_length()) {
            B = repeat_groups(B, b_count, b_axis);
        }
    }

    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, false, true);

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
