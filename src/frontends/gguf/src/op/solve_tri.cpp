// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/model.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// ggml_solve_tri(a, b): solve A X = B for X with A lower triangular and a non-unit diagonal, the only
// variant ggml accepts (left, lower, !uni). OV layout: A [B2, B1, n, n], B and X [B2, B1, n, k].
// Forward substitution row by row: x_i = (b_i - A[i, :] X) / A[i, i]. Rows not yet solved are still
// zero in X and A is zero above the diagonal, so the full-row product is the partial sum.
OutputVector translate_solve_tri(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    // ggml's {left, lower, unit} flags, when the decoder supplies them: only the left, lower, non-unit system
    // is translated.
    const auto params = context.get_attribute<std::vector<int32_t>>("solve_tri_params", {});
    if (!params.empty()) {
        FRONT_END_OP_CONVERSION_CHECK(params.size() == 3, "SOLVE_TRI requires left, lower and unit flags");
        FRONT_END_OP_CONVERSION_CHECK(params[0] != 0 && params[1] != 0 && params[2] == 0,
                                      "SOLVE_TRI supports only left, lower, non-unit triangular systems");
    }
    auto a = context.get_input(0);
    auto b = context.get_input(1);
    const auto type = b.get_element_type();
    if (a.get_element_type() != type) {
        a = std::make_shared<ov::op::v0::Convert>(a, type);
    }
    FRONT_END_OP_CONVERSION_CHECK(a.get_partial_shape().rank().is_static() && a.get_partial_shape().rank().get_length() == 4,
                                  "SOLVE_TRI requires a rank-4 coefficient matrix");

    auto zero = ov::op::v0::Constant::create(type, {}, {0.0f});
    auto x_init = std::make_shared<ov::op::v3::Broadcast>(zero, std::make_shared<ov::op::v3::ShapeOf>(b, ov::element::i64));
    // One iteration per row; read at run time so a dynamic chunk length also works.
    auto rows = get_dimensions(a, {2});

    auto body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    auto body_x = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape::dynamic(4));
    auto body_a = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape::dynamic(4));
    auto body_b = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape::dynamic(4));
    auto row_axis = ov::op::v0::Constant::create(ov::element::i64, {}, {2});
    auto col_axis = ov::op::v0::Constant::create(ov::element::i64, {}, {3});

    auto b_i = std::make_shared<ov::op::v8::Gather>(body_b, body_iter, row_axis);       // [B2, B1, 1, k]
    auto a_row = std::make_shared<ov::op::v8::Gather>(body_a, body_iter, row_axis);     // [B2, B1, 1, n]
    auto partial = std::make_shared<ov::op::v0::MatMul>(a_row, body_x, false, false);  // [B2, B1, 1, k]
    auto diag = std::make_shared<ov::op::v8::Gather>(a_row, body_iter, col_axis);       // [B2, B1, 1, 1]
    auto x_i = std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(b_i, partial), diag);
    auto x_next = std::make_shared<ov::op::v3::ScatterUpdate>(body_x, body_iter, x_i, row_axis);
    auto keep_going = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {true});
    auto body = std::make_shared<ov::Model>(ov::OutputVector{keep_going, x_next},
                                            ov::ParameterVector{body_iter, body_x, body_a, body_b});

    auto loop = std::make_shared<ov::op::v5::Loop>(rows, ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {true}));
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});
    loop->set_merged_input(body_x, x_init, x_next);
    loop->set_invariant_input(body_a, a);
    loop->set_invariant_input(body_b, b);
    auto res = loop->get_iter_value(x_next, -1);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
