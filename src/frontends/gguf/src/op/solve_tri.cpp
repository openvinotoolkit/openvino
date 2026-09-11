// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_solve_tri(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    const auto params = context.get_attribute<std::vector<int32_t>>("solve_tri_params");
    FRONT_END_OP_CONVERSION_CHECK(params.size() == 3, "SOLVE_TRI requires left, lower and unit flags");
    FRONT_END_OP_CONVERSION_CHECK(params[0] != 0 && params[1] != 0 && params[2] == 0,
                                  "SOLVE_TRI supports only left, lower, non-unit triangular systems");

    auto a = context.get_input(0);
    auto b = context.get_input(1);
    const auto a_shape = context.get_input_shape(0).to_shape();
    FRONT_END_OP_CONVERSION_CHECK(a_shape.size() == 4 && a_shape[2] == a_shape[3],
                                  "SOLVE_TRI requires square matrices");
    const int64_t n = static_cast<int64_t>(a_shape[2]);

    auto b_shape = std::make_shared<ov::op::v3::ShapeOf>(b, ov::element::i64);
    auto zero = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    auto x_init = std::make_shared<ov::op::v3::Broadcast>(zero, b_shape);

    auto iteration = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    auto body_x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    auto body_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    auto body_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));

    auto axis2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {int64_t{2}});
    auto axis3 = ov::op::v0::Constant::create(ov::element::i64, {1}, {int64_t{3}});
    auto axis2_scalar = ov::op::v0::Constant::create(ov::element::i64, {}, {int64_t{2}});
    auto b_i = std::make_shared<ov::op::v8::Gather>(body_b, iteration, axis2);
    auto a_row_i = std::make_shared<ov::op::v8::Gather>(body_a, iteration, axis2);
    auto partial = std::make_shared<ov::op::v0::MatMul>(a_row_i, body_x, false, false);
    auto diagonal = std::make_shared<ov::op::v8::Gather>(a_row_i, iteration, axis3);
    auto x_i = std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(b_i, partial), diagonal);
    auto x_updated = std::make_shared<ov::op::v3::ScatterUpdate>(body_x, iteration, x_i, axis2_scalar);
    auto body_condition = ov::op::v0::Constant::create(ov::element::boolean, {1}, {true});

    auto body = std::make_shared<ov::Model>(ov::OutputVector{body_condition, x_updated},
                                            ov::ParameterVector{iteration, body_x, body_a, body_b});
    auto trip_count = ov::op::v0::Constant::create(ov::element::i64, {1}, {n});
    auto execution_condition = ov::op::v0::Constant::create(ov::element::boolean, {1}, {true});
    auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, execution_condition);
    loop->set_function(body);
    loop->set_special_body_ports({0, 0});
    loop->set_merged_input(body_x, x_init, x_updated);
    loop->set_invariant_input(body_a, a);
    loop->set_invariant_input(body_b, b);

    return rename_outputs_with_suffix({loop->get_iter_value(x_updated, -1)}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
