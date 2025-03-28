// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_take_op(const NodeContext& context) {
    num_inputs_check(context, 2, 2, true);
    auto input = context.get_input(0);
    auto indices = context.get_input(1);
    auto input_shape = input.get_partial_shape();
    if (input_shape.rank().is_static() && input_shape.rank().get_length() == 0) {
        FRONT_END_OP_CONVERSION_CHECK(false, "input tensor MUST be non-scalar");
    }
    auto new_shape = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1}));
    input = context.mark_node(std::make_shared<v1::Reshape>(input, new_shape, false));
    indices = context.mark_node(std::make_shared<v0::Convert>(indices, element::i64));
    auto input_size = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i64));
    indices = normalize_axis(context, indices, input_size);
    auto indices_const = ov::util::get_constant_from_source(indices);
    // check for out of bounds indices
    if (indices_const && input_shape.is_static()) {
        int64_t flat_size = input_shape[0].get_length();
        auto indices_vals = indices_const->cast_vector<int64_t>();
        for (const auto& idx : indices_vals) {
            FRONT_END_OP_CONVERSION_CHECK(idx >= 0 && idx < flat_size,
                                          "Index out of bounds: ",
                                          idx,
                                          ". Valid range: [0, ",
                                          flat_size,
                                          ").");
        }
    }
    auto axis_constant = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto gather = context.mark_node(std::make_shared<v8::Gather>(input, indices, axis_constant));

    return {gather};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
