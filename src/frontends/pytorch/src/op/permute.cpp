// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_permute(const NodeContext& context) {
    num_inputs_check(context, 2, 2, true);
    auto data = context.get_input(0);
    auto order = get_input_concat_if_list(context, 1);

    Output<Node> rank;
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(data.get_node_shared_ptr());
    if (complex_type_mark) {
        data = complex_type_mark->get_data();
        rank = std::get<1>(get_shape_rank(context, data));
        auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
        rank = context.mark_node(std::make_shared<v1::Subtract>(rank, const_1));
    } else {
        rank = std::get<1>(get_shape_rank(context, data));
    }

    auto rank_converted = context.mark_node(std::make_shared<v1::ConvertLike>(rank, order));
    auto order_normalized = normalize_axis(context, order, rank_converted);

    if (complex_type_mark) {
        auto to_concat = OutputVector{order_normalized, rank_converted};
        order_normalized = context.mark_node(std::make_shared<v0::Concat>(to_concat, 0));
    }

    if (const auto order_const = ov::util::get_constant_from_source(order_normalized)) {
        order_normalized = order_const;
    }
    auto permute = context.mark_node(std::make_shared<v1::Transpose>(data, order_normalized));
    if (complex_type_mark) {
        const auto& complex_dtype = complex_type_mark->get_complex_part_type();
        permute = context.mark_node(std::make_shared<ComplexTypeMark>(permute, complex_dtype));
    }
    return {permute};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
