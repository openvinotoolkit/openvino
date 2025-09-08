// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roll.hpp"

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
ov::OutputVector translate_roll_op(const NodeContext& node) {
    default_op_checks(node, 3, {"Roll"}, true);
    auto input = node.get_input(0);
    auto shift = node.get_input(1);
    auto axis = node.get_input(2);

    // check if complex type mark is set
    // if yes, sinking it through Roll operation further
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    element::Type complex_part_type = element::dynamic;
    if (complex_type_mark) {
        input = complex_type_mark->get_data();
        complex_part_type = complex_type_mark->get_complex_part_type();

        // axes can be negative so we need to adjust them
        // since the last dimension for complex type case is auxiliary (not real)
        axis = make_shared<v0::Convert>(axis, element::i64);
        auto input_rank = compute_subgraph_scalar_rank(input, element::i64, true);
        auto const_one = make_shared<v0::Constant>(element::i64, Shape{}, 1);
        auto input_rank_minus_one = make_shared<v1::Subtract>(input_rank, const_one)->output(0);

        // adjust axis to make them non-negative
        axis = make_shared<v1::FloorMod>(axis, input_rank_minus_one);
    }

    auto roll = std::make_shared<v7::Roll>(input, shift, axis)->output(0);
    set_node_name(node.get_name(), roll.get_node_shared_ptr());

    if (complex_type_mark) {
        roll = make_shared<ComplexTypeMark>(roll, complex_part_type)->output(0);
    }

    return {roll};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
