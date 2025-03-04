// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_expand_dims_op(const NodeContext& node) {
    default_op_checks(node, 2, {"ExpandDims", "EXPAND_DIMS"}, true);
    auto input = node.get_input(0);
    auto axis = node.get_input(1);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());

    if (complex_type_mark) {
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        input = complex_type_mark->get_data();

        auto const_zero = create_same_type_const_scalar<int32_t>(axis, 0);

        auto is_axis_neg = make_shared<v1::Less>(axis, const_zero);

        auto const_one = create_same_type_const_scalar<int32_t>(axis, 1);
        auto axis_min_one = make_shared<v1::Subtract>(axis, const_one);

        auto new_axis = make_shared<v1::Select>(is_axis_neg, axis_min_one, axis);

        auto unsqueeze = make_shared<v0::Unsqueeze>(input, new_axis);

        set_node_name(node.get_name(), unsqueeze);
        auto complex_result = make_shared<ComplexTypeMark>(unsqueeze, complex_part_type);
        return {complex_result};
    }

    auto unsqueeze = make_shared<v0::Unsqueeze>(input, axis);
    set_node_name(node.get_name(), unsqueeze);
    return {unsqueeze};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
