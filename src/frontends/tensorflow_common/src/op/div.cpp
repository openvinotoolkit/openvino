// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/select.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_div_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Div"});
    auto x = node.get_input(0);
    auto y = node.get_input(1);
    // Check if the element type is a signed integer
    if (x.get_element_type().is_integral_number() && x.get_element_type().is_signed()) {
        // prepare auxiliary zero constants of the same type as the inputs
        auto const_zero = create_same_type_const_scalar<int32_t>(x, 0);

        // compute the modulus of x and y
        auto mod_result = make_shared<v1::Mod>(x, y);
        // compute a mask to get positions of non-zero values of mod result
        auto mod_non_zero = make_shared<v1::NotEqual>(mod_result, const_zero);

        // compute the division of x and y
        auto divide = make_shared<v1::Divide>(x, y);
        // compute a mask to get positions of negative values of division result
        auto div_is_neg = make_shared<v1::Less>(divide, const_zero);

        // compute a boolean mask of elements for non-zero values of Mod result and negative values of Divide result
        auto mask = make_shared<v1::LogicalAnd>(mod_non_zero, div_is_neg);

        // prepare auxiliary one constants of the same type as the inputs
        auto const_one = create_same_type_const_scalar<int32_t>(x, 1);
        // add 1 to the divide result
        auto add_result = make_shared<v1::Add>(divide, const_one);

        // select division results based on the mask
        // - perform floor division for non-negative values.
        // - round negative values to the nearest zero.
        auto div = make_shared<v1::Select>(mask, add_result, divide);
        set_node_name(node.get_name(), div);
        return div->outputs();
    } else {
        // for other cases (non-signed-integer types)
        // compute regular division of x and y
        auto div = make_shared<v1::Divide>(x, y);
        set_node_name(node.get_name(), div);
        return div->outputs();
    }
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
