// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/divide.hpp"

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
    bool m_pythondiv = false;

    // Check if the element type is a integer
    if (x.get_element_type().is_integral_number()) {
        // prepare auxiliary zero constants of the same type as the inputs
        auto zero = make_shared<v0::Constant>(element::i32, Shape{}, 0)->output(0);
        zero = make_shared<v1::ConvertLike>(zero, x);

        auto mod_result = make_shared<v1::Mod>(x, y);
        auto mod_non_zero = make_shared<v1::NotEqual>(mod_result, zero);

        auto divide = make_shared<v1::Divide>(x, y, m_pythondiv);
        auto div_is_neg = make_shared<v1::Less>(divide, zero);

        // generate a boolean mask of elements for non-zero values of Mod result and negative values of Divide result
        auto mask = make_shared<v1::LogicalAnd>(mod_non_zero, div_is_neg);

        // add 1 to the divide result
        auto one = make_shared<v0::Constant>(element::i32, Shape{}, 1)->output(0);
        one = make_shared<v1::ConvertLike>(one, x);
        auto add_result = make_shared<v1::Add>(divide, one);

        // select elements based on the mask
        auto div = make_shared<v1::Select>(mask, add_result, divide);
        set_node_name(node.get_name(), div);
        return div->outputs();
    } else {
        // for other cases
        auto div = make_shared<v1::Divide>(x, y, m_pythondiv);
        set_node_name(node.get_name(), div);
        return div->outputs();
    }
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
