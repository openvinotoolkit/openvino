// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/cum_sum.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/add.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_cumsum_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Cumsum"});
    auto x = node.get_input(0);
    auto axis = node.get_input(1);
    auto exclusive = node.get_attribute<bool>("exclusive", false);
    auto reverse = node.get_attribute<bool>("reverse", false);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    if (complex_type_mark) {
        x = complex_type_mark->input_value(0);
        auto zero = create_same_type_const_scalar<int32_t>(x, 0);
        
        auto is_zero = make_shared<v1::Equal>(axis, zero);
        auto greater_than_zero = make_shared<v1::Greater>(axis, zero);

        auto logical_or = make_shared<v1::LogicalOr>(is_zero, greater_than_zero);

        auto const_one = make_shared<v0::Constant>(element::i32, Shape{}, 1);
        auto const_minus_one = make_shared<v0::Constant>(element::i32, Shape{}, -1);

        auto axis_update = make_shared<v1::Select>(logical_or, const_one, const_minus_one);

        auto new_axis = make_shared<v1::Add>(axis, axis_update);    
    }

    auto cum_sum = make_shared<v0::CumSum>(x, axis, exclusive, reverse);
    set_node_name(node.get_name(), cum_sum);
    return cum_sum->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
