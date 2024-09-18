// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_cumsum_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Cumsum", "CUMSUM"}, true);

    auto x = node.get_input(0);
    auto axis = node.get_input(1);
    auto exclusive = node.get_attribute<bool>("exclusive", false);
    auto reverse = node.get_attribute<bool>("reverse", false);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    if (complex_type_mark) {
        x = complex_type_mark->input_value(0);
        auto zero = create_same_type_const_scalar<int32_t>(axis, 0);
        auto less_than_zero = make_shared<v1::Less>(axis, zero);
        auto const_one = create_same_type_const_scalar<int32_t>(axis, 1);

        auto axis_update = make_shared<v1::Select>(less_than_zero, const_one, zero);
        axis = make_shared<v1::Subtract>(axis, axis_update)->output(0);
    }

    auto cum_sum = make_shared<v0::CumSum>(x, axis, exclusive, reverse);
    set_node_name(node.get_name(), cum_sum);
    if (complex_type_mark) {
        auto cum_sum_complex = make_shared<ComplexTypeMark>(cum_sum, complex_type_mark->get_complex_part_type());
        return {cum_sum_complex};
    }
    return cum_sum->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
