// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/reshape.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_cumsum_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Cumsum"}, true);

    auto x = node.get_input(0);
    auto axis = node.get_input(1);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    if (complex_type_mark) {
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        x = complex_type_mark->input_value(0);

        OutputVector concat_inputs;
        concat_inputs.push_back(axis);
        concat_inputs.push_back(make_shared<v0::Constant>(axis.get_element_type(), Shape{1}, 2));

        auto concat = make_shared<v0::Concat>(concat_inputs, 0);
        auto reshape = make_shared<v1::Reshape>(x, concat, false);
        set_node_name(node.get_name(), reshape);
        auto complex_reshape = make_shared<ComplexTypeMark>(reshape, complex_part_type);
        return {complex_reshape->output(0)};
    }

    auto reshape = make_shared<v1::Reshape>(x, axis, false);
    set_node_name(node.get_name(), reshape);
    return {reshape};
    auto exclusive = node.get_attribute<bool>("exclusive", false);
    auto reverse = node.get_attribute<bool>("reverse", false);

    auto cum_sum = make_shared<v0::CumSum>(x, axis, exclusive, reverse);
    set_node_name(node.get_name(), cum_sum);
    return cum_sum->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
