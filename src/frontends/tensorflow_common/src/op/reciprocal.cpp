// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/power.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_reciprocal_op(const NodeContext& node) {
    // computes element-wise 1/x, where x - input
    default_op_checks(node, 1, {"Reciprocal"}, true);
    auto x = node.get_input(0);
    auto minus_one_const = create_same_type_const_scalar<int32_t>(x, -1);

    auto complex_type_mark_x = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());

    if (complex_type_mark_x) {
        x = complex_type_mark_x->input_value(0);
    }

    auto reciprocal = make_shared<v1::Power>(complex_type_mark_x, minus_one_const);

    set_node_name(node.get_name(), reciprocal);
    if (complex_type_mark_x) {
        auto complex_reciprocal = make_shared<ComplexTypeMark>(reciprocal, complex_type_mark_x);
        return {complex_reciprocal->output(0)};
    }

    return {reciprocal};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
