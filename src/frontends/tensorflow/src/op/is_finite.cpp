// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_is_finite_op(const NodeContext& node) {
    // TODO: Refactor this code. Not sure about in == in.
    // Implemented tf.is_finite by checking:
    // (in != inf) && (in != -inf) && (in == in)
    //                                 ^^^^^^^^ checks for NaN's

    auto input = node.get_input(0);
    auto el_type = input.get_element_type();

    auto inf = make_shared<Constant>(el_type, Shape{}, vector<float>{numeric_limits<float>::infinity()});
    auto neg_inf = make_shared<Constant>(el_type, Shape{}, vector<float>{-numeric_limits<float>::infinity()});

    auto neq_inf = make_shared<NotEqual>(input, inf);
    auto neq_neg_inf = make_shared<NotEqual>(input, neg_inf);
    auto eq_nan = make_shared<Equal>(input, input);

    auto neq_inf_and_neq_neg_inf = make_shared<LogicalAnd>(neq_inf, neq_neg_inf);
    auto res = make_shared<LogicalAnd>(neq_inf_and_neq_neg_inf, eq_nan);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
