// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateIsFiniteOp(const NodeContext& node) {
    // TODO (itikhono): Refactor this code. Not sure about in == in.
    // Implemented tf.is_finite by checking:
    // (in != inf) && (in != -inf) && (in == in)
    //                                 ^^^^^^^^ checks for NaN's

    auto input = node.get_ng_input(0);
    auto el_type = input.get_element_type();

    auto inf = make_shared<Constant>(el_type, Shape{}, vector<float>{numeric_limits<float>::infinity()});
    auto neg_inf = make_shared<Constant>(el_type, Shape{}, vector<float>{-numeric_limits<float>::infinity()});

    auto neq_inf = make_shared<NotEqual>(input, inf);
    auto neq_neg_inf = make_shared<NotEqual>(input, neg_inf);
    auto eq_nan = make_shared<Equal>(input, input);

    auto neq_inf_and_neq_neg_inf = make_shared<LogicalAnd>(neq_inf, neq_neg_inf);
    auto is_finite = make_shared<LogicalAnd>(neq_inf_and_neq_neg_inf, eq_nan);
    is_finite->set_friendly_name(node.get_name());
    return is_finite->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
