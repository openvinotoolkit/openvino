// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/select.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_div_no_nan_op(const NodeContext& node) {
    default_op_checks(node, 2, {"DivNoNan"});
    auto numer = node.get_input(0);
    auto denom = node.get_input(1);

    // prepare auxiliary zero and one constants of the same type as the inputs
    auto zero = make_shared<v0::Constant>(element::f32, Shape{}, 0.0f)->output(0);
    auto one = make_shared<v0::Constant>(element::f32, Shape{}, 1.0f)->output(0);
    zero = make_shared<v1::ConvertLike>(zero, denom);
    one = make_shared<v1::ConvertLike>(one, denom);

    // compute a mask to get positions of Nan values of division result
    auto is_zero = make_shared<v1::Equal>(denom, zero);

    // fix zeros in the denomimator to avoid undefined behaviour
    auto fixed_denom = make_shared<v1::Select>(is_zero, one, denom);

    // compute Division and do not afraid division by zero
    // since all of them fixed
    auto div = make_shared<v1::Divide>(numer, fixed_denom);

    // set zero to the result where initially the denomimator is zero
    auto div_no_nan = make_shared<v1::Select>(is_zero, zero, div);
    set_node_name(node.get_name(), div_no_nan);
    return div_no_nan->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
