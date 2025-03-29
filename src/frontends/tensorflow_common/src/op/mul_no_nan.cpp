// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_mul_no_nan_op(const NodeContext& node) {
    default_op_checks(node, 2, {"MulNoNan"});

    // first = x, second = y
    auto x = node.get_input(0);
    auto y = node.get_input(1);

    // prepare zero constant of the same type as the inputs
    auto const_zero = create_same_type_const_scalar<int32_t>(x, 0);

    // get mask where y equals 0
    auto is_zero = make_shared<v1::Equal>(y, const_zero);

    // replace all values in x at is_zero mask with zeros
    auto x_zeros = make_shared<v1::Select>(is_zero, const_zero, x);

    // multiply y with the updated x
    auto mul_no_nan = make_shared<v1::Multiply>(x_zeros, y);

    set_node_name(node.get_name(), mul_no_nan);
    return mul_no_nan->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov