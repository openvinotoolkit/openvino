// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_xlogy_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Xlogy"});
    auto x = node.get_input(0);
    auto y = node.get_input(1);

    // prepare auxiliary zero constant of the same type as the input
    auto zero = create_same_type_const_scalar<int32_t>(x, 0);

    // compute a mask to identify where x is equal to 0
    auto is_zero = make_shared<v1::Equal>(x, zero);

    // compute x * log(y) elementwise
    auto xlog_y = make_shared<v1::Multiply>(x, make_shared<v0::Log>(y));

    // create the output tensor using Select to handle the x == 0 condition
    auto result = make_shared<v1::Select>(is_zero, zero, xlog_y);

    set_node_name(node.get_name(), result);
    return result->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
