// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
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
OutputVector translate_xlog1py_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Xlog1py"});
    auto x = node.get_input(0);
    auto y = node.get_input(1);

    // prepare auxiliary constants of the same type as the input
    auto zero = create_same_type_const_scalar<int32_t>(x, 0);
    auto one = create_same_type_const_scalar<int32_t>(y, 1);

    // compute a mask to identify where x is equal to 0
    auto is_zero = make_shared<v1::Equal>(x, zero);

    // compute x * log(y + 1) elementwise
    auto xlog1py = make_shared<v1::Multiply>(x, make_shared<v0::Log>(make_shared<v1::Add>(y, one)));

    // create the output tensor using Select to handle the x == 0 condition
    auto result = make_shared<v1::Select>(is_zero, zero, xlog1py);

    set_node_name(node.get_name(), result);
    return result->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
