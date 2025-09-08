// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_x_div_y_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Xdivy"});
    auto x = node.get_input(0);
    auto y = node.get_input(1);

    // create auxiliary constants
    auto const_zero = create_same_type_const_scalar<int32_t>(x, 0);
    auto const_one = create_same_type_const_scalar<int32_t>(x, 1);

    auto x_is_zero = make_shared<v1::Equal>(x, const_zero);
    auto select = make_shared<v1::Select>(x_is_zero, const_one, y);
    auto xdivy = make_shared<v1::Divide>(x, select);
    set_node_name(node.get_name(), xdivy);
    return {xdivy};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
