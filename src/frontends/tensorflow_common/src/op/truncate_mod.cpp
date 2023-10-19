// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace ov::opset10;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_truncate_mod_op(const NodeContext& node) {
    default_op_checks(node, 2, {"TruncateMod"});
    auto x = node.get_input(0);
    auto y = node.get_input(1);

    auto is_x_negative = make_shared<Less>(x, create_same_type_const_scalar(x, 0));
    auto is_y_negative = make_shared<Less>(y, create_same_type_const_scalar(y, 0));

    // if (y < 0) {y = -y}
    auto negative_y = make_shared<Negative>(y);
    y = make_shared<Select>(is_y_negative, negative_y, y);

    // check if floor_mod == zero
    auto floor_mod = make_shared<FloorMod>(x, y);
    auto is_zero = make_shared<Equal>(floor_mod, create_same_type_const_scalar(floor_mod, 0));

    // floor_mod - y
    auto other_res = make_shared<Subtract>(floor_mod, y);

    // select operation to handle the sign
    auto result = make_shared<Select>(is_zero, floor_mod, make_shared<Select>(is_x_negative, other_res, floor_mod));

    set_node_name(node.get_name(), result);
    return result->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
