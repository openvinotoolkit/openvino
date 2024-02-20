// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_inv_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Inv"});
    auto x = node.get_input(0);

    // prepare auxiliary one constants of the same type as the inputs
    auto one = create_same_type_const_scalar<int32_t>(x, 1);

    auto inv = make_shared<v1::Divide>(one, x);
    set_node_name(node.get_name(), inv);
    return inv->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov