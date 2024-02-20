// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
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
    default_op_checks(node, 1, {"Reciprocal"});
    auto x = node.get_input(0);
    auto minus_one_const = create_same_type_const_scalar<int32_t>(x, -1);
    auto reciprocal = make_shared<v1::Power>(x, minus_one_const);
    set_node_name(node.get_name(), reciprocal);
    return {reciprocal};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
