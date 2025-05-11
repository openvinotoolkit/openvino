// Copyright (C) 2018-2025 Intel Corporation
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

OutputVector translate_sqrt_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Sqrt", "SQRT"});
    auto input = node.get_input(0);
    auto exponent = create_same_type_const_scalar<float>(input, 0.5f);
    auto sqrt_node = make_shared<v1::Power>(input, exponent);
    set_node_name(node.get_name(), sqrt_node);
    return {sqrt_node};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
