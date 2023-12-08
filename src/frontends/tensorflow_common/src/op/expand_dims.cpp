// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_expand_dims_op(const NodeContext& node) {
    default_op_checks(node, 2, {"ExpandDims"});
    auto input = node.get_input(0);
    auto axis = node.get_input(1);
    auto unsqueeze = make_shared<v0::Unsqueeze>(input, axis);
    set_node_name(node.get_name(), unsqueeze);
    return {unsqueeze};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
