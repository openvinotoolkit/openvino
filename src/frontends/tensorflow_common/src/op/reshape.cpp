// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include "common_op_table.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_reshape_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Reshape"});
    auto tensor = node.get_input(0);
    auto shape = node.get_input(1);
    auto reshape = make_shared<v1::Reshape>(tensor, shape, false);
    set_node_name(node.get_name(), reshape);
    return {reshape};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
