// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/floor_mod.hpp"

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

    auto trunc_mod = make_shared<FloorMod>(x, y);

    set_node_name(node.get_name(), trunc_mod);
    return trunc_mod->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
