// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/keep_in_graph_op.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_write_file(const NodeContext& node) {
    default_op_checks(node, 1, {"WriteFile"});

    auto write_file_node = make_shared<KeepInGraphOp>("WriteFile");
    set_node_name(node.get_name(), write_file_node);
    return {write_file_node};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
