// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/util/keep_in_graph_op.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_write_file(const NodeContext& node) {
    default_op_checks(node, 1, {"WriteFile"});
    auto input = node.get_input(0);
    OutputVector inputs = {input};

    auto write_file_node = std::make_shared<ov::op::util::KeepInGraphOp>(inputs, 1, "WriteFile");
    set_node_name(node.get_name(), write_file_node);
    return {write_file_node};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
