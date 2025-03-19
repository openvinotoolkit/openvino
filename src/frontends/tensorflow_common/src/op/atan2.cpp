// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common_op_table.hpp"
#include "common_translators.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_atan2_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Atan2"});
    auto y = node.get_input(0);
    auto x = node.get_input(1);

    auto result = common_translators::translate_atan2(node);

    set_node_name(node.get_name(), result[0].get_node_shared_ptr());
    return result;
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
