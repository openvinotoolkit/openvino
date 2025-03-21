// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "common_translators.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_erfc_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Erfc"});
    auto res = common_translators::translate_erfc(node);

    set_node_name(node.get_name(), res[0].get_node_shared_ptr());
    return res;
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
