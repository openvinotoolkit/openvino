// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"

#include "common_op_table.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_transpose_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Transpose", "TRANSPOSE"});
    auto x = node.get_input(0);
    auto perm = node.get_input(1);
    auto transpose = make_shared<v1::Transpose>(x, perm);
    set_node_name(node.get_name(), transpose);
    return {transpose};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
