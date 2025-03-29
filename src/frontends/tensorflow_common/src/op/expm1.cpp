// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_expm1_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Expm1"});
    auto input = node.get_input(0);
    auto const_one = create_same_type_const_scalar<int>(input, 1);
    auto exp = make_shared<v0::Exp>(input);
    auto res = make_shared<v1::Subtract>(exp, const_one);
    set_node_name(node.get_name(), res);
    return {res};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
