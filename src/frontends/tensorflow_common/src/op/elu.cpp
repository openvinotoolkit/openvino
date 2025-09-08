// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/elu.hpp"

#include "common_op_table.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_elu_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Elu", "ELU"});
    auto input = node.get_input(0);
    auto alpha = node.get_attribute<float>("alpha", 1.0);
    auto res = make_shared<v0::Elu>(input, alpha);

    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
