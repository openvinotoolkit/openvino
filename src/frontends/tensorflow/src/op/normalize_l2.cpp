// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_normalize_l2_op(const NodeContext& node) {
    default_op_checks(node, 2, {"NormalizeL2"});
    auto x = node.get_input(0);
    auto axes = node.get_input(1);

    // retrieve attribute
    auto eps = node.get_attribute<float>("epsilon");

    auto normalize_l2 = make_shared<NormalizeL2>(x, axes, eps, ov::op::EpsMode::MAX);
    set_node_name(node.get_name(), normalize_l2);
    return {normalize_l2};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
