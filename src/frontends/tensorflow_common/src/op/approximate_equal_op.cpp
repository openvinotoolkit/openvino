// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_approximate_equal_op(const NodeContext& node) {
    default_op_checks(node, 2, {"ApproximateEqual"});
    auto x = node.get_input(0);
    auto y = node.get_input(1);
    auto tolerance_value = node.get_attribute<float>("tolerance", 1e-5f);
    auto tolerance = create_same_type_const_scalar<float>(x, tolerance_value);
    // Implement the logic for ApproximateEqual
    auto difference = make_shared<v1::Subtract>(x, y);
    auto absolute = make_shared<v0::Abs>(difference);
    auto is_less = make_shared<v1::Less>(absolute, tolerance);

    // Create and return the corresponding OpenVINO operation
    set_node_name(node.get_name(), is_less);
    return {is_less};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov