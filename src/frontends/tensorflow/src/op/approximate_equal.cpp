// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_op_table.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_approximate_equal_op(const NodeContext& node) {
    // Perform default checks for the operation
    default_op_checks(node, 2, {"ApproximateEqual"});

    // Extract necessary attributes and inputs
    // Set default value for tolerance
    auto tolerance = node.get_attribute<float>("tolerance", 1e-5f);
    // Access inputs directly
    auto x = node.get_input(0); 
    auto y = node.get_input(1);

    // Implement the logic for ApproximateEqual
    auto difference = make_shared<Subtract>(x, y);
    auto absolute = make_shared<Abs>(difference);
    auto tolerance_constant = create_same_type_const_scalar(x, tolerance);
     auto less = make_shared<Less>(absolute, tolerance_constant);

    // Create and return the corresponding OpenVINO operation
    set_node_name(node.get_name(), less);
    return {less};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
