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
    
    auto tolerance = node.get_attribute<float>("tolerance", 1e-5f);
    // Access inputs directly
    auto x = node.get_input(0); 
    auto y = node.get_input(1);

    // Implement the logic for ApproximateEqual
    auto difference = make_shared<ov::op::v1::Subtract>(x, y);
    auto absolute = make_shared<ov::op::v0::Abs>(difference);  
    auto tolerance_constant = make_shared<ov::op::v0::Constant>(x.get_element_type(), Shape{}, vector<float>{tolerance});
    auto less = make_shared<ov::op::v1::Less>(absolute, tolerance_constant);

    // Create and return the corresponding OpenVINO operation
    set_node_name(node.get_name(), less);
    return {less};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
