// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common_op_table.hpp"
#include "openvino/op/angle.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/select.hpp"

using namespace std;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_angle_op(const NodeContext& node) {
    // Perform default checks for the operation
    default_op_checks(node, 2, {"Angle"});
    
    // Access inputs directly
    auto y = node.get_input(0); 
    auto x = node.get_input(1);

    // Calculate atan2(y, x)
    auto atan2_op = make_shared<ov::op::Atan2>(y, x);

    // Convert radians to degrees
    auto radians_to_degrees = make_shared<ov::op::Constant>(ov::element::f32, Shape{}, vector<float>{180.0 / M_PI});
    auto degrees = make_shared<ov::op::Divide>(atan2_op, radians_to_degrees);

    // Ensure that the result is within [0, 360) degrees
    auto zero = make_shared<ov::op::Constant>(ov::element::f32, Shape{}, vector<float>{0.0});
    auto three_sixty = make_shared<ov::op::Constant>(ov::element::f32, Shape{}, vector<float>{360.0});
    auto modulo = make_shared<ov::op::Mod>(degrees, three_sixty);
    auto angle = make_shared<ov::op::Select>(make_shared<ov::op::Greater>(modulo, zero), modulo, make_shared<ov::op::Add>(modulo, three_sixty));

    // Set node name
    set_node_name(node.get_name(), angle);

    // Return output vector
    return {angle};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
