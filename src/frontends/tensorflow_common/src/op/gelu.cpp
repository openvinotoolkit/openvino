// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/gelu.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_gelu_op(const NodeContext& node) {
    default_op_checks(node, 1, {"GELU"});
    auto x = node.get_input(0);

    // Get the approximate method
    auto approximate = node.get_attribute<GeluApproximationMode>("approximate");
    // auto approximation_mode = approximate ? GeluApproximationMode::ERF : GeluApproximationMode::TANH;
    auto res = make_shared<v7::Gelu>(x, approximate);

    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov