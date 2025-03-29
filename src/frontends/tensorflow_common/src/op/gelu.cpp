// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gelu.hpp"

#include "common_op_table.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_gelu_op(const NodeContext& node) {
    default_op_checks(node, 1, {"GELU"});
    auto input = node.get_input(0);
    bool approximate = node.get_attribute<bool>("approximate", false);
    const auto mode = (approximate == true) ? ov::op::GeluApproximationMode::TANH : ov::op::GeluApproximationMode::ERF;
    auto res = make_shared<ov::op::v7::Gelu>(input, mode);
    set_node_name(node.get_name(), res);
    return res->outputs();
};

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
