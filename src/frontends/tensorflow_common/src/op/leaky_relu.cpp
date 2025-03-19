// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/prelu.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
ov::OutputVector translate_leaky_relu_op(const NodeContext& node) {
    default_op_checks(node, 1, {"LeakyRelu", "LEAKY_RELU"});
    auto features = node.get_input(0);
    auto alpha = node.get_attribute<float>("alpha", 0.2f);
    auto alpha_const = create_same_type_const_scalar<float>(features, alpha);
    auto leaky_relu = make_shared<v0::PRelu>(features, alpha_const);
    set_node_name(node.get_name(), leaky_relu);
    return {leaky_relu};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
