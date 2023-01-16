// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
ov::OutputVector translate_random_uniform_op(const NodeContext& node) {
    default_op_checks(node, 1, {"RandomUniform"});
    auto shape = node.get_input(0);

    // retrieve attributes
    auto seed = node.get_attribute<int64_t>("seed", 0);
    auto seed2 = node.get_attribute<int64_t>("seed2", 0);
    auto output_type = node.get_attribute<ov::element::Type>("dtype");

    auto minval = make_shared<Constant>(output_type, Shape{}, 0);
    auto maxval = make_shared<Constant>(output_type, Shape{}, 1);
    auto random = std::make_shared<RandomUniform>(shape, minval, maxval, output_type, seed, seed2);

    set_node_name(node.get_name(), random);
    return random->outputs();
}

ov::OutputVector translate_random_uniform_int_op(const NodeContext& node) {
    default_op_checks(node, 3, {"RandomUniformInt"});
    auto shape = node.get_input(0);
    auto minval = node.get_input(1);
    auto maxval = node.get_input(2);

    // retrieve attributes
    auto seed = node.get_attribute<int64_t>("seed", 0);
    auto seed2 = node.get_attribute<int64_t>("seed2", 0);

    auto output_type = minval.get_element_type();
    auto random = std::make_shared<RandomUniform>(shape, minval, maxval, output_type, seed, seed2);

    set_node_name(node.get_name(), random);
    return random->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
