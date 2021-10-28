// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {
ov::OutputVector TranslateRandomUniformOp(const NodeContext& node) {
    auto shape = node.get_ng_input(0);
    auto seed = node.get_attribute<int64_t>("seed", 0);
    auto seed2 = node.get_attribute<int64_t>("seed2", 0);
    auto minval_const = make_shared<Constant>(element::f32, Shape{}, 0);
    auto maxval_const = make_shared<Constant>(element::f32, Shape{}, 1);
    auto ng_et = node.get_attribute<ov::element::Type>("dtype");
    auto res = std::make_shared<RandomUniform>(shape, minval_const, maxval_const, ng_et, seed, seed2);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

ov::OutputVector TranslateRandomUniformIntOp(const NodeContext& node) {
    auto shape = node.get_ng_input(0);
    auto minval = node.get_ng_input(1);
    auto maxval = node.get_ng_input(2);
    auto seed = node.get_attribute<int64_t>("seed", 0);
    auto seed2 = node.get_attribute<int64_t>("seed2", 0);
    auto ng_et = minval.get_element_type();
    auto res = std::make_shared<RandomUniform>(shape, minval, maxval, ng_et, seed, seed2);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
