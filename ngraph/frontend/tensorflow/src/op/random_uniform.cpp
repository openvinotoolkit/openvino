// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {
ov::OutputVector TranslateRandomUniformOp(const NodeContext& node) {
    auto data = node.get_ng_input(0);
    auto seed = node.get_attribute<int64_t>("seed");
    auto seed2 = node.get_attribute<int64_t>("seed2");
    auto minval_const = make_shared<Constant>(element::f32, Shape{}, 0);
    auto maxval_const = make_shared<Constant>(element::f32, Shape{}, 1);
    auto ng_et = node.get_attribute<ov::element::Type>("dtype");
    auto random_uniform = std::make_shared<RandomUniform>(data, minval_const, maxval_const, ng_et, seed, seed2);
    random_uniform->set_friendly_name(node.get_name());
    return random_uniform->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
