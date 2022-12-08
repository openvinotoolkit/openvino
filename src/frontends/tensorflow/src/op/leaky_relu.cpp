// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
ov::OutputVector translate_leaky_relu_op(const NodeContext& node) {
    auto in = node.get_input(0);
    auto alpha_attr = node.get_attribute<float>("alpha", 0.f);
    auto alpha_const = make_shared<Constant>(element::f32, Shape{1}, alpha_attr);

    auto leaky_relu = make_shared<PRelu>(in, alpha_const);
    return leaky_relu->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
