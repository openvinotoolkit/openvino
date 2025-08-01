// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

NamedOutputs gap(const NodeContext& node) {
    // 取输入
    auto data = node.get_input("X");

    // 检查输入维度，必须是4D
    const auto input_shape = data.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(input_shape.rank().is_static() && input_shape.rank().get_length() == 4,
                                  "Input to GAP must be 4D tensor");

    // 获取数据的形状: [N, C, H, W]
    auto shape_of = std::make_shared<default_opset::ShapeOf>(data);
    auto axes = default_opset::Constant::create(ov::element::i64, {2}, {2, 3});  // 对 H 和 W 做 average

    // ReduceMean: 在 H, W 上做平均 -> [N, C]
    auto reduced = std::make_shared<default_opset::ReduceMean>(data, axes, true);  // keep_dims = true => [N, C, 1, 1]

    return node.default_single_output_mapping({reduced}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov