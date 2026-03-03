// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/percent_format.hpp"   // Correct path (since file is in core/include/openvino/op/)
#include "openvino/op/constant.hpp"         // Needed for Constant op
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov;
using namespace ov::op;

/// Translates PyTorch aten::percentFormat to OpenVINO PercentFormat op
OutputVector translate_percent_format(const NodeContext& context) {
    // aten::percentFormat(self, precision)
    //
    // PyTorch inputs:
    // input(0): value tensor (scalar float) to format as percentage
    // input(1): precision tensor (optional scalar int) specifying decimal places
    //
    // This translator converts the PyTorch aten::percentFormat operator
    // into an OpenVINO custom PercentFormat node.

    FRONT_END_OP_CONVERSION_CHECK(
        context.get_input_size() >= 1,
        "aten::percentFormat requires at least 1 input (value)"
    );

    // Get input value tensor
    auto value = context.get_input(0);

    // Default precision is 2 if not provided
    int precision_value = 2;

    // If precision argument exists, extract it
    if (context.get_input_size() >= 2) {
        auto precision_node = context.get_input(1);

        // Try to read constant precision value
        if (auto const_precision =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(
                    precision_node.get_node_shared_ptr())) {

            precision_value = const_precision->cast_vector<int>()[0];
        }
    }

    // Create OpenVINO constant node for precision
    auto precision_const =
        context.mark_node(
            ov::op::v0::Constant::create(
                ov::element::i32,
                ov::Shape{},
                {precision_value}
            )
        );

    // Create custom PercentFormat node
    auto percent_node =
        context.mark_node(
            std::make_shared<ov::op::custom::PercentFormat>(
                value,
                precision_const
            )
        );

    // Return created node
    return {percent_node};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov