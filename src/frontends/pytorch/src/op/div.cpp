// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_div(NodeContext& context) {
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    align_eltwise_input_types(context, &x, &y);
    auto res = context.mark_node(std::make_shared<opset10::Divide>(x, y, true));
    if (!context.input_is_none(2)) {
        auto rounding_mode = context.const_input<std::string>(2);
        if (rounding_mode == "floor") {
            res = context.mark_node(std::make_shared<opset10::Floor>(res));
        } else if (rounding_mode == "trunc") {
            const auto convert = context.mark_node(std::make_shared<opset10::Convert>(res, element::i64));
            res = context.mark_node(std::make_shared<opset10::ConvertLike>(convert, x));
        } else {
            FRONT_END_OP_CONVERSION_CHECK(false,
                                          "Openvino Pytorch Frontend doesn't support rounding mode ",
                                          rounding_mode,
                                          " for aten::div");
        }
    }
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov