// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_clamp(NodeContext& context) {
    auto x = context.get_input(0);
    if (!context.input_is_none(1)) {
        auto min_clip = context.get_input(1);
        min_clip = context.mark_node(std::make_shared<opset8::ConvertLike>(min_clip, x));
        x = context.mark_node(std::make_shared<opset8::Maximum>(x, min_clip));
    }
    if (!context.input_is_none(2)) {
        auto max_clip = context.get_input(2);
        max_clip = context.mark_node(std::make_shared<opset8::ConvertLike>(max_clip, x));
        x = context.mark_node(std::make_shared<opset8::Minimum>(x, max_clip));
    }
    return {x};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov