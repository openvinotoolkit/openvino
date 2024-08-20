// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_clamp(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    if (!context.input_is_none(1)) {
        auto min_clip = context.get_input(1);
        min_clip = context.mark_node(std::make_shared<v1::ConvertLike>(min_clip, x));
        x = context.mark_node(std::make_shared<v1::Maximum>(x, min_clip));
    }
    if (!context.input_is_none(2)) {
        auto max_clip = context.get_input(2);
        max_clip = context.mark_node(std::make_shared<v1::ConvertLike>(max_clip, x));
        x = context.mark_node(std::make_shared<v1::Minimum>(x, max_clip));
    }
    return {std::move(x)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov