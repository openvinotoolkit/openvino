// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "utils.hpp"

#include <limits>

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_clamp(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    
    // Check if min and max are constants
    bool min_is_const = !context.input_is_none(1) && ov::as_type_ptr<ov::op::v0::Constant>(context.get_input(1).get_node_shared_ptr());
    bool max_is_const = !context.input_is_none(2) && ov::as_type_ptr<ov::op::v0::Constant>(context.get_input(2).get_node_shared_ptr());
    
    if (min_is_const && max_is_const) {
        // Both min and max are constants, use Clamp op
        auto min_const = ov::as_type_ptr<ov::op::v0::Constant>(context.get_input(1).get_node_shared_ptr());
        double min_val = min_const->cast_vector<double>()[0];
        auto max_const = ov::as_type_ptr<ov::op::v0::Constant>(context.get_input(2).get_node_shared_ptr());
        double max_val = max_const->cast_vector<double>()[0];
        return {context.mark_node(std::make_shared<v0::Clamp>(x, min_val, max_val))};
    } else {
        // Fallback to Maximum/Minimum for tensor min/max or missing min/max
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
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov