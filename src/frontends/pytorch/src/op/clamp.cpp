// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_clamp(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);

    bool min_is_const_scalar = context.input_is_none(1);
    bool max_is_const_scalar = context.input_is_none(2);
    double min_val = std::numeric_limits<double>::lowest();
    double max_val = std::numeric_limits<double>::max();

    // Check if 'min' is a constant scalar
    if (!context.input_is_none(1)) {
        if (auto min_const = ov::as_type_ptr<v0::Constant>(context.get_input(1).get_node_shared_ptr())) {
            auto min_vals = min_const->cast_vector<double>();
            if (min_vals.size() == 1) {
                min_is_const_scalar = true;
                min_val = min_vals[0];
            }
        }
    }

    // Check if 'max' is a constant scalar
    if (!context.input_is_none(2)) {
        if (auto max_const = ov::as_type_ptr<v0::Constant>(context.get_input(2).get_node_shared_ptr())) {
            auto max_vals = max_const->cast_vector<double>();
            if (max_vals.size() == 1) {
                max_is_const_scalar = true;
                max_val = max_vals[0];
            }
        }
    }

    // Use native Clamp if bounds are constants
    if (min_is_const_scalar && max_is_const_scalar && (!context.input_is_none(1) || !context.input_is_none(2))) {
        return {context.mark_node(std::make_shared<v0::Clamp>(x, min_val, max_val))};
    }

    // Fallback for dynamic tensor bounds
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
