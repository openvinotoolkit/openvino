// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_logit(const NodeContext& context) {
    // torch.logit(input, eps=None) computes logit(x) = log(x / (1-x))
    // If eps is not None, input is clamped to [eps, 1-eps]
    num_inputs_check(context, 1, 2);
    
    auto x = get_input_with_floating_type(context, 0);
    
    // Check if eps parameter is provided and clamp if needed
    if (!context.input_is_none(1)) {
        auto eps = context.get_input(1);
        eps = context.mark_node(std::make_shared<v1::ConvertLike>(eps, x));
        
        // Create (1 - eps) for max clamp
        auto one = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
        one = context.mark_node(std::make_shared<v1::ConvertLike>(one, x));
        auto one_minus_eps = context.mark_node(std::make_shared<v1::Subtract>(one, eps));
        
        // Clamp x to [eps, 1-eps]
        x = context.mark_node(std::make_shared<v1::Maximum>(x, eps));
        x = context.mark_node(std::make_shared<v1::Minimum>(x, one_minus_eps));
    }
    
    // Compute logit: log(x / (1-x)) = log(x) - log(1-x)
    auto one = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    one = context.mark_node(std::make_shared<v1::ConvertLike>(one, x));
    auto one_minus_x = context.mark_node(std::make_shared<v1::Subtract>(one, x));
    
    auto log_x = context.mark_node(std::make_shared<v0::Log>(x));
    auto log_one_minus_x = context.mark_node(std::make_shared<v0::Log>(one_minus_x));
    
    auto result = context.mark_node(std::make_shared<v1::Subtract>(log_x, log_one_minus_x));
    
    return {result};
};

OutputVector translate_special_logit(const NodeContext& context) {
    return translate_logit(context);
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
