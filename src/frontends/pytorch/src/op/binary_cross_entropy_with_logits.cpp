// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
using namespace std;

OutputVector translate_binary_cross_entropy_with_logits(const NodeContext& context) {
    num_inputs_check(context, 2, 5);
    auto input = context.get_input(0);
    auto target = context.get_input(1);

    auto zero = v0::Constant::create(input.get_element_type(), Shape{}, {0.0});
    auto one = v0::Constant::create(input.get_element_type(), Shape{}, {1.0});
    auto neg_one = v0::Constant::create(input.get_element_type(), Shape{}, {-1.0});

    // Numerically stable BCE:
    auto max_val = context.mark_node(std::make_shared<v1::Maximum>(input, zero));
    auto abs_input = context.mark_node(std::make_shared<v0::Abs>(input));
    auto neg_abs_input = context.mark_node(std::make_shared<v1::Multiply>(abs_input, neg_one));
    auto exp_neg_abs = context.mark_node(std::make_shared<v0::Exp>(neg_abs_input));
    auto exp_plus_one = context.mark_node(std::make_shared<v1::Add>(exp_neg_abs, one));
    auto log_term = context.mark_node(std::make_shared<v0::Log>(exp_plus_one));

    // base_loss = max_val - input * target + log(1 + exp(-abs(input)))
    auto input_mul_target = context.mark_node(std::make_shared<v1::Multiply>(input, target));
    auto base = context.mark_node(std::make_shared<v1::Subtract>(max_val, input_mul_target));
    auto base_loss = context.mark_node(std::make_shared<v1::Add>(base, log_term));

    // Optional: pos_weight (for positive class scaling)
    Output<Node> final_loss = base_loss;
    if (!context.input_is_none(3)) {
        auto pos_weight = context.get_input(3);
        // pos_weight term: input * target * (1 - sigmoid(input)) * pos_weight
        // Use: final_loss = base_loss + log_term * target * (pos_weight - 1)
        auto pos_weight_minus_one = context.mark_node(std::make_shared<v1::Subtract>(pos_weight, one));
        auto scale_term = context.mark_node(std::make_shared<v1::Multiply>(log_term, target));
        auto weighted_term = context.mark_node(std::make_shared<v1::Multiply>(scale_term, pos_weight_minus_one));
        final_loss = context.mark_node(std::make_shared<v1::Add>(base_loss, weighted_term));
    }

    // Optional: element-wise weight
    if (!context.input_is_none(2)) {
        auto weight = context.get_input(2);
        final_loss = context.mark_node(std::make_shared<v1::Multiply>(final_loss, weight));
    }

    // Optional: reduction
    if (!context.input_is_none(4)) {
        auto reduction_node = context.get_input(4).get_node_shared_ptr();
        auto reduction_const = ov::as_type_ptr<v0::Constant>(reduction_node);
        FRONT_END_OP_CONVERSION_CHECK(reduction_const, "Reduction must be a constant int.");
        int64_t reduction_mode = reduction_const->cast_vector<int64_t>()[0];
        switch (reduction_mode) {
        case 0:     // none
            break;  // do nothing
        case 1: {   // mean
            auto numel =
                context.mark_node(std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{-1}));
            auto reduce_mean = context.mark_node(std::make_shared<v1::ReduceMean>(final_loss, numel, true));
            final_loss = reduce_mean;
            break;
        }
        case 2: {  // sum
            auto numel =
                context.mark_node(std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{-1}));
            auto reduce_sum = context.mark_node(std::make_shared<v1::ReduceSum>(final_loss, numel, true));
            final_loss = reduce_sum;
            break;
        }
        default:
            FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported reduction mode: ", reduction_mode);
        }
    }

    return {final_loss};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
