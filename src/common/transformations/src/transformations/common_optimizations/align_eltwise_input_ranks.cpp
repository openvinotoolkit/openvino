// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/align_eltwise_input_ranks.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::AlignEltwiseInputRanks, "AlignEltwiseInputRanks", 0);

ngraph::pass::AlignEltwiseInputRanks::AlignEltwiseInputRanks() {
    auto input_pattern = pattern::any_input();
    auto const_pattern = pattern::wrap_type<opset8::Constant>();
    auto eltwise_pattern = pattern::wrap_type<opset8::SquaredDifference,
                                              op::util::BinaryElementwiseComparison,
                                              op::util::BinaryElementwiseLogical,
                                              op::util::BinaryElementwiseArithmetic>({input_pattern, const_pattern}, pattern::has_static_rank());

    auto input_low_pattern = pattern::wrap_type<opset8::Constant>();
    auto input_high_pattern = pattern::wrap_type<opset8::Constant>();
    auto output_low_pattern = pattern::wrap_type<opset8::Constant>();
    auto output_high_pattern = pattern::wrap_type<opset8::Constant>();
    auto fq_pattern = pattern::wrap_type<opset8::FakeQuantize>({input_pattern, input_low_pattern, input_high_pattern,
                                                                output_low_pattern, output_high_pattern}, pattern::has_static_rank());
    auto eltwise_or_fq = std::make_shared<pattern::op::Or>(OutputVector{eltwise_pattern, fq_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();

        auto fq = dynamic_cast<opset8::FakeQuantize*>(node.get());
        if (fq) {
            if (fq->get_auto_broadcast() != ngraph::op::AutoBroadcastType::NUMPY) {
                return false;
            }
        } else if (node->get_autob() != ngraph::op::AutoBroadcastType::NUMPY) {
            return false;
        }

        const auto rank = node->get_output_partial_shape(0).size();

        for (size_t i = 0; i < node->get_input_size(); i++) {
            auto const_node = dynamic_cast<op::Constant*>(node->get_input_node_ptr(i));
            if (const_node == nullptr)
                continue;
            const auto& const_shape = const_node->get_shape();
            auto diff = rank - const_shape.size();
            if (diff > 0) {
                Shape new_shape = const_shape;
                new_shape.insert(new_shape.begin(), diff, 1);
                auto new_const = std::make_shared<op::Constant>(*const_node, new_shape);
                node->input(i).replace_source_output(new_const);
            }
        }

        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(eltwise_or_fq, "AlignEltwiseInputRanks");
    this->register_matcher(m, callback);
}
