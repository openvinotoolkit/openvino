// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/align_eltwise_input_ranks.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::AlignEltwiseInputRanks::AlignEltwiseInputRanks() {
    auto eltwise_pattern = pattern::wrap_type<ov::op::v0::SquaredDifference,
                                              op::util::BinaryElementwiseComparison,
                                              op::util::BinaryElementwiseLogical,
                                              op::util::BinaryElementwiseArithmetic,
                                              ov::op::v0::FakeQuantize>(pattern::has_static_rank());

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();

        auto fq = as_type<ov::op::v0::FakeQuantize>(node.get());
        if (fq) {
            if (fq->get_auto_broadcast() != ov::op::AutoBroadcastType::NUMPY) {
                return false;
            }
        } else if (node->get_autob() != ov::op::AutoBroadcastType::NUMPY) {
            return false;
        }

        // NormalizeL2 and Multiply can be fused to NormalizeIE.
        // NormalizeIE has an attribute called channel_shared, which is set
        // based on Multiply's constant input rank - it's true if the rank is 1.
        // So we skip extending Multiply's constant input rank here.
        if (ov::is_type<ov::op::v1::Multiply>(node)) {
            auto inputs = node->input_values();
            if (std::any_of(inputs.begin(), inputs.end(), [](const Output<Node>& input) -> bool {
                    return ov::is_type<ov::op::v0::NormalizeL2>(input.get_node());
                }))
                return false;
        }

        const auto rank = static_cast<int64_t>(node->get_output_partial_shape(0).size());

        for (size_t i = 0; i < node->get_input_size(); i++) {
            auto const_node = as_type<ov::op::v0::Constant>(node->get_input_node_ptr(i));
            if (const_node == nullptr)
                continue;
            const auto& const_shape = const_node->get_shape();
            auto diff = rank - static_cast<int64_t>(const_shape.size());
            if (diff > 0) {
                Shape new_shape = const_shape;
                new_shape.insert(new_shape.begin(), diff, 1);
                auto new_const = std::make_shared<ov::op::v0::Constant>(*const_node, new_shape);
                copy_runtime_info(node->get_input_node_shared_ptr(i), new_const);
                node->input(i).replace_source_output(new_const);
            }
        }

        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(eltwise_pattern, "AlignEltwiseInputRanks");
    this->register_matcher(m, callback);
}
