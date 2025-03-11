// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sink_reshape.hpp"

#include "intel_gpu/op/convolution.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

SinkReshape::SinkReshape() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;
    using namespace ov::op;

    auto reshape_predicate = [](const ov::Output<ov::Node>& output) -> bool {
        auto supported_conv_act_post_ops_for_fuse = [](const std::shared_ptr<const Node>& node) -> bool {
            return ov::is_type<v0::Relu>(node) || ov::is_type<v0::Elu>(node) || ov::is_type<v0::Sigmoid>(node) ||
                   ov::is_type<v5::HSigmoid>(node) || ov::is_type<v0::Clamp>(node) || ov::is_type<v4::Swish>(node) ||
                   ov::is_type<v4::HSwish>(node) || ov::is_type<v4::Mish>(node) || ov::is_type<v5::Round>(node) ||
                   ov::is_type<v4::Mish>(node) || ov::is_type<v5::Round>(node);
        };
        auto supported_conv_eltwise_post_ops_for_fuse = [](const std::shared_ptr<const Node>& node) -> bool {
            if (ov::is_type<v1::Add>(node) || ov::is_type<v1::Subtract>(node) || ov::is_type<v1::Multiply>(node) ||
                ov::is_type<v1::Divide>(node))
                return std::dynamic_pointer_cast<v0::Constant>(node->get_input_node_shared_ptr(1)) != nullptr;
            return ov::is_type<v0::Exp>(node);
        };
        std::function<bool(const std::shared_ptr<ov::Node>&)> is_suitable_parent;
        is_suitable_parent = [&](const std::shared_ptr<ov::Node>& node) -> bool {
            if (node->get_users().size() != 1 || node->is_dynamic())
                return false;
            if (ov::as_type_ptr<op::Convolution>(node))
                return true;
            for (size_t idx = 0; idx < node->get_input_size(); idx++) {
                auto input = node->get_input_node_shared_ptr(idx);
                if (ov::as_type_ptr<v0::Constant>(node))
                    continue;
                if (supported_conv_eltwise_post_ops_for_fuse(node)) {
                    return is_suitable_parent(input);
                } else if (supported_conv_act_post_ops_for_fuse(node)) {
                    return is_suitable_parent(input);
                }
                return false;
            }
            return false;
        };
        // reshape supported only in one case, if two consecutive input dims are merged into 1
        auto is_suitable_reshape = [](const std::shared_ptr<ov::Node>& node) -> bool {
            if (node->is_dynamic())
                return false;
            auto& in_ps = node->get_input_partial_shape(0);
            auto& out_ps = node->get_output_partial_shape(0);
            if (in_ps.size() - out_ps.size() != 1)
                return false;
            size_t mismatch_count = 0;
            for (size_t i = 0; i < out_ps.size(); ++i) {
                if (i + mismatch_count >= in_ps.size())
                    return false;
                if (out_ps[i] != in_ps[i + mismatch_count]) {
                    mismatch_count++;
                }
            }
            return mismatch_count == 1;
        };
        const auto reshape = ov::as_type_ptr<v1::Reshape>(output.get_node_shared_ptr());
        return is_suitable_reshape(reshape) && is_suitable_parent(reshape->get_input_node_shared_ptr(0));
    };

    auto reshape_m = wrap_type<v1::Reshape>(reshape_predicate && consumers_count(1));
    auto transpose_const_m = wrap_type<v0::Constant>();
    auto transpose_m = wrap_type<v1::Transpose>({reshape_m, transpose_const_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto reshape = std::dynamic_pointer_cast<v1::Reshape>(pattern_map.at(reshape_m).get_node_shared_ptr());
        if (!reshape || transformation_callback(reshape)) {
            return false;
        }

        auto update_order = [](std::vector<uint16_t> original_order, const std::shared_ptr<v1::Reshape>& reshape_node) {
            // Example. For this sequence, there is Reshape node which merges 2 consecutive dims into one
            // order must be updated like permute is done before reshape
            // [1,3,4,6] -> Reshape[1,3,24]-> permute(0,2,1) -> [1,24,3]
            // updated order must be (0,2,3,1):
            // dim with index=2 is split into 2 parts: 2 and 3
            auto reshape_in_shape = reshape_node->get_input_partial_shape(0).to_shape();
            auto reshape_out_shape = reshape_node->get_output_partial_shape(0).to_shape();
            auto transformed_order = original_order;
            ov::Shape new_shape(transformed_order.size());
            const uint16_t merge_dim_idx = [&]() {
                for (uint16_t i = 0; i < reshape_out_shape.size(); ++i) {
                    if (reshape_in_shape[i] != reshape_out_shape[i])
                        return i;
                }
                OPENVINO_THROW("same input/output for reshape node");
            }();
            auto insertIt = transformed_order.end();
            for (auto it = transformed_order.begin(); it != transformed_order.end(); ++it) {
                auto& elem = *it;
                if (elem > merge_dim_idx) {
                    elem++;
                } else if (elem == merge_dim_idx) {
                    insertIt = it + 1;
                }
            }
            transformed_order.insert(insertIt, merge_dim_idx + 1);
            return transformed_order;
        };

        // allow tranposes which rotate feature dim to back to be taken as inner-most axis
        auto check_transpose_order = [](std::vector<uint16_t>& order) -> bool {
            if (order.size() <= 2)
                return false;
            if ((int32_t)order[order.size() - 2] != order.size() - 1)
                return false;
            if ((int32_t)order[0] != 0)
                return false;
            for (int32_t i = 2; i < (int32_t)order.size(); ++i) {
                if ((int32_t)order[i - 1] != i)
                    return false;
            }
            return true;
        };

        auto transpose = std::dynamic_pointer_cast<v1::Transpose>(pattern_map.at(transpose_m).get_node_shared_ptr());
        if (pattern_map.count(transpose_const_m) > 0) {
            auto org_transpose_m = pattern_map.at(transpose_const_m).get_node_shared_ptr();
            auto org_transpose_os = transpose->get_output_shape(0);
            auto tranpose_order = std::dynamic_pointer_cast<v0::Constant>(org_transpose_m);
            auto updated_order = update_order(tranpose_order->cast_vector<uint16_t>(), reshape);
            if (check_transpose_order(updated_order)) {
                auto updated_transpose_order = std::make_shared<v0::Constant>(tranpose_order->get_element_type(),
                                                                              ov::Shape(1, updated_order.size()),
                                                                              updated_order);
                updated_transpose_order->set_friendly_name(tranpose_order->get_friendly_name() + "_updated");
                auto new_transpose =
                    std::make_shared<v1::Transpose>(reshape->input(0).get_source_output(), updated_transpose_order);
                new_transpose->set_friendly_name(transpose->get_friendly_name() + "_with_updated_order");
                copy_runtime_info(transpose, new_transpose);
                ov::replace_node(reshape, new_transpose);
                auto new_pattern_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                                ov::Shape{org_transpose_os.size()},
                                                                                org_transpose_os);
                auto new_reshape = std::make_shared<ov::op::v1::Reshape>(new_transpose,
                                                                         new_pattern_const,
                                                                         reshape->get_special_zero());
                new_reshape->set_friendly_name(reshape->get_friendly_name() + "_sinked_after_transpose");
                copy_runtime_info(reshape, new_reshape);
                ov::replace_node(transpose, new_reshape);
            }
        }
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_m, "SinkReshapeIfNeeded");
    this->register_matcher(m, callback);
}
}  // namespace intel_gpu
}  // namespace ov
