// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_broadcast3.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace {

bool make_compatible_shape(const ov::PartialShape& input_shape, std::vector<size_t>& target_shape) {
    if (input_shape.rank().is_dynamic()) {
        return false;
    }
    const int64_t& input_shape_rank = input_shape.rank().get_length();
    if (input_shape_rank > static_cast<int64_t>(target_shape.size())) {
        // target_shape rank must greater or equal to input_shape rank, so in case when it's less we
        // insert missing input_shape dimensions to the beginning of the target_shape.
        const int64_t& dims_to_add_count = input_shape_rank - target_shape.size();
        std::vector<size_t> dims_to_add(dims_to_add_count);
        for (int64_t dim = 0; dim < dims_to_add_count; ++dim) {
            if (input_shape[dim].is_dynamic()) {
                return false;
            }
            dims_to_add[dim] = input_shape[dim].get_length();
        }
        target_shape.insert(target_shape.begin(), dims_to_add.begin(), dims_to_add.end());
    }
    for (int64_t i_dim = input_shape_rank - 1, t_dim = target_shape.size() - 1; i_dim >= 0 && t_dim >= 0;
         --i_dim, --t_dim) {
        if (input_shape[i_dim].is_static()) {
            const auto& input_dim = input_shape[i_dim].get_length();
            if (static_cast<size_t>(input_dim) != target_shape[t_dim] && input_dim != 1 && target_shape[t_dim] != 1) {
                // this dimensions are not broadcastable
                return false;
            }
            target_shape[t_dim] = std::max(target_shape[t_dim], static_cast<size_t>(input_dim));
        } else {
            if (target_shape[t_dim] == 1) {
                // For example:    |
                //                \/
                // input_shape  [DYN, 3, 4]
                // target_shape [  1, 3, 4] - broadcasted first dimension is unknown
                return false;
            }
        }
    }
    return true;
}

}  // namespace

ov::pass::ConvertBroadcast3::ConvertBroadcast3() {
    MATCHER_SCOPE(ConvertBroadcast3);
    auto broadcast = pattern::wrap_type<ov::op::v3::Broadcast>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto broadcast = ov::as_type_ptr<ov::op::v3::Broadcast>(m.get_match_root());
        if (!broadcast) {
            return false;
        }

        auto input = broadcast->input_value(0);
        auto target_shape_input = broadcast->input_value(1);
        const auto& broadcast_type = broadcast->get_broadcast_spec();
        const auto& input_element_type = input.get_element_type();

        if (broadcast_type == op::BroadcastType::NUMPY) {
            input = std::make_shared<ov::op::v1::Broadcast>(input, target_shape_input, op::AutoBroadcastType::NUMPY);
        } else if (broadcast_type == op::BroadcastType::PDPD) {
            input = std::make_shared<ov::op::v1::Broadcast>(input, target_shape_input, op::AutoBroadcastType::PDPD);
        } else if (broadcast_type == op::BroadcastType::NONE) {
            input = std::make_shared<ov::op::v1::Broadcast>(input,
                                                            target_shape_input,
                                                            broadcast->input_value(2),
                                                            op::AutoBroadcastType::NONE);
        } else if (broadcast_type == op::BroadcastType::BIDIRECTIONAL) {
            if (auto const_target_shape =
                    ov::as_type_ptr<ov::op::v0::Constant>(target_shape_input.get_node_shared_ptr())) {
                const auto& input_shape = input.get_partial_shape();
                const auto& target_shape = const_target_shape->cast_vector<size_t>();
                std::vector<size_t> aligned_target_shape{target_shape};
                if (make_compatible_shape(input_shape, aligned_target_shape)) {
                    input = std::make_shared<ov::op::v1::Broadcast>(
                        input,
                        ov::op::v0::Constant::create(element::i64,
                                                     Shape({aligned_target_shape.size()}),
                                                     aligned_target_shape));
                } else {
                    if (input_element_type == element::boolean) {
                        input = std::make_shared<ov::op::v1::LogicalAnd>(
                            input,
                            ov::op::v0::Constant::create(input_element_type, target_shape, {1}));
                    } else {
                        input = std::make_shared<ov::op::v1::Multiply>(
                            input,
                            ov::op::v0::Constant::create(input_element_type, target_shape, {1}));
                    }
                }
            } else {
                auto constant_one = ov::op::v0::Constant::create(input_element_type, {1}, {1});
                auto broadcast_ones = std::make_shared<ov::op::v1::Broadcast>(constant_one, target_shape_input);
                if (input_element_type == element::boolean) {
                    input = std::make_shared<ov::op::v1::LogicalAnd>(input, broadcast_ones);
                } else {
                    input = std::make_shared<ov::op::v1::Multiply>(input, broadcast_ones);
                }
                copy_runtime_info(broadcast, broadcast_ones);
            }
        } else {
            return false;
        }

        input.get_node_shared_ptr()->set_friendly_name(broadcast->get_friendly_name());
        copy_runtime_info(broadcast, input.get_node_shared_ptr());
        replace_node(broadcast, {input});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(broadcast, matcher_name);
    register_matcher(m, callback);
}
