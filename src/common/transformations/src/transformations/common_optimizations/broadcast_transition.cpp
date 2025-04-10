// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/broadcast_transition.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::BroadcastTransition::BroadcastTransition() {
    MATCHER_SCOPE(BroadcastTransition);
    auto bcast_m =
        pass::pattern::wrap_type<ov::op::v1::Broadcast, ov::op::v3::Broadcast>(pass::pattern::consumers_count(1));
    auto eltwise_input_m = pass::pattern::any_input(pass::pattern::has_static_rank());
    auto eltwise_1 = pass::pattern::wrap_type<op::util::BinaryElementwiseArithmetic>({eltwise_input_m, bcast_m});
    auto eltwise_2 = pass::pattern::wrap_type<op::util::BinaryElementwiseArithmetic>({bcast_m, eltwise_input_m});
    auto eltwise_m = std::make_shared<pass::pattern::op::Or>(OutputVector{eltwise_1, eltwise_2});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto eltwise = ov::as_type_ptr<ov::op::util::BinaryElementwiseArithmetic>(m.get_match_root());
        if (eltwise->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            return false;
        }

        const auto bcast = ov::as_type_ptr<ov::op::util::BroadcastBase>(pattern_map.at(bcast_m).get_node_shared_ptr());
        if (!bcast || (bcast->get_broadcast_spec().m_type != ov::op::BroadcastType::NUMPY &&
                       bcast->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL)) {
            return false;
        }

        const auto& eltwise_input = pattern_map.at(eltwise_input_m);
        const auto& bcast_data = bcast->input_value(0);
        // inputs order mustn't be changed because an eltwise might be not commutative
        ov::OutputVector new_inputs{
            eltwise->get_input_node_ptr(0) == eltwise_input.get_node() ? eltwise_input : bcast_data,
            eltwise->get_input_node_ptr(1) == bcast.get() ? bcast_data : eltwise_input};
        const auto new_eltwise = eltwise->clone_with_new_inputs(new_inputs);
        ov::copy_runtime_info(eltwise, new_eltwise);

        auto target_shape = bcast->input_value(1);
        const auto& target_shape_et = target_shape.get_element_type();

        std::shared_ptr<ov::Node> data_shape_path;
        if (target_shape_et == ov::element::i32 || target_shape_et == ov::element::i64) {
            data_shape_path = ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(new_eltwise, target_shape_et);
            ov::copy_runtime_info(eltwise, data_shape_path);
        } else {
            auto shapeof = ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(new_eltwise);
            data_shape_path = ov::op::util::make_try_fold<ov::op::v0::Convert>(shapeof, target_shape_et);
            ov::copy_runtime_info(eltwise, {shapeof, data_shape_path});
        }

        const size_t target_shape_rank = target_shape.get_partial_shape()[0].get_length();
        const size_t input_rank = new_eltwise->get_output_partial_shape(0).size();
        if (input_rank != target_shape_rank) {
            auto align_rank = [&](const ov::Output<ov::Node>& out, const size_t count) {
                const auto constant = ov::op::v0::Constant::create(target_shape_et, {count}, {1});
                const auto res = ov::op::util::make_try_fold<ov::op::v0::Concat>(ov::OutputVector{constant, out}, 0);
                ov::copy_runtime_info(out.get_node_shared_ptr(), {constant, res});
                return res;
            };
            if (input_rank < target_shape_rank) {
                data_shape_path = align_rank(data_shape_path, target_shape_rank - input_rank);
            } else {
                target_shape = align_rank(target_shape, input_rank - target_shape_rank);
            }
        }
        const auto new_target_shape = ov::op::util::make_try_fold<ov::op::v1::Maximum>(data_shape_path, target_shape);
        ov::copy_runtime_info(eltwise, new_target_shape);

        const auto new_bcast = std::make_shared<ov::op::v3::Broadcast>(new_eltwise, new_target_shape);
        new_bcast->set_friendly_name(eltwise->get_friendly_name());
        ov::copy_runtime_info(eltwise, {new_eltwise, new_bcast});
        ov::replace_node(eltwise, new_bcast);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, matcher_name);
    register_matcher(m, callback);
}
