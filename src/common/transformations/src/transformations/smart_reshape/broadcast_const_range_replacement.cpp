// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/broadcast_const_range_replacement.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::BroadcastConstRangeReplacement::BroadcastConstRangeReplacement() {
    MATCHER_SCOPE(BroadcastConstRangeReplacement);
    auto data_input = pattern::wrap_type<ov::op::v0::Constant>();
    auto target_shape = pattern::any_input();
    auto broadcast_pattern_node = pattern::wrap_type<ov::op::v3::Broadcast>({data_input, target_shape});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto broadcast = m.get_match_root();
        // The transformation was requested only for models with BroadcastType::BIDIRECTIONAL
        // Further analysis is needed for other broadcast modes enablement
        const auto broadcast_ptr = ov::as_type_ptr<ov::op::v3::Broadcast>(broadcast);
        if (!broadcast_ptr || broadcast_ptr->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL)
            return false;

        const auto data_const_out = broadcast->get_input_source_output(0);
        const auto target_shape_out = broadcast->get_input_source_output(1);

        const auto const_node = ov::as_type_ptr<ov::op::v0::Constant>(data_const_out.get_node_shared_ptr());
        if (!const_node || !const_node->get_element_type().is_integral_number())
            return false;

        const auto& const_node_shape = const_node->get_output_shape(0);
        const auto elem_count = shape_size(const_node_shape);
        const auto one_dims_count = std::count(const_node_shape.cbegin(), const_node_shape.cend(), 1);

        constexpr size_t dim_low_limit = 5;
        constexpr size_t dim_up_limit = 500;
        const auto const_rank = const_node_shape.size();

        // To affect less models, the transformation is applied to Constants with elements count in range (5:500)
        if (const_rank - one_dims_count != 1 || elem_count <= dim_low_limit || elem_count >= dim_up_limit)
            return false;

        std::vector<int64_t> sequence_pattern(elem_count);
        std::iota(sequence_pattern.begin(), sequence_pattern.end(), 0);
        const auto& const_values = const_node->cast_vector<int64_t>();

        // Check if the value sequence is contiguous
        if (const_values != sequence_pattern)
            return false;

        const auto data_elem_type = data_const_out.get_element_type();
        const auto target_dim_index =
            std::distance(const_node_shape.cbegin(),
                          std::find(const_node_shape.cbegin(), const_node_shape.cend(), elem_count));
        const int64_t target_dim_neg_index = target_dim_index - static_cast<int64_t>(const_rank);

        NodeRegistry node_registry;

        const auto axis_node = node_registry.add(ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
        const auto target_dim_index_node =
            node_registry.add(ov::op::v0::Constant::create(ov::element::i64, {}, {target_dim_neg_index}));
        const auto gather_dim =
            node_registry.make<ov::op::v8::Gather>(target_shape_out, target_dim_index_node, axis_node);

        // If the corresponding target dim is 1, use the original end of range
        const auto one_dim_const =
            node_registry.add(ov::op::v0::Constant::create(target_shape_out.get_element_type(), {}, {1}));
        const auto dim_check_one = node_registry.make<ov::op::v1::Equal>(gather_dim, one_dim_const);

        const auto start = node_registry.add(ov::op::v0::Constant::create(data_elem_type, {}, {0}));
        const auto original_end = node_registry.add(ov::op::v0::Constant::create(data_elem_type, {}, {elem_count}));

        const auto cast_gather_dim = node_registry.make<ov::op::v0::Convert>(gather_dim, data_elem_type);
        const auto select_end = node_registry.make<ov::op::v1::Select>(dim_check_one, original_end, cast_gather_dim);

        const auto default_range_step = node_registry.add(ov::op::v0::Constant::create(data_elem_type, {}, {1}));
        std::shared_ptr<Node> replacement =
            node_registry.make<ov::op::v4::Range>(start, select_end, default_range_step, data_elem_type);

        if (const_rank > 1) {
            // Unsqueeze the output of the Range op to the original shape of data input
            std::vector<int64_t> final_shape_axes(const_rank);
            std::iota(final_shape_axes.begin(), final_shape_axes.end(), 0);
            final_shape_axes.erase(final_shape_axes.begin() + target_dim_index);
            const auto axes_to_unsqueeze = node_registry.add(
                ov::op::v0::Constant::create(ov::element::i64, {final_shape_axes.size()}, final_shape_axes));
            replacement = node_registry.make<ov::op::v0::Unsqueeze>(replacement, axes_to_unsqueeze);
        }

        copy_runtime_info(const_node, node_registry.get());
        broadcast->input(0).replace_source_output(replacement);
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(broadcast_pattern_node, matcher_name);
    this->register_matcher(m, callback);
}
