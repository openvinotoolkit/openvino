// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include "transformations/common_optimizations/variadic_split_merge.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::pass {

struct split_candidate {
    std::shared_ptr<ov::Node> node;
    int64_t axis;
    int64_t begin;
    int64_t end;
};

static bool mask_zero(const std::vector<int64_t>& mask) {
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i] != 0)
            return false;
    }
    return true;
}

static std::optional<int64_t> axis_scan(const std::vector<int64_t>& axis_mask) {
    int64_t axis = 0;
    int n = 0;
    for (size_t i = 0; i < axis_mask.size(); i++) {
        auto mask = axis_mask[i];
        if (mask == 0) {
            n++;
            axis = i;
        }
        if (n >= 2)
            return std::nullopt;
    }

    if (n == 0)
        return std::nullopt;

    return axis;
}

static std::optional<int64_t> get_scalar(const std::shared_ptr<ov::Node>& node) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant)
        return std::nullopt;
    if (shape_size(constant->get_shape()) != 1)
        return std::nullopt;
    auto value = constant->cast_vector<int64_t>()[0];
    return value;
}

static int64_t convert_value(int64_t value, int64_t max) {
    if (value >= 0) {
        if (value > max)
            return max;

        return value;
    }
    else {
        int64_t temp = max + (value + 1);
        if (temp < 0)
            return 0;

        return max + (value + 1);
    }
}

VariadicSplitMerge::VariadicSplitMerge() {
    MATCHER_SCOPE(VariadicSplitMerge);

    auto any_node = ov::pass::pattern::wrap_type<ov::op::Op>([](const ov::Output<ov::Node>&) {
        return true;  // Accept all
    });

    auto m = std::make_shared<ov::pass::pattern::Matcher>(any_node, "MatchAnyNode");

    register_matcher(m, [](ov::pass::pattern::Matcher& m) {
        auto op = m.get_match_root();

        if (op->get_output_size() != 1)
            return false;  // Cannot safely apply transformation

        auto users = op->get_users();
        // Multiple users are needed
        if (users.size() <= 1)
            return false;

        std::vector<split_candidate> splitlist;
        for (auto user : users) {
            if (ov::is_type<ov::op::v1::StridedSlice>(user) || ov::is_type<ov::op::v8::Slice>(user)) {
                struct split_candidate candidate;
                candidate.node = user;
                candidate.axis = 0;
                candidate.begin = 0;
                candidate.end = 0;
                splitlist.push_back(candidate);
            } else
                return false;
        }

        // Check whether they have one axis
        for (size_t i = 0; i < splitlist.size(); i++) {
            auto node = splitlist[i].node;

            // Ranks have to be static
            auto input_ps = node->get_input_partial_shape(0);

            if (!input_ps.rank().is_static())
                return false;

            auto output_ps = node->get_output_partial_shape(0);

            if (!output_ps.rank().is_static())
                return false;

            if (input_ps.rank().get_length() != output_ps.rank().get_length())
                return false;

            if (input_ps.rank().get_length() == 0)
                return false;

            if (ov::is_type<ov::op::v1::StridedSlice>(node)) {
                auto slice = ov::as_type_ptr<ov::op::v1::StridedSlice>(node);
                uint64_t rank = input_ps.rank().get_length();

                auto begin_input = slice->get_input_node_shared_ptr(1);
                if (!ov::as_type_ptr<ov::op::v0::Constant>(begin_input))
                    return false;

                auto end_input = slice->get_input_node_shared_ptr(2);
                if (!ov::as_type_ptr<ov::op::v0::Constant>(end_input))
                    return false;

                if (slice->get_input_size() > 3) {
                    auto stride_input = slice->get_input_node_shared_ptr(3);
                    if (!ov::as_type_ptr<ov::op::v0::Constant>(stride_input))
                        return false;
                }

                // We cannot convert if we have any of these
                const auto new_axis_mask = slice->get_new_axis_mask();
                if (!new_axis_mask.empty()) {
                    if (!mask_zero(new_axis_mask))
                        return false;
                }

                const auto shrink_axis_mask = slice->get_shrink_axis_mask();
                if (!shrink_axis_mask.empty()) {
                    if (!mask_zero(shrink_axis_mask))
                        return false;
                }

                const auto ellipsis_mask = slice->get_ellipsis_mask();
                if (!ellipsis_mask.empty()) {
                    if (!mask_zero(ellipsis_mask))
                        return false;
                }

                // Read the axis
                const auto begin_mask = slice->get_begin_mask();
                if (begin_mask.size() != rank)
                    return false;

                auto begin_axis = axis_scan(begin_mask);

                if (!begin_axis.has_value())
                    return false;

                const auto end_mask = slice->get_end_mask();
                if (end_mask.size() != rank)
                    return false;

                auto end_axis = axis_scan(end_mask);
                if (!end_axis.has_value())
                    return false;

                // They both need to have the same axis
                if (begin_axis.value() != end_axis.value())
                    return false;

                splitlist[i].axis = begin_axis.value();
            } else if (ov::is_type<ov::op::v8::Slice>(node)) {
                auto slice = ov::as_type_ptr<ov::op::v8::Slice>(node);

                auto input_ps = node->get_input_partial_shape(0);
                int64_t rank = static_cast<int64_t>(input_ps.rank().get_length());

                if (slice->get_input_size() >= 5) {
                    auto axis_input = slice->get_input_node_shared_ptr(4);
                    auto axis_ret = get_scalar(axis_input);
                    if (!axis_ret.has_value())
                        return false;
                  
                    if(axis_ret.value() < -rank || axis_ret.value() > (rank - 1))
                        return false;

                    splitlist[i].axis = convert_value(axis_ret.value(), rank - 1);
                }


                auto step_input = slice->get_input_node_shared_ptr(3);
                auto step_ret = get_scalar(step_input);
                if (!step_ret.has_value())
                    return false;

                if (step_ret.value() != 1)
                    return false;
            }
        }

        auto axis = splitlist[0].axis;

        // All axis have to be the same
        for (size_t i = 1; i < splitlist.size(); i++) {
            auto split = splitlist[i];
            if (split.axis != axis)
                return false;
        }

        int64_t total_len = 0;

        // Axis partial shape has to be static
        for (size_t i = 0; i < splitlist.size(); i++) {
            auto node = splitlist[i].node;

            auto input_ps = node->get_input_partial_shape(0);
            auto output_ps = node->get_output_partial_shape(0);

            if (!input_ps[axis].is_static() || !output_ps[axis].is_static())
                return false;

            if (i == 0) {
                auto& total_length = input_ps[axis];
                total_len = total_length.get_length();
                if (total_len <= 0)
                    return false;
            } else {
                auto& total_length = input_ps[axis];
                if (total_len != total_length.get_length())
                    return false;
            }
        }

        // Get begin and end
        for (auto& split : splitlist) {
            auto node = split.node;
            if (ov::is_type<ov::op::v1::StridedSlice>(node)) {
                auto slice = ov::as_type_ptr<ov::op::v1::StridedSlice>(node);

                auto begin_input = slice->get_input_node_shared_ptr(1);
                auto end_input = slice->get_input_node_shared_ptr(2);

                auto input_ps = node->get_input_partial_shape(0);
                uint64_t rank = input_ps.rank().get_length();

                if (slice->get_input_size() > 3) {
                    auto stride_input = slice->get_input_node_shared_ptr(3);
                    auto stride_const = ov::as_type_ptr<ov::op::v0::Constant>(stride_input);

                    auto stride_vec = stride_const->cast_vector<int64_t>();

                    if (stride_vec.size() != rank)
                        return false;

                    if (stride_vec[axis] != 1)
                        return false;
                }

                auto begin_const = ov::as_type_ptr<ov::op::v0::Constant>(begin_input);
                auto end_const = ov::as_type_ptr<ov::op::v0::Constant>(end_input);

                auto begin_vec = begin_const->cast_vector<int64_t>();
                if (begin_vec.size() != rank)
                    return false;

                auto end_vec = end_const->cast_vector<int64_t>();
                if (end_vec.size() != rank)
                    return false;

                split.begin = convert_value(begin_vec[axis], total_len);
                split.end = convert_value(end_vec[axis], total_len);
            } else if (ov::is_type<ov::op::v8::Slice>(node)) {
                auto slice = ov::as_type_ptr<ov::op::v8::Slice>(node);

                auto begin_input = slice->get_input_node_shared_ptr(1);
                auto end_input = slice->get_input_node_shared_ptr(2);

                auto begin_ret = get_scalar(begin_input);
                if (!begin_ret.has_value())
                    return false;

                auto end_ret = get_scalar(end_input);
                if (!end_ret.has_value())
                    return false;

                split.begin = convert_value(begin_ret.value(), total_len);
                split.end = convert_value(end_ret.value(), total_len);
            }
        }

        // Sort by ascending begin index
        std::sort(splitlist.begin(), splitlist.end(), [](const split_candidate& a, const split_candidate& b) {
            return a.begin < b.begin;
        });

        int64_t last = 0;
        for (auto& split : splitlist) {
            // Check if we are not going out of bounds
            if (split.begin > total_len || split.begin < 0)
                return false;
            if (split.end > total_len || split.end < 0)
                return false;

            // Check for continuity
            if (split.begin != last)
                return false;

            last = split.end;
        }

        // Check if we are filling the entire output
        if (last != total_len)
            return false;

        // Rebuild lengths & nodes in sorted order
        std::vector<int64_t> split_lengths;
        for (auto& cand : splitlist) {
            auto len = cand.end - cand.begin;
            if (len == 0 || len < 0)
                return false;
            split_lengths.push_back(len);
        }

        // Now create the split using the correctly ordered lengths
        auto name = op->get_friendly_name() + "_split";
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
        auto split_lengths_const =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{split_lengths.size()}, split_lengths);
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(op, axis_const, split_lengths_const);
        variadic_split->set_friendly_name(name);

        for (size_t i = 0; i < splitlist.size(); ++i) {
            auto node = splitlist[i].node;
            ov::replace_output_update_name(node->output(0), variadic_split->output(i));
            ov::copy_runtime_info(node, variadic_split);
        }

        return true;
    });
}

}  // namespace ov::pass
