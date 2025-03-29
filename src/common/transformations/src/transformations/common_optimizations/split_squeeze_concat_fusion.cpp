// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/split_squeeze_concat_fusion.hpp"

#include <memory>
#include <numeric>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

static bool is_axis_squeezed_by_node(const std::shared_ptr<ov::Node>& squeeze_node, int64_t axis, bool use_shapes);

ov::pass::SplitSqueezeConcatFusion::SplitSqueezeConcatFusion(bool use_shapes) {
    MATCHER_SCOPE(SplitSqueezeConcatFusion);
    // Detect only concat, because we don't know how many inputs will go into concat
    auto concat_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Concat>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto concat = ov::as_type_ptr<ov::op::v0::Concat>(pattern_to_output.at(concat_pattern).get_node_shared_ptr());
        if (!concat)
            return false;

        NodeVector nodes_to_delete{concat};

        std::shared_ptr<ov::op::v1::Split> split;
        int64_t split_axis = 0;

        for (size_t i = 0; i < concat->get_input_size(); i++) {
            auto squeeze_node = concat->get_input_node_shared_ptr(i);
            if (!ov::is_type<ov::op::v0::Squeeze>(squeeze_node) && !ov::is_type<ov::op::v1::Reshape>(squeeze_node))
                return false;
            auto split_to_check = ov::as_type_ptr<ov::op::v1::Split>(squeeze_node->get_input_node_shared_ptr(0));
            if (!split_to_check)
                return false;

            if (i == 0) {
                nodes_to_delete.push_back(split_to_check);
                split = split_to_check;
                auto split_axis_node = ov::as_type_ptr<ov::op::v0::Constant>(split->get_input_node_shared_ptr(1));
                if (!split_axis_node)
                    return false;
                auto axis_vec = split_axis_node->cast_vector<int64_t>();
                if (axis_vec.size() != 1)
                    return false;
                split_axis = axis_vec[0];
                if (split_axis < 0) {
                    auto rank = split->get_output_partial_shape(0).rank();
                    if (rank.is_dynamic())
                        return false;
                    split_axis += rank.get_length();
                }
            } else if (split_to_check != split) {
                return false;
            }

            if (!is_axis_squeezed_by_node(squeeze_node, split_axis, use_shapes)) {
                return false;
            }

            nodes_to_delete.push_back(squeeze_node);

            auto split_output = squeeze_node->input_value(0);
            if (split_output.get_target_inputs().size() != 1 || split_output.get_index() != i)
                return false;
        }

        if (split->get_num_splits() != concat->get_input_size())
            return false;

        auto input = split->input_value(0);

        auto concat_axis = concat->get_axis();
        auto rank = input.get_partial_shape().rank();
        if (!rank.is_static())
            return false;
        std::vector<int64_t> order(rank.get_length());
        std::iota(order.begin(), order.end(), 0);
        order.erase(order.begin() + split_axis);
        order.insert(order.begin() + concat_axis, split_axis);

        auto transpose_order = ov::op::v0::Constant::create(element::i64, {(size_t)rank.get_length()}, order);
        auto transpose = register_new_node<ov::op::v1::Transpose>(input, transpose_order);
        auto shape_after =
            ov::op::v0::Constant::create(element::i64, {(size_t)rank.get_length() - 1}, concat->get_output_shape(0));
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose, shape_after, false);

        reshape->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(nodes_to_delete, {transpose, reshape});
        ov::replace_node(m.get_match_root(), reshape);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_pattern, matcher_name);
    register_matcher(m, callback);
}

bool is_axis_squeezed_by_node(const std::shared_ptr<ov::Node>& squeeze_node, int64_t axis, bool use_shapes) {
    const auto& input_shape = squeeze_node->get_input_partial_shape(0);
    const auto& output_shape = squeeze_node->get_output_partial_shape(0);
    if (input_shape.rank().is_dynamic() || output_shape.rank().is_dynamic())
        return false;

    auto input_rank = input_shape.rank().get_length();
    auto output_rank = output_shape.rank().get_length();
    // check if output_rank == input_rank - 1
    // to make sure the node actually squeezes a dimension
    if (input_rank != output_rank + 1)
        return false;

    // check if squeezed dimension equals to 1
    if (input_shape[axis].is_dynamic() || input_shape[axis] != 1)
        return false;

    if (ov::is_type<ov::op::v1::Reshape>(squeeze_node)) {
        if (!use_shapes)
            return false;
        // clang-format off
        // check if the dimensions surrounding squeezed axis match
        // function returns false if input_shape[:axis] != output_shape[:axis] or input_shape[(axis + 1):] != output_shape[axis:]
        // clang-format on
        if (input_shape.is_dynamic() || output_shape.is_dynamic())
            return false;

        if (!std::equal(input_shape.begin(), input_shape.begin() + axis, output_shape.begin()))
            return false;

        if (!std::equal(input_shape.begin() + axis + 1, input_shape.end(), output_shape.begin() + axis))
            return false;
    } else {
        if (squeeze_node->get_input_size() == 1) {
            // The case when Squeeze has only one input so every dimension == 1 is squeezed
            if (input_shape.is_dynamic())
                return false;
            size_t num_squeezed_axes = 0;
            for (size_t i = 0; i < input_shape.size(); i++) {
                if (input_shape[i].get_length() == 1) {
                    num_squeezed_axes++;
                    if (num_squeezed_axes > 1)
                        return false;
                    if (static_cast<int64_t>(i) != axis)
                        return false;
                }
            }
        } else {
            // The second Squeeze input has explicit axes
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(squeeze_node->get_input_node_shared_ptr(1));
            if (!constant)
                return false;
            if (ov::shape_size(constant->get_shape()) != 1)
                return false;
            auto squeezed_axis = constant->cast_vector<int64_t>()[0];
            squeezed_axis = squeezed_axis < 0 ? squeezed_axis + input_rank : squeezed_axis;
            if (axis != squeezed_axis)
                return false;
        }
    }

    return true;
}
