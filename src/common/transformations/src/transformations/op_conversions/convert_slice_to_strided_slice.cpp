// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_slice_to_strided_slice.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

namespace {
Output<ov::Node> align_indices(const Output<ov::Node>& indices,
                               const Output<ov::Node>& slice_axes,
                               const Output<ov::Node>& scatter_axis,
                               size_t slice_indices_length,
                               int64_t fill_in_value,
                               NodeVector& new_ops) {
    // Handle a case when starts/ends/steps lengths are less than provided axes
    // in order to ensure compatibility with `StridedSlice:v1` interface
    // Example:
    // data_shape: {3, 3, 3, 3}
    // starts: [1, 1] - after extending --> [0, 0, 1, 1]
    // ends: [2, 2] - after extending --> [0, 0, 2, 2]
    // steps : [1, 1] - after extending --> [1, 1, 1, 1]
    // axes: [2, 3] - apply slice values to 2 and 3 dimension of input data
    // expected_output_shape: {3, 3, 1, 1}

    const auto default_indices =
        ov::op::v0::Constant::create(indices.get_element_type(), Shape{slice_indices_length}, {fill_in_value});
    std::shared_ptr<ov::Node> adjusted_indices =
        ov::op::util::make_try_fold<ov::op::v3::ScatterUpdate>(default_indices,
                                                               slice_axes,
                                                               indices,  // updates
                                                               scatter_axis);

    if (!ov::op::util::is_constant(adjusted_indices)) {
        new_ops.push_back(default_indices);
    }
    return adjusted_indices;
}

std::vector<int64_t> axes_to_mask(const std::vector<int64_t>& axes, size_t slice_indices_length) {
    std::vector<int64_t> mask(slice_indices_length, 1);
    for (auto axis : axes) {
        mask[axis] = 0;
    }
    return mask;
}

}  // namespace

ov::pass::SliceToStridedSlice::SliceToStridedSlice(bool use_shapes) {
    MATCHER_SCOPE(SliceToStridedSlice);
    auto slice = pattern::wrap_type<ov::op::v8::Slice>();
    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto slice_node = ov::as_type_ptr<ov::op::v8::Slice>(m.get_match_root());
        if (!slice_node)
            return false;

        if (slice_node->get_input_size() < 4)
            return false;

        auto arg = slice_node->input_value(0);

        std::shared_ptr<ov::op::v0::Constant> start_const;
        std::shared_ptr<ov::op::v0::Constant> stop_const;
        std::shared_ptr<ov::op::v0::Constant> step_const;

        if (use_shapes) {
            start_const = ov::util::get_constant_from_source(slice_node->input_value(1));
            stop_const = ov::util::get_constant_from_source(slice_node->input_value(2));
            step_const = ov::util::get_constant_from_source(slice_node->input_value(3));
        } else {
            start_const = ov::as_type_ptr<ov::op::v0::Constant>(slice_node->input_value(1).get_node_shared_ptr());
            stop_const = ov::as_type_ptr<ov::op::v0::Constant>(slice_node->input_value(2).get_node_shared_ptr());
            step_const = ov::as_type_ptr<ov::op::v0::Constant>(slice_node->input_value(3).get_node_shared_ptr());
        }

        auto start_input = start_const ? start_const : slice_node->input_value(1);
        auto stop_input = stop_const ? stop_const : slice_node->input_value(2);
        auto step_input = step_const ? step_const : slice_node->input_value(3);

        std::shared_ptr<ov::op::v0::Constant> axes_const;
        if (slice_node->get_input_size() > 4) {
            axes_const = use_shapes
                             ? ov::util::get_constant_from_source(slice_node->input_value(4))
                             : ov::as_type_ptr<ov::op::v0::Constant>(slice_node->input_value(4).get_node_shared_ptr());
        } else {
            axes_const = slice_node->get_default_const_axes(start_input);
        }
        if (!axes_const)
            return false;

        const auto& data_shape = slice_node->get_input_partial_shape(0);
        auto axes_vec = axes_const->cast_vector<int64_t>();
        if (data_shape.rank().is_static()) {
            ov::util::try_normalize_axes(axes_vec, data_shape.rank(), *slice_node);
        } else {
            const bool need_normalization = std::any_of(axes_vec.begin(), axes_vec.end(), [](int64_t axis) {
                return axis < 0;
            });
            if (need_normalization)
                return false;
        }
        const size_t slice_indices_length = *std::max_element(std::begin(axes_vec), std::end(axes_vec)) + 1;
        const auto begin_end_mask = axes_to_mask(axes_vec, slice_indices_length);

        const bool are_axes_sorted = std::is_sorted(axes_vec.begin(), axes_vec.end());
        const bool are_indices_aligned = are_axes_sorted && (axes_vec.size() == slice_indices_length);

        NodeVector new_ops;
        if (!are_indices_aligned) {
            const auto scatter_axis = ov::op::v0::Constant::create(element::i32, Shape{1}, {0});
            const auto slice_axes = ov::op::v0::Constant::create(element::i64, Shape{axes_vec.size()}, axes_vec);
            new_ops.insert(new_ops.end(), {scatter_axis, slice_axes});

            start_input = align_indices(start_input, slice_axes, scatter_axis, slice_indices_length, 0, new_ops);
            stop_input = align_indices(stop_input, slice_axes, scatter_axis, slice_indices_length, 0, new_ops);
            step_input = align_indices(step_input, slice_axes, scatter_axis, slice_indices_length, 1, new_ops);
        }
        new_ops.insert(
            new_ops.end(),
            {start_input.get_node_shared_ptr(), stop_input.get_node_shared_ptr(), step_input.get_node_shared_ptr()});

        const auto strided_slice = std::make_shared<ov::op::v1::StridedSlice>(arg,
                                                                              start_input,
                                                                              stop_input,
                                                                              step_input,
                                                                              begin_end_mask,
                                                                              begin_end_mask);
        new_ops.push_back(strided_slice);

        strided_slice->set_friendly_name(slice_node->get_friendly_name());
        ov::copy_runtime_info(slice_node, new_ops);
        ov::replace_node(slice_node, strided_slice);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(slice, matcher_name);
    register_matcher(m, callback);
}
