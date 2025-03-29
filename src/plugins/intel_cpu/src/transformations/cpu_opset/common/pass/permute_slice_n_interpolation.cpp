// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_slice_n_interpolation.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils/general_utils.h"

namespace ov {

intel_cpu::PermuteSliceAndInterpolation::PermuteSliceAndInterpolation() {
    MATCHER_SCOPE(PermuteSliceAndInterpolation);

    element::TypeVector param_precisions{element::i8, element::u8};
    auto input_m = pass::pattern::wrap_type<op::v0::Parameter>(pass::pattern::type_matches_any(param_precisions));
    auto const_m = pass::pattern::wrap_type<op::v0::Constant>();
    auto slice_m = pass::pattern::wrap_type<op::v8::Slice>({input_m, const_m, const_m, const_m, const_m},
                                                           ov::pass::pattern::consumers_count(1));
    auto transpose_m =
        pass::pattern::wrap_type<op::v1::Transpose>({slice_m, const_m}, ov::pass::pattern::consumers_count(1));
    auto interpolate_m = pass::pattern::wrap_type<op::v0::Interpolate, op::v4::Interpolate, op::v11::Interpolate>(
        {transpose_m, const_m, const_m, const_m},
        pass::pattern::consumers_count(1));

    matcher_pass_callback callback = [=](pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& input = pattern_map.at(input_m);
        auto interpolate = pattern_map.at(interpolate_m).get_node_shared_ptr();
        auto transpose = pattern_map.at(transpose_m).get_node_shared_ptr();
        auto slice = pattern_map.at(slice_m).get_node_shared_ptr();

        // Check Slice axes
        auto in_rank = slice->get_input_partial_shape(0).rank();
        if (in_rank == Rank::dynamic()) {
            return false;
        }
        auto axes = (as_type<op::v0::Constant>(slice->get_input_node_ptr(4)))->cast_vector<int64_t>();
        if (axes[0] < 0L) {
            axes[0] += in_rank.get_length();
        }
        if (!one_of(in_rank.get_length(), 3L, 4L, 5L) || axes.size() != 1L || axes[0] != (in_rank.get_length() - 1L)) {
            return false;
        }
        // Check Transpose order
        auto order = (as_type<op::v0::Constant>(transpose->get_input_node_ptr(1)))->cast_vector<int64_t>();
        if (!one_of(order,
                    std::vector<int64_t>{0, 2, 1},
                    std::vector<int64_t>{0, 3, 1, 2},
                    std::vector<int64_t>{0, 4, 1, 2, 3})) {
            return false;
        }

        auto transpose_inputs = transpose->input_values();
        transpose_inputs[0] = input;
        const auto new_transpose = transpose->clone_with_new_inputs(transpose_inputs);
        copy_runtime_info(transpose, new_transpose);

        auto interpolate_inputs = interpolate->input_values();
        interpolate_inputs[0] = new_transpose->output(0);
        const auto new_interpolate = interpolate->clone_with_new_inputs(interpolate_inputs);
        copy_runtime_info(interpolate, new_interpolate);

        auto slice_inputs = slice->input_values();
        auto new_slice_axes = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{1}, 1lu);

        slice_inputs[0] = new_interpolate;
        slice_inputs[4] = new_slice_axes;

        auto new_slice = slice->clone_with_new_inputs(slice_inputs);
        replace_output_update_name(interpolate->output(0), new_slice->output(0));
        copy_runtime_info({slice, interpolate}, new_interpolate);

        return true;
    };

    auto m = std::make_shared<pass::pattern::Matcher>(interpolate_m, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov
