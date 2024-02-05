// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_slice_n_interpolation.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "itt.hpp"
#include "utils/general_utils.h"


namespace ov {

intel_cpu::PermuteSliceAndInterpolation::PermuteSliceAndInterpolation() {
    MATCHER_SCOPE(PermuteSliceAndInterpolation);
    auto interpolate_pattern = pass::pattern::wrap_type<op::v0::Interpolate,
                                                        op::v4::Interpolate,
                                                        op::v11::Interpolate>(pass::pattern::has_static_rank());

    matcher_pass_callback callback = [=](pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto interpolate = pattern_map.at(interpolate_pattern).get_node_shared_ptr();
        if (transformation_callback(interpolate)) {
            return false;
        }

        if (interpolate->get_output_target_inputs(0).size() != 1) {
            return false;
        }

        for (size_t i = 1lu; i < interpolate->get_input_size(); ++i) {
            if (!is_type<op::v0::Constant>(interpolate->get_input_node_shared_ptr(i))) {
                return false;
            }
        }

        auto in_0 = interpolate->get_input_node_ptr(0);

        if (is_type<op::v1::Transpose>(in_0) && is_type<op::v8::Slice>(in_0->get_input_node_ptr(0))) {
            auto slice = as_type_ptr<op::v8::Slice>(in_0->get_input_node_shared_ptr(0));
            if (slice->get_input_size() < 5 ||
                slice->get_output_size() != 1 ||
                slice->get_output_target_inputs(0).size() != 1) {
                return false;
            }
            for (size_t i = 1lu; i < slice->get_input_size(); ++i) {
                if (!is_type<op::v0::Constant>(slice->get_input_node_ptr(i))) {
                    return false;
                }
            }
            auto in_rank = slice->get_input_partial_shape(0).rank().get_length();
            auto axes = (as_type<op::v0::Constant>(slice->get_input_node_ptr(4)))->cast_vector<int64_t>();
            if (!one_of(in_rank, 3, 4, 5)
                    || axes.size() != 1lu
                    || axes[0] != (in_rank - 1lu)) {
                return false;
            }

            auto ss_in_0 = slice->get_input_node_shared_ptr(0);
            replace_output_update_name(slice->output(0), slice->input_value(0));

            auto slice_inputs = slice->input_values();
            auto old_slice_axes = as_type_ptr<op::v0::Constant>(slice->get_input_node_shared_ptr(4));
            auto new_slice_axes = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{1}, 1lu);
            replace_node_update_name(old_slice_axes, new_slice_axes);

            auto intp_out = interpolate->outputs()[0];
            auto target_inputs = intp_out.get_target_inputs();
            replace_output_update_name(interpolate->output(0), slice->output(0));
            slice->set_argument(0, interpolate->output(0));

            return true;
        }

        return false;
    };

    auto m = std::make_shared<pass::pattern::Matcher>(interpolate_pattern, matcher_name);
    register_matcher(m, callback);
}

} // namespace ov
