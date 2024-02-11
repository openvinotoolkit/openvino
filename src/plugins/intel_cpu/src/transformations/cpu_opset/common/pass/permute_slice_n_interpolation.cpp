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

    element::TypeVector param_precisions{ element::i8, element::u8 };
    auto input_m = pass::pattern::wrap_type<op::v0::Parameter>(pass::pattern::type_matches_any(param_precisions));
    auto const_m = pass::pattern::wrap_type<op::v0::Constant>();
    auto slice_m = pass::pattern::wrap_type<op::v8::Slice>({input_m, const_m, const_m, const_m, const_m});
    auto transpose_m = pass::pattern::wrap_type<op::v1::Transpose>({slice_m, const_m});
    auto interpolate_m = pass::pattern::wrap_type<op::v0::Interpolate,
                                                  op::v4::Interpolate,
                                                  op::v11::Interpolate>({transpose_m, const_m, const_m, const_m});

    matcher_pass_callback callback = [=](pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto interpolate = pattern_map.at(interpolate_m).get_node_shared_ptr();

        if (interpolate->get_output_size() != 1lu) {
            return false;
        }

        auto slice = as_type_ptr<op::v8::Slice>(interpolate->get_input_node_ptr(0)->get_input_node_shared_ptr(0));
        auto in_rank = static_cast<int64_t>(slice->get_input_partial_shape(0).rank().get_length());
        auto axes = (as_type<op::v0::Constant>(slice->get_input_node_ptr(4)))->cast_vector<int64_t>();
        if (axes[0] < 0L) {
            axes[0] += in_rank;
        }
        if (!one_of(in_rank, 3L, 4L, 5L)
                || axes.size() != 1L
                || axes[0] != (in_rank - 1L)) {
            return false;
        }

        // Remove Slice op
        const auto slice_out_tensor_names = slice->get_output_tensor(0).get_names();
        slice->get_output_target_inputs(0).begin()->replace_source_output(slice->get_input_source_output(0));

        auto intp_out_ins = interpolate->get_output_target_inputs(0);
        std::unordered_set<std::string> result_tensor_names;
        if (std::any_of(intp_out_ins.cbegin(), intp_out_ins.cend(), [](const Input<Node>& consumer) {
                return ov::is_type<op::v0::Result>(consumer.get_node());
            })) {
            result_tensor_names = interpolate->get_output_tensor(0).get_names();
        }

        auto slice_inputs = slice->input_values();
        auto new_slice_axes = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{1}, 1lu);

        slice_inputs[0] = interpolate;
        slice_inputs[4] = new_slice_axes;

        auto new_slice = slice->clone_with_new_inputs(slice_inputs);
        for (auto inp : intp_out_ins) {
            inp.replace_source_output(new_slice->output(0));
        }

        interpolate->get_output_tensor(0).set_names(slice_out_tensor_names);
        if (!result_tensor_names.empty()) {
            new_slice->get_output_tensor(0).add_names(result_tensor_names);
        }
        copy_runtime_info({new_slice, interpolate}, new_slice);

        return true;
    };

    auto m = std::make_shared<pass::pattern::Matcher>(interpolate_m, matcher_name);
    register_matcher(m, callback);
}

} // namespace ov
