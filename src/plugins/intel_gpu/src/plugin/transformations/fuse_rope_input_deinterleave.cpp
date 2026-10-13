// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_rope_input_deinterleave.hpp"

#include <algorithm>
#include <limits>

#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
namespace {

bool is_zero_mask(const std::vector<int64_t>& mask) {
    return std::all_of(mask.begin(), mask.end(), [](int64_t value) {
        return value == 0;
    });
}

bool is_full_axis_mask(const std::vector<int64_t>& mask, size_t rank) {
    if (mask.size() != rank || mask.back() != 0) {
        return false;
    }
    return std::all_of(mask.begin(), mask.end() - 1, [](int64_t value) {
        return value == 1;
    });
}

bool is_deinterleave_slice(const std::shared_ptr<ov::op::v1::StridedSlice>& slice, int64_t begin_last, int64_t input_width) {
    const auto& input_shape = slice->get_input_partial_shape(0);
    const auto& output_shape = slice->get_output_partial_shape(0);
    if (input_shape.rank() != 4 || output_shape.rank() != 4 || !input_shape[3].is_static() || !output_shape[3].is_static() ||
        input_shape[3].get_length() != input_width || output_shape[3].get_length() * 2 != input_width) {
        return false;
    }

    if (!is_full_axis_mask(slice->get_begin_mask(), 4) || !is_full_axis_mask(slice->get_end_mask(), 4) || !is_zero_mask(slice->get_new_axis_mask()) ||
        !is_zero_mask(slice->get_shrink_axis_mask()) || !is_zero_mask(slice->get_ellipsis_mask())) {
        return false;
    }

    const auto begin = ov::util::get_constant_from_source(slice->input_value(1));
    const auto end = ov::util::get_constant_from_source(slice->input_value(2));
    const auto strides = ov::util::get_constant_from_source(slice->input_value(3));
    if (!begin || !end || !strides) {
        return false;
    }

    const auto begin_values = begin->cast_vector<int64_t>();
    const auto end_values = end->cast_vector<int64_t>();
    const auto stride_values = strides->cast_vector<int64_t>();
    if (begin_values.size() != 4 || end_values.size() != 4 || stride_values != std::vector<int64_t>({1, 1, 1, 2}) || begin_values.back() != begin_last) {
        return false;
    }

    const auto end_last = end_values.back();
    return end_last == input_width || end_last == std::numeric_limits<int32_t>::max() || end_last == std::numeric_limits<int64_t>::max();
}

}  // namespace

FuseRoPEInputDeinterleave::FuseRoPEInputDeinterleave() {
    using namespace ov::pass::pattern;

    auto rope_pattern = wrap_type<ov::op::internal::RoPE>();
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& matcher) {
        auto rope = ov::as_type_ptr<ov::op::internal::RoPE>(matcher.get_match_root());
        if (!rope || transformation_callback(rope) || rope->get_input_size() != 3) {
            return false;
        }

        const auto config = rope->get_config();
        if (config.is_interleaved || config.input_interleaved || config.is_chatglm || config.is_qwen || config.is_ltx_video || config.support_2d_rope ||
            config.support_3d_rope || config.input_trans0213 || config.output_trans0213 || config.slice_start != 0 || config.slice_stop != 0) {
            return false;
        }

        auto concat = ov::as_type_ptr<ov::op::v0::Concat>(rope->get_input_node_shared_ptr(0));
        if (!concat || concat->get_input_size() != 2 || concat->get_output_target_inputs(0).size() != 1) {
            return false;
        }

        const auto& concat_shape = concat->get_output_partial_shape(0);
        if (concat_shape.rank() != 4 || !concat_shape[3].is_static()) {
            return false;
        }
        const int64_t input_width = concat_shape[3].get_length();
        int64_t axis = concat->get_axis();
        if (axis < 0) {
            axis += 4;
        }
        if (axis != 3 || input_width <= 0 || input_width % 2 != 0 || config.rotary_ndims != static_cast<size_t>(input_width)) {
            return false;
        }

        auto even = ov::as_type_ptr<ov::op::v1::StridedSlice>(concat->get_input_node_shared_ptr(0));
        auto odd = ov::as_type_ptr<ov::op::v1::StridedSlice>(concat->get_input_node_shared_ptr(1));
        if (!even || !odd || even->get_output_target_inputs(0).size() != 1 || odd->get_output_target_inputs(0).size() != 1 ||
            even->input_value(0) != odd->input_value(0) || !is_deinterleave_slice(even, 0, input_width) || !is_deinterleave_slice(odd, 1, input_width)) {
            return false;
        }

        auto new_config = config;
        new_config.input_interleaved = true;
        rope->set_argument(0, even->input_value(0));
        rope->set_config(new_config);
        rope->validate_and_infer_types();
        register_new_node(rope);
        return true;
    };

    auto matcher = std::make_shared<Matcher>(rope_pattern, "FuseRoPEInputDeinterleave");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
