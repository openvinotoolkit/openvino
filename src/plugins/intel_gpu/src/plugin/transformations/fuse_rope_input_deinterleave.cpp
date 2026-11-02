// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_rope_input_deinterleave.hpp"

#include <algorithm>
#include <limits>

#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
namespace {

bool is_full_axis_mask(const std::vector<int64_t>& mask, size_t rank) {
    if (mask.size() != rank || mask.back() != 0) {
        return false;
    }
    return std::all_of(mask.begin(), mask.end() - 1, [](int64_t value) {
        return value == 1;
    });
}

bool is_deinterleave_slice(const ov::Output<ov::Node>& output, int64_t begin_last) {
    const auto slice = ov::as_type_ptr<ov::op::v1::StridedSlice>(output.get_node_shared_ptr());
    if (!slice) {
        return false;
    }

    const auto& input_shape = slice->get_input_partial_shape(0);
    const auto& output_shape = slice->get_output_partial_shape(0);
    if (input_shape.rank() != 4 || output_shape.rank() != 4 || !input_shape[3].is_static() || !output_shape[3].is_static()) {
        return false;
    }

    const auto input_width = input_shape[3].get_length();
    if (output_shape[3].get_length() * 2 != input_width) {
        return false;
    }

    if (!is_full_axis_mask(slice->get_begin_mask(), 4) || !is_full_axis_mask(slice->get_end_mask(), 4) || !cldnn::all_zeroes(slice->get_new_axis_mask()) ||
        !cldnn::all_zeroes(slice->get_shrink_axis_mask()) || !cldnn::all_zeroes(slice->get_ellipsis_mask())) {
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

    auto input_pattern = any_input(rank_equals(4) && has_static_dim(3));
    auto even_pattern = wrap_type<ov::op::v1::StridedSlice>(
        {input_pattern, any_input(), any_input(), any_input()},
        consumers_count(1) && [](const ov::Output<ov::Node>& output) {
            return is_deinterleave_slice(output, 0);
        });
    auto odd_pattern = wrap_type<ov::op::v1::StridedSlice>(
        {input_pattern, any_input(), any_input(), any_input()},
        consumers_count(1) && [](const ov::Output<ov::Node>& output) {
            return is_deinterleave_slice(output, 1);
        });
    auto concat_pattern = wrap_type<ov::op::v0::Concat>(
        {even_pattern, odd_pattern},
        consumers_count(1) && rank_equals(4) && has_static_dim(3) && [](const ov::Output<ov::Node>& output) {
            const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(output.get_node_shared_ptr());
            const auto& concat_shape = output.get_partial_shape();
            const auto input_width = concat_shape[3].get_length();
            const auto axis = ov::util::try_normalize_axis(concat->get_axis(), concat_shape.rank(), *concat);
            return axis == 3 && input_width > 0 && input_width % 2 == 0;
        });
    auto rope_pattern = wrap_type<ov::op::internal::RoPE>(
        {concat_pattern, any_input(), any_input()},
        rank_equals(4) && has_static_dim(3) && [](const ov::Output<ov::Node>& output) {
            const auto rope = ov::as_type_ptr<ov::op::internal::RoPE>(output.get_node_shared_ptr());
            const auto& config = rope->get_config();
            if (config.input_interleaved || !ov::op::internal::is_plain_rotate_half(config)) {
                return false;
            }

            const auto input_width = output.get_partial_shape()[3].get_length();
            return config.rotary_ndims == static_cast<size_t>(input_width);
        });
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& matcher) {
        const auto& pattern_map = matcher.get_pattern_value_map();
        auto rope = ov::as_type_ptr<ov::op::internal::RoPE>(pattern_map.at(rope_pattern).get_node_shared_ptr());
        OPENVINO_ASSERT(rope, "RoPE node is expected");
        if (transformation_callback(rope)) {
            return false;
        }

        auto new_config = rope->get_config();
        new_config.input_interleaved = true;
        rope->input(0).replace_source_output(pattern_map.at(input_pattern));
        rope->set_config(new_config);
        rope->validate_and_infer_types();
        register_new_node(rope);
        return true;
    };

    auto matcher = std::make_shared<Matcher>(rope_pattern, "FuseRoPEInputDeinterleave");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
