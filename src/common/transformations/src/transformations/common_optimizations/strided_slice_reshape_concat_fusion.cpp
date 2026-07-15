// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/strided_slice_reshape_concat_fusion.hpp"

#include <cstdint>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass {

namespace {

bool get_scalar_i64(const Output<Node>& value, int64_t& out) {
    const auto constant = ov::as_type_ptr<op::v0::Constant>(value.get_node_shared_ptr());
    if (!constant || ov::shape_size(constant->get_shape()) != 1 || !constant->get_element_type().is_integral_number()) {
        return false;
    }

    out = constant->cast_vector<int64_t>()[0];
    return true;
}

bool get_vector_i64(const Output<Node>& value, std::vector<int64_t>& out) {
    const auto constant = ov::as_type_ptr<op::v0::Constant>(value.get_node_shared_ptr());
    if (!constant || !constant->get_element_type().is_integral_number()) {
        return false;
    }

    out = constant->cast_vector<int64_t>();
    return !out.empty();
}

bool get_mask(const std::vector<int64_t>& mask, size_t idx) {
    return idx < mask.size() ? mask[idx] != 0 : false;
}

bool parse_slice_window(const std::shared_ptr<ov::Node>& node, int64_t& start, int64_t& stop, Output<Node>& data) {
    if (const auto strided_slice = ov::as_type_ptr<op::v1::StridedSlice>(node)) {
        data = strided_slice->input_value(0);
        const auto pshape = data.get_partial_shape();
        if (pshape.rank().is_dynamic() || pshape.rank().get_length() != 2 || pshape[0].is_dynamic() ||
            pshape[1].is_dynamic()) {
            return false;
        }

        std::vector<int64_t> begin;
        std::vector<int64_t> end;
        std::vector<int64_t> strides;
        if (!get_vector_i64(strided_slice->input_value(1), begin) ||
            !get_vector_i64(strided_slice->input_value(2), end) ||
            !get_vector_i64(strided_slice->input_value(3), strides)) {
            return false;
        }

        if (begin.size() != 2 || end.size() != 2 || strides.size() != 2) {
            return false;
        }

        const auto& new_axis_mask = strided_slice->get_new_axis_mask();
        const auto& shrink_axis_mask = strided_slice->get_shrink_axis_mask();
        const auto& ellipsis_mask = strided_slice->get_ellipsis_mask();

        auto pred = [](int64_t a) {
            return a != 0;
        };
        if (std::any_of(new_axis_mask.begin(), new_axis_mask.end(), pred) ||
            std::any_of(shrink_axis_mask.begin(), shrink_axis_mask.end(), pred) ||
            std::any_of(ellipsis_mask.begin(), ellipsis_mask.end(), pred)) {
            return false;
        }

        if (strides[0] != 1 || strides[1] != 1) {
            return false;
        }

        const int64_t dim0 = pshape[0].get_length();
        const int64_t dim1 = pshape[1].get_length();
        const auto& begin_mask = strided_slice->get_begin_mask();
        const auto& end_mask = strided_slice->get_end_mask();

        int64_t b0 = begin[0];
        int64_t e0 = end[0];
        if (get_mask(begin_mask, 0)) {
            b0 = 0;
        } else if (b0 < 0) {
            b0 += dim0;
        }
        if (get_mask(end_mask, 0)) {
            e0 = dim0;
        } else if (e0 < 0) {
            e0 += dim0;
        }

        if (b0 != 0 || e0 != dim0) {
            return false;
        }

        if (get_mask(begin_mask, 1) || get_mask(end_mask, 1)) {
            return false;
        }

        int64_t b1 = begin[1];
        int64_t e1 = end[1];
        if (b1 < 0) {
            b1 += dim1;
        }
        if (e1 < 0) {
            e1 += dim1;
        }

        if (b1 < 0 || e1 > dim1 || e1 <= b1) {
            return false;
        }

        start = b1;
        stop = e1;
        return true;
    }

    if (const auto slice = ov::as_type_ptr<op::v8::Slice>(node)) {
        data = slice->input_value(0);
        const auto data_rank = data.get_partial_shape().rank();
        if (data_rank.is_dynamic() || data_rank.get_length() != 2) {
            return false;
        }

        int64_t axis = 0;
        int64_t step = 0;
        if (!get_scalar_i64(slice->input_value(1), start) || !get_scalar_i64(slice->input_value(2), stop) ||
            !get_scalar_i64(slice->input_value(3), step) || !get_scalar_i64(slice->input_value(4), axis)) {
            return false;
        }

        const auto rank = data_rank.get_length();
        if (!ov::util::is_axis_valid(axis, rank)) {
            return false;
        }
        axis = static_cast<int64_t>(ov::util::normalize_axis(axis, rank));

        if (axis != 1 || step != 1 || stop <= start) {
            return false;
        }

        return true;
    }

    return false;
}

bool is_reshape_to_unsqueeze_on_axis_1(const std::shared_ptr<op::v1::Reshape>& reshape) {
    const auto input_pshape = reshape->get_input_partial_shape(0);
    const auto output_pshape = reshape->get_output_partial_shape(0);
    if (input_pshape.rank().is_dynamic() || output_pshape.rank().is_dynamic()) {
        return false;
    }

    if (input_pshape.rank().get_length() != 2 || output_pshape.rank().get_length() != 3) {
        return false;
    }

    if (!input_pshape.is_static() || !output_pshape.is_static()) {
        return false;
    }

    return output_pshape[0].get_length() == input_pshape[0].get_length() && output_pshape[1].get_length() == 1 &&
           output_pshape[2].get_length() == input_pshape[1].get_length();
}

}  // namespace

StridedSliceReshapeConcatFusion::StridedSliceReshapeConcatFusion() {
    MATCHER_SCOPE(StridedSliceReshapeConcatFusion);

    auto concat_pattern = pattern::wrap_type<op::v0::Concat>(pattern::has_static_rank());

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto concat = ov::as_type_ptr<op::v0::Concat>(pattern_to_output.at(concat_pattern).get_node_shared_ptr());
        if (!concat || concat->get_input_size() < 2) {
            return false;
        }

        const auto output_rank = concat->get_output_partial_shape(0).rank();
        int64_t concat_axis = concat->get_axis();
        const auto rank = output_rank.get_length();
        if (!ov::util::is_axis_valid(concat_axis, rank)) {
            return false;
        }
        concat_axis = static_cast<int64_t>(ov::util::normalize_axis(concat_axis, rank));
        if (concat_axis != 1) {
            return false;
        }

        Output<Node> common_data;
        int64_t window_size = -1;
        std::vector<int64_t> starts;
        starts.reserve(concat->get_input_size());

        NodeVector matched_nodes{concat};

        for (size_t i = 0; i < concat->get_input_size(); ++i) {
            auto reshape = ov::as_type_ptr<op::v1::Reshape>(concat->get_input_node_shared_ptr(i));
            if (!reshape || reshape->output(0).get_target_inputs().size() != 1 ||
                !is_reshape_to_unsqueeze_on_axis_1(reshape)) {
                return false;
            }

            int64_t start = 0;
            int64_t stop = 0;
            Output<Node> data;
            auto slice_like = reshape->input_value(0).get_node_shared_ptr();
            if (slice_like->output(0).get_target_inputs().size() != 1) {
                return false;
            }
            if (!parse_slice_window(slice_like, start, stop, data)) {
                return false;
            }

            if (!common_data.get_node_shared_ptr()) {
                common_data = data;
            } else if (common_data != data) {
                return false;
            }

            const int64_t current_window = stop - start;
            if (window_size < 0) {
                window_size = current_window;
            } else if (window_size != current_window) {
                return false;
            }

            matched_nodes.push_back(std::move(reshape));
            matched_nodes.push_back(std::move(slice_like));
            starts.push_back(start);
        }

        if (window_size <= 0 || starts.empty()) {
            return false;
        }

        std::vector<int64_t> gather_indices;
        gather_indices.reserve(starts.size() * static_cast<size_t>(window_size));
        for (const auto start : starts) {
            for (int64_t i = 0; i < window_size; ++i) {
                gather_indices.push_back(start + i);
            }
        }

        auto indices = op::v0::Constant::create(element::i64,
                                                Shape{starts.size(), static_cast<size_t>(window_size)},
                                                gather_indices);
        auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<op::v8::Gather>(common_data, indices, axis, 0);
        gather->set_friendly_name(concat->get_friendly_name());

        ov::copy_runtime_info(matched_nodes, {indices, axis, gather});
        ov::replace_node(concat, gather);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(concat_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass
