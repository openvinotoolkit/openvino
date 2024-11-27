// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/wrap_interpolate_into_transposes.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace {
std::vector<int64_t> reverse_permutation(const std::vector<int64_t>& perm) {
    if (perm.empty())
        return {};

    std::vector<int64_t> result(perm.size());
    for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
        result[perm[i]] = i;
    }

    return result;
}

std::vector<int64_t> build_transposition_for_axes(const std::vector<int64_t>& axes, size_t input_rank) {
    std::set<int64_t> non_interpolated_axes_set;
    for (size_t i = 0; i < input_rank; ++i) {
        non_interpolated_axes_set.insert(static_cast<int64_t>(i));
    }
    for (const auto& axis : axes) {
        non_interpolated_axes_set.erase(axis);
    }
    std::vector<int64_t> result(non_interpolated_axes_set.begin(), non_interpolated_axes_set.end());
    result.insert(result.end(), axes.begin(), axes.end());

    return result;
}

std::vector<int64_t> build_new_axes(size_t num_of_axes, size_t rank) {
    std::vector<int64_t> result(num_of_axes);
    std::iota(result.begin(), result.end(), static_cast<int64_t>(rank - num_of_axes));
    return result;
}
}  // namespace

ov::pass::WrapInterpolateIntoTransposes::WrapInterpolateIntoTransposes() {
    MATCHER_SCOPE(WrapInterpolateIntoTransposes);
    auto interpolate_pattern = ov::pass::pattern::wrap_type<ov::op::v4::Interpolate>();
    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto interpolate = ov::as_type_ptr<ov::op::v4::Interpolate>(m.get_match_root());
        if (!interpolate || interpolate->get_input_partial_shape(0).rank().is_dynamic() ||
            interpolate->inputs().size() != 4)
            return false;

        int64_t input_rank = interpolate->get_input_partial_shape(0).rank().get_length();
        // If the input rank is equal to 1 or 2, then such Interpolate is supported by OneDNN.
        if (input_rank < 3)
            return false;

        auto axes_node = ov::as_type_ptr<ov::op::v0::Constant>(interpolate->input_value(3).get_node_shared_ptr());
        if (!axes_node)
            return false;

        const auto axes = axes_node->cast_vector<int64_t>();
        if (static_cast<int64_t>(axes.size()) > input_rank - 2 ||
            std::all_of(axes.begin(), axes.end(), [](int64_t axis) {
                return axis != 0 && axis != 1;
            })) {
            return false;
        }

        const auto first_perm = build_transposition_for_axes(axes, input_rank);
        const auto last_perm = reverse_permutation(first_perm);

        auto first_transpose_perm = ov::op::v0::Constant::create(element::i64, {first_perm.size()}, first_perm);
        auto first_transpose =
            std::make_shared<ov::op::v1::Transpose>(interpolate->input_value(0), first_transpose_perm);
        auto new_axes = build_new_axes(axes.size(), input_rank);
        auto new_axes_node = ov::op::v0::Constant::create(element::i64, {new_axes.size()}, new_axes);
        auto new_interpolate = interpolate->clone_with_new_inputs(
            {first_transpose, interpolate->input_value(1), interpolate->input_value(2), new_axes_node});
        auto last_transpose_perm = ov::op::v0::Constant::create(element::i64, {last_perm.size()}, last_perm);
        auto last_transpose = std::make_shared<ov::op::v1::Transpose>(new_interpolate, last_transpose_perm);

        last_transpose->set_friendly_name(interpolate->get_friendly_name());
        copy_runtime_info(interpolate,
                          {first_transpose_perm,
                           first_transpose,
                           new_axes_node,
                           new_interpolate,
                           last_transpose_perm,
                           last_transpose});
        replace_node(interpolate, last_transpose);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(interpolate_pattern, matcher_name);
    register_matcher(m, callback);
}
