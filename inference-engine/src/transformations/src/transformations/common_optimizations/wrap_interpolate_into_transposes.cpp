// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/wrap_interpolate_into_transposes.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <tuple>
#include <set>
#include <utility>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

namespace {
bool transformation_is_applicable(const std::shared_ptr<ngraph::opset8::Interpolate>& interpolate) {
    std::cout << "Check whether Interpolate wrapping is applicable..\n";
    if (interpolate->get_input_partial_shape(0).rank().is_dynamic() || interpolate->inputs().size() != 4) return false;

    int64_t input_rank = interpolate->get_input_partial_shape(0).rank().get_length();
    std::cout <<"input_rank: " << input_rank << "\n";
    if (input_rank < 4) return false;

    auto axes_node = std::dynamic_pointer_cast<ngraph::opset8::Constant>(interpolate->input_value(3).get_node_shared_ptr());
    if (!axes_node) return false;

    const auto axes = axes_node->cast_vector<int64_t>();
    std::cout << "static_cast<int64_t>(axes.size()) > input_rank - 2: " << ((static_cast<int64_t>(axes.size()) > input_rank - 2) ? "true\n" : "false\n");
    if (static_cast<int64_t>(axes.size()) > input_rank - 2) return false;
    std::cout << "axes: ";
    for (const auto a : axes) {
        std::cout << a << " ";
    }
    std::cout << "\n";
    std::cout << "std::any_of(axes.begin(), axes.end(), [](int64_t axis){ return axis != 0 && axis != 1; }): "
              << ((std::any_of(axes.begin(), axes.end(), [](int64_t axis){ return axis != 0 && axis != 1; })) ? "true\n" : "false\n");
    std::cout << "std::any_of(axes.begin(), axes.end(), [](int64_t axis){ return axis == 0 || axis == 1; }): "
              << ((std::any_of(axes.begin(), axes.end(), [](int64_t axis){ return axis == 0 || axis == 1; })) ? "true\n" : "false\n");

    return std::any_of(axes.begin(), axes.end(), [](int64_t axis){ return axis == 0 || axis == 1; });
}

std::vector<int64_t> reverse_permutation(const std::vector<int64_t>& perm) {
    if (perm.empty()) return {};

    std::vector<int64_t> result(perm.size());
    for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
        result[perm[i]] = i;
    }

    return result;
}

std::vector<int64_t> new_non_interpolated_axes(const std::vector<int64_t>& axes, int64_t input_rank) {
    std::vector<int64_t> result {0, 1};
    for (int64_t i = 2 + static_cast<int64_t>(axes.size()); i < input_rank; ++i) {
        result.push_back(i);
    }
    return result;
}

std::set<int64_t> create_set_of_all_axes(int64_t input_rank) {
    std::set<int64_t> result;
    for (int64_t i = 0; i < input_rank; ++i) {
        result.insert(i);
    }
    return result;
}

std::vector<int64_t> old_non_interpolated_axes(const std::vector<int64_t>& axes, int64_t input_rank) {
    auto non_interpolated_axes = create_set_of_all_axes(input_rank);
    for (auto axis : axes) {
        non_interpolated_axes.erase(axis);
    }
    return std::vector<int64_t>(non_interpolated_axes.begin(), non_interpolated_axes.end());
}

std::vector<int64_t> build_transposition_for_axes(const std::vector<int64_t>& axes, int64_t input_rank) {
    const auto new_non_interpolated_axes_vector = new_non_interpolated_axes(axes, input_rank);
    const auto old_non_interpolated_axes_vector = old_non_interpolated_axes(axes, input_rank);
    const std::set<int64_t> old_interpolated_axes_set(axes.begin(), axes.end());

    std::vector<int64_t> result(input_rank);

    int64_t idx = 2;
    for (auto axis : old_interpolated_axes_set) {
        result[axis] = idx++;
    }

    for (uint64_t i = 0; i < static_cast<uint64_t>(new_non_interpolated_axes_vector.size()); ++i) {
        result[old_non_interpolated_axes_vector[i]] = new_non_interpolated_axes_vector[i];
    }

    return result;
}

// std::shared_ptr<ngraph::opset8::Constant> build_new_axes_node(const std::vector<int64_t>& axes, const std::vector<int64_t>& perm) {
//     auto value = axes;
//     for (uint64_t i = 0; i < static_cast<uint64_t>(axes.size()); ++i) {
//         value[i] = perm[axes[i]];
//     }
//     return ngraph::opset8::Constant::create(ngraph::element::i64, {axes.size()}, value);
// }

std::vector<int64_t> build_new_axes(const std::vector<int64_t>& axes, const std::vector<int64_t>& perm) {
    auto value = axes;
    for (uint64_t i = 0; i < static_cast<uint64_t>(axes.size()); ++i) {
        value[i] = perm[axes[i]];
    }
    return value;
}
} // namespace

NGRAPH_RTTI_DEFINITION(ngraph::pass::WrapInterpolateIntoTransposes, "WrapInterpolateIntoTransposes", 0);

ngraph::pass::WrapInterpolateIntoTransposes::WrapInterpolateIntoTransposes() {
    MATCHER_SCOPE(WrapInterpolateIntoTransposes);
    auto interpolate_pattern = ngraph::pattern::wrap_type<ngraph::opset8::Interpolate>();
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto interpolate = std::dynamic_pointer_cast<opset8::Interpolate>(m.get_match_root());
        if (!interpolate || !transformation_is_applicable(interpolate)) return false;

        auto axes_node = std::dynamic_pointer_cast<ngraph::opset8::Constant>(interpolate->input_value(3).get_node_shared_ptr());
        if (!axes_node) return false;
        std::cout << "Transformation is applicable.\n";

        std::cout << "Source axes: ";
        const auto axes = axes_node->cast_vector<int64_t>();
        for (const auto a : axes) {
            std::cout << a << " ";
        }
        std::cout << "\n";
        const int64_t input_rank = interpolate->get_input_partial_shape(0).rank().get_length();
        std::cout << "input_rank: " << input_rank << "\n";
        const auto first_perm = build_transposition_for_axes(axes, input_rank);
        const auto last_perm = reverse_permutation(first_perm);
        std::cout << "first_perm: ";
        for (const auto a : first_perm) {
            std::cout << a << " ";
        }
        std::cout << "\n";
        std::cout << "last_perm: ";
        for (const auto a : last_perm) {
            std::cout << a << " ";
        }
        std::cout << "\n";
        std::cout << "Source input shape: " << interpolate->get_input_partial_shape(0) << "\n";

        auto first_transpose_perm = opset8::Constant::create(element::i64, {first_perm.size()}, first_perm);
        auto first_transpose = std::make_shared<opset8::Transpose>(interpolate->input_value(0), first_transpose_perm);
        auto new_axes = build_new_axes(axes, first_perm);
        std::cout << "new_axes: ";
        for (const auto a : new_axes) {
            std::cout << a << " ";
        }
        std::cout << "\n";
        auto new_axes_node = opset8::Constant::create(element::i64, {new_axes.size()}, new_axes);
//         auto new_interpolate = std::make_shared<opset8::Interpolate>(first_transpose, interpolate->input_value(1), interpolate->input_value(2),
//                                                                      build_new_axes_node(axes, first_perm), interpolate->get_attrs());
        auto new_interpolate = interpolate->clone_with_new_inputs({first_transpose, interpolate->input_value(1), interpolate->input_value(2), new_axes_node});
//         auto new_interpolate = std::make_shared<opset8::Interpolate>(first_transpose, interpolate->input_value(1), interpolate->input_value(2),
//                                                                      new_axes_node, interpolate->get_attrs());
        auto last_transpose_perm = opset8::Constant::create(element::i64, {last_perm.size()}, last_perm);
        auto last_transpose = std::make_shared<opset8::Transpose>(new_interpolate, last_transpose_perm);

        last_transpose->set_friendly_name(interpolate->get_friendly_name());
//         copy_runtime_info(interpolate, {first_transpose_perm, first_transpose, new_interpolate, last_transpose_perm, last_transpose});
        copy_runtime_info(interpolate, {first_transpose_perm, first_transpose, new_axes_node, new_interpolate, last_transpose_perm, last_transpose});
        replace_node(interpolate, last_transpose);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(interpolate_pattern, matcher_name);
    register_matcher(m, callback);
}
