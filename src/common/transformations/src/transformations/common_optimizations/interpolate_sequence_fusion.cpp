// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/interpolate_sequence_fusion.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace {
using namespace ov;

bool compatible_axes(const std::vector<int64_t>& fst_axes_vector, const std::vector<int64_t>& snd_axes_vector) {
    std::set<int64_t> fst_axes_set(fst_axes_vector.begin(), fst_axes_vector.end());
    for (const auto& a : snd_axes_vector) {
        if (fst_axes_set.count(a) != 0)
            return false;
    }
    return true;
}

bool shape_calculation_mode_can_use_constant_inputs(const std::shared_ptr<ov::op::v4::Interpolate>& interpolate) {
    const auto& attrs = interpolate->get_attrs();
    if (attrs.shape_calculation_mode == ov::op::v4::Interpolate::ShapeCalcMode::SIZES) {
        return ov::as_type_ptr<ov::op::v0::Constant>(interpolate->input_value(1).get_node_shared_ptr()) != nullptr;
    }
    return ov::as_type_ptr<ov::op::v0::Constant>(interpolate->input_value(2).get_node_shared_ptr()) != nullptr;
}

bool is_candidate_for_fusion(const std::shared_ptr<ov::op::v4::Interpolate>& interpolate) {
    return (interpolate->get_input_partial_shape(0).rank().is_static()) &&
           (interpolate->inputs().size() != 4 ||
            ov::as_type_ptr<ov::op::v0::Constant>(interpolate->input_value(3).get_node_shared_ptr())) &&
           shape_calculation_mode_can_use_constant_inputs(interpolate);
}

std::vector<int64_t> get_interpolated_axes(const std::shared_ptr<ov::op::v4::Interpolate>& interpolate) {
    if (interpolate->inputs().size() != 4) {
        const auto input_rank = interpolate->get_input_partial_shape(0).rank().get_length();

        std::vector<int64_t> default_value(input_rank);
        std::iota(default_value.begin(), default_value.end(), 0);

        return default_value;
    }
    return ov::as_type_ptr<ov::op::v0::Constant>(interpolate->input_value(3).get_node_shared_ptr())
        ->cast_vector<int64_t>();
}

bool can_be_fused(const std::shared_ptr<ov::op::v4::Interpolate>& fst,
                  const std::shared_ptr<ov::op::v4::Interpolate>& snd) {
    // The first Interpolate (fst) must have only one consumer.
    for (const auto& output : fst->outputs()) {
        for (const auto& consumer : output.get_target_inputs()) {
            if (consumer.get_node() != snd.get())
                return false;
        }
    }

    if (fst->get_attrs() != snd->get_attrs() || !is_candidate_for_fusion(fst) || !is_candidate_for_fusion(snd))
        return false;

    const auto fst_axes = get_interpolated_axes(fst);
    const auto snd_axes = get_interpolated_axes(snd);
    return compatible_axes(fst_axes, snd_axes);
}

ov::NodeVector subgraph_for_sizes_calculation_mode(const std::shared_ptr<ov::op::v4::Interpolate>& fst,
                                                   const std::shared_ptr<ov::op::v4::Interpolate>& snd,
                                                   pass::MatcherPass* matcherPass) {
    const auto fst_axes = get_interpolated_axes(fst);
    const auto snd_axes = get_interpolated_axes(snd);
    const auto fst_sizes_node = ov::as_type_ptr<ov::op::v0::Constant>(fst->input_value(1).get_node_shared_ptr());
    const auto snd_sizes_node = ov::as_type_ptr<ov::op::v0::Constant>(snd->input_value(1).get_node_shared_ptr());
    if (!fst_sizes_node || !snd_sizes_node)
        return {};

    const auto fst_sizes = fst_sizes_node->cast_vector<int64_t>();
    const auto snd_sizes = snd_sizes_node->cast_vector<int64_t>();
    std::vector<std::pair<int64_t, int64_t>> axes_and_sizes;
    for (size_t i = 0; i < fst_axes.size(); ++i) {
        axes_and_sizes.emplace_back(std::make_pair(fst_axes[i], fst_sizes[i]));
    }
    for (size_t i = 0; i < snd_axes.size(); ++i) {
        axes_and_sizes.emplace_back(std::make_pair(snd_axes[i], snd_sizes[i]));
    }
    std::sort(axes_and_sizes.begin(),
              axes_and_sizes.end(),
              [](const std::pair<int64_t, int64_t>& a, const std::pair<int64_t, int64_t>& b) {
                  return a.first < b.first;
              });
    std::vector<int64_t> new_axes;
    std::vector<int64_t> new_sizes;
    for (const auto& as : axes_and_sizes) {
        new_axes.emplace_back(as.first);
        new_sizes.emplace_back(as.second);
    }

    auto new_sizes_node = ov::op::v0::Constant::create(element::i64, {new_sizes.size()}, new_sizes);
    auto new_axes_node = ov::op::v0::Constant::create(element::i64, {new_axes.size()}, new_axes);
    auto new_sizes_cast = std::make_shared<ov::op::v0::Convert>(new_sizes_node, element::f32);
    auto shape_node = std::make_shared<ov::op::v3::ShapeOf>(fst->input_value(0));

    auto gather_axis_node = ov::op::v0::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
    auto gather_node = std::make_shared<ov::op::v8::Gather>(shape_node, new_axes_node, gather_axis_node);
    auto cast_shape_to_float = std::make_shared<ov::op::v0::Convert>(gather_node, element::f32);

    auto div_node = std::make_shared<ov::op::v1::Divide>(new_sizes_cast, cast_shape_to_float);

    const auto new_interpolate = ov::as_type_ptr<ov::op::v4::Interpolate>(
        fst->clone_with_new_inputs({fst->input_value(0), new_sizes_node, div_node, new_axes_node}));
    matcherPass->register_new_node(new_interpolate);

    return {new_sizes_node,
            new_axes_node,
            new_sizes_cast,
            shape_node,
            gather_axis_node,
            gather_node,
            cast_shape_to_float,
            div_node,
            new_interpolate};
}

ov::NodeVector subgraph_for_scales_calculation_mode(const std::shared_ptr<ov::op::v4::Interpolate>& fst,
                                                    const std::shared_ptr<ov::op::v4::Interpolate>& snd,
                                                    pass::MatcherPass* matcherPass) {
    const auto fst_axes = get_interpolated_axes(fst);
    const auto snd_axes = get_interpolated_axes(snd);
    const auto fst_scales_node = ov::as_type_ptr<ov::op::v0::Constant>(fst->input_value(2).get_node_shared_ptr());
    const auto snd_scales_node = ov::as_type_ptr<ov::op::v0::Constant>(snd->input_value(2).get_node_shared_ptr());
    if (!fst_scales_node || !snd_scales_node)
        return {};

    const auto fst_scales = fst_scales_node->cast_vector<float>();
    const auto snd_scales = snd_scales_node->cast_vector<float>();
    std::vector<std::pair<int64_t, float>> axes_and_scales;
    for (size_t i = 0; i < fst_axes.size(); ++i) {
        axes_and_scales.emplace_back(std::make_pair(fst_axes[i], fst_scales[i]));
    }
    for (size_t i = 0; i < snd_axes.size(); ++i) {
        axes_and_scales.emplace_back(std::make_pair(snd_axes[i], snd_scales[i]));
    }
    std::sort(axes_and_scales.begin(),
              axes_and_scales.end(),
              [](const std::pair<int64_t, float>& a, const std::pair<int64_t, float>& b) {
                  return a.first < b.first;
              });
    std::vector<int64_t> new_axes;
    std::vector<float> new_scales;
    for (const auto& as : axes_and_scales) {
        new_axes.emplace_back(as.first);
        new_scales.emplace_back(as.second);
    }

    auto new_scales_node = ov::op::v0::Constant::create(element::f32, {new_scales.size()}, new_scales);
    auto new_axes_node = ov::op::v0::Constant::create(element::i64, {new_axes.size()}, new_axes);
    auto shape_node = std::make_shared<ov::op::v3::ShapeOf>(fst->input_value(0));

    auto gather_axis_node = ov::op::v0::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
    auto gather_node = std::make_shared<ov::op::v8::Gather>(shape_node, new_axes_node, gather_axis_node);
    auto cast_shape_to_float = std::make_shared<ov::op::v0::Convert>(gather_node, element::f32);

    auto mul_node = std::make_shared<ov::op::v1::Multiply>(cast_shape_to_float, new_scales_node);
    auto eps_node = ov::op::v0::Constant::create(element::f32, {}, std::vector<float>{1.0e-5f});
    auto add_node = std::make_shared<ov::op::v1::Multiply>(mul_node, eps_node);
    auto floor_node = std::make_shared<ov::op::v0::Floor>(add_node);
    auto cast_mul_result_to_int = std::make_shared<ov::op::v0::Convert>(floor_node, element::i64);

    const auto new_interpolate = ov::as_type_ptr<ov::op::v4::Interpolate>(
        fst->clone_with_new_inputs({fst->input_value(0), cast_mul_result_to_int, new_scales_node, new_axes_node}));
    matcherPass->register_new_node(new_interpolate);

    return {new_scales_node,
            new_axes_node,
            shape_node,
            gather_axis_node,
            gather_node,
            cast_shape_to_float,
            mul_node,
            eps_node,
            add_node,
            floor_node,
            cast_mul_result_to_int,
            new_interpolate};
}
}  // namespace

ov::pass::InterpolateSequenceFusion::InterpolateSequenceFusion() {
    MATCHER_SCOPE(InterpolateSequenceFusion);
    auto interpolate_pattern = ov::pass::pattern::wrap_type<ov::op::v4::Interpolate>();
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto snd_interpolate = ov::as_type_ptr<ov::op::v4::Interpolate>(m.get_match_root());
        if (!snd_interpolate)
            return false;

        auto fst_interpolate =
            ov::as_type_ptr<ov::op::v4::Interpolate>(snd_interpolate->input_value(0).get_node_shared_ptr());
        if (!fst_interpolate)
            return false;

        if (!can_be_fused(fst_interpolate, snd_interpolate))
            return false;

        NodeVector new_subgraph;
        if (fst_interpolate->get_attrs().shape_calculation_mode == ov::op::v4::Interpolate::ShapeCalcMode::SIZES) {
            new_subgraph = subgraph_for_sizes_calculation_mode(fst_interpolate, snd_interpolate, this);
        } else {
            new_subgraph = subgraph_for_scales_calculation_mode(fst_interpolate, snd_interpolate, this);
        }
        if (new_subgraph.empty())
            return false;

        auto& new_interpolate = new_subgraph.back();
        new_interpolate->set_friendly_name(snd_interpolate->get_friendly_name());
        copy_runtime_info({fst_interpolate, snd_interpolate}, new_subgraph);
        replace_node(snd_interpolate, new_interpolate);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(interpolate_pattern, matcher_name);
    register_matcher(m, callback);
}
