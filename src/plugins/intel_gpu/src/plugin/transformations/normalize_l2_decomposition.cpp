// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalize_l2_decomposition.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {
    std::shared_ptr<ov::Node> create_eps_node(const std::shared_ptr<ov::Node>& reduce_sum,
                                             const std::shared_ptr<ov::op::v0::NormalizeL2>& normalize_l2,
                                             ov::element::Type eps_type) {
        auto eps_const_node = ov::op::v0::Constant::create(eps_type, Shape{}, {normalize_l2->get_eps()});

        switch (normalize_l2->get_eps_mode()) {
        case ov::op::EpsMode::ADD:
            return std::make_shared<ov::op::v1::Add>(reduce_sum, eps_const_node);
        case ov::op::EpsMode::MAX:
            return std::make_shared<ov::op::v1::Maximum>(reduce_sum, eps_const_node);
        default:
            return nullptr;
        }
    }
} // namespace

NormalizeL2LastAxisToFeature::NormalizeL2LastAxisToFeature() {
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v0::NormalizeL2>();
    matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& matcher) {
        const auto normalize = ov::as_type_ptr<ov::op::v0::NormalizeL2>(matcher.get_match_root());
        if (!normalize || transformation_callback(normalize)) {
            return false;
        }

        const auto& partial_shape = normalize->get_input_partial_shape(0);
        const auto type = normalize->get_input_element_type(0);
        if (!partial_shape.is_static() || (partial_shape.size() != 3 && partial_shape.size() != 4) ||
            (type != ov::element::f16 && type != ov::element::f32) ||
            normalize->get_eps_mode() != ov::op::EpsMode::ADD) {
            return false;
        }
        const auto axes_node = ov::as_type_ptr<ov::op::v0::Constant>(normalize->get_input_node_shared_ptr(1));
        if (!axes_node) {
            return false;
        }
        const auto axes = axes_node->cast_vector<int64_t>();
        if (axes.size() != 1 || (axes[0] != -1 && axes[0] != static_cast<int64_t>(partial_shape.size() - 1))) {
            return false;
        }
        const auto shape = partial_shape.to_shape();
        if (ov::shape_size(shape) == 0) {
            return false;
        }

        const std::vector<int64_t> feature_shape{ static_cast<int64_t>(ov::shape_size(shape) / shape.back()),
                                                  static_cast<int64_t>(shape.back()),
                                                  1,
                                                  1 };
        const auto input_pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 4 }, feature_shape);
        const auto input_reshape = std::make_shared<ov::op::v1::Reshape>(normalize->input_value(0), input_pattern, false);
        const auto feature_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 1 });
        const auto feature_normalize = std::make_shared<ov::op::v0::NormalizeL2>(input_reshape,
            feature_axis,
            normalize->get_eps(),
            normalize->get_eps_mode());
        const auto output_pattern = ov::op::v0::Constant::create(ov::element::i64,
            ov::Shape{ shape.size() },
            std::vector<int64_t>(shape.begin(), shape.end()));
        const auto output_reshape = std::make_shared<ov::op::v1::Reshape>(feature_normalize, output_pattern, false);
        input_reshape->set_friendly_name(normalize->get_friendly_name() + "/to_feature");
        feature_normalize->set_friendly_name(normalize->get_friendly_name() + "/feature_normalize");
        output_reshape->set_friendly_name(normalize->get_friendly_name());
        ov::copy_runtime_info(normalize, { input_reshape, feature_normalize, output_reshape });
        ov::replace_node(normalize, output_reshape);
        return true;
        };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "NormalizeL2LastAxisToFeatureGPU"), callback);
}

NormalizeL2Decomposition::NormalizeL2Decomposition(bool use_fp32_reducesum) {
    auto normalize_l2_pattern = ov::pass::pattern::wrap_type<ov::op::v0::NormalizeL2>();

    matcher_pass_callback callback = [this, use_fp32_reducesum](ov::pass::pattern::Matcher& m) {
        auto normalize_l2 = std::dynamic_pointer_cast<ov::op::v0::NormalizeL2>(m.get_match_root());

        if (!normalize_l2 || transformation_callback(normalize_l2)) {
            return false;
        }

        // Create power operation
        auto input_type = normalize_l2->get_input_element_type(0);

        std::shared_ptr<ov::Node> input = normalize_l2->input_value(0).get_node_shared_ptr();
        // Convert to fp32 if requested
        if (use_fp32_reducesum) {
            input = std::make_shared<ov::op::v0::Convert>(normalize_l2->input_value(0), ov::element::f32);
        }

        ov::element::Type decomposition_data_type = use_fp32_reducesum ? ov::element::f32 : input_type;

        auto power = std::make_shared<ov::op::v1::Power>(input,
            ov::op::v0::Constant::create(decomposition_data_type, Shape{}, {2.0}));

        // Create reduce sum
        auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(
            power, normalize_l2->input_value(1), true);


        auto eps_node = create_eps_node(reduce_sum, normalize_l2, decomposition_data_type);
        if (!eps_node) {
            return false;
        }

        // Create sqrt
        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(eps_node);

        // Create division
        std::shared_ptr<ov::Node> dividend = normalize_l2->input_value(0).get_node_shared_ptr();
        std::shared_ptr<ov::Node> final_convert = nullptr;

        auto div = std::make_shared<ov::op::v1::Divide>(input, sqrt);
        div->set_friendly_name(normalize_l2->get_friendly_name());

        std::shared_ptr<ov::Node> final_result = div;

        // Convert back to original type if we used fp32
        if (use_fp32_reducesum && input_type != ov::element::f32) {
            final_convert = std::make_shared<ov::op::v0::Convert>(div, input_type);
            final_result = final_convert;
        }

        // Copy runtime info and replace node
        std::vector<std::shared_ptr<ov::Node>> new_ops = {power, reduce_sum, eps_node, sqrt, div};
        if (final_convert) new_ops.push_back(final_convert);

        ov::copy_runtime_info(normalize_l2, new_ops);
        ov::replace_node(normalize_l2, final_result);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(normalize_l2_pattern, "NormalizeL2DecompositionGPU");
    register_matcher(m, callback);
}

} // namespace ov::intel_gpu
