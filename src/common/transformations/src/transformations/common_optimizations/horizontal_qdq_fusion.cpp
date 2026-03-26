// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/horizontal_qdq_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace {

// Represents a dequantization subgraph starting from a consumer of convert2:
// [Subtract(zero_point)] -> Multiply(scale)
struct DQSubgraph {
    std::shared_ptr<ov::Node> subtract;  // Optional Subtract node (nullptr if absent)
    std::shared_ptr<ov::Node> multiply;  // Multiply (scale) node - the output of the DQ chain

    ov::Output<ov::Node> zero_point;  // zero_point input to Subtract (invalid if no subtract)
    ov::Output<ov::Node> scale;       // scale input to Multiply
};

// Check if two outputs represent the same constant value by building a Subtract
// and attempting to constant-fold it to zero.
bool outputs_are_equal(const ov::Output<ov::Node>& a, const ov::Output<ov::Node>& b) {
    auto diff = std::make_shared<ov::op::v1::Subtract>(a, b);
    auto folded = ov::util::get_constant_from_source(diff);
    if (!folded)
        return false;
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(folded);
    if (!constant)
        return false;
    float value;
    if (!ov::op::util::get_single_value(constant, value))
        return false;
    return value == 0.0f;
}

bool dq_subgraphs_are_identical(const DQSubgraph& a, const DQSubgraph& b) {
    // Both must have the same structure (both with or without subtract)
    const bool a_has_sub = (a.subtract != nullptr);
    const bool b_has_sub = (b.subtract != nullptr);
    if (a_has_sub != b_has_sub)
        return false;

    // Compare zero points
    if (a_has_sub) {
        if (!outputs_are_equal(a.zero_point, b.zero_point))
            return false;
    }

    // Compare scales
    if (!outputs_are_equal(a.scale, b.scale))
        return false;

    return true;
}

// Try to extract a DQ subgraph from a consumer of convert2.
// The consumer should be either Subtract (with Multiply after) or Multiply directly.
// Pattern: convert2 -> Subtract(zero_point) -> Multiply(scale)
// or:      convert2 -> Multiply(scale)
std::optional<DQSubgraph> try_extract_dq(const ov::Input<ov::Node>& consumer_input,
                                         const std::shared_ptr<ov::Node>& convert2) {
    auto consumer_node = consumer_input.get_node()->shared_from_this();

    // Try: convert2 -> Subtract -> Multiply
    if (auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(consumer_node)) {
        if (subtract->get_input_node_shared_ptr(0) != convert2)
            return std::nullopt;

        auto sub_consumers = subtract->get_output_target_inputs(0);
        if (sub_consumers.size() != 1)
            return std::nullopt;

        auto mul_node = sub_consumers.begin()->get_node()->shared_from_this();
        auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(mul_node);
        if (!multiply || multiply->get_input_node_shared_ptr(0) != subtract)
            return std::nullopt;

        DQSubgraph dq;
        dq.subtract = subtract;
        dq.multiply = multiply;
        dq.zero_point = subtract->input_value(1);
        dq.scale = multiply->input_value(1);
        return dq;
    }

    // Try: convert2 -> Multiply (no subtract)
    if (auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(consumer_node)) {
        if (multiply->get_input_node_shared_ptr(0) != convert2)
            return std::nullopt;

        DQSubgraph dq;
        dq.subtract = nullptr;
        dq.multiply = multiply;
        dq.scale = multiply->input_value(1);
        return dq;
    }

    return std::nullopt;
}

}  // namespace

namespace ov::pass {

HorizontalQDQFusion::HorizontalQDQFusion(const ov::element::TypeVector& supported_low_precisions,
                                          const ov::element::TypeVector& supported_original_precisions) {
    MATCHER_SCOPE(HorizontalQDQFusion);
    auto data_pattern = pattern::any_input(pattern::type_matches_any(supported_original_precisions));
    auto input_low_pattern = pattern::any_input();
    auto input_high_pattern = pattern::any_input();
    auto output_low_pattern = pattern::wrap_type<ov::op::v0::Constant>();
    auto output_high_pattern = pattern::wrap_type<ov::op::v0::Constant>();
    auto fq_pattern = pattern::wrap_type<ov::op::v0::FakeQuantize>(
        {data_pattern, input_low_pattern, input_high_pattern, output_low_pattern, output_high_pattern});

    // convert1_pattern: Convert from original to low precision
    auto convert1_predicate = pattern::type_matches_any(supported_low_precisions);
    auto convert1_pattern =
        pattern::wrap_type<ov::op::v0::Convert>(ov::OutputVector{fq_pattern}, convert1_predicate);

    // convert2_pattern: Convert back to original precision, must have more than 1 consumer
    // (SharedOpOptimization merges identical Convert2 nodes, so convert2 accumulates all DQ branches)
    auto convert2_predicate =
        pattern::type_matches_any(supported_original_precisions) && pattern::consumers_more_than(1);
    auto convert2_pattern =
        pattern::wrap_type<ov::op::v0::Convert>(ov::OutputVector{convert1_pattern}, convert2_predicate);

    auto zero_point_pattern = pattern::any_input();
    auto sub_pattern =
        pattern::optional<ov::op::v1::Subtract>(ov::OutputVector{convert2_pattern, zero_point_pattern});
    auto scale_pattern = pattern::any_input();
    auto mul_pattern =
        pattern::wrap_type<ov::op::v1::Multiply>(ov::OutputVector{sub_pattern, scale_pattern});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto convert2 = pattern_map.at(convert2_pattern).get_node_shared_ptr();
        auto mul = pattern_map.at(mul_pattern).get_node_shared_ptr();

        // Collect all DQ subgraphs originating from convert2's consumers.
        // Each consumer of convert2 should start a DQ chain:
        // [Subtract(zero_point)] -> Multiply(scale)
        std::vector<DQSubgraph> dq_subgraphs;

        auto convert2_consumers = convert2->get_output_target_inputs(0);
        for (const auto& consumer_input : convert2_consumers) {
            auto dq = try_extract_dq(consumer_input, convert2);
            if (dq.has_value()) {
                dq_subgraphs.push_back(std::move(dq.value()));
            }
        }

        if (dq_subgraphs.size() <= 1)
            return false;

        // Find the "reference" DQ subgraph (the one matched by the pattern)
        DQSubgraph* reference = nullptr;
        for (auto& dq : dq_subgraphs) {
            if (dq.multiply == mul) {
                reference = &dq;
                break;
            }
        }

        if (!reference)
            return false;

        bool changed = false;

        // Find all DQ subgraphs identical to the reference and replace them
        for (auto& dq : dq_subgraphs) {
            if (dq.multiply == reference->multiply)
                continue;  // Skip the reference itself

            if (!dq_subgraphs_are_identical(*reference, dq))
                continue;

            // Replace duplicate DQ output (multiply) with the reference DQ output (multiply)
            ov::replace_output_update_name(dq.multiply->output(0), reference->multiply->output(0));
            changed = true;
        }

        return changed;
    };

    auto m = std::make_shared<pattern::Matcher>(mul_pattern, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
