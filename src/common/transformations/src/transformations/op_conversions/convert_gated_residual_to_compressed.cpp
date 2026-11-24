// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gated_residual_to_compressed.hpp"

#include <memory>
#include <optional>
#include <set>
#include <utility>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/gated_residual.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/op_conversions/convert_fc_to_compressed.hpp"
#include "transformations/pattern_blocks/compressed_weights_block.hpp"

namespace ov::pass {

namespace {

using ov::pass::pattern::op::CompressedWeightsBlock;

/// A compressed weight of an [N, K] matrix, laid out as the op takes it.
struct CompressedWeight {
    std::shared_ptr<ov::Node> weights;  // i4/u4/i8/u8 [N, K]
    std::shared_ptr<ov::Node> scale;    // floating point [N, G] or one element
    std::shared_ptr<ov::Node> zp;       // weight type [N, G] or one element; null without zero points
};

/// Prepares the weight matched by `block` (N rows). Returns nullopt for what the op would reject: a Transpose, a
/// non-floating-point scale, a scale not [N, G] or single-element, a zero point not of the weight type or not shaped
/// like the scale (or single-element).
std::optional<CompressedWeight> prepare_compressed_weight(const std::shared_ptr<CompressedWeightsBlock>& block,
                                                          const ov::pass::pattern::PatternValueMap& pm,
                                                          size_t N,
                                                          ov::NodeVector& result_nodes) {
    if (block->get_anchor("transpose", pm)) {
        return std::nullopt;
    }
    const auto weights = block->get_anchor("weights", pm).value();
    const auto scale = block->get_anchor("mul_const", pm).value();
    const auto zp = block->get_anchor("sub_const", pm);
    // Grouped weights are stored [N, G, K / G], the others [N, K].
    const bool grouped = weights.get_shape().size() == 3;

    const auto [prepared_weights, prepared_scale, prepared_zp] =
        ov::pass::ConvertFullyConnectedToFullyConnectedCompressed::process_compressed_weights(
            block,
            pm,
            /*convert_u4zp_to_u8=*/false,
            /*has_transpose=*/false,
            grouped,
            /*batched_weights=*/false,
            result_nodes);
    const auto& scale_shape = prepared_scale->get_output_shape(0);
    if (!scale.get_element_type().is_real() ||
        (ov::shape_size(scale_shape) != 1 && (scale_shape.size() != 2 || scale_shape[0] != N))) {
        return std::nullopt;
    }
    if (zp &&
        (zp->get_element_type() != weights.get_element_type() ||
         (ov::shape_size(prepared_zp->get_output_shape(0)) != 1 && prepared_zp->get_output_shape(0) != scale_shape))) {
        return std::nullopt;
    }
    // The combined groups are new constants: they take part in the runtime info merge of the converted op.
    for (const auto& [prepared, original] : {std::pair{prepared_weights, weights.get_node_shared_ptr()},
                                             std::pair{prepared_scale, scale.get_node_shared_ptr()},
                                             std::pair{prepared_zp, zp ? zp->get_node_shared_ptr() : nullptr}}) {
        if (prepared && prepared != original) {
            result_nodes.push_back(prepared);
        }
    }
    return CompressedWeight{prepared_weights, prepared_scale, zp ? prepared_zp : nullptr};
}

}  // namespace

ConvertGatedResidualToCompressed::ConvertGatedResidualToCompressed(
    const ov::element::TypeVector& supported_weights_types,
    SupportsPredicate supports_config) {
    MATCHER_SCOPE(ConvertGatedResidualToCompressed);
    using namespace ov::pass::pattern;

    const auto make_weights_block = [&] {
        return std::make_shared<CompressedWeightsBlock>(supported_weights_types, std::set<size_t>{2});
    };
    auto residual = any_input();
    auto gain = any_input();
    auto w_down = make_weights_block();
    auto w_up = make_weights_block();
    auto w_inject = make_weights_block();
    // The inputs are the floating-point form: residual, gain, W_down, W_up, with W_inject if the op produces the write
    // gate.
    auto gated_residual = wrap_type<ov::op::internal::GatedResidual>({residual, gain, w_down, w_up});
    auto gated_residual_with_gate =
        wrap_type<ov::op::internal::GatedResidual>({residual, gain, w_down, w_up, w_inject});
    auto root_pattern = gated_residual | gated_residual_with_gate;

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        const auto gr = ov::as_type_ptr<ov::op::internal::GatedResidual>(m.get_match_root());
        OPENVINO_ASSERT(gr, "ConvertGatedResidualToCompressed: matched node is expected to be GatedResidual");
        const bool with_gate = pm.count(gated_residual_with_gate) != 0;

        const auto& gr_config = gr->get_config();
        const size_t C = gr_config.hc_count;
        const size_t CD = C * gr_config.hidden_size;
        const size_t L = gr_config.lowrank;
        ov::NodeVector result_nodes;
        const auto down = prepare_compressed_weight(w_down, pm, L, result_nodes);
        const auto up = prepare_compressed_weight(w_up, pm, CD, result_nodes);
        const auto inject = with_gate ? prepare_compressed_weight(w_inject, pm, C, result_nodes) : std::nullopt;
        if (!down || !up || (with_gate && !inject)) {
            return false;
        }
        // One form for all weights: the type and the presence of zero points are shared.
        const auto same_form = [&](const CompressedWeight& other) {
            return other.weights->get_output_element_type(0) == down->weights->get_output_element_type(0) &&
                   (other.zp != nullptr) == (down->zp != nullptr);
        };
        if (!same_form(*up) || (with_gate && !same_form(*inject))) {
            return false;
        }

        // residual, gain, W_down, W_up, scales [, zero points] [, W_inject, its scale [, its zero point]]
        ov::OutputVector inputs{pm.at(residual), pm.at(gain), down->weights, up->weights, down->scale, up->scale};
        if (down->zp) {
            inputs.push_back(down->zp);
            inputs.push_back(up->zp);
        }
        if (with_gate) {
            inputs.push_back(inject->weights);
            inputs.push_back(inject->scale);
            if (inject->zp) {
                inputs.push_back(inject->zp);
            }
        }

        auto compressed = std::make_shared<ov::op::internal::GatedResidual>(inputs, gr_config);
        if (supports_config && !supports_config(compressed)) {
            return false;
        }

        result_nodes.push_back(compressed);
        compressed->set_friendly_name(gr->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result_nodes);
        ov::replace_node(gr, compressed);
        return true;
    };

    auto matcher = std::make_shared<Matcher>(root_pattern, matcher_name);
    this->register_matcher(matcher, callback);
}

}  // namespace ov::pass
