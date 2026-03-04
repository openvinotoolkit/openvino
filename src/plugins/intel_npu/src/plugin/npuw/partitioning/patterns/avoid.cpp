// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "avoid.hpp"

#include "../../logging.hpp"
#include "../online/group.hpp"     // online::Group
#include "../online/snapshot.hpp"  // online::Snapshot
#include "openvino/op/ops.hpp"
#include "openvino/pass/pattern/op/label.hpp"  // any_input
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace avoid {

namespace opp = ov::pass::pattern;

using NodeToGroupMap = std::unordered_map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::npuw::online::Group>>;

//------------------------------------------------------------------------------
// Common helpers for avoid-pattern callbacks
//------------------------------------------------------------------------------

// Tag a node for avoidance.  Constants and decompression Converts
// are not in the node-to-group map (isOp() excludes them) and are
// automatically colocated with their consumer by the ov::Model
// constructor, so there is no need to tag them explicitly.
static void avoid_node(const std::shared_ptr<ov::Node>& node,
                       const std::shared_ptr<NodeToGroupMap>& node_to_gptr,
                       const std::string& avoid_device) {
    auto it = node_to_gptr->find(node);
    if (it == node_to_gptr->end())
        return;
    it->second->avoid(avoid_device);
}

// WORKAROUND(bool→u8): NPU converts boolean outputs to u8 at
// partition boundaries, which the CPU partition cannot consume as
// bool.  When a tagged node produces a boolean output we also tag
// its input producers so the boolean value is produced entirely
// within the CPU partition.
// TODO(CVS-XXXXX): remove this function once NPU preserves bool
// dtype at partition boundaries, and replace all call-sites with
// plain avoid_node().
static void avoid_node_bool_wa(const std::shared_ptr<ov::Node>& node,
                               const std::shared_ptr<NodeToGroupMap>& node_to_gptr,
                               const std::string& avoid_device) {
    avoid_node(node, node_to_gptr, avoid_device);

    if (node->get_output_size() > 0 && node->get_output_element_type(0) == ov::element::boolean) {
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            auto pred = node->input_value(i).get_node_shared_ptr();
            avoid_node(pred, node_to_gptr, avoid_device);
        }
    }
}

//------------------------------------------------------------------------------
// Pattern: RMSNorm, from LLaMa-v2-7b model
//
//            Power     Const
//               :        :
//               V        V
//               ReduceMean
//                    :
//                    V
//                   Add
//                    :
//                    V
//                   Sqrt
//                    :
//                    V
//
RMSNorm::RMSNorm(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device) {
    auto power = opp::wrap_type<ov::op::v1::Power>({opp::any_input(), opp::any_input()});
    auto reduce = opp::wrap_type<ov::op::v1::ReduceMean>({power, opp::wrap_type<ov::op::v0::Constant>()});
    auto add = opp::wrap_type<ov::op::v1::Add>({reduce, opp::any_input()});
    auto sqrt = opp::wrap_type<ov::op::v0::Sqrt>({add});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_power = node_to_output.at(power).get_node_shared_ptr();
        auto matched_reduce = node_to_output.at(reduce).get_node_shared_ptr();
        auto matched_add = node_to_output.at(add).get_node_shared_ptr();
        auto matched_sqrt = node_to_output.at(sqrt).get_node_shared_ptr();

        node_to_gptr->at(matched_power)->avoid(avoid_device);
        node_to_gptr->at(matched_reduce)->avoid(avoid_device);
        node_to_gptr->at(matched_add)->avoid(avoid_device);
        node_to_gptr->at(matched_sqrt)->avoid(avoid_device);

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(sqrt, "TagRMSNormAvoid"), std::move(callback));
}

// From DeepSeek
SinCos::SinCos(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device) {
    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat_1 = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});
    auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({opp::wrap_type<ov::op::v0::Constant>(), concat_1});
    auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({opp::any_input(), opp::wrap_type<ov::op::v0::Constant>()});
    auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, convert});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    auto concat_2 = opp::wrap_type<ov::op::v0::Concat>({transpose, opp::any_input()});
    auto sin_cos = opp::wrap_type<ov::op::v0::Sin, ov::op::v0::Cos>({concat_2});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_shape_of = node_to_output.at(shape_of).get_node_shared_ptr();
        auto matched_gather = node_to_output.at(gather).get_node_shared_ptr();
        auto matched_concat_1 = node_to_output.at(concat_1).get_node_shared_ptr();
        auto matched_broadcast = node_to_output.at(broadcast).get_node_shared_ptr();
        auto matched_unsqueeze = node_to_output.at(unsqueeze).get_node_shared_ptr();
        auto matched_convert = node_to_output.at(convert).get_node_shared_ptr();
        auto matched_matmul = node_to_output.at(matmul).get_node_shared_ptr();
        auto matched_transpose = node_to_output.at(transpose).get_node_shared_ptr();
        auto matched_concat_2 = node_to_output.at(concat_2).get_node_shared_ptr();
        auto matched_sin_cos = node_to_output.at(sin_cos).get_node_shared_ptr();

        node_to_gptr->at(matched_shape_of)->avoid(avoid_device);
        node_to_gptr->at(matched_gather)->avoid(avoid_device);
        node_to_gptr->at(matched_concat_1)->avoid(avoid_device);
        node_to_gptr->at(matched_broadcast)->avoid(avoid_device);
        node_to_gptr->at(matched_unsqueeze)->avoid(avoid_device);
        node_to_gptr->at(matched_convert)->avoid(avoid_device);
        node_to_gptr->at(matched_matmul)->avoid(avoid_device);
        node_to_gptr->at(matched_transpose)->avoid(avoid_device);
        node_to_gptr->at(matched_concat_2)->avoid(avoid_device);
        node_to_gptr->at(matched_sin_cos)->avoid(avoid_device);

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(sin_cos, "TagSinCos"), std::move(callback));
}
GemmaRoPE::GemmaRoPE(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device) {
    auto power = opp::wrap_type<ov::op::v1::Power>({opp::any_input(), opp::any_input()});
    auto unsqueeze1 = opp::wrap_type<ov::op::v0::Unsqueeze>({power, opp::wrap_type<ov::op::v0::Constant>()});
    auto unsqueeze2 = opp::wrap_type<ov::op::v0::Unsqueeze>({unsqueeze1, opp::wrap_type<ov::op::v0::Constant>()});
    auto divide = opp::wrap_type<ov::op::v1::Divide>({opp::wrap_type<ov::op::v0::Convert>(), unsqueeze2});
    auto unsqueeze3 = opp::wrap_type<ov::op::v0::Unsqueeze>({divide, opp::wrap_type<ov::op::v0::Constant>()});
    auto sin_cos = opp::wrap_type<ov::op::v0::Sin, ov::op::v0::Cos>({unsqueeze3});
    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_power = node_to_output.at(power).get_node_shared_ptr();
        auto matched_unsqueeze1 = node_to_output.at(unsqueeze1).get_node_shared_ptr();
        auto matched_unsqueeze2 = node_to_output.at(unsqueeze2).get_node_shared_ptr();
        auto matched_divide = node_to_output.at(divide).get_node_shared_ptr();
        auto matched_unsqueeze3 = node_to_output.at(unsqueeze3).get_node_shared_ptr();
        auto matched_sin_cos = node_to_output.at(sin_cos).get_node_shared_ptr();

        node_to_gptr->at(matched_power)->avoid(avoid_device);
        node_to_gptr->at(matched_unsqueeze1)->avoid(avoid_device);
        node_to_gptr->at(matched_unsqueeze2)->avoid(avoid_device);
        node_to_gptr->at(matched_divide)->avoid(avoid_device);
        node_to_gptr->at(matched_unsqueeze3)->avoid(avoid_device);
        node_to_gptr->at(matched_sin_cos)->avoid(avoid_device);
        return false;
    };
    register_matcher(std::make_shared<opp::Matcher>(sin_cos, "TagGemmaRoPE"), std::move(callback));
}
//------------------------------------------------------------------------------
// Pattern: Interpolate in downsampling case
//
// Matches any Interpolate (v4/v11) node. The callback inspects the actual
// partial shapes: if at least one spatial dimension shrinks (input > output),
// the node is marked as "avoid" for the given device.
//
DownsampleInterpolate::DownsampleInterpolate(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                                             const std::string& avoid_device) {
    auto interpolate = opp::wrap_type<ov::op::v4::Interpolate, ov::op::v11::Interpolate>();

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_node = node_to_output.at(interpolate).get_node_shared_ptr();

        // Compare input and output spatial shapes to detect downsampling
        const auto& in_shape = matched_node->get_input_partial_shape(0);
        const auto& out_shape = matched_node->get_output_partial_shape(0);

        if (in_shape.rank().is_static() && out_shape.rank().is_static() &&
            in_shape.rank().get_length() == out_shape.rank().get_length()) {
            bool is_downsample = false;
            // Check spatial dimensions (skip batch and channel: indices 2..rank-1)
            for (auto i = 2; i < in_shape.rank().get_length(); ++i) {
                if (in_shape[i].is_static() && out_shape[i].is_static() &&
                    in_shape[i].get_length() > out_shape[i].get_length()) {
                    is_downsample = true;
                    break;
                }
            }
            if (is_downsample) {
                LOG_DEBUG("Avoiding downsampling Interpolate: " << matched_node->get_friendly_name());
                node_to_gptr->at(matched_node)->avoid(avoid_device);
            }
        }

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(interpolate, "TagDownsampleInterpolateAvoid"), std::move(callback));
}

//------------------------------------------------------------------------------
// Pattern: FloorMod and its direct input producer
//
// FloorMod requires FP32 precision to produce correct results.  The layer
// feeding into it typically also needs to stay in FP32 to avoid a precision
// mismatch, so both the FloorMod and its first-input producer are avoided.
//
FloorModFP32::FloorModFP32(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                           const std::string& avoid_device) {
    auto producer = opp::any_input();
    auto floor_mod = opp::wrap_type<ov::op::v1::FloorMod>({producer, opp::any_input()});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_floor_mod = node_to_output.at(floor_mod).get_node_shared_ptr();
        auto matched_producer = node_to_output.at(producer).get_node_shared_ptr();

        LOG_DEBUG("Avoiding FloorMod: " << matched_floor_mod->get_friendly_name());
        node_to_gptr->at(matched_floor_mod)->avoid(avoid_device);

        // Also avoid the direct producer of the first input if it is a real op
        auto it = node_to_gptr->find(matched_producer);
        if (it != node_to_gptr->end()) {
            LOG_DEBUG("Avoiding FloorMod producer: " << matched_producer->get_friendly_name());
            it->second->avoid(avoid_device);
        }

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(floor_mod, "TagFloorModFP32Avoid"), std::move(callback));
}

//------------------------------------------------------------------------------
// Pattern: CumSumSinGen — Sinusoidal position encoding generator
//
// From Kokoro-82M (and similar TTS models). The l_sin_gen module
// accumulates phase via CumSum, scales it, upsamples with Interpolate,
// and applies Sin. CumSum output can reach ~70 000 which overflows
// FP16 max (65 504), corrupting all downstream Sin values.
//
// Matched sub-chain (data-flow direction →):
//
//   any → CumSum → Multiply → Transpose → Multiply → Interpolate → Transpose → Sin
//
// All matched nodes are marked "avoid" so the partitioner offloads
// them to CPU (or another FP32-capable device).
//
CumSumSinGen::CumSumSinGen(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                           const std::string& avoid_device) {
    auto cumsum = opp::wrap_type<ov::op::v0::CumSum>({opp::any_input(), opp::any_input()});
    auto mul_1 = opp::wrap_type<ov::op::v1::Multiply>({cumsum, opp::any_input()});
    auto transpose_2 = opp::wrap_type<ov::op::v1::Transpose>({mul_1, opp::any_input()});
    auto mul_2 = opp::wrap_type<ov::op::v1::Multiply>({transpose_2, opp::any_input()});
    auto interpolate =
        opp::wrap_type<ov::op::v4::Interpolate, ov::op::v11::Interpolate>({mul_2, opp::any_input(), opp::any_input()});
    auto transpose_3 = opp::wrap_type<ov::op::v1::Transpose>({interpolate, opp::any_input()});
    auto sin = opp::wrap_type<ov::op::v0::Sin>({transpose_3});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_cumsum = node_to_output.at(cumsum).get_node_shared_ptr();
        auto matched_mul_1 = node_to_output.at(mul_1).get_node_shared_ptr();
        auto matched_transpose_2 = node_to_output.at(transpose_2).get_node_shared_ptr();
        auto matched_mul_2 = node_to_output.at(mul_2).get_node_shared_ptr();
        auto matched_interpolate = node_to_output.at(interpolate).get_node_shared_ptr();
        auto matched_transpose_3 = node_to_output.at(transpose_3).get_node_shared_ptr();
        auto matched_sin = node_to_output.at(sin).get_node_shared_ptr();

        // Also avoid the Transpose feeding CumSum if present
        auto cumsum_input = matched_cumsum->input_value(0).get_node_shared_ptr();
        avoid_node(cumsum_input, node_to_gptr, avoid_device);

        LOG_DEBUG("Avoiding CumSumSinGen chain: " << matched_cumsum->get_friendly_name() << " \u2192 "
                                                  << matched_sin->get_friendly_name());

        avoid_node(matched_cumsum, node_to_gptr, avoid_device);
        avoid_node(matched_mul_1, node_to_gptr, avoid_device);
        avoid_node(matched_transpose_2, node_to_gptr, avoid_device);
        avoid_node(matched_mul_2, node_to_gptr, avoid_device);
        avoid_node(matched_interpolate, node_to_gptr, avoid_device);
        avoid_node(matched_transpose_3, node_to_gptr, avoid_device);
        avoid_node(matched_sin, node_to_gptr, avoid_device);

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(sin, "TagCumSumSinGenAvoid"), std::move(callback));
}

//------------------------------------------------------------------------------
// Pattern: BoxMullerNoise — Box-Muller normal-distribution noise generator
//
// PyTorch's randn_like decomposes to two RandomUniform ops feeding a
// Box-Muller transform:
//
//   RandomUniform → Log → Multiply(×-2) → Sqrt ─┐
//                                                 ├→ Multiply (= randn output)
//   RandomUniform ───────────────────────── Cos ──┘
//
// On FP16, very small uniform values flush to zero (subnormal), causing
// Log(0) = -Inf which propagates through the entire noise path.
//
// The pattern root is the final Multiply that combines Sqrt and Cos.
// We match from the Sqrt branch upward and tag all matched nodes.
//
BoxMullerNoise::BoxMullerNoise(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                               const std::string& avoid_device) {
    auto rand_uniform_1 = opp::wrap_type<ov::op::v8::RandomUniform>();
    auto log = opp::wrap_type<ov::op::v0::Log>({rand_uniform_1});
    auto mul_neg2 = opp::wrap_type<ov::op::v1::Multiply>({log, opp::any_input()});
    auto sqrt = opp::wrap_type<ov::op::v0::Sqrt>({mul_neg2});
    auto cos = opp::wrap_type<ov::op::v0::Cos>({opp::any_input()});
    auto mul_out = opp::wrap_type<ov::op::v1::Multiply>({sqrt, cos});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_rand_1 = node_to_output.at(rand_uniform_1).get_node_shared_ptr();
        auto matched_log = node_to_output.at(log).get_node_shared_ptr();
        auto matched_mul_neg2 = node_to_output.at(mul_neg2).get_node_shared_ptr();
        auto matched_sqrt = node_to_output.at(sqrt).get_node_shared_ptr();
        auto matched_cos = node_to_output.at(cos).get_node_shared_ptr();
        auto matched_mul_out = node_to_output.at(mul_out).get_node_shared_ptr();

        // Also tag the RandomUniform feeding Cos (the second uniform sample)
        auto cos_input = matched_cos->input_value(0).get_node_shared_ptr();
        avoid_node(cos_input, node_to_gptr, avoid_device);

        LOG_DEBUG("Avoiding BoxMullerNoise chain: " << matched_rand_1->get_friendly_name() << " \u2192 "
                                                    << matched_mul_out->get_friendly_name());

        avoid_node(matched_rand_1, node_to_gptr, avoid_device);
        avoid_node(matched_log, node_to_gptr, avoid_device);
        avoid_node(matched_mul_neg2, node_to_gptr, avoid_device);
        avoid_node(matched_sqrt, node_to_gptr, avoid_device);
        avoid_node(matched_cos, node_to_gptr, avoid_device);
        avoid_node(matched_mul_out, node_to_gptr, avoid_device);

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(mul_out, "TagBoxMullerNoiseAvoid"), std::move(callback));
}

//------------------------------------------------------------------------------
// Pattern: AngleComplex — aten::angle decomposition (complex phase angle)
//
// PyTorch's torch.angle (aten::angle) on complex STFT output is decomposed
// into real/imaginary arithmetic in OpenVINO IR:
//
//   Divide(imag, real) → Atan → quadrant correction → Select cascade
//
// The quadrant correction uses Greater, Less, GreaterEqual, LogicalAnd,
// LogicalOr, Add (atan+pi), Add (atan-pi), and four cascaded Selects.
//
// On FP16, when the real part (denominator) contains values in the
// subnormal range (~9% of elements), the Divide output reaches
// ±200 000 — far exceeding FP16 max (65 504).  This corrupts Atan
// and every downstream comparison and Select.
//
// We match the core chain: Divide → Atan → ... → Select (4 deep).
// The callback also walks backward from Divide inputs and forward
// through all consumers to tag the complete angle subgraph.
//
AngleComplex::AngleComplex(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                           const std::string& avoid_device) {
    // Match the core arithmetic spine: Divide → Atan
    auto divide = opp::wrap_type<ov::op::v1::Divide>({opp::any_input(), opp::any_input()});
    auto atan = opp::wrap_type<ov::op::v0::Atan>({divide});

    // Quadrant correction: Atan feeds into two Add ops (atan+pi, atan-pi)
    auto add_pos = opp::wrap_type<ov::op::v1::Add>({atan, opp::any_input()});
    auto add_neg = opp::wrap_type<ov::op::v1::Add>({atan, opp::any_input()});

    // Select_1: picks between add_pos and add_neg based on LogicalAnd_2
    auto select_1 = opp::wrap_type<ov::op::v1::Select>({opp::any_input(), add_pos, add_neg});

    // Select_2: picks between atan and select_1 based on Greater
    auto select_2 = opp::wrap_type<ov::op::v1::Select>({opp::any_input(), atan, select_1});

    // Select_3: final output picks between a constant and select_2 based on LogicalOr
    auto select_3 = opp::wrap_type<ov::op::v1::Select>({opp::any_input(), opp::any_input(), select_2});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_divide = node_to_output.at(divide).get_node_shared_ptr();
        auto matched_atan = node_to_output.at(atan).get_node_shared_ptr();
        auto matched_add_pos = node_to_output.at(add_pos).get_node_shared_ptr();
        auto matched_add_neg = node_to_output.at(add_neg).get_node_shared_ptr();
        auto matched_select_1 = node_to_output.at(select_1).get_node_shared_ptr();
        auto matched_select_2 = node_to_output.at(select_2).get_node_shared_ptr();
        auto matched_select_3 = node_to_output.at(select_3).get_node_shared_ptr();

        LOG_DEBUG("Avoiding AngleComplex chain: " << matched_divide->get_friendly_name() << " -> "
                                                  << matched_select_3->get_friendly_name());

        // Tag core arithmetic chain (no boolean outputs here)
        avoid_node(matched_divide, node_to_gptr, avoid_device);
        avoid_node(matched_atan, node_to_gptr, avoid_device);
        avoid_node(matched_add_pos, node_to_gptr, avoid_device);
        avoid_node(matched_add_neg, node_to_gptr, avoid_device);
        avoid_node(matched_select_1, node_to_gptr, avoid_device);
        avoid_node(matched_select_2, node_to_gptr, avoid_device);
        avoid_node(matched_select_3, node_to_gptr, avoid_device);

        // Tag the condition inputs of the Selects (Greater, Less,
        // GreaterEqual, LogicalAnd, LogicalOr) and walk one more
        // level to reach their predecessors.
        // Uses avoid_node_bool_wa: these nodes may output boolean
        // which NPU converts to u8 at partition boundaries.
        for (auto sel : {matched_select_1, matched_select_2, matched_select_3}) {
            auto cond = sel->input_value(0).get_node_shared_ptr();
            avoid_node_bool_wa(cond, node_to_gptr, avoid_device);
            for (size_t i = 0; i < cond->get_input_size(); ++i) {
                auto pred = cond->input_value(i).get_node_shared_ptr();
                avoid_node_bool_wa(pred, node_to_gptr, avoid_device);
            }
        }

        // Tag the Select used as input 1 of select_3
        auto sel3_input1 = matched_select_3->input_value(1).get_node_shared_ptr();
        avoid_node(sel3_input1, node_to_gptr, avoid_device);
        // Walk Select's own condition and value inputs
        if (node_to_gptr->find(sel3_input1) != node_to_gptr->end()) {
            for (size_t i = 0; i < sel3_input1->get_input_size(); ++i) {
                auto pred = sel3_input1->input_value(i).get_node_shared_ptr();
                avoid_node_bool_wa(pred, node_to_gptr, avoid_device);
            }
        }

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(select_3, "TagAngleComplexAvoid"), std::move(callback));
}

}  // namespace avoid
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
