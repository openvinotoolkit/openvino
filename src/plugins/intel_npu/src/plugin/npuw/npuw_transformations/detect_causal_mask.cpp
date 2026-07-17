// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detect_causal_mask.hpp"

#include <cstdlib>

#include "openvino/op/ops.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

// Builds the Range chain shared by causal/sliding window mask pattern below:
//
//   Range(start, stop, step)
//     -> opt Add(range, offset)
//     -> opt Unsqueeze (up to 3x)
//     -> opt Reshape
//     -> opt Convert
//
// Real model shapes covered by this single chain:
//   Range -> Unsqueeze x(0-3) -> opt Convert          (Llama, Tril, Whisper)
//   Range -> Add(range, offset) -> Reshape            (MiniCPM, Q side)
//   Range -> Add(range, offset) -> Unsqueeze x3       (Gemma-4 cache_position)
std::shared_ptr<ov::Node> make_range_chain() {
    auto range = opp::wrap_type<ov::op::v4::Range>({opp::any_input(), opp::any_input(), opp::any_input()});
    auto add = opp::optional<ov::op::v1::Add>({range, opp::any_input()});
    auto unsqueeze1 = opp::optional<ov::op::v0::Unsqueeze>({add, opp::any_input()});
    auto unsqueeze2 = opp::optional<ov::op::v0::Unsqueeze>({unsqueeze1, opp::any_input()});
    auto unsqueeze3 = opp::optional<ov::op::v0::Unsqueeze>({unsqueeze2, opp::any_input()});
    auto reshape = opp::optional<ov::op::v1::Reshape>({unsqueeze3, opp::any_input()});
    auto convert = opp::optional<ov::op::v0::Convert>({reshape});
    return convert;
}

int64_t get_window_size(const std::shared_ptr<ov::Node>& node) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant)
        return 0;
    const auto vals = constant->cast_vector<int64_t>();
    return vals.empty() ? 0 : std::llabs(vals.front());
}

#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

// ============================================================================
// Matches: ScaledDotProductAttention(is_causal=true)
//
// The case when causality is an SDPA attribute, not an explicit mask
// subgraph.
// ============================================================================
class SDPACausalMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::SDPACausalMatcher");
    explicit SDPACausalMatcher(ov::npuw::MaskInfo& mask_info) {
        auto sdpa = opp::wrap_type<ov::op::v13::ScaledDotProductAttention>();
        auto callback = [&mask_info](opp::Matcher& m) {
            auto node = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(m.get_match_root());
            if (node && node->get_causal() && mask_info.mask_type != ov::npuw::MaskInfo::MaskType::SlidingWindow)
                mask_info = {ov::npuw::MaskInfo::MaskType::Causal, 0};
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(sdpa, "DetectSDPACausal"), callback);
    }
};

// ============================================================================
// Matches: LessEqual|Less(K = range_chain, Q = range_chain)
// Two Range chains compared directly,
// with no extra offset between K and Q. This is the most common causal-mask
// shape, seen (with minor chain variations) in:
//   - Llama    : Range -> Unsqueeze x3
//   - Tril
//   - MiniCPM  : Less(Range, Reshape(Add(Range, offset)))
//   - Whisper  : Range -> Unsqueeze x3
//
// ============================================================================
class StandardCausalMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::StandardCausalMatcher");
    explicit StandardCausalMatcher(ov::npuw::MaskInfo& mask_info) {
        auto cmp = opp::wrap_type<ov::op::v1::LessEqual, ov::op::v1::Less>({make_range_chain(), make_range_chain()});
        auto callback = [&mask_info](opp::Matcher& /*m*/) {
            if (mask_info.mask_type != ov::npuw::MaskInfo::MaskType::SlidingWindow)
                mask_info = {ov::npuw::MaskInfo::MaskType::Causal, 0};
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(cmp, "StandardCausal"), callback);
    }
};

// ============================================================================
// Matches: LessEqual|Less(K = range_chain, Q = Add(any, range_chain))
//
// StandardCausalMatcher with extra Add: Add(cache_len, range_chain), with the range chain as the
// Add's 2nd input.
// ============================================================================
class Qwen3CausalMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::Qwen3CausalMatcher");
    explicit Qwen3CausalMatcher(ov::npuw::MaskInfo& mask_info) {
        auto add = opp::wrap_type<ov::op::v1::Add>({opp::any_input(), make_range_chain()});
        auto cmp = opp::wrap_type<ov::op::v1::LessEqual, ov::op::v1::Less>({make_range_chain(), add});
        auto callback = [&mask_info](opp::Matcher& /*m*/) {
            if (mask_info.mask_type != ov::npuw::MaskInfo::MaskType::SlidingWindow)
                mask_info = {ov::npuw::MaskInfo::MaskType::Causal, 0};
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(cmp, "Qwen3Causal"), callback);
    }
};

// ============================================================================
// Matches a generic sliding-window mask, built from two comparisons ANDed
// together:
//
//   window_check = Greater(K, Add(Q, neg_window))
//   causal_check = LessEqual(K, Q)
//   mask         = BitwiseAnd(BitwiseAnd(any, window_check), causal_check)
//
// Covers: Phi-3 / Gemma-2 / Gemma-3 / Gemma-4 models.
// ============================================================================
class BitwiseAndSlidingMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::BitwiseAndSlidingMatcher");
    explicit BitwiseAndSlidingMatcher(ov::npuw::MaskInfo& mask_info) {
        auto q_chain = make_range_chain();
        auto k_chain = make_range_chain();
        auto window_constant = opp::wrap_type<ov::op::v0::Constant>();
        auto add = opp::wrap_type<ov::op::v1::Add>({q_chain, window_constant});
        auto greater = opp::wrap_type<ov::op::v1::Greater>({k_chain, add});
        auto and_win = opp::wrap_type<ov::op::v13::BitwiseAnd>({opp::any_input(), greater});
        auto causal = opp::wrap_type<ov::op::v1::LessEqual>({k_chain, q_chain});
        auto anchor = opp::wrap_type<ov::op::v13::BitwiseAnd>({and_win, causal});
        auto callback = [&mask_info, window_constant](opp::Matcher& m) {
            const int64_t window_size =
                get_window_size(m.get_pattern_value_map().at(window_constant).get_node_shared_ptr());
            if (window_size > 0)
                mask_info = {ov::npuw::MaskInfo::MaskType::SlidingWindow, window_size};
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(anchor, "BitwiseAndSliding"), callback);
    }
};

// ============================================================================
// Matches the legacy Phi-3 inverted sliding-window mask:
//
//   K = Convert(Convert(Range(0, atten_mask_len, step)))   // K_f32
//   Q = Reshape(Range(past, full_ctx, step), [-1, 1])      // Q_col
//
//   causal_check  = Greater(K, Q)
//   sliding_check = LessEqual(K, Add(Q, neg_window))
//   mask          = BitwiseOr(causal_check, sliding_check)
//
// ============================================================================
class OldPhi3SlidingMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::OldPhi3SlidingMatcher");
    explicit OldPhi3SlidingMatcher(ov::npuw::MaskInfo& mask_info) {
        auto k_constant = opp::wrap_type<ov::op::v0::Constant>();
        auto gather = opp::wrap_type<ov::op::v8::Gather>({opp::any_input(), opp::any_input(), opp::any_input()});
        auto k_range = opp::wrap_type<ov::op::v4::Range>({k_constant, gather, opp::any_input()});
        auto k_convert = opp::wrap_type<ov::op::v0::Convert>({k_range});
        auto k_f32 = opp::wrap_type<ov::op::v0::Convert>({k_convert});
        auto q_range = opp::wrap_type<ov::op::v4::Range>({opp::any_input(), opp::any_input(), opp::any_input()});
        auto q_reshape = opp::wrap_type<ov::op::v1::Reshape>({q_range, opp::any_input()});
        auto q_constant = opp::wrap_type<ov::op::v0::Constant>();
        auto q_add = opp::wrap_type<ov::op::v1::Add>({q_reshape, q_constant});
        auto sliding_mask = opp::wrap_type<ov::op::v1::Greater>({k_f32, q_reshape});
        auto causal_mask = opp::wrap_type<ov::op::v1::LessEqual>({k_f32, q_add});
        auto anchor = opp::wrap_type<ov::op::v13::BitwiseOr>({sliding_mask, causal_mask});

        auto callback = [=, &mask_info](opp::Matcher& m) {
            const int64_t w = get_window_size(m.get_pattern_value_map().at(q_constant).get_node_shared_ptr());
            if (w > 0)
                mask_info = {ov::npuw::MaskInfo::MaskType::SlidingWindow, w};
            return false;
        };

        register_matcher(std::make_shared<opp::Matcher>(anchor, "OldPhi3Sliding"), callback);
    }
};

// ============================================================================
// Matches the default float sliding-window mask:
//
//   causal_check  = LessEqual(K, Q)
//   sliding_check = Greater(K, Subtract(Q, window))
//   mask          = LogicalAnd(causal_check, sliding_check)
//
// ============================================================================
class DefaultSWAMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::DefaultSWAMatcher");
    explicit DefaultSWAMatcher(ov::npuw::MaskInfo& mask_info) {
        auto k_chain = make_range_chain();
        auto q_chain = make_range_chain();
        auto window_const = opp::wrap_type<ov::op::v0::Constant>();
        auto causal_mask = opp::wrap_type<ov::op::v1::LessEqual>({k_chain, q_chain});
        auto subtract = opp::wrap_type<ov::op::v1::Subtract>({q_chain, window_const});
        auto sliding_mask = opp::wrap_type<ov::op::v1::Greater>({k_chain, subtract});
        auto anchor = opp::wrap_type<ov::op::v1::LogicalAnd>({causal_mask, sliding_mask});
        auto callback = [=, &mask_info](opp::Matcher& m) {
            const int64_t w = get_window_size(m.get_pattern_value_map().at(window_const).get_node_shared_ptr());
            if (w > 0)
                mask_info = {ov::npuw::MaskInfo::MaskType::SlidingWindow, w};
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(anchor, "DefaultSWA"), callback);
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // namespace

namespace ov::npuw {

bool DetectAttentionMask::run_on_model(const std::shared_ptr<ov::Model>& model) {
    m_mask_info = MaskInfo{};

    ov::pass::GraphRewrite detector;
    detector.add_matcher<BitwiseAndSlidingMatcher>(m_mask_info);
    detector.add_matcher<OldPhi3SlidingMatcher>(m_mask_info);
    detector.add_matcher<DefaultSWAMatcher>(m_mask_info);
    detector.add_matcher<SDPACausalMatcher>(m_mask_info);
    detector.add_matcher<StandardCausalMatcher>(m_mask_info);
    detector.add_matcher<Qwen3CausalMatcher>(m_mask_info);
    detector.run_on_model(model);

    return false;
}

}  // namespace ov::npuw
