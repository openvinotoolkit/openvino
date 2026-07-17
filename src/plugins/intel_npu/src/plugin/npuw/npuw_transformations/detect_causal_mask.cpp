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

// Builds: Range(any,any,any)
//         -> opt Add(range, any)
//         -> opt Unsqueeze x3
//         -> opt Reshape
//         -> opt Convert
// Covers:
//   Range -> Unsqueeze x (0-3) -> opt Convert           (Llama, Tril, Whisper)
//   Range -> Add(Range, offset) -> Unsqueeze x (0-3)    (NPUW Standard LLM)
//   Range -> Add(Range, offset) -> Reshape               (MiniCPM q-side)
//   Range -> Add(Range, offset) -> Unsqueeze x3          (Gemma-4 cache_position)
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
// Detects SDPA with is_causal=true attribute.
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
// Detects standard causal: LessEqual/Less(range_chain, range_chain).
// Covers: Llama (Range->Unsq x3), NPUW LLM (Add(Range,off)->Unsq), Tril,
//         MiniCPM (Less(Range, Reshape(Add(Range,off)))), Whisper (Range->Unsq x3).
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
// Detects Qwen3-style causal: LessEqual/Less(range_chain, Add(any, range_chain)).
// The threshold is Add(cache_len, Unsqueeze(Range)) -- Range is the 2nd Add input.
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
// Generic BitwiseAnd-based sliding window:
//   BitwiseAnd(BitwiseAnd(any, Greater(K, Add(Q, neg_window))), LessEqual(K, Q))
// K and Q are shared (diamond) to enforce "same node" in both comparison arms.
// Covers: Phi-3 / Gemma-2 / Gemma-3 / Gemma-4 and hand-built real-model patterns.
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
// Detects old Phi-3 (transformers 4.51) inverted sliding window:
//   BitwiseOr(Greater(K_f32, Q_col), LessEqual(K_f32, Add(Q_col, neg_window)))
// K: Range(0, atten_mask_len) -> Convert -> Convert
// Q: Range(past, full_ctx) -> Reshape([-1, 1])
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
        auto causal_mask = opp::wrap_type<ov::op::v1::Greater>({k_f32, q_reshape});
        auto sliding_mask = opp::wrap_type<ov::op::v1::LessEqual>({k_f32, q_add});
        auto anchor = opp::wrap_type<ov::op::v13::BitwiseOr>({causal_mask, sliding_mask});

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
// Detects default float sliding window mask:
//   LogicalAnd(LessEqual(K, Q), Greater(K, Subtract(Q, window_const)))
// K and Q are each shared between two branches (diamond) to enforce the
// "same node" constraint.
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
