// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detect_causal_mask.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <functional>
#include <queue>
#include <regex>
#include <string>
#include <unordered_set>
#include <vector>

#include "../logging.hpp"
#include "../util.hpp"
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

// Writes `encoded_value` (see NPUW_SDPA_MASK_RT_KEY for the encoding) onto `sdpa`'s
// rt_info. A node may only be annotated once: if it is already annotated with a
// *different* value, that means two matchers produced genuinely contradictory
// evidence for the same SDPA node (e.g. an is_causal=true node also fed an
// explicit sliding-window mask) -- fail loudly instead of silently picking one.
// Re-annotating with the *same* value (e.g. two causal matchers agreeing) is a
// harmless no-op.
void assign_mask_rt_info(const std::shared_ptr<ov::Node>& sdpa, int64_t encoded_value) {
    auto& rt_info = sdpa->get_rt_info();
    const auto it = rt_info.find(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
    if (it != rt_info.end()) {
        const auto existing = it->second.as<int64_t>();
        NPUW_ASSERT(existing == encoded_value && "NPUW: conflicting attention mask detection for the same SDPA node");
        return;
    }
    rt_info[ov::npuw::NPUW_SDPA_MASK_RT_KEY] = encoded_value;
}

// Forward-propagates `encoded_value` onto every ScaledDotProductAttention node
// that (transitively) consumes `mask_output`, by writing rt_info directly on the
// SDPA node (see assign_mask_rt_info above). ScaledDotProductAttentionDecomposition::
// decompose() later calls copy_runtime_info(node, get_new_nodes()), which carries
// this rt_info onto the newly created Add(QK, mask) node for free -- so no
// separate post-decomposition detection pass is required.
void annotate_sdpa_consumers(const ov::Output<ov::Node>& mask_output, int64_t encoded_value) {
    std::unordered_set<ov::Node*> visited;
    std::queue<ov::Output<ov::Node>> to_visit;
    to_visit.push(mask_output);

    while (!to_visit.empty()) {
        auto output = to_visit.front();
        to_visit.pop();
        for (const auto& input : output.get_target_inputs()) {
            auto consumer = input.get_node()->shared_from_this();
            if (!visited.insert(consumer.get()).second)
                continue;
            if (auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(consumer)) {
                assign_mask_rt_info(sdpa, encoded_value);
                continue;  // don't cross into the SDPA's own outputs
            }
            for (const auto& out : consumer->outputs())
                to_visit.push(out);
        }
    }
}

bool is_boolean_combine_op(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::op::v13::BitwiseAnd>(node) || ov::is_type<ov::op::v13::BitwiseOr>(node) ||
           ov::is_type<ov::op::v1::LogicalAnd>(node);
}

// True when `node` is (or transitively nests, through boolean-combine ops) a
// sliding-window bound check (a Greater comparison). Used to tell apart a
// boolean-combine consumer that's a genuine SWA anchor -- owned by the SWA
// matchers below -- from one that's just ANDing/ORing the causal comparison
// with something unrelated (a `new_ones` identity constant, the padding
// attention_mask, ...), which is still a plain causal mask in disguise. Real
// exports commonly wrap even a "pure causal, no window" layer's mask in such an
// identity/padding combine (see e.g. Gemma's non-sliding layers), so this needs
// to be decided per boolean-combine node, not per matched causal comparison.
bool contains_window_check(const std::shared_ptr<ov::Node>& node) {
    if (ov::is_type<ov::op::v1::Greater>(node))
        return true;
    if (is_boolean_combine_op(node)) {
        return contains_window_check(node->get_input_node_shared_ptr(0)) ||
               contains_window_check(node->get_input_node_shared_ptr(1));
    }
    return false;
}

// True when `node` is (or, through single-input passthrough ops like Unsqueeze/
// Reshape/Convert/Broadcast, transitively wraps) a Gemma-4-12B-style decomposed
// sliding-window bound check: GreaterEqual(Subtract(...), window_const). Plays
// the same role as contains_window_check() above, but for the Select-based
// mask family matched by TriuCausalMatcher/TriuSlidingMatcher below (Gemma-4-12B's
// traced torch.triu()/masked_fill() decomposition uses GreaterEqual + Select
// instead of LessEqual/Greater + BitwiseAnd).
bool contains_triu_window_check(const std::shared_ptr<ov::Node>& node) {
    if (auto ge = ov::as_type_ptr<ov::op::v1::GreaterEqual>(node)) {
        return ov::is_type<ov::op::v1::Subtract>(ge->get_input_node_shared_ptr(0)) ||
               ov::is_type<ov::op::v1::Subtract>(ge->get_input_node_shared_ptr(1));
    }
    if (ov::is_type<ov::op::v0::Unsqueeze>(node) || ov::is_type<ov::op::v1::Reshape>(node) ||
        ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v3::Broadcast>(node)) {
        return contains_triu_window_check(node->get_input_node_shared_ptr(0));
    }
    return false;
}

// True when `node` is (or, transitively through single-input/pass-through ops
// -- Add, Unsqueeze, Reshape, Convert, Broadcast -- within `depth` steps) derived
// from a Range op. Used by TriuCausalMatcher below as a post-match guard: unlike
// make_range_chain() (which anchors the LessEqual/Less family to an *exact*
// Range->Add->Unsqueeze->Reshape->Convert op ordering), this walks the graph
// looking for *any* Range reachable within a bounded number of hops, so it still
// rejects arbitrary/unrelated GreaterEqual(any_input, any_input) matches without
// having to match Gemma-4-12B's specific chain shape one op at a time. `depth` is
// capped to keep the walk local to the comparison's immediate operands.
bool traces_to_range(const std::shared_ptr<ov::Node>& node, int depth = 8) {
    if (!node || depth <= 0)
        return false;
    if (ov::is_type<ov::op::v4::Range>(node))
        return true;
    if (ov::is_type<ov::op::v1::Add>(node) || ov::is_type<ov::op::v0::Unsqueeze>(node) ||
        ov::is_type<ov::op::v1::Reshape>(node) || ov::is_type<ov::op::v0::Convert>(node) ||
        ov::is_type<ov::op::v3::Broadcast>(node)) {
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            if (traces_to_range(node->get_input_node_shared_ptr(i), depth - 1))
                return true;
        }
    }
    return false;
}

// Shared BFS skeleton behind annotate_causal_mask()/annotate_triu_causal_mask()
// below: forward-propagates a Causal annotation from `mask_output` onto every
// ScaledDotProductAttention node it (transitively) feeds. For every non-SDPA
// consumer reached during the walk, `should_skip_branch(consumer, output)` decides
// whether that consumer is a genuine SWA anchor already owned by a sliding-window
// matcher -- if so, traversal doesn't cross into it, leaving that branch alone;
// otherwise traversal continues through it (e.g. a combine with a `new_ones`
// identity constant or a padding mask is still just a plain causal mask).
void annotate_causal_mask_impl(
    const ov::Output<ov::Node>& mask_output,
    const std::function<bool(const std::shared_ptr<ov::Node>&, const ov::Output<ov::Node>&)>& should_skip_branch) {
    std::unordered_set<ov::Node*> visited;
    std::queue<ov::Output<ov::Node>> to_visit;
    to_visit.push(mask_output);

    while (!to_visit.empty()) {
        auto output = to_visit.front();
        to_visit.pop();
        for (const auto& input : output.get_target_inputs()) {
            auto consumer = input.get_node()->shared_from_this();
            if (!visited.insert(consumer.get()).second)
                continue;
            if (auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(consumer)) {
                assign_mask_rt_info(sdpa, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
                continue;  // don't cross into the SDPA's own outputs
            }
            if (should_skip_branch(consumer, output))
                continue;  // this branch belongs to a SWA matcher, don't propagate into it
            for (const auto& out : consumer->outputs())
                to_visit.push(out);
        }
    }
}

// Forward-propagates a Causal annotation from a matched causal comparison's
// output (LessEqual/Less family). A boolean-combine op (BitwiseAnd/BitwiseOr/
// LogicalAnd) is only treated as SWA-owned when its *other* operand
// (transitively) contains a sliding-window bound check (see contains_window_check
// above) -- meaning this specific combine node is a genuine SWA anchor.
void annotate_causal_mask(const ov::Output<ov::Node>& mask_output) {
    annotate_causal_mask_impl(mask_output,
                              [](const std::shared_ptr<ov::Node>& consumer, const ov::Output<ov::Node>& output) {
                                  if (!is_boolean_combine_op(consumer))
                                      return false;
                                  for (size_t i = 0; i < consumer->get_input_size(); ++i) {
                                      if (consumer->input_value(i) != output &&
                                          contains_window_check(consumer->input_value(i).get_node_shared_ptr())) {
                                          return true;
                                      }
                                  }
                                  return false;
                              });
}

// Forward-propagates a Causal annotation from a matched Gemma-4-12B-style triu
// causal Select's output (see TriuCausalMatcher below). Real exports build the
// final per-layer mask by repeatedly Select()-ing the plain causal mask against
// further conditions (a sliding-window bound check, a user-supplied padding
// mask, ...); a Select is only treated as SWA-owned when our matched mask feeds
// one of its *data* operands (then/else, not the condition) AND that Select's
// condition is a genuine sliding-window bound check (see contains_triu_window_check
// above) -- meaning TriuSlidingMatcher already owns that branch.
void annotate_triu_causal_mask(const ov::Output<ov::Node>& mask_output) {
    annotate_causal_mask_impl(
        mask_output,
        [](const std::shared_ptr<ov::Node>& consumer, const ov::Output<ov::Node>& output) {
            auto select = ov::as_type_ptr<ov::op::v1::Select>(consumer);
            if (!select)
                return false;
            const bool feeds_data_operand = select->input_value(1) == output || select->input_value(2) == output;
            return feeds_data_operand && contains_triu_window_check(select->input_value(0).get_node_shared_ptr());
        });
}

#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

// ============================================================================
// Matches: ScaledDotProductAttention(is_causal=true)
//
// The case when causality is an SDPA attribute, not an explicit mask
// subgraph. Anchored directly on the SDPA node itself (not reached via mask
// traversal), so it can genuinely conflict with a SlidingWindow annotation
// coming from an SWA matcher on the same node -- assign_mask_rt_info() will
// assert in that case, since it means the model has contradictory evidence
// (an explicit is_causal=true attribute together with an explicit
// sliding-window mask feeding the same SDPA).
// ============================================================================
class SDPACausalMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::SDPACausalMatcher");
    SDPACausalMatcher() {
        auto sdpa = opp::wrap_type<ov::op::v13::ScaledDotProductAttention>();
        auto callback = [](opp::Matcher& m) {
            auto node = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(m.get_match_root());
            if (node && node->get_causal())
                assign_mask_rt_info(node, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
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
// Uses annotate_causal_mask() rather than a blanket skip when the matched
// comparison feeds a boolean-combine op: only branches that are genuine SWA
// anchors (see contains_window_check above) are left for the SWA matchers to
// own; other combine consumers (e.g. ANDing with a `new_ones` identity or the
// padding attention_mask, as seen on Gemma's non-sliding layers) still get
// annotated Causal.
// ============================================================================
class StandardCausalMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::StandardCausalMatcher");
    StandardCausalMatcher() {
        auto cmp = opp::wrap_type<ov::op::v1::LessEqual, ov::op::v1::Less>({make_range_chain(), make_range_chain()});
        auto callback = [](opp::Matcher& m) {
            annotate_causal_mask(m.get_match_root()->output(0));
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
//
// Same annotate_causal_mask() branch-level handling as StandardCausalMatcher above.
// ============================================================================
class Qwen3CausalMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::Qwen3CausalMatcher");
    Qwen3CausalMatcher() {
        auto add = opp::wrap_type<ov::op::v1::Add>({opp::any_input(), make_range_chain()});
        auto cmp = opp::wrap_type<ov::op::v1::LessEqual, ov::op::v1::Less>({make_range_chain(), add});
        auto callback = [](opp::Matcher& m) {
            annotate_causal_mask(m.get_match_root()->output(0));
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
    BitwiseAndSlidingMatcher() {
        auto q_chain = make_range_chain();
        auto k_chain = make_range_chain();
        auto window_constant = opp::wrap_type<ov::op::v0::Constant>();
        auto add = opp::wrap_type<ov::op::v1::Add>({q_chain, window_constant});
        auto greater = opp::wrap_type<ov::op::v1::Greater>({k_chain, add});
        auto and_win = opp::wrap_type<ov::op::v13::BitwiseAnd>({opp::any_input(), greater});
        auto causal = opp::wrap_type<ov::op::v1::LessEqual>({k_chain, q_chain});
        auto anchor = opp::wrap_type<ov::op::v13::BitwiseAnd>({and_win, causal});
        auto callback = [window_constant](opp::Matcher& m) {
            const int64_t window_size =
                get_window_size(m.get_pattern_value_map().at(window_constant).get_node_shared_ptr());
            if (window_size > 0) {
                annotate_sdpa_consumers(m.get_match_root()->output(0), window_size);
            }
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
    OldPhi3SlidingMatcher() {
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

        auto callback = [=](opp::Matcher& m) {
            const int64_t w = get_window_size(m.get_pattern_value_map().at(q_constant).get_node_shared_ptr());
            if (w > 0) {
                annotate_sdpa_consumers(m.get_match_root()->output(0), w);
            }
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
    DefaultSWAMatcher() {
        auto k_chain = make_range_chain();
        auto q_chain = make_range_chain();
        auto window_const = opp::wrap_type<ov::op::v0::Constant>();
        auto causal_mask = opp::wrap_type<ov::op::v1::LessEqual>({k_chain, q_chain});
        auto subtract = opp::wrap_type<ov::op::v1::Subtract>({q_chain, window_const});
        auto sliding_mask = opp::wrap_type<ov::op::v1::Greater>({k_chain, subtract});
        auto anchor = opp::wrap_type<ov::op::v1::LogicalAnd>({causal_mask, sliding_mask});
        auto callback = [=](opp::Matcher& m) {
            const int64_t w = get_window_size(m.get_pattern_value_map().at(window_const).get_node_shared_ptr());
            if (w > 0) {
                annotate_sdpa_consumers(m.get_match_root()->output(0), w);
            }
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(anchor, "DefaultSWA"), callback);
    }
};

// ============================================================================
// Matches Gemma-4-12B's decomposed torch.triu(...)-style causal mask:
//
//   Select(GreaterEqual(row, col), any_input, any_input)
//
// Unlike every other causal matcher above (LessEqual/Less feeding a boolean
// combine op), Gemma-4-12B's trace decomposes torch.triu()/masked_fill() into a
// GreaterEqual boolean feeding a Select that directly picks between the
// unmasked (0) and masked (-inf) float fill values -- there is no
// BitwiseAnd/BitwiseOr/LogicalAnd anywhere in this family. Row/col operands are
// left as any_input() rather than make_range_chain(), since this export's
// Range/Unsqueeze/Add ordering doesn't match that chain's grammar.
//
// Uses annotate_triu_causal_mask() (see its own doc comment above) rather than a
// blanket skip when the matched Select feeds another Select down the line.
//
// The GreaterEqual/Select shapes themselves are common enough (unlike e.g.
// LessEqual/Less feeding a boolean-combine op) that this anchor alone is too
// permissive -- nothing here ties `ge` to actual position indices the way
// make_range_chain() does for the other matchers. The callback additionally
// requires the Select's output to be a floating-point tensor (a real mask
// always selects between float fill values, 0 / -inf) and that at least one of
// `ge`'s operands (transitively) derives from a Range op (see traces_to_range
// above), i.e. is a genuine row/col position-index computation -- ruling out
// unrelated boolean/integer Select(GreaterEqual(...)) uses elsewhere in the
// model without requiring an exact match on this export's Range/Unsqueeze/Add
// chain ordering.
// ============================================================================
class TriuCausalMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::TriuCausalMatcher");
    TriuCausalMatcher() {
        auto ge = opp::wrap_type<ov::op::v1::GreaterEqual>({opp::any_input(), opp::any_input()});
        auto sel = opp::wrap_type<ov::op::v1::Select>({ge, opp::any_input(), opp::any_input()});
        auto callback = [ge](opp::Matcher& m) {
            auto root = m.get_match_root();
            if (!root->get_output_element_type(0).is_real())
                return false;
            auto ge_node = m.get_pattern_value_map().at(ge).get_node_shared_ptr();
            if (!traces_to_range(ge_node->get_input_node_shared_ptr(0)) &&
                !traces_to_range(ge_node->get_input_node_shared_ptr(1)))
                return false;
            annotate_triu_causal_mask(root->output(0));
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(sel, "TriuCausal"), callback);
    }
};

// ============================================================================
// Matches Gemma-4-12B's decomposed sliding-window "beyond window" overwrite:
//
//   diff          = Subtract(row, col)
//   beyond_window = GreaterEqual(diff, window_const)
//   windowed      = Select(opt Unsqueeze x2(beyond_window), any_input, any_input)
//
// `windowed`'s data operands are the plain triu-causal mask matched by
// TriuCausalMatcher above (as either its then or else operand -- Gemma-4-12B
// puts it in the "else" slot, but that's not load-bearing here) and a fill
// constant; this Select overwrites the causal mask with the fill value
// wherever the key is further back than `window_const` positions. Anchored
// (and annotated) independently of TriuCausalMatcher, same as
// BitwiseAndSlidingMatcher/DefaultSWAMatcher above vs. the LessEqual family.
// ============================================================================
class TriuSlidingMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::TriuSlidingMatcher");
    TriuSlidingMatcher() {
        auto diff = opp::wrap_type<ov::op::v1::Subtract>({opp::any_input(), opp::any_input()});
        auto window_const = opp::wrap_type<ov::op::v0::Constant>();
        auto beyond_window = opp::wrap_type<ov::op::v1::GreaterEqual>({diff, window_const});
        auto beyond_window_unsq1 = opp::optional<ov::op::v0::Unsqueeze>({beyond_window, opp::any_input()});
        auto beyond_window_unsq2 = opp::optional<ov::op::v0::Unsqueeze>({beyond_window_unsq1, opp::any_input()});
        auto windowed = opp::wrap_type<ov::op::v1::Select>({beyond_window_unsq2, opp::any_input(), opp::any_input()});
        auto callback = [=](opp::Matcher& m) {
            if (!m.get_match_root()->get_output_element_type(0).is_real())
                return false;
            const int64_t w = get_window_size(m.get_pattern_value_map().at(window_const).get_node_shared_ptr());
            if (w > 0) {
                annotate_sdpa_consumers(m.get_match_root()->output(0), w);
            }
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(windowed, "TriuSliding"), callback);
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // namespace

namespace ov::npuw {

bool DetectAttentionMask::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // Matchers below annotate individual SDPA nodes as they recognize Causal or
    // SlidingWindow patterns feeding that node's mask input (see
    // NPUW_SDPA_MASK_RT_KEY in the header for the annotation/encoding contract); a
    // node that no matcher recognizes stays Unknown.
    ov::pass::GraphRewrite detector;
    detector.add_matcher<BitwiseAndSlidingMatcher>();
    detector.add_matcher<OldPhi3SlidingMatcher>();
    detector.add_matcher<DefaultSWAMatcher>();
    detector.add_matcher<TriuSlidingMatcher>();
    detector.add_matcher<SDPACausalMatcher>();
    detector.add_matcher<StandardCausalMatcher>();
    detector.add_matcher<Qwen3CausalMatcher>();
    detector.add_matcher<TriuCausalMatcher>();
    detector.run_on_model(model);

    return false;
}

void log_detected_masks(const std::shared_ptr<ov::Model>& model) {
    if (ov::npuw::get_log_level() < ov::npuw::LogLevel::Debug) {
        return;
    }

    // Best-effort: pull a transformer layer index out of the SDPA node's friendly
    // name (HF/ONNX exports commonly retain the originating module's scope, e.g.
    // "__module.model.layers.4.self_attn/aten::scaled_dot_product_attention").
    // Falls back to topological position when no such index can be found (e.g. in
    // standalone/synthetic test graphs).
    static const std::regex layer_idx_re(R"([Ll]ayers?[._/]([0-9]+))");

    struct Entry {
        std::string name;
        std::string type;
        int64_t sort_key;
    };
    std::vector<Entry> entries;

    int64_t position = 0;
    for (const auto& node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node);
        if (!sdpa)
            continue;

        std::string type;
        const auto& rt_info = sdpa->get_rt_info();
        const auto it = rt_info.find(NPUW_SDPA_MASK_RT_KEY);
        if (it == rt_info.end()) {
            type = "Unknown";
        } else {
            const auto encoded = it->second.as<int64_t>();
            type = (encoded < 0) ? "Causal" : ("SlidingWindow(" + std::to_string(encoded) + ")");
        }

        const auto& name = sdpa->get_friendly_name();
        std::smatch match;
        int64_t sort_key = position;
        if (std::regex_search(name, match, layer_idx_re) && match.size() > 1) {
            const auto& digits = match[1].str();
            int64_t parsed = 0;
            const auto res = std::from_chars(digits.data(), digits.data() + digits.size(), parsed);
            if (res.ec == std::errc{}) {
                sort_key = parsed;
            }
        }
        entries.push_back({name, std::move(type), sort_key});
        ++position;
    }

    std::stable_sort(entries.begin(), entries.end(), [](const Entry& lhs, const Entry& rhs) {
        return lhs.sort_key < rhs.sort_key;
    });
    for (const auto& entry : entries) {
        LOG_DEBUG("layer " << entry.sort_key << " (" << entry.name << "): " << entry.type);
    }
}

std::map<size_t, int64_t> get_layer_mask_annotations(const std::shared_ptr<ov::Model>& model) {
    std::map<size_t, int64_t> result;
    for (const auto& node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node);
        if (!sdpa)
            continue;

        size_t layer_idx = 0;
        if (!ov::npuw::util::try_parse_self_attn_layer_idx(sdpa->get_friendly_name(), layer_idx))
            continue;

        const auto& rt_info = sdpa->get_rt_info();
        const auto it = rt_info.find(NPUW_SDPA_MASK_RT_KEY);
        if (it == rt_info.end())
            continue;  // Unknown - omitted from the map

        result[layer_idx] = it->second.as<int64_t>();
    }
    return result;
}

}  // namespace ov::npuw


