// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pre_compute.hpp"

#include "../../logging.hpp"
#include "../../orc.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"

namespace opp = ov::pass::pattern;
namespace pre_compute = ov::npuw::patterns::pre_compute;

namespace {
// TODO: copied from common tests
// Builds a [1, max_position_embeddings, row_width] sin/cos LUT.
// When duplicate=true (LLama2-style rotate_half), rotary_ndims = inv_freq_size*2
// and the second half mirrors the first (torch.cat([freqs, freqs], dim=-1)).
// When duplicate=false (GPT-style), rotary_ndims = inv_freq_size with no mirroring.
//
// row_width is rotary_ndims unless pad_to is larger, in which case the extra trailing
// columns are filled with identity values (cos=1, sin=0). That padding is what lets
// the LongRoPE unrotated-KV rewrite multiply the FULL (rotary + passthrough) K by a
// single table - see CacheRawKeyPattern below.
//
// Coefficients are always computed in f32 and stored as f16 (the precision the
// graph-side RoPE cache has always used).
static std::pair<ov::Tensor, ov::Tensor> makeCosSinTables(const size_t max_position_embeddings,
                                                          const std::vector<float>& inverse_freq_fp32,
                                                          bool duplicate = true,
                                                          size_t pad_to = 0) {
    const size_t inv_freq_size = inverse_freq_fp32.size();
    const size_t rotary_ndims = duplicate ? inv_freq_size * 2 : inv_freq_size;
    const size_t row_width = std::max(pad_to, rotary_ndims);

    const ov::Shape table_shape{1, max_position_embeddings, row_width};
    ov::Tensor lut_cos(ov::element::f16, table_shape);
    ov::Tensor lut_sin(ov::element::f16, table_shape);
    // Identity everywhere; the rotary columns are overwritten below, the (optional)
    // passthrough columns keep cos=1/sin=0 so rotating them is a no-op.
    std::fill_n(lut_cos.data<ov::float16>(), lut_cos.get_size(), ov::float16{1.0f});
    std::fill_n(lut_sin.data<ov::float16>(), lut_sin.get_size(), ov::float16{0.0f});

    // rotate_half style cos/sin table:
    //   y1 = cos(m*xita_i) * x1 - sin(m*xita_i) * x2
    //   y2 = cos(m*xita_i) * x2 + sin(m*xita_i) * x1
    //
    for (size_t k = 0; k < inv_freq_size; k++) {
        auto xita_i = inverse_freq_fp32[k];
        ov::float16* psin = lut_sin.data<ov::float16>();
        ov::float16* pcos = lut_cos.data<ov::float16>();
        for (size_t m = 0; m < max_position_embeddings; m++, psin += row_width, pcos += row_width) {
            pcos[k] = ov::float16{std::cos(xita_i * static_cast<float>(m))};
            psin[k] = ov::float16{std::sin(xita_i * static_cast<float>(m))};
            if (duplicate) {
                pcos[k + inv_freq_size] = pcos[k];
                psin[k + inv_freq_size] = psin[k];
            }
        }
    }

    return {lut_cos, lut_sin};
}

// Wraps makeCosSinTables()' output into graph Constants. Used by the ordinary
// (rotated-KV) RoPE-cache path only - the LongRoPE unrotated-KV path deliberately
// creates no cos/sin Constants at all, see RopeCacheMatcher.
static ov::OutputVector makeCosSinCache(const size_t max_position_embeddings,
                                        const std::shared_ptr<ov::Node> inverse_frequencies,
                                        bool duplicate = true) {
    const auto inverse_freq_fp32 = ov::as_type_ptr<ov::op::v0::Constant>(inverse_frequencies)->cast_vector<float>();
    auto tables = makeCosSinTables(max_position_embeddings, inverse_freq_fp32, duplicate);

    auto Cos = std::make_shared<ov::op::v0::Constant>(tables.first);
    auto Sin = std::make_shared<ov::op::v0::Constant>(tables.second);

    return {Cos, Sin};
}

static ov::NodeVector calculate_freq(const std::shared_ptr<ov::Node> short_factor_node,
                                     const std::shared_ptr<ov::Node> long_factor_node,
                                     const std::shared_ptr<ov::Node> multiply_node,
                                     const std::shared_ptr<ov::Node> power_node) {
    const auto short_factor = ov::as_type_ptr<ov::op::v0::Constant>(short_factor_node)->cast_vector<float>();
    const auto long_factor = ov::as_type_ptr<ov::op::v0::Constant>(long_factor_node)->cast_vector<float>();
    const auto multiply_const = ov::as_type_ptr<ov::op::v0::Constant>(multiply_node)->cast_vector<float>();
    const auto power_const = ov::as_type_ptr<ov::op::v0::Constant>(power_node)->cast_vector<float>();
    auto factor_size = short_factor.size();

    OPENVINO_ASSERT(short_factor.size() == multiply_const.size() && long_factor.size() == multiply_const.size(),
                    "Invalid constants for LongRopePatternPhi_v5, expected same size for short_factor, long_factor "
                    "and multiply_const");
    OPENVINO_ASSERT(power_const.size() == 1,
                    "Invalid constants for LongRopePatternPhi_v5, expected single value for power_const");

    std::vector<float> freq(factor_size, 0.0f);
    std::vector<float> freq_long(factor_size, 0.0f);
    for (size_t i = 0; i < factor_size; i++) {
        freq[i] = std::pow(short_factor[i] * multiply_const[i], power_const[0]);
        freq_long[i] = std::pow(long_factor[i] * multiply_const[i], power_const[0]);
    }

    auto inv_freq = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape({factor_size}), freq);
    auto inv_freq_long = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape({factor_size}), freq_long);

    return {inv_freq, inv_freq_long};
}

struct GatheredCosSin {
    std::shared_ptr<ov::Node> cos;  // [1, query_len, rotary_ndims], f32 - gathered via matched_position_ids
    std::shared_ptr<ov::Node> sin;
};

GatheredCosSin replaceSinCosByCache(int max_prompt_len,
                                    const ov::OutputVector& cache,
                                    const pre_compute::RopePatternDesc* rpe) {
    auto inv_freq_size = ov::shape_size(rpe->matched_inv_freq->get_shape());

    LOG_VERB("Making sin-cos cache of size: " << max_prompt_len << "x" << inv_freq_size);

    // Step 1: Define axis (gather along axis 1)
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});

    // Step 2: Apply Gather for cos and sin
    auto gather_cos = std::make_shared<ov::op::v8::Gather>(cache[0], rpe->matched_position_ids, axis);
    auto gather_sin = std::make_shared<ov::op::v8::Gather>(cache[1], rpe->matched_position_ids, axis);
    LOG_VERB("Created gather op facilitate LUT search: " << gather_cos->get_name() << ", " << gather_cos->get_shape());

    // Step 2: convert fp16->fp32
    auto cos_fp32 = std::make_shared<ov::op::v0::Convert>(gather_cos, ov::element::f32);
    auto sin_fp32 = std::make_shared<ov::op::v0::Convert>(gather_sin, ov::element::f32);

    // Create the squeeze operation required after gather
    auto squeeze_cos = std::make_shared<ov::op::v0::Squeeze>(cos_fp32, axis);
    auto squeeze_sin = std::make_shared<ov::op::v0::Squeeze>(sin_fp32, axis);

    LOG_VERB("Created squeeze_cos op to reduce axis=1: " << squeeze_cos->get_name() << ", "
                                                         << squeeze_cos->get_shape());
    LOG_VERB("Created squeeze_sin op to reduce axis=1: " << squeeze_sin->get_name() << ", "
                                                         << squeeze_sin->get_shape());

    LOG_VERB("Rope cos detected at: " << rpe->matched_cos->get_name() << ", replacing by cache node: "
                                      << gather_cos->get_name() << ", " << gather_cos->get_shape());
    LOG_VERB("Rope sin detected at: " << rpe->matched_sin->get_name() << ", replacing by cache node: "
                                      << gather_sin->get_name() << ", " << gather_sin->get_shape());

    // replacing sin with gather op
    ov::replace_node(rpe->matched_cos, squeeze_cos);
    ov::replace_node(rpe->matched_sin, squeeze_sin);

    // disconnecting gather from rest or subgraph started from concat_1
    auto gather_input_to_concat = rpe->matched_concat->input(0);
    gather_input_to_concat.get_source_output().remove_target_input(gather_input_to_concat);

    return {squeeze_cos, squeeze_sin};
}

// Matches the Phi-style partial-rotary K-embedding subgraph:
//   raw_k --Slice(rotary)--> rotary_part --Slice(2nd half)--neg-----> Concat(rotate_half)
//                                         --Slice(1st half)--------->
//   rotary_part * cos = mul_cos ; rotate_half(rotary_part) * sin = mul_sin
//   mul_cos + mul_sin = add
//   raw_k --Slice(passthrough)--> passthrough_part
//   Concat(add, passthrough_part) = k_embed  (rotated K for the current token(s))
//
// k_embed normally either (a) feeds a Convert -> Result "present.*.key" directly
// (whole/STATIC prefill - no past-key concat inside a single call), or (b) ALSO
// feeds a Concat(past_key_values.*.key Parameter, k_embed) used internally for
// attention (generate/chunked-prefill - the past+current KV used for QK^T).
//
// This pass rewrites both so the persisted KV cache stores the RAW (pre-RoPE) key:
//  - case (a): redirect the Convert's input from k_embed to raw_k (present.*.key
//    becomes raw). No further change - prefill's own attention still uses the
//    (unchanged) rotated k_embed, which is internally self-consistent (one call =
//    one global short/long-factor decision for the whole span).
//  - case (b): redirect the past-key Concat's 2nd input from k_embed to raw_k (so
//    it now concatenates raw past + raw current = fully raw KV), then re-apply RoPE
//    to the WHOLE raw concat output right before attention.
//
//    Rather than re-deriving the rotary/passthrough split (which would need a
//    second Slice consumer of the raw concat, and a reassembly Concat at the
//    end - both awkward for downstream passes that expect a KV-cache Concat to
//    have exactly one consumer, e.g. the attention-isolation pattern matchers
//    in sdpa.cpp), this extends cos/sin to cover the FULL head_dim (not just
//    the rotary_ndims) with identity values (cos=1, sin=0) on the passthrough
//    dims, and applies the ordinary rotate_half formula to the whole tensor:
//        out = raw_full * cos_param + rotate_half_full(raw_full) * sin_param
//    For the passthrough dims this reduces to raw_full*1 + (anything)*0 =
//    raw_full, i.e. exactly the original passthrough behaviour - the identity
//    padding makes the two formulations equivalent. The terminal node is now a
//    plain Add (no reassembly Concat), and raw_full's own consumer count stays
//    at the same "one direct split, one direct multiply" shape as the Q-side
//    rope-apply already has (see sdpa.cpp's matching `Or` alternative, and
//    util.cpp's find_concat_from_matmul - both taught to see past this chain
//    without needing to distinguish "real" vs. "incidental" Concats, since
//    there's no longer an incidental Concat directly touching the raw KV).
//
// cos_param/sin_param are genuine model Parameters (element type f16, shape
// [1,1,max_len,head_dim], identity-padded on the passthrough dims), NOT Constants -
// this is deliberate for two independent reasons.
//
// 1. Naming. NPUW's repeated-function/closure-extraction (partitioning.cpp) silently
//    discards a Constant's friendly_name when promoting it to a shared closure
//    Parameter (there's no set_friendly_name() call on the newly-created Parameter
//    there), which broke an earlier attempt at this feature that tried to
//    find-and-truncate these LUTs by name for Pyramid attention's per-bucket variants.
//    Genuine, from-the-start Parameters take a different code path in partitioning.cpp
//    (matched against body_sg._parameters by identity, not promoted) and keep their
//    name - the same mechanism npuw_longrope_input already relies on.
//
// 2. Single source of truth. These two tensors are the ONLY place the RoPE
//    coefficients exist for a transformed variant: replaceSinCosByLutTail() rewires the
//    Q-side cos/sin to a tail slice of these very Parameters, so this variant gets no
//    cos/sin Constant at all (and no Select, and no npuw_longrope_input scalar). A
//    second, graph-Constant copy would be both redundant and unreliable - a compiled
//    blob's constants may legitimately be repacked or otherwise transformed by the
//    driver, so host-side data must never be assumed to still match them.
//
// f16 is used because the Q-side RoPE apply has always consumed an f16 table converted
// back to f32 right before the rotation; a per-layer Convert to the K element type
// follows each Parameter, mirroring that chain. Values are supplied by the host every
// call (see the lr_lut helpers in llm_infer_request.cpp) - since the host already knows
// position_ids and the LongRoPE regime decision, no in-graph Select/Gather is needed.
//
// This is intentionally scoped to whole/STATIC prefill, chunked prefill and
// generate (see docs/CONTINUOUS_PREFILL... discussion): cos_param/sin_param
// must be indexed directly by cache slot (slot i == absolute position i),
// which holds for the historical prefix in all three (KV stored contiguously
// from slot 0); the current call's own token(s) are taken by the host from the
// table rows named by the real position_ids, since their absolute position
// isn't known at compile time.
class CacheRawKeyPattern : public ov::pass::MatcherPass {
    // Cached after the first match (same for every decoder layer) so the two
    // new Parameters are only built once per model, not once per layer.
    bool m_have_ext = false;
    size_t m_max_len = 0;
    pre_compute::LongRopeHostLut* m_out_lut = nullptr;

public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::precompute::CacheRawKey");

    // Parameters created on first match; nullptr until then. Retrieved by
    // applyCacheRawKeyAtAttention() after the GraphRewrite finishes, to be added
    // to the model (a MatcherPass has no direct model-level access of its own).
    std::shared_ptr<ov::op::v0::Parameter> cos_param;
    std::shared_ptr<ov::op::v0::Parameter> sin_param;

    CacheRawKeyPattern(size_t max_len, pre_compute::LongRopeHostLut* out_lut)
        : m_max_len(max_len),
          m_out_lut(out_lut) {
        auto raw_k = opp::any_input();
        auto slice_inputs = [](const ov::Output<ov::Node>& data) {
            return ov::OutputVector{data, opp::any_input(), opp::any_input(), opp::any_input(), opp::any_input()};
        };
        auto rotary_part = opp::wrap_type<ov::op::v8::Slice>(slice_inputs(raw_k));
        auto passthrough_part = opp::wrap_type<ov::op::v8::Slice>(slice_inputs(raw_k));
        auto first_half = opp::wrap_type<ov::op::v8::Slice>(slice_inputs(rotary_part));
        auto second_half = opp::wrap_type<ov::op::v8::Slice>(slice_inputs(rotary_part));
        auto neg = opp::wrap_type<ov::op::v1::Multiply>({second_half, opp::any_input()});
        auto rotate_half = opp::wrap_type<ov::op::v0::Concat>({neg, first_half});
        auto mul_cos = opp::wrap_type<ov::op::v1::Multiply>({rotary_part, opp::any_input()});
        auto mul_sin = opp::wrap_type<ov::op::v1::Multiply>({rotate_half, opp::any_input()});
        auto add = opp::wrap_type<ov::op::v1::Add>({mul_cos, mul_sin});
        auto k_embed = opp::wrap_type<ov::op::v0::Concat>({add, passthrough_part});

        ov::matcher_pass_callback callback = [=](opp::Matcher& m) {
            auto& pm = m.get_pattern_value_map();
            auto raw_k_out = pm.at(raw_k);
            auto neg_node = pm.at(neg).get_node_shared_ptr();
            auto first_half_node = pm.at(first_half).get_node_shared_ptr();
            auto second_half_node = pm.at(second_half).get_node_shared_ptr();
            auto passthrough_part_node = pm.at(passthrough_part).get_node_shared_ptr();
            auto k_embed_node = pm.at(k_embed).get_node_shared_ptr();

            auto raw_k_matching_dtype = [&]() -> ov::Output<ov::Node> {
                if (raw_k_out.get_element_type() == k_embed_node->get_output_element_type(0)) {
                    return raw_k_out;
                }
                return std::make_shared<ov::op::v0::Convert>(raw_k_out, k_embed_node->get_output_element_type(0))
                    ->output(0);
            }();

            // Peels a chain of zero-or-more Converts to see if `out` ultimately
            // comes from a Parameter (e.g. past_key_values.*.key -> Convert(f16->f32)
            // -> Concat) - the past-key Parameter is not always Concat's DIRECT input.
            auto originates_from_parameter = [](ov::Output<ov::Node> out) {
                while (ov::is_type<ov::op::v0::Convert>(out.get_node())) {
                    out = out.get_node()->input_value(0);
                }
                return ov::is_type<ov::op::v0::Parameter>(out.get_node());
            };

            bool changed = false;
            // Copy: we mutate consumers below, so iterate over a snapshot.
            auto k_embed_targets = k_embed_node->output(0).get_target_inputs();
            for (auto target_input : k_embed_targets) {
                auto consumer = target_input.get_node()->shared_from_this();

                // Case (a): Convert feeding a Result (present.*.key) - cache raw K.
                if (ov::is_type<ov::op::v0::Convert>(consumer)) {
                    bool feeds_result = false;
                    for (auto& ct : consumer->output(0).get_target_inputs()) {
                        if (ov::is_type<ov::op::v0::Result>(ct.get_node())) {
                            feeds_result = true;
                            break;
                        }
                    }
                    if (feeds_result) {
                        target_input.replace_source_output(raw_k_matching_dtype);
                        changed = true;
                        continue;
                    }
                }

                // Case (b): Concat with a sibling input that ultimately originates from a
                // Parameter (possibly through a dtype Convert) - this is the internal
                // past_key_values.*.key + current-token concat used for attention.
                if (auto concat = ov::as_type_ptr<ov::op::v0::Concat>(consumer)) {
                    bool has_param_sibling = false;
                    for (auto& in : concat->inputs()) {
                        if (in.get_source_output().get_node() != k_embed_node.get() &&
                            originates_from_parameter(in.get_source_output())) {
                            has_param_sibling = true;
                            break;
                        }
                    }
                    if (!has_param_sibling) {
                        continue;
                    }

                    target_input.replace_source_output(raw_k_matching_dtype);
                    changed = true;

                    // Capture attention consumers BEFORE we add new consumers below.
                    auto attn_targets = concat->output(0).get_target_inputs();
                    auto raw_full = concat->output(0);

                    if (!m_have_ext) {
                        const auto passthrough_width = passthrough_part_node->get_output_shape(0).back();
                        const auto first_half_width0 = first_half_node->get_output_shape(0).back();
                        const auto second_half_width0 = second_half_node->get_output_shape(0).back();
                        const auto rotary_ndims = first_half_width0 + second_half_width0;
                        const auto head_dim = rotary_ndims + passthrough_width;

                        cos_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                                            ov::Shape{1, 1, m_max_len, head_dim});
                        sin_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                                            ov::Shape{1, 1, m_max_len, head_dim});
                        // Named so the runtime (process_longrope_lut, llm_infer_request.cpp) and
                        // Pyramid attention's per-variant construction (pyramid_attention.cpp)
                        // can find them. Unlike the earlier Constant-based attempt, these
                        // names survive closure-promotion since they're genuine Parameters
                        // from the start (see the class comment above).
                        cos_param->set_friendly_name("npuw_lr_full_cos");
                        sin_param->set_friendly_name("npuw_lr_full_sin");

                        if (m_out_lut) {
                            // Layout the runtime needs to fill the two Parameters above.
                            // The coefficient values themselves are built afterwards, on
                            // the host, into host-owned buffers (see RopeCacheMatcher).
                            m_out_lut->rotary_ndims = rotary_ndims;
                            m_out_lut->head_dim = head_dim;
                        }

                        m_have_ext = true;
                    }

                    const auto first_half_width = first_half_node->get_output_shape(0).back();
                    const auto second_half_width = second_half_node->get_output_shape(0).back();
                    const auto passthrough_width = passthrough_part_node->get_output_shape(0).back();
                    auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
                    auto split_lengths = ov::op::v0::Constant::create(
                        ov::element::i64,
                        ov::Shape{3},
                        std::vector<int64_t>{static_cast<int64_t>(first_half_width),
                                             static_cast<int64_t>(second_half_width),
                                             static_cast<int64_t>(passthrough_width)});
                    auto split_full =
                        std::make_shared<ov::op::v1::VariadicSplit>(raw_full, split_axis, split_lengths);

                    auto neg_full =
                        neg_node->clone_with_new_inputs({split_full->output(1), neg_node->input_value(1)});
                    auto rotate_half_full = std::make_shared<ov::op::v0::Concat>(
                        ov::OutputVector{neg_full->output(0), split_full->output(0), split_full->output(2)},
                        -1);
                    // Per-layer Convert (f16 LUT -> K's own element type), mirroring the
                    // Q-side Gather->Convert(f32) chain. It has to be per layer (not one
                    // shared Convert) so it can be isolated into THIS layer's attention
                    // block together with the rest of the rotation chain - see sdpa.cpp.
                    const auto& k_type = raw_full.get_element_type();
                    auto to_k_type = [&](const std::shared_ptr<ov::op::v0::Parameter>& p) -> ov::Output<ov::Node> {
                        if (p->get_element_type() == k_type) {
                            return p->output(0);
                        }
                        return std::make_shared<ov::op::v0::Convert>(p->output(0), k_type)->output(0);
                    };
                    auto mul_cos_full = std::make_shared<ov::op::v1::Multiply>(raw_full, to_k_type(cos_param));
                    auto mul_sin_full =
                        std::make_shared<ov::op::v1::Multiply>(rotate_half_full->output(0), to_k_type(sin_param));
                    auto k_for_attention = std::make_shared<ov::op::v1::Add>(mul_cos_full, mul_sin_full);

                    for (auto attn_input : attn_targets) {
                        attn_input.replace_source_output(k_for_attention->output(0));
                    }
                }
            }

            return changed;
        };

        register_matcher(std::make_shared<opp::Matcher>(k_embed, "CacheRawKeyPattern"), callback);
    }
};

// The two host-fed LUT Parameters created by applyCacheRawKeyAtAttention(), or a pair
// of nullptrs when the model has no past+present K Concat to re-rotate (e.g. whole/
// STATIC prefill) and therefore keeps the ordinary rotated-KV RoPE-cache path.
struct LongRopeLutParams {
    std::shared_ptr<ov::op::v0::Parameter> cos;
    std::shared_ptr<ov::op::v0::Parameter> sin;

    explicit operator bool() const {
        return cos != nullptr && sin != nullptr;
    }
};

LongRopeLutParams applyCacheRawKeyAtAttention(const std::shared_ptr<ov::Model>& model,
                                              size_t max_len,
                                              pre_compute::LongRopeHostLut* out_lut) {
    ov::pass::GraphRewrite grw;
    auto matcher = grw.add_matcher<CacheRawKeyPattern>(max_len, out_lut);
    grw.run_on_model(model);

    if (matcher->cos_param && matcher->sin_param) {
        model->add_parameters({matcher->cos_param, matcher->sin_param});
        for (auto&& input : model->inputs()) {
            if (input.get_node() == matcher->cos_param.get()) {
                input.set_names({matcher->cos_param->get_friendly_name()});
            } else if (input.get_node() == matcher->sin_param.get()) {
                input.set_names({matcher->sin_param->get_friendly_name()});
            }
        }
        return {matcher->cos_param, matcher->sin_param};
    }
    return {};
}

// Rewires the Q-side RoPE cos/sin to read from the SAME host-fed full-K LUT the raw-K
// rotation uses, instead of building a second (graph-Constant) cos/sin table for them.
//
// This is exact, not an approximation: Q's positions for a given call are, by
// construction, this call's own new tokens - and those are precisely the LAST
// query_len rows of the LUT (the rows the K rotation uses for its "present" part, see
// lr_lut::refresh_tail in llm_infer_request.cpp). So a tail slice of the Parameter is
// all Q needs. Negative Slice indices are used so the chain stays valid regardless of
// the LUT's seq-dim size.
//
// Consequences: the transformed variant contains no cos/sin Constants, no short/long
// Select, and no npuw_longrope_input scalar - the regime is picked once by the host
// when it fills the LUT, and Q and K then physically share the same numbers.
void replaceSinCosByLutTail(const pre_compute::RopePatternDesc* rpe,
                            const std::shared_ptr<ov::op::v0::Parameter>& cos_param,
                            const std::shared_ptr<ov::op::v0::Parameter>& sin_param,
                            size_t rotary_ndims) {
    const auto& cos_shape = rpe->matched_cos->get_output_shape(0);
    OPENVINO_ASSERT(cos_shape.size() == 3 && cos_shape.back() == rotary_ndims,
                    "LongRoPE unrotated-KV: unexpected Q-side cos/sin shape, expected [1, query_len, rotary_ndims]");
    const auto query_len = static_cast<int64_t>(cos_shape[1]);

    const auto i64 = ov::element::i64;
    auto step = ov::op::v0::Constant::create(i64, ov::Shape{1}, {1});
    auto seq_axis = ov::op::v0::Constant::create(i64, ov::Shape{1}, {2});
    auto last_axis = ov::op::v0::Constant::create(i64, ov::Shape{1}, {-1});
    auto tail_start = ov::op::v0::Constant::create(i64, ov::Shape{1}, {-query_len});
    auto tail_stop = ov::op::v0::Constant::create(i64, ov::Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto rot_start = ov::op::v0::Constant::create(i64, ov::Shape{1}, {0});
    auto rot_stop = ov::op::v0::Constant::create(i64, ov::Shape{1}, {static_cast<int64_t>(rotary_ndims)});
    auto squeeze_axis = ov::op::v0::Constant::create(i64, ov::Shape{1}, {1});

    auto build = [&](const std::shared_ptr<ov::op::v0::Parameter>& param, const std::shared_ptr<ov::Node>& matched) {
        // [1,1,F,head_dim] -> [1,1,query_len,head_dim] -> [1,1,query_len,rotary_ndims]
        auto tail = std::make_shared<ov::op::v8::Slice>(param, tail_start, tail_stop, step, seq_axis);
        auto rotary = std::make_shared<ov::op::v8::Slice>(tail, rot_start, rot_stop, step, last_axis);
        // -> [1,query_len,rotary_ndims], in the element type the native Sin/Cos had
        auto squeezed = std::make_shared<ov::op::v0::Squeeze>(rotary, squeeze_axis);
        return std::make_shared<ov::op::v0::Convert>(squeezed, matched->get_output_element_type(0));
    };

    ov::replace_node(rpe->matched_cos, build(cos_param, rpe->matched_cos));
    ov::replace_node(rpe->matched_sin, build(sin_param, rpe->matched_sin));

    // Disconnect the now-dead native inv_freq/Select/MatMul chain, same as
    // replaceSinCosByCache() does.
    auto gather_input_to_concat = rpe->matched_concat->input(0);
    gather_input_to_concat.get_source_output().remove_target_input(gather_input_to_concat);
}

}  // namespace

ov::npuw::patterns::pre_compute::RopePatternLLama2::RopePatternLLama2() : matcher("sin-cos-matcher") {
    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat_1 = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});
    // here we can seen inverse frequencies as a parameter or constant depending on partitioner passes
    auto inv_freq = opp::wrap_type<ov::op::v0::Constant>();
    auto inv_freq_convert = opp::optional<ov::op::v0::Convert>({inv_freq->output(0)});
    auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({inv_freq_convert, concat_1});
    auto position_ids = opp::wrap_type<ov::op::v0::Parameter>();
    auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({position_ids, opp::wrap_type<ov::op::v0::Constant>()});
    auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, convert});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    // Optional Concat between Transpose and Sin/Cos: present in LLama2-style RoPE
    // (torch.cat([freqs, freqs], dim=-1)), absent in GPT-style RoPE (inv_freq shape [1, n, 1]).
    auto concat_2 = opp::optional<ov::op::v0::Concat>({transpose->output(0), opp::any_input()});
    auto output_sin = opp::wrap_type<ov::op::v0::Sin>({concat_2});
    auto output_cos = opp::wrap_type<ov::op::v0::Cos>({concat_2});

    init_cb = [=](const auto& matches) {
        const auto& map_sin = matches.at(output_sin)[0];
        const auto& map_cos = matches.at(output_cos)[0];

        this->matched_position_ids = map_sin.at(position_ids).get_node_shared_ptr();
        this->matched_concat = map_sin.at(concat_1).get_node_shared_ptr();
        this->matched_inv_freq = map_sin.at(inv_freq).get_node_shared_ptr();

        this->matched_cos = map_cos.at(output_cos).get_node_shared_ptr();
        this->matched_sin = map_sin.at(output_sin).get_node_shared_ptr();

        // Determine if freq duplication was applied by inspecting the actual graph:
        // sin's direct input is Concat (LLama2-style) or Transpose (GPT-style, no dup).
        auto sin_input = this->matched_sin->input_value(0).get_node_shared_ptr();
        this->duplicate_freqs = ov::is_type<ov::op::v0::Concat>(sin_input);

        LOG_VERB("Rope found : sin=" << matched_sin->get_name() << ", cos=" << matched_cos->get_name()
                                     << " (duplicate_freqs=" << duplicate_freqs << ")");

        return true;
    };

    matcher.register_patterns({output_sin, output_cos}, make_matcher_callback());
}

ov::npuw::patterns::pre_compute::LongRopePatternPhi::LongRopePatternPhi() : matcher("sin-cos-matcher") {
    auto MakeConstant = []() {
        return opp::wrap_type<ov::op::v0::Constant>();
    };

    auto make_select_pattern = [&](const std::shared_ptr<ov::Node>& position_ids,
                                   const std::shared_ptr<ov::Node>& inv_freq_short,
                                   const std::shared_ptr<ov::Node>& inv_freq_long) {
        auto red_max = opp::wrap_type<ov::op::v1::ReduceMax>({position_ids, MakeConstant()});
        auto add = opp::wrap_type<ov::op::v1::Add>({red_max, MakeConstant()});
        // max(position_ids) + 1 <= original_max_position_embeddings
        auto leq = opp::wrap_type<ov::op::v1::LessEqual>({add, MakeConstant()});

        auto inv_freq_short_conv = opp::optional<ov::op::v0::Convert>({inv_freq_short->output(0)});
        auto inv_freq_long_conv = opp::optional<ov::op::v0::Convert>({inv_freq_long->output(0)});

        // max(position_ids) + 1 <= original_max_position_embeddings ? short_factor : long_factor;
        auto select = opp::wrap_type<ov::op::v1::Select>({leq, inv_freq_short_conv, inv_freq_long_conv});
        auto unsqueeze = opp::optional<ov::op::v0::Unsqueeze>({select, MakeConstant()});
        auto unsqueeze_1 = opp::optional<ov::op::v0::Unsqueeze>({unsqueeze, MakeConstant()});

        return std::make_tuple(unsqueeze_1, leq, red_max);
    };

    auto position_ids = opp::wrap_type<ov::op::v0::Parameter>();

    auto inv_freq_short = MakeConstant();
    auto inv_freq_long = MakeConstant();

    auto select_cond_max_pos_id = make_select_pattern(position_ids, inv_freq_short, inv_freq_long);
    auto select = std::get<0>(select_cond_max_pos_id);
    auto cond = std::get<1>(select_cond_max_pos_id);
    auto max_pos_id = std::get<2>(select_cond_max_pos_id);

    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat_1 = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});
    // here we can seen inverse frequencies as a parameter or constant depending on partitioner passes
    auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({select, concat_1});
    auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({position_ids, MakeConstant()});
    auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, convert});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    auto concat_2 = opp::wrap_type<ov::op::v0::Concat>({transpose, opp::any_input()});
    auto output_sin = opp::wrap_type<ov::op::v0::Sin>({concat_2});
    auto output_cos = opp::wrap_type<ov::op::v0::Cos>({concat_2});

    init_cb = [=](const auto& matches) {
        const auto& map_sin = matches.at(output_sin)[0];
        const auto& map_cos = matches.at(output_cos)[0];

        this->matched_position_ids = map_sin.at(position_ids).get_node_shared_ptr();
        this->matched_concat = map_sin.at(concat_1).get_node_shared_ptr();
        this->matched_inv_freq = map_sin.at(inv_freq_short).get_node_shared_ptr();
        this->matched_inv_freq_long = map_sin.at(inv_freq_long).get_node_shared_ptr();
        this->matched_cond = map_sin.at(cond).get_node_shared_ptr();
        this->max_pos_id = map_sin.at(max_pos_id).get_node_shared_ptr();

        this->matched_cos = map_cos.at(output_cos).get_node_shared_ptr();
        this->matched_sin = map_sin.at(output_sin).get_node_shared_ptr();

        LOG_VERB("Rope found : sin=" << matched_sin->get_name() << ", cos=" << matched_cos->get_name());

        return true;
    };

    matcher.register_patterns({output_sin, output_cos}, make_matcher_callback());
}

ov::npuw::patterns::pre_compute::LongRopePatternPhi_v5::LongRopePatternPhi_v5() : matcher("sin-cos-matcher") {
    auto MakeConstant = []() {
        return opp::wrap_type<ov::op::v0::Constant>();
    };

    auto make_select_pattern = [&](const std::shared_ptr<ov::Node>& position_ids,
                                   const std::shared_ptr<ov::Node>& short_factor,
                                   const std::shared_ptr<ov::Node>& long_factor,
                                   const std::shared_ptr<ov::Node>& multiply_const,
                                   const std::shared_ptr<ov::Node>& power_const) {
        auto red_max = opp::wrap_type<ov::op::v1::ReduceMax>({position_ids, MakeConstant()});
        auto add = opp::wrap_type<ov::op::v1::Add>({red_max, MakeConstant()});
        auto context_limit = MakeConstant();
        // max(position_ids) + 1 > original_max_position_embeddings
        auto greater = opp::wrap_type<ov::op::v1::Greater>({add, context_limit});

        auto short_factor_conv = opp::optional<ov::op::v0::Convert>({short_factor->output(0)});
        auto long_factor_conv = opp::optional<ov::op::v0::Convert>({long_factor->output(0)});

        // max(position_ids) + 1 > original_max_position_embeddings ? long_factor : short_factor;
        auto select = opp::wrap_type<ov::op::v1::Select>({greater, long_factor_conv, short_factor_conv});
        auto multiply = opp::wrap_type<ov::op::v1::Multiply>({select, multiply_const});
        auto power = opp::wrap_type<ov::op::v1::Power>({multiply, power_const});
        auto unsqueeze = opp::optional<ov::op::v0::Unsqueeze>({power, MakeConstant()});
        auto unsqueeze_1 = opp::optional<ov::op::v0::Unsqueeze>({unsqueeze, MakeConstant()});

        return std::make_tuple(unsqueeze_1, greater, red_max, context_limit);
    };

    auto position_ids = opp::wrap_type<ov::op::v0::Parameter>();

    auto short_factor = MakeConstant();
    auto long_factor = MakeConstant();

    auto multiply_const = MakeConstant();
    auto power_const = MakeConstant();

    auto select_cond_max_pos_id =
        make_select_pattern(position_ids, short_factor, long_factor, multiply_const, power_const);
    auto select = std::get<0>(select_cond_max_pos_id);
    auto cond = std::get<1>(select_cond_max_pos_id);
    auto max_pos_id = std::get<2>(select_cond_max_pos_id);
    auto context_limit = std::get<3>(select_cond_max_pos_id);

    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat_1 = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});
    // here we can seen inverse frequencies as a parameter or constant depending on partitioner passes
    auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({select, concat_1});
    auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({position_ids, MakeConstant()});
    auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, convert});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    auto concat_2 = opp::wrap_type<ov::op::v0::Concat>({transpose, opp::any_input()});
    auto output_sin = opp::wrap_type<ov::op::v0::Sin>({concat_2});
    auto output_cos = opp::wrap_type<ov::op::v0::Cos>({concat_2});

    init_cb = [=](const auto& matches) {
        const auto& map_sin = matches.at(output_sin)[0];
        const auto& map_cos = matches.at(output_cos)[0];

        this->matched_position_ids = map_sin.at(position_ids).get_node_shared_ptr();
        this->matched_concat = map_sin.at(concat_1).get_node_shared_ptr();
        this->matched_short_factor = map_sin.at(short_factor).get_node_shared_ptr();
        this->matched_long_factor = map_sin.at(long_factor).get_node_shared_ptr();
        this->matched_context_limit = map_sin.at(context_limit).get_node_shared_ptr();
        this->matched_multiply_const = map_sin.at(multiply_const).get_node_shared_ptr();
        this->matched_power_const = map_sin.at(power_const).get_node_shared_ptr();
        this->matched_cond = map_sin.at(cond).get_node_shared_ptr();
        this->max_pos_id = map_sin.at(max_pos_id).get_node_shared_ptr();

        this->matched_cos = map_cos.at(output_cos).get_node_shared_ptr();
        this->matched_sin = map_sin.at(output_sin).get_node_shared_ptr();

        LOG_VERB("Rope found : sin=" << matched_sin->get_name() << ", cos=" << matched_cos->get_name());

        return true;
    };

    matcher.register_patterns({output_sin, output_cos}, make_matcher_callback());
}

std::optional<uint64_t> ov::npuw::patterns::pre_compute::extract_phi_v5_longrope_context_limit(
    const std::shared_ptr<ov::Model>& model) {
    auto long_rope = std::make_shared<ov::npuw::patterns::pre_compute::LongRopePatternPhi_v5>();
    std::optional<uint64_t> context_limit;
    long_rope->transform_cb = [&]() {
        auto limit_const = ov::as_type_ptr<ov::op::v0::Constant>(long_rope->matched_context_limit);
        OPENVINO_ASSERT(limit_const, "Invalid LongRopePatternPhi_v5 match, expected constant context limit");

        const auto limit_values = limit_const->cast_vector<int64_t>();
        OPENVINO_ASSERT(limit_values.size() == 1,
                        "Invalid LongRopePatternPhi_v5 context limit, expected a single scalar value");
        OPENVINO_ASSERT(limit_values.front() >= 0,
                        "Invalid LongRopePatternPhi_v5 context limit, expected a non-negative value");

        const auto matched_limit = static_cast<uint64_t>(limit_values.front());
        if (context_limit.has_value()) {
            OPENVINO_ASSERT(context_limit.value() == matched_limit,
                            "Inconsistent LongRopePatternPhi_v5 context limits detected in the model");
        } else {
            context_limit = matched_limit;
        }
    };
    long_rope->run_on_model(model);
    return context_limit;
}

ov::npuw::patterns::pre_compute::RopeCacheMatcher::RopeCacheMatcher(const uint32_t max_prompt_len,
                                                                    const std::shared_ptr<ov::Model>& model,
                                                                    const std::string& longrope_input_name,
                                                                    bool cache_raw_key_at_attention,
                                                                    LongRopeHostLut* out_lut) {
    auto rpe = std::make_shared<RopePatternLLama2>();

    rpe->transform_cb = [&]() {
        auto cache = makeCosSinCache(max_prompt_len, rpe->matched_inv_freq, rpe->duplicate_freqs);
        replaceSinCosByCache(max_prompt_len, cache, rpe.get());
    };
    rpe->run_on_model(model);

    auto long_rpe = std::make_shared<LongRopePatternPhi>();

    std::shared_ptr<ov::op::v0::Parameter> max_pos_id_param;
    long_rpe->transform_cb = [&]() {
        auto cache_short = makeCosSinCache(max_prompt_len, long_rpe->matched_inv_freq);
        auto cache_long = makeCosSinCache(max_prompt_len, long_rpe->matched_inv_freq_long);

        auto select_cos = std::make_shared<ov::op::v1::Select>(long_rpe->matched_cond, cache_short[0], cache_long[0]);
        auto select_sin = std::make_shared<ov::op::v1::Select>(long_rpe->matched_cond, cache_short[1], cache_long[1]);

        replaceSinCosByCache(max_prompt_len, {select_cos, select_sin}, long_rpe.get());

        auto max_pos_id_out = long_rpe->max_pos_id->output(0);
        max_pos_id_param.reset(new ov::op::v0::Parameter(max_pos_id_out.get_element_type(), {1}));
        max_pos_id_param->set_friendly_name(longrope_input_name);
        max_pos_id_out.replace(max_pos_id_param->output(0));
    };
    long_rpe->run_on_model(model);

    auto long_rpe_v5 = std::make_shared<LongRopePatternPhi_v5>();

    // The v5 transform is deferred: when the unrotated-KV mitigation is requested we
    // first have to find out whether it actually applies to this variant (and what
    // head_dim it has) before deciding HOW to feed the Q side - either from the
    // ordinary graph-Constant cache, or from the very same host-fed LUT the raw-K
    // rotation uses, so that one single table serves both.
    ov::NodeVector v5_inv_freq;
    long_rpe_v5->transform_cb = [&]() {
        if (!v5_inv_freq.empty()) {
            return;  // one cache per model
        }
        v5_inv_freq = calculate_freq(long_rpe_v5->matched_short_factor,
                                     long_rpe_v5->matched_long_factor,
                                     long_rpe_v5->matched_multiply_const,
                                     long_rpe_v5->matched_power_const);
        // WA: to get correct sin-cos cache size
        long_rpe_v5->matched_inv_freq = v5_inv_freq[0];
    };
    long_rpe_v5->run_on_model(model);

    if (!v5_inv_freq.empty()) {
        LongRopeLutParams lut_params;
        if (cache_raw_key_at_attention && out_lut) {
            lut_params = applyCacheRawKeyAtAttention(model, max_prompt_len, out_lut);
        }

        if (lut_params) {
            LOG_DEBUG("Caching raw (pre-RoPE) K, rotating at attention time");
            const auto inv_freq_short = ov::as_type_ptr<ov::op::v0::Constant>(v5_inv_freq[0])->cast_vector<float>();
            const auto inv_freq_long = ov::as_type_ptr<ov::op::v0::Constant>(v5_inv_freq[1])->cast_vector<float>();
            OPENVINO_ASSERT(inv_freq_short.size() * 2 == out_lut->rotary_ndims,
                            "LongRoPE unrotated-KV: the K-side rotary width does not match the inverse-frequency "
                            "array size");

            // Q reads the tail of the same Parameters K is rotated by - no cos/sin
            // Constants, no Select, and no npuw_longrope_input scalar are created for
            // this variant at all.
            replaceSinCosByLutTail(long_rpe_v5.get(), lut_params.cos, lut_params.sin, out_lut->rotary_ndims);

            out_lut->max_len = max_prompt_len;
            out_lut->inv_freq_short = inv_freq_short;
            out_lut->inv_freq_long = inv_freq_long;
            out_lut->rebuild_tables();
        } else {
            if (out_lut) {
                // No past+present K Concat here (e.g. whole/STATIC prefill) - nothing was
                // rewritten, so the runtime must not try to bind anything.
                *out_lut = LongRopeHostLut{};
            }

            auto cache_short = makeCosSinCache(max_prompt_len, v5_inv_freq[0]);
            auto cache_long = makeCosSinCache(max_prompt_len, v5_inv_freq[1]);

            auto select_cos =
                std::make_shared<ov::op::v1::Select>(long_rpe_v5->matched_cond, cache_long[0], cache_short[0]);
            auto select_sin =
                std::make_shared<ov::op::v1::Select>(long_rpe_v5->matched_cond, cache_long[1], cache_short[1]);

            replaceSinCosByCache(max_prompt_len, {select_cos, select_sin}, long_rpe_v5.get());

            auto max_pos_id_out = long_rpe_v5->max_pos_id->output(0);
            max_pos_id_param.reset(new ov::op::v0::Parameter(max_pos_id_out.get_element_type(), {1}));
            max_pos_id_param->set_friendly_name(longrope_input_name);
            max_pos_id_out.replace(max_pos_id_param->output(0));
        }
    }

    if (max_pos_id_param) {
        model->add_parameters({max_pos_id_param});
        for (auto&& input : model->inputs()) {
            if (input.get_node() == max_pos_id_param.get()) {
                input.set_names({max_pos_id_param->get_friendly_name()});
            }
        }
    }
    model->validate_nodes_and_infer_types();
}

ov::npuw::patterns::pre_compute::RopeInverseFreq::RopeInverseFreq(
    ov::npuw::patterns::pre_compute::RopeInverseFreq::Results need_freq_consts,
    const std::shared_ptr<ov::Model>& model) {
    auto rpe = std::make_shared<ov::npuw::patterns::pre_compute::RopePatternLLama2>();

    rpe->transform_cb = [&]() {
        if (auto inverse_freq_constant = ov::as_type_ptr<ov::op::v0::Constant>(rpe->matched_inv_freq)) {
            LOG_VERB("Inverse Frequences Constant found: " << inverse_freq_constant->get_name());
            need_freq_consts.get().push_back(inverse_freq_constant);
            return true;
        }
        return false;  // root hasnt changed
    };
    rpe->run_on_model(model);
}

bool ov::npuw::patterns::pre_compute::RopeCache::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::npuw::patterns::pre_compute::RopeCacheMatcher ropeCache(m_max_prompt_len,
                                                                model,
                                                                m_longrope_input_name,
                                                                m_cache_raw_key_at_attention,
                                                                &m_host_lut);
    return true;
}

void ov::npuw::patterns::pre_compute::LongRopeHostLut::rebuild_tables() {
    if (max_len == 0 || rotary_ndims == 0 || head_dim < rotary_ndims || inv_freq_short.empty() ||
        inv_freq_long.empty()) {
        return;
    }
    // Host-owned buffers, laid out exactly like the npuw_lr_full_cos/sin Parameters
    // ([1, max_len, head_dim], identity on the passthrough columns) so filling them at
    // runtime is a plain memcpy. Nothing here aliases anything in the compiled graph.
    auto tables_short = makeCosSinTables(max_len, inv_freq_short, true, head_dim);
    auto tables_long = makeCosSinTables(max_len, inv_freq_long, true, head_dim);
    cos_short = tables_short.first;
    sin_short = tables_short.second;
    cos_long = tables_long.first;
    sin_long = tables_long.second;
}

void ov::npuw::patterns::pre_compute::LongRopeHostLut::serialize(ov::npuw::orc::Stream& stream) {
    stream & max_len & rotary_ndims & head_dim & inv_freq_short & inv_freq_long;
    if (stream.input()) {
        // Deserialization constructs a dummy LLMCompiledModel and imports already
        // compiled child models, so RopeCache never runs again - rebuild the tables
        // here instead, otherwise the imported graph's npuw_lr_full_cos/sin inputs
        // would be left uninitialized (see LLMInferRequest's validity assert).
        rebuild_tables();
    }
}
