// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decompose_gqa.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/range.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

class GroupQueryAttentionDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::GroupQueryAttentionDecomposition");
    GroupQueryAttentionDecomposition(bool is_prefill_model) {
        auto pattern_node = opp::wrap_type<ov::op::internal::GroupQueryAttention>();

        ov::matcher_pass_callback callback = [=](opp::Matcher& m) {
            auto& pattern_to_output = m.get_pattern_value_map();
            auto node = ov::as_type_ptr<ov::op::internal::GroupQueryAttention>(
                pattern_to_output.at(pattern_node).get_node_shared_ptr());

            if (node == nullptr || transformation_callback(node)) {
                return false;
            }

            auto new_output_node = decompose(node, is_prefill_model);
            ov::replace_node(node, new_output_node);
            return true;
        };

        auto m = std::make_shared<opp::Matcher>(pattern_node, "GroupQueryAttentionDecomposition");
        register_matcher(m, std::move(callback));
    }

    ov::OutputVector decompose(std::shared_ptr<ov::op::internal::GroupQueryAttention> node, bool is_prefill_model) {
        using namespace ov::op;
        using namespace ov;

        const auto num_heads = node->get_num_heads();
        const auto kv_num_heads = node->get_kv_num_heads();
        const auto scale = node->get_scale();
        const auto do_rotary = node->get_do_rotary();
        const auto rotary_interleaved = node->get_rotary_interleaved();
        const auto local_window_size = node->get_local_window_size();
        const auto smooth_softmax = node->get_smooth_softmax();
        // A sliding window is active for local_window_size >= 1; -1 disables it. sliding_window_cache (the
        // physical rolling KV buffer) is intentionally NOT honored here: NPUW manages the KV cache statically
        // on the host, so the window is applied as a mask over the full cache (same attention result).
        // TODO: add softcap support

        const auto has_input = [&](ov::op::internal::GroupQueryAttentionInputs input_pos) {
            const auto pos = static_cast<size_t>(input_pos);
            return pos < node->get_input_size() && !ov::util::is_empty_constant_tensor(node->input_value(pos));
        };

        const auto get_input = [&](ov::op::internal::GroupQueryAttentionInputs input_pos) -> ov::Output<ov::Node> {
            const auto original_pos = static_cast<int64_t>(input_pos);
            const bool exists = has_input(input_pos);
            OPENVINO_ASSERT(exists, "Missing required GroupQueryAttention input at original position ", original_pos);
            return node->input_value(static_cast<size_t>(original_pos));
        };

        auto Q = get_input(ov::op::internal::GroupQueryAttentionInputs::QUERY);
        auto K = get_input(ov::op::internal::GroupQueryAttentionInputs::KEY);
        auto V = get_input(ov::op::internal::GroupQueryAttentionInputs::VALUE);
        auto past_key = get_input(ov::op::internal::GroupQueryAttentionInputs::PAST_KEY);
        auto past_value = get_input(ov::op::internal::GroupQueryAttentionInputs::PAST_VALUE);
        auto seqlens_k = get_input(ov::op::internal::GroupQueryAttentionInputs::SEQLENS_K);

        // The length of all tokens (past + current) is `seqlens_k` + 1.
        // current = Q.shape[2], past = `seqlens_k` + 1 - current

        const auto T = Q.get_element_type();
        const auto q_shape = register_new_node<v3::ShapeOf>(Q);
        const auto current_seqlen = get_dimensions(q_shape, {2});
        const auto head_size_node = get_dimensions(q_shape, {3});

        const auto zero = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
        const auto zero_without_shape = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        const auto one = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
        const auto one_without_shape = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
        const auto two = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
        const auto seqlens_elemi64 = register_new_node<v0::Convert>(seqlens_k, ov::element::i64);
        const auto real_seqlens = register_new_node<v1::Add>(seqlens_elemi64, one);

        // Only consider batch is 1
        const auto seqlens_1d = register_new_node<v1::Reshape>(real_seqlens, one, false);
        const auto past_seqlen = register_new_node<v1::Subtract>(seqlens_1d, current_seqlen);
        const auto curr_seqlen_scalar = register_new_node<v0::Squeeze>(current_seqlen);

        if (do_rotary) {
            auto cos_cache = get_input(ov::op::internal::GroupQueryAttentionInputs::COS_CACHE);
            auto sin_cache = get_input(ov::op::internal::GroupQueryAttentionInputs::SIN_CACHE);
            ov::Output<ov::Node> position_ids = register_new_node<v4::Range>(zero_without_shape,
                                                                             curr_seqlen_scalar,
                                                                             one_without_shape,
                                                                             ov::element::i64);
            position_ids = register_new_node<v1::Add>(position_ids, past_seqlen);

            const auto cos = register_new_node<v8::Gather>(cos_cache, position_ids, zero);
            const auto sin = register_new_node<v8::Gather>(sin_cache, position_ids, zero);
            Q = rotaryEmbedding(Q, cos, sin, rotary_interleaved);
            K = rotaryEmbedding(K, cos, sin, rotary_interleaved);
        }

        auto construct_kv_cache = [&](const ov::Output<ov::Node>& past, const ov::Output<ov::Node>& current) {
            return register_new_node<v0::Concat>(ov::OutputVector{past, current}, 2);
        };
        K = construct_kv_cache(past_key, K);
        V = construct_kv_cache(past_value, V);

        ov::Output<ov::Node> present_k = K;
        ov::Output<ov::Node> present_v = V;

        const auto concat_kv_len = get_dimensions(K.get_node_shared_ptr(), {2});
        const auto concat_kv_len_scalar = register_new_node<v0::Squeeze>(concat_kv_len);

        // Broadcast KV if grouped query attention
        const size_t kv_num_heads_factor = num_heads / kv_num_heads;
        if (kv_num_heads_factor > 1) {
            const auto kv_shape = register_new_node<v3::ShapeOf>(K);
            const auto kv_shape_prev_2 = get_dimensions(kv_shape, {0, 1});
            const auto kv_shape_last_2 = get_dimensions(kv_shape, {2, 3});
            auto new_kv_shape = register_new_node<v0::Concat>(ov::NodeVector{kv_shape_prev_2, one, kv_shape_last_2}, 0);
            K = register_new_node<v1::Reshape>(K, new_kv_shape, false);
            V = register_new_node<v1::Reshape>(V, new_kv_shape, false);
            K = register_new_node<v0::Concat>(ov::OutputVector(kv_num_heads_factor, K), 2);
            V = register_new_node<v0::Concat>(ov::OutputVector(kv_num_heads_factor, V), 2);
            const auto q_shape = register_new_node<v3::ShapeOf>(Q);
            const auto q_shape_prev_2 = get_dimensions(q_shape, {0, 1});
            auto extended_kv_shape = register_new_node<v0::Concat>(ov::NodeVector{q_shape_prev_2, kv_shape_last_2}, 0);
            K = register_new_node<v1::Reshape>(K, extended_kv_shape, false);
            V = register_new_node<v1::Reshape>(V, extended_kv_shape, false);
        }

        // Make attention mask
        std::shared_ptr<ov::Node> mask;

        std::shared_ptr<ov::Node> hori_range =
            register_new_node<v4::Range>(zero_without_shape, concat_kv_len_scalar, one_without_shape, ov::element::i64);
        hori_range = register_new_node<v0::Unsqueeze>(hori_range, zero);

        std::shared_ptr<ov::Node> vert_range =
            register_new_node<v4::Range>(zero_without_shape, curr_seqlen_scalar, one_without_shape, ov::element::i64);
        vert_range = register_new_node<v0::Unsqueeze>(vert_range, one);
        const auto past_k_node_len = get_dimensions(past_key.get_node_shared_ptr(), {2});
        vert_range = register_new_node<v1::Add>(vert_range, past_k_node_len);

        const auto triu = register_new_node<v1::Greater>(hori_range, vert_range);
        const auto typed_zero = register_new_node(v0::Constant::create(T, ov::Shape{}, {0}));
        // cf. make_attention_mask@src\plugins\intel_gpu\tests\common\subgraphs_builders.hpp
        std::shared_ptr<ov::Node> minus_inf = nullptr;
        if (T == ov::element::f32)
            minus_inf =
                register_new_node(v0::Constant::create(T, ov::Shape{}, {-std::numeric_limits<float>::infinity()}));
        else if (T == ov::element::f16)
            minus_inf =
                register_new_node(v0::Constant::create(T, ov::Shape{}, {std::numeric_limits<ov::float16>::lowest()}));
        mask = register_new_node<v1::Select>(triu, minus_inf, typed_zero);

        if (is_prefill_model) {
            // prefill model
            const auto padding_len = register_new_node<v1::Subtract>(concat_kv_len, seqlens_1d);
            const auto padding_mask_vert_shape = register_new_node<v0::Concat>(ov::NodeVector{current_seqlen, one}, 0);
            const auto padding_mask_vert = register_new_node<v3::Broadcast>(padding_len, padding_mask_vert_shape);
            const auto padding_mask = register_new_node<v1::GreaterEqual>(hori_range, padding_mask_vert);
            mask = register_new_node<v1::Select>(padding_mask, mask, minus_inf);
            if (local_window_size >= 1) {
                // Sliding window: prefill packs past+current in one right-aligned frame, so (vert - hori) is
                // the true query-key distance. Mask keys older than the window: (q - k) >= local_window_size.
                const auto window =
                    register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {local_window_size}));
                const auto distance = register_new_node<v1::Subtract>(vert_range, hori_range);
                const auto too_old = register_new_node<v1::GreaterEqual>(distance, window);
                mask = register_new_node<v1::Select>(too_old, minus_inf, mask);
            }
        } else {
            // kv cache model. Valid keys are the resident past plus the current block: past occupies the
            // real prefix [0, past_seqlen) (past_seqlen = seqlens_k + 1 - curr; the slots [past_seqlen,
            // capacity) hold stale/garbage KV and must be masked), and the current tokens occupy the block
            // [capacity, capacity + curr). Using seqlens_k / a single diagonal here is only correct for
            // single-token decode (curr == 1); for a multi-token (speculative) step it would keep curr-1
            // garbage past slots and drop earlier current tokens. Gate on past_seqlen and the whole current
            // block; the causal triu above provides intra-current-block causality.
            const auto left_mask = register_new_node<v1::Less>(hori_range, past_seqlen);              // resident past
            const auto righ_mask = register_new_node<v1::GreaterEqual>(hori_range, past_k_node_len);  // current block
            const auto atte_mask = register_new_node<v1::LogicalOr>(left_mask, righ_mask);
            mask = register_new_node<v1::Select>(atte_mask, mask, minus_inf);
            if (local_window_size >= 1) {
                // Sliding window (generate). The mask coordinates here are NOT the true absolute positions:
                // past keys are left-aligned at slots [0, seqlens_k) (slot h holds absolute key h), the
                // current tokens sit at the fixed capacity slots [C, C+curr) (slot h holds absolute key
                // total-curr + (h-C)), and vert = C + i is the capacity slot, not the query position. So the
                // window must be expressed in true absolute positions, computed per query row (correct for
                // curr > 1 speculative decode, not just single-token decode):
                //   q_abs(i)  = (total - curr) + i          , total = seqlens_k + 1
                //   k_abs(h)  = h                            if h < seqlens_k   (past, left-aligned)
                //             = (total - curr) + (h - C)     if h >= C          (current block)
                //   mask (too old) iff q_abs - k_abs >= local_window_size
                const auto total_len = register_new_node<v1::Add>(seqlens_elemi64, one);                // seqlens_k + 1
                const auto past_base = register_new_node<v1::Subtract>(total_len, curr_seqlen_scalar);  // total - curr
                // Per-row query absolute positions [curr, 1]: (total - curr) + [0, curr).
                std::shared_ptr<ov::Node> q_abs = register_new_node<v4::Range>(zero_without_shape,
                                                                               curr_seqlen_scalar,
                                                                               one_without_shape,
                                                                               ov::element::i64);
                q_abs = register_new_node<v0::Unsqueeze>(q_abs, one);
                q_abs = register_new_node<v1::Add>(q_abs, past_base);
                // Per-slot key absolute positions [1, kv]: a past slot h (h < capacity C) holds abs key h;
                // a current-block slot h (h >= C) holds abs key (total - curr) + (h - C). The past/current
                // boundary is the capacity C (past_k_node_len), NOT seqlens_k: when the past overflows the
                // capacity (seqlens_k >= C) a seqlens_k boundary would misclassify current-block slots as past.
                const auto cur_kabs =
                    register_new_node<v1::Add>(register_new_node<v1::Subtract>(hori_range, past_k_node_len), past_base);
                const auto is_past = register_new_node<v1::Less>(hori_range, past_k_node_len);
                const auto k_abs = register_new_node<v1::Select>(is_past, hori_range, cur_kabs);
                const auto window =
                    register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {local_window_size}));
                const auto distance = register_new_node<v1::Subtract>(q_abs, k_abs);
                const auto too_old = register_new_node<v1::GreaterEqual>(distance, window);
                mask = register_new_node<v1::Select>(too_old, minus_inf, mask);
            }
        }

        // head_sink (input 11) or smooth_softmax add an extra logit to the softmax denominator. SDPA models
        // this with its sink input: a [1, num_heads, 1, 1] tensor included in the softmax then sliced out.
        // head_sink provides a per-head value; plain smooth_softmax uses 0.
        ov::Output<ov::Node> sink;
        const bool has_head_sink = node->get_input_size() > 11 && !is_null(node->input_value(11));
        if (has_head_sink || smooth_softmax) {
            const auto sink_shape =
                register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, -1, 1, 1}));
            if (has_head_sink) {
                auto head_sink = node->input_value(11);
                if (head_sink.get_element_type() != T) {
                    head_sink = register_new_node<v0::Convert>(head_sink, T);
                }
                sink = register_new_node<v1::Reshape>(head_sink, sink_shape, false);
            } else {
                const auto num_heads_1d =
                    register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_heads}));
                sink = register_new_node<v3::Broadcast>(
                    register_new_node(v0::Constant::create(T, ov::Shape{}, {0})),
                    register_new_node<v0::Concat>(ov::NodeVector{one, num_heads_1d, one, one}, 0));
            }
        }

        std::shared_ptr<ov::Node> qga_output;
        if (sink.get_node_shared_ptr()) {
            // SDPA's 6-input form requires an explicit scale; use the op scale or the default 1/sqrt(head_size).
            ov::Output<ov::Node> scale_node;
            if (scale != 0.0f) {
                scale_node = register_new_node(v0::Constant::create(T, Shape{}, {scale}));
            } else {
                const auto head_size_t = register_new_node<v0::Convert>(head_size_node, T);
                const auto neg_half = register_new_node(v0::Constant::create(T, Shape{}, {-0.5f}));
                scale_node = register_new_node<v0::Squeeze>(register_new_node<v1::Power>(head_size_t, neg_half));
            }
            qga_output = register_new_node<v13::ScaledDotProductAttention>(Q, K, V, mask, scale_node, sink, false);
        } else if (scale != 0.0f) {
            auto scale_node = register_new_node(v0::Constant::create(T, Shape{}, {scale}));
            qga_output = register_new_node<v13::ScaledDotProductAttention>(Q, K, V, mask, scale_node, false);
        } else {
            qga_output = register_new_node<v13::ScaledDotProductAttention>(Q, K, V, mask, false);
        }

        // transpose the result from (batch_size, num_heads, sequence_length, head_size)
        // to (batch_size, sequence_length, num_heads * head_size)
        auto perm = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3}));
        auto qga_output_transposed = register_new_node<v1::Transpose>(qga_output, perm);
        auto dim_merge_shape = register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, -1}));
        auto output = register_new_node<v1::Reshape>(qga_output_transposed, dim_merge_shape, true)->output(0);

        return {std::move(output), std::move(present_k), std::move(present_v)};
    }

    // make split functions is a copy-paste from ONNX FE. TODO: move it to one place
    ov::OutputVector make_split(const ov::Output<ov::Node>& value, int64_t num_splits, int64_t axis) {
        using namespace ov::op;
        const auto axis_node = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {axis}));
        const auto split = register_new_node<v1::Split>(value, axis_node, num_splits);

        return split->outputs();
    }

    std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
                                             const std::vector<int>& dims) {
        using namespace ov::op;
        const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
        return register_new_node<v8::Gather>(shape, dims_const, zero);
    }

    std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims) {
        return get_dimensions(register_new_node<ov::op::v3::ShapeOf>(node), dims);
    }

    std::shared_ptr<ov::Node> rotaryEmbedding(ov::Output<ov::Node> input,
                                              ov::Output<ov::Node> cos,
                                              ov::Output<ov::Node> sin,
                                              bool interleaved) {
        using namespace ov::op;
        auto zero = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});

        if (interleaved) {
            auto two = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
            auto cos_last_dim = get_dimensions(cos.get_node_shared_ptr(), {-1});
            auto input_shape = register_new_node<v3::ShapeOf>(input);
            auto dim_bns = get_dimensions(input_shape, {0, 1, 2});

            auto negtive_one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
            auto split_input_shape = register_new_node<v0::Concat>(ov::NodeVector{dim_bns, cos_last_dim, two}, 0);
            auto reshaped_input = register_new_node<v1::Reshape>(input, split_input_shape, false);

            auto in_split = make_split(reshaped_input, 2, -1);
            split_input_shape = register_new_node<v0::Concat>(ov::NodeVector{dim_bns, cos_last_dim}, 0);
            auto in_split_0 = register_new_node<v1::Reshape>(in_split[0], split_input_shape, false);
            auto in_split_1 = register_new_node<v1::Reshape>(in_split[1], split_input_shape, false);

            auto res_0 = register_new_node<v1::Subtract>(register_new_node<v1::Multiply>(in_split_0, cos),
                                                         register_new_node<v1::Multiply>(in_split_1, sin));
            auto res_1 = register_new_node<v1::Add>(register_new_node<v1::Multiply>(in_split_0, sin),
                                                    register_new_node<v1::Multiply>(in_split_1, cos));

            split_input_shape = register_new_node<v0::Concat>(ov::NodeVector{dim_bns, cos_last_dim, one}, 0);
            auto res_0_5d = register_new_node<v1::Reshape>(res_0, split_input_shape, false);
            auto res_1_5d = register_new_node<v1::Reshape>(res_1, split_input_shape, false);

            auto concat_ret = register_new_node<v0::Concat>(ov::NodeVector{res_0_5d, res_1_5d}, -1);
            return register_new_node<v1::Reshape>(concat_ret, input_shape, false);
        } else {
            auto in_split = make_split(input, 2, -1);
            auto res_0 = register_new_node<v1::Subtract>(register_new_node<v1::Multiply>(in_split[0], cos),
                                                         register_new_node<v1::Multiply>(in_split[1], sin));
            auto res_1 = register_new_node<v1::Add>(register_new_node<v1::Multiply>(in_split[0], sin),
                                                    register_new_node<v1::Multiply>(in_split[1], cos));

            return register_new_node<v0::Concat>(ov::NodeVector{res_0, res_1}, -1);
        }
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

bool decompose_GQA(std::shared_ptr<ov::Model> model, bool is_prefill_model) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<GroupQueryAttentionDecomposition>(is_prefill_model);
    return rewr.run_on_model(model);
}

}  // namespace

namespace ov::npuw {

DecomposeGQA::DecomposeGQA(bool is_prefill_model) : m_is_prefill_model(is_prefill_model) {}

bool DecomposeGQA::run_on_model(const std::shared_ptr<ov::Model>& model) {
    return decompose_GQA(model, m_is_prefill_model);
}

}  // namespace ov::npuw
