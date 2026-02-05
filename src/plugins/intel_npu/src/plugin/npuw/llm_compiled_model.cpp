// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "llm_compiled_model.hpp"

#include "embedding_infer_request.hpp"
#include "embedding_model_utils.hpp"
#include "llm_infer_request.hpp"
#include "logging.hpp"
#include "low_precision/concat.hpp"
#include "low_precision/kv_cache_concat.hpp"
#include "low_precision/low_precision.hpp"
#include "low_precision/move_fake_convert_up_through_kv_cache_concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/util/node_util.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "openvino/pass/validate.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "partitioning/patterns/moe.hpp"
#include "partitioning/patterns/pre_compute.hpp"
#include "partitioning/patterns/sdpa.hpp"
#include "serialization.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/op_conversions/fake_convert_decomposition.hpp"
#include "util.hpp"
#include "whisper_infer_request.hpp"

namespace opp = ov::pass::pattern;

class RemoveEmptyKVTensors : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::RemoveEmptyKVTensors");

    struct Context {
        std::vector<std::shared_ptr<ov::opset13::Parameter>> old_params;
        using Ref = std::reference_wrapper<Context>;
    };

    RemoveEmptyKVTensors(Context::Ref ctx) {
        auto param = opp::wrap_type<ov::op::v0::Parameter>();
        auto concat = opp::wrap_type<ov::op::v0::Concat>({param, opp::any_input()});

        auto callback = [=](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();
            auto matched_param = ov::as_type_ptr<ov::op::v0::Parameter>(node_to_output.at(param).get_node_shared_ptr());
            auto matched_node_concat = node_to_output.at(concat).get_node_shared_ptr();

            ctx.get().old_params.push_back(matched_param);

            auto users = matched_param->get_users();
            if (users.size() == 2u) {
                auto shapeof_node = ov::is_type<ov::op::v3::ShapeOf>(users[0]) ? users[0] : users[1];
                NPUW_ASSERT(ov::is_type<ov::op::v3::ShapeOf>(shapeof_node));
                auto cst_node =
                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, matched_param->get_shape());
                ov::replace_node(shapeof_node, cst_node);
            } else {
                NPUW_ASSERT(users.size() == 1u);
            }

            // Redirect second concat input to every node which reads from concat
            auto curr_kv_tensor = matched_node_concat->input(1).get_source_output();
            for (auto target_input : matched_node_concat->output(0u).get_target_inputs()) {
                target_input.replace_source_output(curr_kv_tensor);
            }

            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(concat, "RemoveEmptyKVTensors"), std::move(callback));
    }
};

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
        // TODO: add softcap support

        auto Q = node->input_value(0);
        auto K = node->input_value(1);
        auto V = node->input_value(2);
        auto past_key = node->input_value(3);
        auto past_value = node->input_value(4);
        auto seqlens_k = node->input_value(5);
        auto total_sequence_length = node->input_value(6);
        auto cos_cache = node->input_value(7);
        auto sin_cache = node->input_value(8);

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
        } else {
            // kv cache model
            const auto left_mask = register_new_node<v1::Less>(hori_range, seqlens_elemi64);     // first N
            const auto righ_mask = register_new_node<v1::GreaterEqual>(hori_range, vert_range);  // last 1
            const auto atte_mask = register_new_node<v1::LogicalOr>(left_mask, righ_mask);       // [1,1,1,..., 0,0,0,1]
            mask = register_new_node<v1::Select>(atte_mask, mask, minus_inf);
        }

        std::shared_ptr<ov::Node> qga_output;
        if (scale != 0.0f) {
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

    // make split functions is a copy-past from ONNX FE. TODO: move it to one place
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

class RedirectNewKvToOutput : public ov::pass::MatcherPass {
public:
    RedirectNewKvToOutput() {
        auto match_down_up_convert_subgraph = [](const ov::Output<ov::Node>& input) {
            auto upconvert = opp::wrap_type<ov::op::v0::Convert>({input}, opp::type_matches(ov::element::f32));

            auto upscale = opp::wrap_type<ov::op::v0::Constant>(opp::rank_equals(0));
            auto upmul = opp::wrap_type<ov::op::v1::Multiply>({upconvert, upscale});

            auto downscale = opp::wrap_type<ov::op::v0::Constant>(opp::rank_equals(0));
            auto downmul = opp::wrap_type<ov::op::v1::Multiply>({upmul, downscale});

            auto downconvert =
                opp::wrap_type<ov::op::v0::Convert>({downmul},
                                                    opp::type_matches_any({ov::element::f8e4m3, ov::element::f8e5m2}));

            return downconvert;
        };

        // example of fp8 inputs to concat
        // input0 : float8e4m3[1,32,1151,96]
        // input1 : float8e4m3[1,32,1,96]

        // parameter, or down_up subgraph case always works for current models-set
        // with both with fp8-optimisation flag enabled and without it
        // TODO: this matcher logic better to cover with unit-tests
        auto input0 = opp::wrap_type<ov::op::v0::Parameter>();
        auto input0_or =
            std::make_shared<opp::op::Or>(ov::OutputVector{input0, match_down_up_convert_subgraph(input0)});

        auto input1 = opp::any_input();

        auto kv_concat = opp::wrap_type<ov::op::v0::Concat>({input0_or, input1});
        auto result1 = opp::wrap_type<ov::op::v0::Result>(kv_concat);
        auto result2 = opp::wrap_type<ov::op::v0::Result>(match_down_up_convert_subgraph(kv_concat));

        auto result_or = std::make_shared<opp::op::Or>(ov::OutputVector{result1, result2});

        ov::matcher_pass_callback callback = [=](opp::Matcher& m) {
            auto& pattern_to_output = m.get_pattern_value_map();
            auto matched_concat = pattern_to_output.at(kv_concat).get_node_shared_ptr();

            auto c0 = matched_concat->input(0).get_source_output();
            auto c1 = matched_concat->input(1).get_source_output();

            LOG_DEBUG(m.get_name() << ": input0.shape=" << c0.get_shape());
            LOG_DEBUG(m.get_name() << ": input1.shape=" << c1.get_shape());
            LOG_DEBUG(m.get_name() << ": concat=" << matched_concat->get_friendly_name());
            LOG_DEBUG(m.get_name() << ": new_kv=" << c1.get_node_shared_ptr()->get_friendly_name());

            std::shared_ptr<ov::Node> matched_result;
            if (pattern_to_output.count(result1)) {
                matched_result = pattern_to_output.at(result1).get_node_shared_ptr();
            } else if (pattern_to_output.count(result2)) {
                matched_result = pattern_to_output.at(result2).get_node_shared_ptr();
                // TODO: need to check that upscale * downscale = 1
                // TODO: need to check input type is f8e5m2 or f8e4m3 if we use this version of concat
            }
            LOG_DEBUG(m.get_name() << ": matched_result=" << matched_result->get_friendly_name());

            matched_result->inputs()[0].replace_source_output(c1);

            return true;
        };

        register_matcher(std::make_shared<opp::Matcher>(result_or, "RedirectNewKvToOutput"), callback);
    }
};

class OldPhi3SlidingMaskMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::OldPhi3SlidingMaskMatcher");

    OldPhi3SlidingMaskMatcher() {
        // Search for the Phi3 old sliding mask pattern to extend it to work with right-padded
        // past tokens and left-padded present tokens.
        //
        // Mask creation is simply done via "less_equal" and "greater" operations between
        // row K range: [0,... mask_len] and column Q range: [current_pos_id,... mask_len].T
        // and sliding window length.
        // Due to broadcasting rules these two operation form two triangular masks.
        //
        // -  "less_equal" forms a sliding window mask, more precisely, it has following expression:
        //
        //        row range [0,... mask_len] <= column range [current_pos_id - sliding_window_size,
        //                                                    ...,
        //                                                    mask_len    -    sliding_window_size]
        //
        //       forming, under example conditions, the mask below:
        //        past tokens = 3
        //        present tokens = 5 (starting with current_pos_id = 3)
        //        sliding window len = 4
        //                    K0 K1 K2 K3 K4 K5 K6 K7
        //                   [ 0  1  2  3  4  5  6  7 ]
        //        Q3[ 3 - 4 ]  0  0  0  0  0  0  0  0
        //        Q4[ 4 - 4 ]  1  0  0  0  0  0  0  0
        //        Q5[ 5 - 4 ]  1  1  0  0  0  0  0  0
        //        Q6[ 6 - 4 ]  1  1  1  0  0  0  0  0
        //        Q7[ 7 - 4 ]  1  1  1  1  0  0  0  0
        //       where 1 at [i, j] means that j token should be forgotten as it can't fit into the sliding
        //       window from the left of i-th token.
        //
        // -   "greater" forms a similar to self-attention mask:
        //
        //        row range [0,... mask_len] > column range [current_pos_id,
        //                                                   ...,
        //                                                   mask_len]
        //
        //       forming, under example conditions, the mask below:
        //        past tokens = 3
        //        present tokens = 5 (starting with current_pos_id = 3)
        //                K0 K1 K2 K3 K4 K5 K6 K7
        //               [ 0  1  2  3  4  5  6  7 ]
        //        Q3[ 3 ]  0  0  0  0  1  1  1  1
        //        Q4[ 4 ]  0  0  0  0  0  1  1  1
        //        Q5[ 5 ]  0  0  0  0  0  0  1  1
        //        Q6[ 6 ]  0  0  0  0  0  0  0  1
        //        Q7[ 7 ]  0  0  0  0  0  0  0  0
        //       where 1 at [i, j] means that j token is a future token for i-th token, that we shouldn't attend to.
        //
        // Together, via "bitwise_or" this two masks forms the inverted sliding attention mask:
        //        past tokens = 3
        //        present tokens = 5 (starting with current_pos_id = 3)
        //        sliding window len = 4
        //                    K0 K1 K2 K3 K4 K5 K6 K7
        //                   [ 0  1  2  3  4  5  6  7 ]
        //        Q3[ 3 - 4 ]  0  0  0  0  1  1  1  1
        //        Q4[ 4 - 4 ]  1  0  0  0  0  1  1  1
        //        Q5[ 5 - 4 ]  1  1  0  0  0  0  1  1
        //        Q6[ 6 - 4 ]  1  1  1  0  0  0  0  1
        //        Q7[ 7 - 4 ]  1  1  1  1  0  0  0  0
        //
        // Issue with sliding attention mask appears when we work with static shapes and different
        // paddings for past and present tokens.
        // More precisely, issue appears with sliding window mask, as Q column range is created
        // from length of past key/values tensor (2175 for 2K case) as start point and the length
        // of attention mask (2176 for 2K) as an end point. This is okay for inverted
        // self-attention mask by means of "greater" operation, as our present tokens exactly
        // left-padded and located on the right in the attention mask.
        // However, for the sliding window mask created by means of "less_equal" operation, given
        // Q range will behave as if position ids of new Q tokens will start from 2175 and not from
        // 3 as in example above and therefore, 2175 - 2047 = 128 first tokens should be forgotten.
        // To fix it a new formula is suggested:
        // 1. (K range <= (Q_pos range - sliding window).T) | (K range > Q range.T)
        // 2. (K range <= (Q range - sliding window).T) & (K range >= len(past_key_values))
        // 3. Resulting mask = 1 | 2,
        // where K range and Q range are created by the same rules as before and Q_pos range is
        // a position_ids array.
        // 4. We also clean mask in places where paddings used instead of real tokens via:
        //    Clean mask = 3 | !(attention_mask_input[past_kv_len:]).T
        auto past_kv_len = opp::wrap_type<ov::op::v8::Gather>({opp::any_input(), opp::any_input(), opp::any_input()});
        auto pos_ids_param = opp::wrap_type<ov::op::v0::Parameter>();
        auto pos_ids_shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({pos_ids_param});
        auto pos_ids_len = opp::wrap_type<ov::op::v8::Gather>({pos_ids_shape_of, opp::any_input(), opp::any_input()});
        auto full_ctx_len = opp::wrap_type<ov::op::v1::Add>({past_kv_len, pos_ids_len});
        auto query_range = opp::wrap_type<ov::op::v4::Range>({past_kv_len, full_ctx_len, opp::any_input()});
        auto column_shape = opp::wrap_type<ov::op::v0::Constant>();
        auto query_range_column = opp::wrap_type<ov::op::v1::Reshape>({query_range, column_shape});

        auto zero_const = opp::wrap_type<ov::op::v0::Constant>();
        auto atten_mask_param = opp::wrap_type<ov::op::v0::Parameter>();
        auto atten_mask_shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({atten_mask_param});
        auto atten_mask_len =
            opp::wrap_type<ov::op::v8::Gather>({atten_mask_shape_of, opp::any_input(), opp::any_input()});
        auto key_range = opp::wrap_type<ov::op::v4::Range>({zero_const, atten_mask_len, opp::any_input()});
        auto key_range_i64 = opp::wrap_type<ov::op::v0::Convert>({key_range});
        auto key_range_f32 = opp::wrap_type<ov::op::v0::Convert>({key_range_i64});

        auto neg_window_size = opp::wrap_type<ov::op::v0::Constant>();
        auto query_left_bound_range = opp::wrap_type<ov::op::v1::Add>({query_range_column, neg_window_size});
        // False in mask means that we shouldn't forget this token
        auto forget_left_tokens_mask = opp::wrap_type<ov::op::v1::LessEqual>({key_range_f32, query_left_bound_range});
        // Basically it is a reference triangle self-attention mask that
        // forbids tokens to attend to future ones, but values are inverted:
        auto look_only_future_mask = opp::wrap_type<ov::op::v1::Greater>({key_range_f32, query_range_column});

        auto inv_sliding_attention_mask =
            opp::wrap_type<ov::op::v13::BitwiseOr>({look_only_future_mask, forget_left_tokens_mask});

        auto callback = [=](opp::Matcher& m) {
            LOG_INFO("Found (4.51) pattern for Phi-3 Sliding Window Attention, will be replaced with custom for static "
                     "shapes.");
            auto& node_to_output = m.get_pattern_value_map();
            auto node_past_kv_len = node_to_output.at(past_kv_len).get_node_shared_ptr();
            auto node_pos_ids_param = node_to_output.at(pos_ids_param).get_node_shared_ptr();
            auto node_atten_mask_param = node_to_output.at(atten_mask_param).get_node_shared_ptr();
            auto node_atten_mask_len = node_to_output.at(atten_mask_len).get_node_shared_ptr();
            auto node_key_range_f32 = node_to_output.at(key_range_f32).get_node_shared_ptr();
            auto node_neg_window_size = node_to_output.at(neg_window_size).get_node_shared_ptr();
            auto node_forget_left_tokens_mask = node_to_output.at(forget_left_tokens_mask).get_node_shared_ptr();
            auto node_bitwise_or = node_to_output.at(inv_sliding_attention_mask).get_node_shared_ptr();

            auto matched_past_kv_len = std::static_pointer_cast<ov::op::v8::Gather>(node_past_kv_len);
            auto matched_pos_ids_input = std::static_pointer_cast<ov::op::v0::Parameter>(node_pos_ids_param);
            auto matched_atten_mask_input = std::static_pointer_cast<ov::op::v0::Parameter>(node_atten_mask_param);
            auto matched_atten_mask_len = std::static_pointer_cast<ov::op::v8::Gather>(node_atten_mask_len);
            auto matched_key_range_f32 = std::static_pointer_cast<ov::op::v0::Convert>(node_key_range_f32);
            auto matched_neg_window_size = std::static_pointer_cast<ov::op::v0::Constant>(node_neg_window_size);
            auto matched_forget_left_tokens_mask =
                std::static_pointer_cast<ov::op::v1::LessEqual>(node_forget_left_tokens_mask);
            auto matched_bitwise_or = std::static_pointer_cast<ov::op::v13::BitwiseOr>(node_bitwise_or);
            OPENVINO_ASSERT(matched_neg_window_size->get_output_size() == 1,
                            "Sliding window size constant must be of size 1, but got " +
                                std::to_string(matched_neg_window_size->get_output_size()));

            // 1.(K range <= (Q_pos range - sliding window).T) | (K range > Q range.T)
            auto query_range_as_pos_ids =
                std::make_shared<ov::op::v0::Convert>(matched_pos_ids_input, ov::element::f32);
            std::vector<int64_t> vector_shape{-1, 1};
            auto vector_shape_const =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, vector_shape);
            auto query_range_as_pos_ids_col =
                std::make_shared<ov::op::v1::Reshape>(query_range_as_pos_ids, vector_shape_const, false);
            auto query_range_as_pos_left_bound =
                std::make_shared<ov::op::v1::Add>(query_range_as_pos_ids_col, matched_neg_window_size);
            auto forget_left_mask_for_right_padding =
                std::make_shared<ov::op::v1::LessEqual>(matched_key_range_f32, query_range_as_pos_left_bound);
            matched_bitwise_or->input(1).replace_source_output(forget_left_mask_for_right_padding);

            // 2. (K range <= (Q range - sliding window).T) & (K range >= shape(past_key_values, 2))
            auto past_kv_len_f32 = std::make_shared<ov::op::v0::Convert>(matched_past_kv_len, ov::element::f32);
            auto only_present_tokens_mask =
                std::make_shared<ov::op::v1::GreaterEqual>(matched_key_range_f32, past_kv_len_f32);
            auto bitwise_and =
                std::make_shared<ov::op::v13::BitwiseAnd>(matched_forget_left_tokens_mask, only_present_tokens_mask);

            // 3. Result = 1 | 2
            // Save target inputs first:
            auto target_inputs = matched_bitwise_or->output(0).get_target_inputs();
            auto new_inv_sliding_mask = std::make_shared<ov::op::v13::BitwiseOr>(matched_bitwise_or, bitwise_and);

            // 4. Removing extra padding via : 3 | !(attention_mask_input[past_kv_len:]).T
            std::vector<int64_t> shape_rank_one{1};
            auto shape_rank_one_const =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, shape_rank_one);
            auto past_len_reshaped =
                std::make_shared<ov::op::v1::Reshape>(matched_past_kv_len, shape_rank_one_const, false);
            auto atten_len_reshaped =
                std::make_shared<ov::op::v1::Reshape>(matched_atten_mask_len, shape_rank_one_const, false);
            auto const_one = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 1);
            auto present_atten_mask = std::make_shared<ov::op::v8::Slice>(matched_atten_mask_input,
                                                                          past_len_reshaped,
                                                                          atten_len_reshaped,
                                                                          const_one,
                                                                          const_one);
            auto present_atten_mask_bool =
                std::make_shared<ov::op::v0::Convert>(present_atten_mask, ov::element::boolean);
            auto inv_present_atten_mask = std::make_shared<ov::op::v1::LogicalNot>(present_atten_mask_bool);
            auto inv_present_atten_mask_col =
                std::make_shared<ov::op::v1::Reshape>(inv_present_atten_mask, vector_shape_const, false);
            auto clean_inv_sliding_mask =
                std::make_shared<ov::op::v13::BitwiseOr>(new_inv_sliding_mask, inv_present_atten_mask_col);
            for (auto&& input : target_inputs) {
                input.replace_source_output(clean_inv_sliding_mask);
            }

            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(inv_sliding_attention_mask, "OldPhi3SlidingMaskMatcher"),
                         std::move(callback));
    }
};

class Phi3SlidingMaskMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::Phi3SlidingMaskMatcher");

    Phi3SlidingMaskMatcher(const std::shared_ptr<ov::Node>& attention_mask_node_ptr,
                           const std::shared_ptr<ov::Node>& position_ids_node_ptr) {
        // Search for the Phi3 sliding mask pattern to extend it to work with right-padded
        // past tokens and left-padded present tokens. Logic to replace pattern is the same
        // as in Phi3SlidingMask rewriter, but adjusted to another set of operations for
        // creation of mask, obtained from transformers 4.53.
        // Pattern is the same as in GemmaSlidingMask.
        //
        // Fix is a replace of following pattern:
        // 1. (K range > (Q range - sliding window).T) & (K range <= Q range.T)
        // to
        // 1. (K range > (Q_pos range - sliding window).T) & (K range <= Q range.T)
        // 2. (K range > (Q range - sliding window).T) | (K range < len(past_key_values))
        // 3. Resulting mask = 1 & 2,
        // where K range and Q range are created by the same rules as before and Q_pos range is
        // a position_ids array.
        // 4. We also clean mask in places where paddings used instead of real tokens via:
        //    Clean mask = 3 & attention_mask_input[past_kv_len:].T
        auto unsqueeze_sequence = [&](std::shared_ptr<ov::Node> range) {
            auto unsqueeze1 = opp::wrap_type<ov::op::v0::Unsqueeze>({range, opp::any_input()});
            auto unsqueeze2 = opp::wrap_type<ov::op::v0::Unsqueeze>({unsqueeze1, opp::any_input()});
            auto unsqueeze3 = opp::wrap_type<ov::op::v0::Unsqueeze>({unsqueeze2, opp::any_input()});

            return unsqueeze3;
        };

        auto past_kv_len = opp::wrap_type<ov::op::v8::Gather>({opp::any_input(), opp::any_input(), opp::any_input()});
        auto full_ctx_len = opp::wrap_type<ov::op::v1::Add>({past_kv_len, opp::any_input()});
        auto query_range = opp::wrap_type<ov::op::v4::Range>({past_kv_len, full_ctx_len, opp::any_input()});
        auto query_range_column = unsqueeze_sequence(query_range);

        auto zero_const = opp::wrap_type<ov::op::v0::Constant>();
        auto full_ctx_len_2 = opp::wrap_type<ov::op::v1::Add>({opp::any_input(), past_kv_len});
        auto key_range = opp::wrap_type<ov::op::v4::Range>({zero_const, full_ctx_len_2, opp::any_input()});
        auto key_range_row = unsqueeze_sequence(key_range);
        auto opt_key_range_row_f32 = opp::optional<ov::op::v0::Convert>({key_range_row->output(0)});

        auto neg_window_size = opp::wrap_type<ov::op::v0::Constant>();
        auto query_left_bound_range = opp::wrap_type<ov::op::v1::Add>({query_range_column, neg_window_size});
        // True in mask means that we should attend this token
        auto sliding_mask = opp::wrap_type<ov::op::v1::Greater>({opt_key_range_row_f32, query_left_bound_range});
        auto sliding_and_true = opp::wrap_type<ov::op::v13::BitwiseAnd>({opp::any_input(), sliding_mask});
        // Basically it is a reference triangle self-attention mask
        auto causal_mask = opp::wrap_type<ov::op::v1::LessEqual>({opt_key_range_row_f32, query_range_column});

        auto sliding_and_causal_mask = opp::wrap_type<ov::op::v13::BitwiseAnd>({sliding_and_true, causal_mask});

        auto callback = [=](opp::Matcher& m) {
            LOG_INFO("Found (4.53) pattern for Phi-3 Sliding Window Attention, will be replaced with custom for static "
                     "shapes.");
            auto& node_to_output = m.get_pattern_value_map();
            auto node_past_kv_len = node_to_output.at(past_kv_len).get_node_shared_ptr();
            auto node_full_ctx_len = node_to_output.at(full_ctx_len).get_node_shared_ptr();
            auto node_neg_window_size = node_to_output.at(neg_window_size).get_node_shared_ptr();
            auto node_sliding_mask = node_to_output.at(sliding_mask).get_node_shared_ptr();
            auto node_sliding_and_causal_mask = node_to_output.at(sliding_and_causal_mask).get_node_shared_ptr();

            auto matched_past_kv_len = std::static_pointer_cast<ov::op::v8::Gather>(node_past_kv_len);
            auto matched_full_ctx_len = std::static_pointer_cast<ov::op::v1::Add>(node_full_ctx_len);
            std::shared_ptr<ov::Node> matched_key_range_row = nullptr;
            if (node_to_output.count(opt_key_range_row_f32)) {
                auto node_key_range_row_f32 = node_to_output[opt_key_range_row_f32].get_node_shared_ptr();
                matched_key_range_row = std::static_pointer_cast<ov::op::v0::Convert>(node_key_range_row_f32);
            } else {
                auto node_key_range_row = node_to_output.at(key_range_row).get_node_shared_ptr();
                matched_key_range_row = std::static_pointer_cast<ov::op::v0::Unsqueeze>(node_key_range_row);
            }
            auto matched_neg_window_size = std::static_pointer_cast<ov::op::v0::Constant>(node_neg_window_size);
            auto matched_sliding_mask = std::static_pointer_cast<ov::op::v1::Greater>(node_sliding_mask);
            auto matched_sliding_and_causal_mask =
                std::static_pointer_cast<ov::op::v13::BitwiseAnd>(node_sliding_and_causal_mask);
            OPENVINO_ASSERT(matched_neg_window_size->get_output_size() == 1,
                            "Sliding window size constant must be of size 1, but got " +
                                std::to_string(matched_neg_window_size->get_output_size()));

            std::shared_ptr<ov::Node> passed_attention_mask = attention_mask_node_ptr;
            std::shared_ptr<ov::Node> passed_position_ids = position_ids_node_ptr;
            OPENVINO_ASSERT(passed_attention_mask, "Passed attention_mask node is nullptr!");
            OPENVINO_ASSERT(passed_position_ids, "Passed position_ids node is nullptr!");

            auto const_zero = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
            auto const_one = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 1);
            auto const_three = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 3);

            // 1.(K range > (Q_pos range - sliding window).T) & (K range <= Q range.T)
            std::shared_ptr<ov::Node> query_range_as_pos_ids = passed_position_ids;
            if (matched_neg_window_size->output(0).get_element_type() == ov::element::f32) {
                query_range_as_pos_ids = std::make_shared<ov::op::v0::Convert>(passed_position_ids, ov::element::f32);
            }
            auto query_range_as_pos_ids_unsqueezed =
                std::make_shared<ov::op::v0::Unsqueeze>(query_range_as_pos_ids, const_zero);
            auto query_range_as_pos_ids_col =
                std::make_shared<ov::op::v0::Unsqueeze>(query_range_as_pos_ids_unsqueezed, const_three);
            auto query_range_as_pos_left_bound =
                std::make_shared<ov::op::v1::Add>(query_range_as_pos_ids_col, matched_neg_window_size);
            auto sliding_mask_for_right_padding =
                std::make_shared<ov::op::v1::Greater>(matched_key_range_row, query_range_as_pos_left_bound);
            matched_sliding_and_causal_mask->input(0).replace_source_output(sliding_mask_for_right_padding);

            // 2. (K range > (Q range - sliding window).T) | (K range < shape(past_key_values, 2))
            std::shared_ptr<ov::Node> past_kv_len_argument = matched_past_kv_len;
            if (matched_neg_window_size->output(0).get_element_type() == ov::element::f32) {
                past_kv_len_argument = std::make_shared<ov::op::v0::Convert>(matched_past_kv_len, ov::element::f32);
            }
            auto only_past_tokens_mask =
                std::make_shared<ov::op::v1::Less>(matched_key_range_row, past_kv_len_argument);
            auto sliding_mask_for_left_padding_or_only_past =
                std::make_shared<ov::op::v13::BitwiseOr>(matched_sliding_mask, only_past_tokens_mask);

            // 3. Result = 1 & 2
            // Save target inputs first:
            auto target_inputs = matched_sliding_and_causal_mask->output(0).get_target_inputs();
            auto new_sliding_and_causal_mask =
                std::make_shared<ov::op::v13::BitwiseAnd>(matched_sliding_and_causal_mask,
                                                          sliding_mask_for_left_padding_or_only_past);

            // 4. Removing extra padding via : 3 & attention_mask_input[past_kv_len:].T
            std::vector<int64_t> shape_rank_one{1};
            auto shape_rank_one_const =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, shape_rank_one);
            auto past_len_reshaped =
                std::make_shared<ov::op::v1::Reshape>(matched_past_kv_len, shape_rank_one_const, false);
            auto full_ctx_len_reshaped =
                std::make_shared<ov::op::v1::Reshape>(matched_full_ctx_len, shape_rank_one_const, false);
            auto const_one_rank_one = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 1);
            auto attention_mask_bool =
                std::make_shared<ov::op::v0::Convert>(passed_attention_mask, ov::element::boolean);
            auto present_atten_mask_bool = std::make_shared<ov::op::v8::Slice>(attention_mask_bool,
                                                                               past_len_reshaped,
                                                                               full_ctx_len_reshaped,
                                                                               const_one_rank_one,
                                                                               const_one_rank_one);
            std::vector<int64_t> vector_shape{-1, 1};
            auto vector_shape_const =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, vector_shape);
            auto present_atten_mask_bool_col =
                std::make_shared<ov::op::v1::Reshape>(present_atten_mask_bool, vector_shape_const, false);
            auto clean_sliding_and_causal_mask =
                std::make_shared<ov::op::v13::BitwiseAnd>(new_sliding_and_causal_mask, present_atten_mask_bool_col);
            for (auto&& input : target_inputs) {
                input.replace_source_output(clean_sliding_and_causal_mask);
            }

            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(sliding_and_causal_mask, "Phi3SlidingMaskMatcher"),
                         std::move(callback));
    }
};

class FakeConvertDestinationTypeExtractor : public ov::pass::MatcherPass {
public:
    FakeConvertDestinationTypeExtractor(std::set<ov::element::Type>& fcDestinationTypes) {
        auto fake_convert_m = opp::wrap_type<ov::op::v13::FakeConvert>();

        ov::matcher_pass_callback callback = [=, &fcDestinationTypes](opp::Matcher& m) {
            const auto& pattern_to_output = m.get_pattern_value_map();
            const auto fake_convert =
                ov::as_type_ptr<ov::op::v13::FakeConvert>(pattern_to_output.at(fake_convert_m).get_node_shared_ptr());

            if (fake_convert == nullptr) {
                return false;
            }
            fcDestinationTypes.insert(fake_convert->get_destination_element_type());
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(fake_convert_m, "FakeConvertDestinationTypeExtractor"),
                         callback);
    }
};

class ConvertTypeRelaxedToRegular : public ov::pass::MatcherPass {
public:
    ConvertTypeRelaxedToRegular() {
        auto pattern = opp::wrap_type<ov::op::TypeRelaxed<ov::op::v1::Multiply>>();

        ov::matcher_pass_callback callback = [](opp::Matcher& m) {
            auto tr_mul = std::dynamic_pointer_cast<ov::op::TypeRelaxed<ov::opset1::Multiply>>(m.get_match_root());
            if (tr_mul) {
                auto new_mul = std::make_shared<ov::opset1::Multiply>(tr_mul->input_value(0), tr_mul->input_value(1));
                new_mul->set_friendly_name(tr_mul->get_friendly_name());
                ov::copy_runtime_info(tr_mul, new_mul);
                ov::replace_node(tr_mul, new_mul);
            }
            return true;
        };

        register_matcher(std::make_shared<opp::Matcher>(pattern, "ConvertTypeRelaxedToRegular"), callback);
    }
};
class Phi3SlidingMask : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::LLMCompiledModel::Phi3SlidingMask");
    Phi3SlidingMask() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        std::shared_ptr<ov::Node> attention_mask_node_ptr = nullptr;
        std::shared_ptr<ov::Node> position_ids_node_ptr = nullptr;
        for (const auto& i : model->inputs()) {
            if (i.get_any_name() == "attention_mask") {
                attention_mask_node_ptr = i.get_node_shared_ptr();
            }
            if (i.get_any_name() == "position_ids") {
                position_ids_node_ptr = i.get_node_shared_ptr();
            }
        }
        if (attention_mask_node_ptr == nullptr || position_ids_node_ptr == nullptr) {
            return false;
        }

        auto pos_id_shape = position_ids_node_ptr->get_output_tensor(0).get_partial_shape();
        if (pos_id_shape.size() != 2) {
            // FIXME: Qwen2.5 VL/Omni uses 3D position_ids, which can't be directly used
            //        in creation of sliding window mask.
            return false;
        }

        ov::pass::Manager manager;
        manager.set_per_pass_validation(true);
        const auto rewriter = manager.register_pass<ov::pass::GraphRewrite>();
        rewriter->add_matcher<Phi3SlidingMaskMatcher>(attention_mask_node_ptr, position_ids_node_ptr);
        rewriter->add_matcher<OldPhi3SlidingMaskMatcher>();
        return manager.run_passes(model);
    }
};

namespace {
uint32_t align_to(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}
bool is_aligned_to(uint32_t value, uint32_t alignment) {
    return value % alignment == 0;
}

std::shared_ptr<ov::Model> cvt_kvcache_to_low_precision(const std::shared_ptr<ov::Model>& model,
                                                        const ov::element::Type lptype) {
    ov::preprocess::PrePostProcessor ppp(model);

    for (const auto& tensor : model->inputs()) {
        if (tensor.get_any_name().find("past_key") != std::string::npos) {
            ppp.input(tensor.get_any_name()).tensor().set_element_type(lptype);
        }
    }

    for (const auto& tensor : model->outputs()) {
        if (tensor.get_any_name().find("present") != std::string::npos) {
            ppp.output(tensor.get_any_name()).tensor().set_element_type(lptype);
        }
    }

    return ppp.build();
}

ov::element::Type optimize_kv_cache_storage(const std::shared_ptr<ov::Model>& model) {
    LOG_DEBUG("Running FP8 static quantization on model: " << model->get_name());

    auto kv_kache_storage_type = ov::element::f16;

    ov::pass::low_precision::LayerTransformation::Params params;
    std::set<ov::element::Type> fcTypesInput, fcTypesRemained;

    ov::pass::Manager manager("optimize_fp8");
    params.defaultPrecisions = ov::pass::low_precision::precision_set::get_fp8_support();
    manager.register_pass<FakeConvertDestinationTypeExtractor>(fcTypesInput);
    manager.register_pass<ov::pass::low_precision::MoveFakeConvertUpThroughKVCacheConcat>();
    auto graph_rewrite = manager.register_pass<ov::pass::GraphRewrite>();
    graph_rewrite->add_matcher<ov::pass::FakeConvertDecomposition>();
    graph_rewrite->add_matcher<ov::pass::low_precision::ConcatTransformation>(params);
    graph_rewrite->add_matcher<ov::pass::low_precision::KVCacheConcat>(model);
    manager.register_pass<ConvertTypeRelaxedToRegular>();
    manager.register_pass<FakeConvertDestinationTypeExtractor>(fcTypesRemained);
    manager.run_passes(model);
    if (fcTypesInput.empty()) {
        LOG_WARN("FakeConvert layers not detected - leaving kv-cache in " << kv_kache_storage_type << " precision");
    } else if (!fcTypesRemained.empty()) {
        LOG_WARN(fcTypesRemained.size() << " FakeConvert layers not decomposed - leaving kv-cache in "
                                        << kv_kache_storage_type << " precision");
    } else if (fcTypesInput.size() > 1) {
        auto it2 = std::next(fcTypesInput.begin(), 1);
        LOG_WARN("FakeConvert layers had several precisions (" << fcTypesInput.size() << ")-" << *fcTypesInput.begin()
                                                               << ", " << *it2 << ", ... "
                                                               << "supported only single precision");
        LOG_WARN("Leaving KV-cache in " << kv_kache_storage_type << " precision");
    } else {
        kv_kache_storage_type = *fcTypesInput.begin();
        LOG_DEBUG("KV cache storage precision changed to " << kv_kache_storage_type);
    }
    return kv_kache_storage_type;
}

std::shared_ptr<ov::Model> redirect_new_kv_to_output(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager("redirect_new_kv_to_output");
    manager.register_pass<RedirectNewKvToOutput>();
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();

    return model;
}

bool remove_empty_kv_inputs(std::shared_ptr<ov::Model> model) {
    ov::pass::GraphRewrite rewr;
    RemoveEmptyKVTensors::Context ctx;
    rewr.add_matcher<RemoveEmptyKVTensors>(std::ref(ctx));
    rewr.run_on_model(model);
    for (auto old_param : ctx.old_params) {
        model->remove_parameter(old_param);
    }
    ov::pass::Validate().run_on_model(model);
    // NB: if old_params is not empty - pass has been applied
    return !ctx.old_params.empty();
}

void decompose_GQA(std::shared_ptr<ov::Model> model, bool is_prefill_model) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<GroupQueryAttentionDecomposition>(is_prefill_model);
    rewr.run_on_model(model);
}

void patch_phi3_sliding_mask(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<Phi3SlidingMask>();
    manager.run_passes(model);
}

}  // namespace

class CutLMHead : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::CutLMHead");
    CutLMHead(std::shared_ptr<ov::Model>& lm_head_model) {
        // We are interested at first input to MatMul as a cut point
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), opp::any_input()});

        // There are several patterns for matmul we are looking for:
        // Matmul -> Result
        // Matmul -> Add -> Result
        auto matmul_add = opp::wrap_type<ov::op::v1::Add>({matmul, opp::any_input()});
        // Matmul -> Transpose -> Result
        auto matmul_transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
        //  Matmul -> Convert -> Result
        auto matmul_convert = opp::wrap_type<ov::op::v0::Convert>({matmul});
        // MatMul -> Divide -> Tanh -> Multiply -> Result
        auto div = opp::wrap_type<ov::op::v1::Multiply, ov::op::v1::Divide>({matmul, opp::any_input()});
        auto tanh = opp::wrap_type<ov::op::v0::Tanh>({div});
        auto matmul_multiply = opp::wrap_type<ov::op::v1::Multiply>({tanh, opp::any_input()});

        auto last_op = std::make_shared<opp::op::Or>(ov::OutputVector{matmul->output(0),
                                                                      matmul_add->output(0),
                                                                      matmul_transpose->output(0),
                                                                      matmul_convert->output(0),
                                                                      matmul_multiply->output(0)});
        auto res = opp::wrap_type<ov::op::v0::Result>({last_op->output(0)});

        auto callback = [=, &lm_head_model](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            auto matched_node_matmul = node_to_output.at(matmul).get_node_shared_ptr();
            std::shared_ptr<ov::Node> matched_node_last_op = nullptr;
            if (node_to_output.count(matmul_add)) {
                matched_node_last_op = node_to_output[matmul_add].get_node_shared_ptr();
            } else if (node_to_output.count(matmul_transpose)) {
                matched_node_last_op = node_to_output[matmul_transpose].get_node_shared_ptr();
            } else if (node_to_output.count(matmul_convert)) {
                matched_node_last_op = node_to_output[matmul_convert].get_node_shared_ptr();
            } else if (node_to_output.count(matmul_multiply)) {
                matched_node_last_op = node_to_output[matmul_multiply].get_node_shared_ptr();
            } else {
                matched_node_last_op = matched_node_matmul;
            }
            auto matched_node_result = node_to_output.at(res).get_node_shared_ptr();

            auto matched_matmul = std::static_pointer_cast<ov::op::v0::MatMul>(matched_node_matmul);
            auto matched_result = std::static_pointer_cast<ov::op::v0::Result>(matched_node_result);

            // Some LLMs add intermediate hidden state outputs that can interfere with LM head detection.
            // Skip Result nodes that were manually added (marked with "manually_added_output" in RT_INFO).
            // For example, Eagle-3 target/draft models add "last_hidden_state" output which should be skipped.
            const auto& rt_info = matched_result->get_rt_info();
            if (rt_info.count("manually_added_output")) {
                return false;
            }

            // Cut point:
            auto matmul_first_source = matched_matmul->input(0).get_source_output();

            // Cut original model:
            matched_result->input(0).replace_source_output(matmul_first_source);
            // FIXME: Somehow for KVCache model result output gets renamed in
            //        ICompiledModel::ICompiledModel().
            //        As a WA, setting the same name to output from MatMul
            //        avoids the issue.
            matmul_first_source.set_names({ov::npuw::LLMCompiledModel::output_embeds});
            matched_result->output(0).set_names({ov::npuw::LLMCompiledModel::output_embeds});
            matched_result->validate_and_infer_types();

            // Create an additional model after cut point:
            auto new_param = std::make_shared<ov::op::v0::Parameter>(matmul_first_source.get_element_type(),
                                                                     matmul_first_source.get_partial_shape());
            new_param->output(0).add_names({ov::npuw::LLMCompiledModel::output_embeds});
            matched_matmul->input(0).replace_source_output(new_param);
            auto new_result = std::make_shared<ov::op::v0::Result>(matched_node_last_op);
            lm_head_model =
                std::make_shared<ov::Model>(ov::OutputVector{new_result->output(0)}, ov::ParameterVector{new_param});

            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(res, "CutLMHead"), std::move(callback));
    }
};

namespace {
std::shared_ptr<ov::Model> cut_lm_head(std::shared_ptr<ov::Model>& model) {
    ov::pass::GraphRewrite rewr;
    std::shared_ptr<ov::Model> lm_head_model = nullptr;
    rewr.add_matcher<CutLMHead>(lm_head_model);
    rewr.run_on_model(model);
    if (lm_head_model) {
        lm_head_model->set_friendly_name(model->get_friendly_name() + "_lm_head");
    }
    model->validate_nodes_and_infer_types();

    return lm_head_model;
}

void reshape_to_static(std::shared_ptr<ov::Model> model,
                       const uint32_t input_size,
                       const uint32_t kvcache_size,
                       const KVAxesPosition& kv_axes_position,
                       const uint32_t lora_rank,
                       const uint32_t lhs_seq_size = 0) {
    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : model->inputs()) {
        const auto& input_name = input.get_any_name();
        ov::PartialShape new_shape;
        if (input_name.find("input_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else if (input_name.find("token_type_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else if (input_name.find("inputs_embeds") != std::string::npos) {
            // NB: VLMs case, model accepts inputs_embeds[BATCH, SEQ_LEN, EMB_SIZE]
            NPUW_ASSERT(input.get_partial_shape().size() == 3u);
            NPUW_ASSERT(input.get_partial_shape()[2].is_static());
            new_shape = ov::PartialShape({1, input_size, input.get_partial_shape()[2]});
        } else if (input_name.find("attention_mask") != std::string::npos) {
            new_shape = ov::PartialShape({1, kvcache_size});
            if (lhs_seq_size && kvcache_size > 4)
                // NB: for whisper kvcache model attn mask should be size + 1
                new_shape = ov::PartialShape({1, kvcache_size + 1});
        } else if (input_name.find("position_ids") != std::string::npos) {
            const auto partial_shape_size = input.get_partial_shape().size();
            // NB: Regular LLM uses 2D shapes, Qwen2.5 VL/Omni uses 3D shapes
            // The first dimension (3) represents the three components of position encoding: time, height, and width
            // enabling alignment across multimodal inputs like text, audio, and video
            NPUW_ASSERT(partial_shape_size == 3u || partial_shape_size == 2u);
            new_shape =
                partial_shape_size == 3u ? ov::PartialShape({3, 1, input_size}) : ov::PartialShape({1, input_size});
        } else if (input_name.find("cache_position") != std::string::npos) {
            // NB: Whisper case
            new_shape = ov::PartialShape({1});
        } else if (input_name.find("encoder_hidden_states") != std::string::npos) {
            // NB: Whisper case
            const auto& partial_shape = input.get_partial_shape();
            new_shape = partial_shape;
            new_shape[0] = 1;  // batch_dim
        } else if (ov::npuw::matchEagle3HiddenStatesString(input_name)) {
            new_shape = ov::npuw::Eagle3Extension::get_static_input(model, input, input_size);
        } else if (ov::npuw::util::matchLoRAMatMulAString(input_name)) {
            new_shape = ov::PartialShape({lora_rank, input.get_partial_shape()[1]});
        } else if (ov::npuw::util::matchLoRAMatMulAlphaString(input_name)) {
            new_shape = ov::PartialShape({input.get_partial_shape()[0], lora_rank});
        } else if (ov::npuw::util::matchLoRAMatMulBString(input_name)) {
            new_shape = ov::PartialShape({input.get_partial_shape()[0], lora_rank});
        } else {
            const auto& partial_shape = input.get_partial_shape();
            new_shape = partial_shape;
            new_shape[kv_axes_position.batch] = 1;
            if (lhs_seq_size) {  // Whisper model
                new_shape[kv_axes_position.seq_len] = (input_name.find(".decoder") != std::string::npos)
                                                          ? kvcache_size - input_size  // kv_size for decoder
                                                          : lhs_seq_size;  // sequence size for encoder hidden states
            } else {                                                       // LLM/VLM
                new_shape[kv_axes_position.seq_len] = kvcache_size - input_size;
            }
        }
        new_shapes.emplace(input_name, new_shape);
    }
    model->reshape(new_shapes);
}

void reshape_sliced_head_to_static(std::shared_ptr<ov::Model> lm_head_model,
                                   const uint32_t& batch_dim,
                                   std::size_t max_generation_token_len) {
    // We have only one input with dynamic shapes: output embeds.
    // Output embeds should have "max_generation_token_len" for dimension representing
    // number of embeddings to send to the matmul. Batch size should be equal to "1"
    // for NPU.
    const auto& input = lm_head_model->input(0);
    const auto& partial_shape = input.get_partial_shape();
    NPUW_ASSERT(partial_shape.size() == 3);

    ov::PartialShape new_shape = partial_shape;
    new_shape[batch_dim] = 1;
    // Left dynamic axis will be for number of embeddings
    for (auto i = 0; i < new_shape.rank().get_length(); i++) {
        if (new_shape[i].is_dynamic()) {
            new_shape[i] = max_generation_token_len;
            // Sanity check that only one left dimension is dynamic, as
            // another one should contain embedding space rank
            break;
        }
    }

    lm_head_model->reshape(new_shape);
}

void slice_out_embeds(std::shared_ptr<ov::Model> model,
                      const uint32_t& batch_dim,
                      std::size_t max_generation_token_len) {
    std::shared_ptr<ov::Node> embed_result;
    for (auto&& output : model->outputs()) {
        if (output.get_any_name() == ov::npuw::LLMCompiledModel::output_embeds) {
            embed_result = output.get_node_shared_ptr();
        }
    }

    if (embed_result) {
        auto shape = embed_result->input(0).get_shape();
        // If shape.size() is 3, then last axis should contain the rank of embedding dimension.
        // But 1st and 2nd axes can mean different things.
        // 1st axis can represent the batch size, while 2nd - the number of embeddings,
        // or vice-versa (in chatglm)
        if (shape.size() == 3) {
            OPENVINO_ASSERT(batch_dim <= 1, "Unexpected value of batch_dim: ", batch_dim, ", expected 0 or 1!");
            uint32_t num_embeds_dim = 1 - batch_dim;
            OPENVINO_ASSERT(shape[num_embeds_dim] >= max_generation_token_len,
                            "Number of output embeddings should be greater or equal to the slicing range!");
            if (shape[num_embeds_dim] != max_generation_token_len) {
                std::vector<int32_t> start_pos{
                    static_cast<int32_t>(batch_dim * (shape[num_embeds_dim] - max_generation_token_len)),
                    static_cast<int32_t>(num_embeds_dim * (shape[num_embeds_dim] - max_generation_token_len)),
                    0};
                std::vector<int32_t> stop_pos{static_cast<int32_t>(batch_dim * (shape[num_embeds_dim] - 1)) + 1,
                                              static_cast<int32_t>(num_embeds_dim * (shape[num_embeds_dim] - 1)) + 1,
                                              static_cast<int32_t>(shape[2])};
                auto start = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, start_pos);
                auto stop = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, stop_pos);
                auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                   ov::Shape{3},
                                                                   std::vector<int32_t>{1, 1, 1});

                auto slice = std::make_shared<ov::op::v8::Slice>(embed_result->input_value(0), start, stop, step);

                embed_result->input(0).replace_source_output(slice);
                embed_result->validate_and_infer_types();
                model->validate_nodes_and_infer_types();
            }
        }
    }
}

bool is_cw_compressed(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::string> rt_info_path = {"nncf", "weight_compression", "group_size"};
    if (!model->has_rt_info(rt_info_path)) {
        // NB: Model isn't compressed by NNCF - skip
        return false;
    }
    auto group_size = model->get_rt_info<int>(rt_info_path);
    if (group_size == -1) {
        // NB: Enable DQ for CW quantized models
        return true;
    }
    return false;
}

struct NPUDesc {
    std::string arch;
    int64_t max_tiles = 0;
    bool compiler_dq = false;
    int64_t compiler_ver = 0;
};

std::optional<NPUDesc> extract_npu_descriptor(const std::shared_ptr<const ov::IPlugin>& plugin) {
    const auto all_devices = plugin->get_core()->get_property("NPU", ov::available_devices);
    if (all_devices.empty()) {
        return std::nullopt;
    }

    NPUDesc desc;
    desc.arch = plugin->get_property(ov::device::architecture.name(), ov::AnyMap{}).as<std::string>();
    desc.max_tiles = plugin->get_property(ov::intel_npu::max_tiles.name(), ov::AnyMap{}).as<int64_t>();

    // Don't use reference here!
    const auto supported_properties =
        plugin->get_property(ov::supported_properties.name(), ov::AnyMap{}).as<std::vector<ov::PropertyName>>();
    if (std::find(supported_properties.begin(), supported_properties.end(), "NPU_COMPILER_DYNAMIC_QUANTIZATION") !=
        supported_properties.end()) {
        desc.compiler_dq = true;
    }

    desc.compiler_ver = plugin->get_property(ov::intel_npu::compiler_version.name(), ov::AnyMap{}).as<int64_t>();

    return std::make_optional(std::move(desc));
}

std::optional<ov::Any> pop_option(ov::AnyMap& config, const std::string& option_name) {
    if (auto it = config.find(option_name); it != config.end()) {
        std::optional<ov::Any> found = std::make_optional(it->second);
        config.erase(it);
        return found;
    }
    return std::nullopt;
}

void apply_weights_bank_name(ov::AnyMap& config, const std::string& bank_name) {
    auto it = config.find("NPUW_WEIGHTS_BANK");
    if (it != config.end()) {
        if (it->second.as<std::string>().empty()) {
            NPUW_ASSERT(false && "NPUW_WEIGHTS_BANK is empty in the provided config! Please use non-empty name to "
                                 "share the model weights.");
        }
    } else {
        config["NPUW_WEIGHTS_BANK"] = bank_name;
    }
}

ov::AnyMap get_baseline_common_config(const std::optional<NPUDesc>& npudesc) {
    ov::AnyMap config = {
        {"NPU_COMPILATION_MODE_PARAMS", "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm"},
        {"NPUW_DEVICES", "NPU"},
        {"NPU_USE_NPUW", "YES"},
        {"NPUW_FOLD", "YES"},
        {"NPUW_DCOFF_TYPE", "f16"},
        {"NPUW_DCOFF_SCALE", "YES"},
        {"NPUW_SLICE_OUT", "YES"},
        {"NPUW_FUNCALL_ASYNC", "YES"}};
    // FIXME: this config logic is getting more and more complex
    if (npudesc.has_value() && npudesc->compiler_dq) {
        config.emplace("NPUW_DQ", "YES");
        config.emplace("NPUW_DQ_FULL", "NO");
        config.emplace("NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES");
        config.erase("NPUW_DCOFF_TYPE");
        config.erase("NPUW_DCOFF_SCALE");
    }
    return config;
}

ov::AnyMap get_default_common_config(const std::optional<NPUDesc>& npudesc) {
    // FIXME: add `if_model_contain_slice()` condition for `SLICE_OUT` option.
    auto config = get_baseline_common_config(npudesc);
    const char* npu_l0 = std::getenv("DISABLE_OPENVINO_GENAI_NPU_L0");
    if (npu_l0 && std::atoi(npu_l0) == 1) {
        config.emplace("NPUW_WEIGHTS_BANK_ALLOC", "CPU");
    } else {
        config.emplace("NPUW_FUNCALL_FOR_ALL", "YES");
    }
    return config;
}

ov::AnyMap get_default_prefill_config(const std::shared_ptr<ov::Model>& model, const std::optional<NPUDesc>& npudesc) {
    auto config = get_default_common_config(npudesc);
    if (npudesc.has_value() && npudesc->arch == "4000" && npudesc->max_tiles != -1) {
        config.emplace("NPU_TILES", npudesc->max_tiles);
    }
    // Specify NPUW DQ if Compiler DQ is not enabled
    if (!npudesc.has_value() || !npudesc->compiler_dq) {
        if (is_cw_compressed(model)) {
            config.emplace("NPUW_DQ", "YES");
        } else {
            config.emplace("NPUW_PMM", "NO");
        }
    }
    return config;
}

ov::AnyMap get_default_generate_config(const std::optional<NPUDesc>& npudesc,
                                       const ::intel_npu::npuw::llm::GenerateHint hint) {
    auto config = get_default_common_config(npudesc);
    if (hint == ::intel_npu::npuw::llm::GenerateHint::BEST_PERF) {
        config.emplace("NPUW_ONLINE_PIPELINE", "NONE");
    }
    if (hint == ::intel_npu::npuw::llm::GenerateHint::FAST_COMPILE) {
        config.emplace("NPUW_UNFOLD_IREQS", "YES");
    }
    // Specify NPUW DQ if Compiler DQ is not enabled
    if (!npudesc.has_value() || !npudesc->compiler_dq) {
        config.emplace("NPUW_DQ", "YES");
    }
    // We don't need slice out for kv cache model, especially for speculative decoding which need
    // to generate more than 1 token for each inference
    config.erase("NPUW_SLICE_OUT");
    return config;
}

ov::AnyMap get_default_lm_head_config(const std::optional<NPUDesc>& npudesc) {
    auto config = get_default_common_config(npudesc);
    config.erase("NPUW_SLICE_OUT");
    config.erase("NPUW_FUNCALL_ASYNC");
    config.emplace("NPUW_ONLINE_PIPELINE", "NONE");
    return config;
}

void merge_config_with(ov::AnyMap& lhs, const ov::AnyMap& rhs) {
    for (const auto& [key, value] : rhs) {
        // NB: Overwrite the value if key already exists
        if (auto it = lhs.find(key); it != lhs.end()) {
            it->second = value;
        } else {
            lhs.emplace(key, value);
        }
    }
}

void split_llm_properties(const ov::AnyMap& properties, ov::AnyMap& llm_properties, ov::AnyMap& other_properties) {
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        if (it->first.find("NPUW_LLM") != it->first.npos) {
            llm_properties.insert(*it);
        } else {
            other_properties.insert(*it);
        }
    }
}

void refine_dynamic_props(ov::AnyMap& llm_properties, const std::optional<NPUDesc>& npudesc) {
    if (!npudesc) {
        // No NPU device detected - no idea about the actual capabilities.
        return;
    }

    if (llm_properties.count(ov::intel_npu::npuw::llm::prefill_chunk_size.name())) {
        // The chunk size value is enforced by the config, keep it
        return;
    }

    if (npudesc->compiler_ver < ONEAPI_MAKE_VERSION(7, 22)) {
        // Specify larger chunk size for older compiler versions
        LOG_VERB("Default the prefill chunk size to 1024");
        llm_properties["NPUW_LLM_PREFILL_CHUNK_SIZE"] = 1024;
    }
}

void update_config_for_whisper(ov::AnyMap& config) {
    config.erase("NPUW_SLICE_OUT");
}

void update_config_for_text_embed(ov::AnyMap& config) {
    config.erase("NPUW_SLICE_OUT");
}

std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        result.emplace(value.first, value.second.as<std::string>());
    }
    return result;
}

// Detect if the model is a Mixture-of-Experts (MoE) architecture
// by checking if any node name matches MoE patterns: layers.*.mlp.router or layers.*.mlp.experts
bool is_moe_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ops()) {
        const std::string& node_name = op->get_friendly_name();
        // Check for MoE-specific patterns:
        // - layers.*.mlp.router (router network for expert selection)
        // - layers.*.mlp.experts (expert networks)
        // Note: "expert" also matches "experts" (plural)
        if (node_name.find(ov::npuw::patterns::moe::MLP_ROUTER_NAME) != std::string::npos ||
            node_name.find(ov::npuw::patterns::moe::MLP_EXPERT_NAME) != std::string::npos) {
            LOG_INFO("Detected MoE model: found node with MoE pattern - " << node_name);
            return true;
        }
    }
    LOG_DEBUG("Non-MoE model detected: no .mlp.router or .mlp.expert nodes found");
    return false;
}

// Apply MoE-specific optimizations to stage configuration based on hint
void apply_moe_optimizations(ov::AnyMap& stage_config,
                             ::intel_npu::npuw::llm::MoEHint moe_hint,
                             const std::string& stage_name) {
    // MoE expert and router pattern isolation options
    const ov::AnyMap expert_opts = {
        {"NPUW_ONLINE_PIPELINE", "REP"},
        {"NPUW_ONLINE_ISOLATE", "MOE"},
        {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "4"},
        {"NPUW_UNFOLD_IREQS", "NO"},
    };

    if (moe_hint == ::intel_npu::npuw::llm::MoEHint::HOST_ROUTED) {
        LOG_INFO("MoE architecture optimization for " << stage_name
                                                      << " stage: HOST_ROUTED (host-side expert routing)");
        merge_config_with(stage_config, expert_opts);
    } else if (moe_hint == ::intel_npu::npuw::llm::MoEHint::DEVICE_ROUTED) {
        NPUW_ASSERT(false && "MoE DEVICE_ROUTED is not yet implemented! "
                             "DEVICE_ROUTED will use in-graph gather-based expert selection to avoid "
                             "graph splitting and reduce host-device communication overhead. "
                             "This feature is planned for future releases.");
    } else if (moe_hint == ::intel_npu::npuw::llm::MoEHint::DENSE) {
        LOG_INFO("MoE architecture optimization for " << stage_name << " stage: DENSE (all experts active)");
        // DENSE mode requires CPU-only device due to extremely long NPU compilation time and high resource consumption
        auto npuw_devices =
            stage_config.count("NPUW_DEVICES") ? stage_config.at("NPUW_DEVICES").as<std::string>() : "NPU";
        NPUW_ASSERT(npuw_devices == "CPU" &&
                    "MoE DENSE mode requires CPU-only device (NPUW_DEVICES must be 'CPU'). "
                    "DENSE activates all experts simultaneously, causing extremely long NPU compilation time. "
                    "Please set NPUW_DEVICES to 'CPU'.");
    }
}

}  // namespace

void ov::npuw::LLMCompiledModel::convert_stateful_lora_to_stateless(std::shared_ptr<ov::Model>& model) {
    typedef std::shared_ptr<ov::op::util::AssignBase> PAssign;
    typedef std::shared_ptr<ov::op::util::ReadValueBase> PReadValue;
    std::vector<PReadValue> readValues;
    std::vector<PAssign> assigns;
    auto sinks = model->get_sinks();
    for (size_t i = 0; i < sinks.size(); ++i) {
        if (auto assign = ov::as_type_ptr<ov::op::util::AssignBase>(sinks[i])) {
            auto variable_name = assign->get_variable_id();
            if (!ov::npuw::util::matchLoRAMatMulAString(variable_name) &&
                !ov::npuw::util::matchLoRAMatMulBString(variable_name) &&
                !ov::npuw::util::matchLoRAMatMulAlphaString(variable_name)) {
                continue;
            }

            auto read_value = ov::as_type_ptr<ov::op::util::ReadValueBase>(assign->get_input_node_shared_ptr(0));
            OPENVINO_ASSERT(read_value, "Can't find ReadValue");
            readValues.push_back(read_value);
            assigns.push_back(assign);
        }
    }

    ov::ParameterVector new_parameters;
    new_parameters.reserve(readValues.size());
    for (size_t i = 0; i < readValues.size(); ++i) {
        auto read_value = readValues[i];
        auto variable_name = read_value->get_variable_id();
        const auto element_type = read_value->get_output_element_type(0);
        const auto shape = read_value->get_output_partial_shape(0);

        auto parameter = std::make_shared<ov::op::v0::Parameter>(element_type, shape);
        ov::op::util::set_name(*parameter, variable_name);
        replace_node(read_value, parameter);

        auto assign = assigns[i];
        model->remove_sink(assign);
        model->remove_variable(model->get_variable_by_id(variable_name));
        new_parameters.push_back(parameter);
    }

    model->add_parameters(new_parameters);
}

std::vector<std::shared_ptr<ov::Model>> ov::npuw::LLMCompiledModel::create_generate_model_variants(
    const std::shared_ptr<ov::Model>& generate_model,
    const KVAxesPosition& axes,
    const uint32_t whisper_lhs_seq_size) {
    const uint32_t total_kv_size = m_kvcache_desc.total_size;
    const uint32_t min_response_len = total_kv_size - m_kvcache_desc.max_prompt_size;
    const uint32_t max_generation_token_len = m_kvcache_desc.max_generation_token_len;
    const bool enable_generate_pyramid = m_cfg.get<::intel_npu::NPUW_LLM_GENERATE_PYRAMID>();

    // Check if generate pyramid feature is enabled
    if (enable_generate_pyramid) {
        LOG_INFO("Generate pyramid feature is ENABLED");
        LOG_INFO(
            "Creating multiple generate model variants with stepping: 1K+min_response_len, 2K+min_response_len, etc.");

        // Determine KV cache size steps: (1K + min_response_len), (2K + min_response_len), (4K + min_response_len), (8K
        // + min_response_len), etc.
        std::vector<uint32_t> kv_size_steps;
        for (uint32_t base_size = 1024; base_size + min_response_len <= total_kv_size; base_size *= 2) {
            kv_size_steps.push_back(base_size + min_response_len);
        }
        // Always include the total size if it's not already in the list
        if (kv_size_steps.empty() || kv_size_steps.back() < total_kv_size) {
            kv_size_steps.push_back(total_kv_size);
        }

        LOG_DEBUG("KV cache size variants: ");
        for (const auto& size : kv_size_steps) {
            LOG_DEBUG("  - " << size);
        }

        // Store the sizes for runtime selection
        m_kvcache_sizes = std::move(kv_size_steps);
    } else {
        LOG_INFO("KV cache variants feature is DISABLED - using single model");
        // Use only the total size (traditional single-model approach)
        m_kvcache_sizes = {total_kv_size};
    }

    // Create generate model variants
    LOG_INFO("Creating " << m_kvcache_sizes.size() << " generate model variants...");
    std::vector<std::shared_ptr<ov::Model>> generate_model_variants;
    generate_model_variants.reserve(m_kvcache_sizes.size());

    for (size_t i = 0; i < m_kvcache_sizes.size(); ++i) {
        const uint32_t kv_size = m_kvcache_sizes[i];

        auto generate_variant = (kv_size == total_kv_size) ? generate_model : generate_model->clone();
        LOG_DEBUG("Variant " << (i + 1) << "/" << m_kvcache_sizes.size() << " (size=" << kv_size
                             << "): reshaping to static");

        // Reshape to target size
        reshape_to_static(generate_variant,
                          max_generation_token_len,
                          kv_size,
                          axes,
                          m_max_lora_rank,
                          whisper_lhs_seq_size);

        // Set unique name for this variant
        generate_variant->set_friendly_name(generate_model->get_friendly_name() + "_kv" + std::to_string(kv_size));
        generate_model_variants.push_back(generate_variant);
    }
    LOG_INFO("Created all generate model variants");

    return generate_model_variants;
}

void ov::npuw::LLMCompiledModel::compile_generate_model_variants(
    const std::vector<std::shared_ptr<ov::Model>>& generate_model_variants,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& generate_config) {
    // Compile multiple generate model variants with different sizes
    LOG_INFO("Compiling " << m_kvcache_sizes.size() << " generate model variants...");
    m_generate_compiled_variants.reserve(m_kvcache_sizes.size());

    for (size_t i = 0; i < m_kvcache_sizes.size(); ++i) {
        const uint32_t kv_size = m_kvcache_sizes[i];
        LOG_DEBUG("Compiling generate variant " << (i + 1) << "/" << m_kvcache_sizes.size()
                                                << " with size: " << kv_size);

        // Use the already prepared variant model
        auto& generate_variant = generate_model_variants[i];

        // Compile the variant
        auto compiled_variant = std::dynamic_pointer_cast<ov::npuw::CompiledModel>(
            ov::npuw::ICompiledModel::create(generate_variant, plugin, generate_config));
        NPUW_ASSERT(compiled_variant && "Can't create ov::npuw::CompiledModel for generate variant!");

        m_generate_compiled_variants.push_back(compiled_variant);
        LOG_DEBUG("Successfully compiled generate variant with size: " << kv_size);
    }

    // Keep the original compiled model for backward compatibility (using the largest size)
    m_kvcache_compiled = m_generate_compiled_variants.back();
}

ov::npuw::LLMCompiledModel::LLMCompiledModel(const std::shared_ptr<ov::Model>& model,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             const ov::AnyMap& properties)
    : ov::npuw::ICompiledModel(model, plugin),
      m_name(model->get_friendly_name()),
      m_options_desc(std::make_shared<::intel_npu::OptionsDesc>()),
      m_cfg(m_options_desc) {
    LOG_DEBUG("Creating LLMCompiledModel");
    LOG_BLOCK();

    ::intel_npu::registerNPUWLLMOptions(*m_options_desc);

    const auto npudesc = extract_npu_descriptor(plugin);

    ov::AnyMap npuw_llm_props;
    ov::AnyMap other_props;
    split_llm_properties(properties, npuw_llm_props, other_props);
    auto use_whisper_key = pop_option(other_props, std::string("NPUW_WHISPER"));
    auto use_eagle_key = pop_option(other_props, std::string("NPUW_EAGLE"));
    auto kv_cache_precision_hint = pop_option(other_props, ov::hint::kv_cache_precision.name());

    // Solely used for serialization at the moment
    m_non_llm_props = other_props;

    // Remove "NPUW_LLM_PREFILL_CONFIG", "NPUW_LLM_GENERATE_CONFIG" from map,
    // to not pass them into ::intel_npu::Config object, as we don't need to
    // preserve them somewhere.
    auto prefill_config_opt = pop_option(npuw_llm_props, std::string("NPUW_LLM_PREFILL_CONFIG"));
    auto generate_config_opt = pop_option(npuw_llm_props, std::string("NPUW_LLM_GENERATE_CONFIG"));
    auto prefill_config_addition = pop_option(npuw_llm_props, std::string("++NPUW_LLM_PREFILL_CONFIG"));
    auto generate_config_addition = pop_option(npuw_llm_props, std::string("++NPUW_LLM_GENERATE_CONFIG"));
    // Also make these maps for third: lm head model, in case it will be created:
    auto lm_head_config_opt = pop_option(npuw_llm_props, std::string("NPUW_LLM_SHARED_HEAD_CONFIG"));
    auto lm_head_config_addition = pop_option(npuw_llm_props, std::string("++NPUW_LLM_SHARED_HEAD_CONFIG"));
    refine_dynamic_props(npuw_llm_props, npudesc);
    m_cfg.update(any_copy(npuw_llm_props));

    auto kv_kache_storage_type = ov::element::f16;

    // kv-cache-precision changes to fp8 does make sense unconditionally only if LPT passes succesfully applied
    if (m_cfg.get<::intel_npu::NPUW_LLM_OPTIMIZE_FP8>()) {
        kv_kache_storage_type = optimize_kv_cache_storage(model);
    }

    // ov::kv_cache_precision hint can additionally change kv-cache precision, but it might lead to less accurate
    // results
    if (kv_cache_precision_hint.has_value()) {
        auto suggested_kv_cache_precision = kv_cache_precision_hint.value().as<ov::element::Type>();
        if (kv_kache_storage_type != suggested_kv_cache_precision) {
            LOG_WARN("KV-cache precision HINT: " << suggested_kv_cache_precision << " applied");
            kv_kache_storage_type = suggested_kv_cache_precision;
        }
    }

    m_is_whisper = use_whisper_key.value_or(false).as<bool>() == true;
    if (m_is_whisper) {
        m_cfg.update({{"NPUW_LLM_SHARED_HEAD", "NO"}});
        m_cfg.update({{"NPUW_LLM_PREFILL_CHUNK_SIZE", "0"}});
        m_cfg.update({{"NPUW_LLM_CACHE_ROPE", "NO"}});
        m_cfg.update({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "NO"}});
    }

    m_is_eagle = use_eagle_key.value_or(false).as<bool>() == true;
    if (m_is_eagle) {
        LOG_INFO("Eagle3 speculative decoding mode enabled");
    }

    // NB: PREFILL_HINT is now compatible with the PREFILL_CONFIG section, unlike for
    // the generate model they're not mutually exclusive
    const ::intel_npu::npuw::llm::PrefillHint prefill_hint = m_cfg.get<::intel_npu::NPUW_LLM_PREFILL_HINT>();
    m_prefill_chunk_size = m_cfg.get<::intel_npu::NPUW_LLM_PREFILL_CHUNK_SIZE>();
    m_use_chunk_prefill = (prefill_hint == ::intel_npu::npuw::llm::PrefillHint::DYNAMIC && m_prefill_chunk_size > 0);

    uint32_t max_prompt_len = align_to(m_cfg.get<::intel_npu::NPUW_LLM_MAX_PROMPT_LEN>(), 64u);
    const uint32_t min_response_len = align_to(m_cfg.get<::intel_npu::NPUW_LLM_MIN_RESPONSE_LEN>(), 64u);
    uint32_t max_generation_token_len = m_cfg.get<::intel_npu::NPUW_LLM_MAX_GENERATION_TOKEN_LEN>();
    if (max_generation_token_len != 1) {
        max_generation_token_len = align_to(max_generation_token_len, 8u);
    }

    // If chunk size covers the entire prompt, just follow the static behavior.
    // Otherwise, use chunking and align the prompt size to the chunk size.
    if (m_use_chunk_prefill) {
        OPENVINO_ASSERT(
            !ov::npuw::util::has_input(model, "token_type_ids") || !ov::npuw::util::has_input(model, "inputs_embeds"),
            "Chunking is not implemented for Gemma model family yet. "
            "Please set NPUW_LLM_PREFILL_HINT to 'STATIC'");
        if (m_prefill_chunk_size >= max_prompt_len) {
            m_use_chunk_prefill = false;
        } else {
            const auto is_power_of_two = [](uint64_t n) {
                return n > 0 && (n & (n - 1)) == 0;
            };
            if (!is_power_of_two(m_prefill_chunk_size)) {
                OPENVINO_THROW("Configuration Error: chunk size (",
                               m_prefill_chunk_size,
                               ") is not power of 2. Please adjust NPUW_LLM_PREFILL_CHUNK_SIZE.");
            }
            max_prompt_len = align_to(max_prompt_len, static_cast<uint32_t>(m_prefill_chunk_size));
        }

        m_enable_prefix_caching = m_cfg.get<::intel_npu::NPUW_LLM_ENABLE_PREFIX_CACHING>();
        if (m_enable_prefix_caching) {
            LOG_INFO("Prefix caching is enabled");
            m_prefix_caching_block_size = m_cfg.get<::intel_npu::NPUW_LLM_PREFIX_CACHING_BLOCK_SIZE>();
            if (!is_aligned_to(static_cast<uint32_t>(m_prefill_chunk_size),
                               static_cast<uint32_t>(m_prefix_caching_block_size))) {
                LOG_INFO("Prefix caching block size is adjusted to " << m_prefill_chunk_size);
                m_prefix_caching_block_size = m_prefill_chunk_size;
            }
            m_prefix_caching_max_num_blocks = m_cfg.get<::intel_npu::NPUW_LLM_PREFIX_CACHING_MAX_NUM_BLOCKS>();
            LOG_INFO("Prefix caching block size: " << m_prefix_caching_block_size);
            LOG_INFO("Prefix caching maximum number of blocks: " << m_prefix_caching_max_num_blocks);
        }
    }

    LOG_VERB("Enabled prefill chunking: " << m_use_chunk_prefill);
    LOG_VERB("Prefill chunk size: " << m_prefill_chunk_size);
    LOG_VERB("Maximum prompt length: " << max_prompt_len);

    const uint32_t batch_dim = m_cfg.get<::intel_npu::NPUW_LLM_BATCH_DIM>();
    const uint32_t seq_len_dim = m_cfg.get<::intel_npu::NPUW_LLM_SEQ_LEN_DIM>();
    KVAxesPosition axes{batch_dim, seq_len_dim};

    LOG_DEBUG("Creating kvcache model as clone of passed one.");
    auto kvcache_model = model->clone();

    auto use_text_embed_key = pop_option(other_props, std::string("NPUW_TEXT_EMBED"));
    m_is_embedding = use_text_embed_key.value_or(false).as<bool>() == true;

    if (m_is_embedding) {
        LOG_DEBUG("Text-embedding model rebuild");
        ov::npuw::util::prepare_text_embedding_model(kvcache_model, seq_len_dim);
    } else {
        LOG_DEBUG("Transform kvcache model from stateful to stateless.");
        ov::pass::StatefulToStateless().run_on_model(kvcache_model);
    }

    convert_stateful_lora_to_stateless(kvcache_model);

    LOG_DEBUG("   ...also convert BF16 to FP16");
    // Note: we need to identify original bf16 constants for potential weightless deserialization later
    // And only then do bf16 to f16 transformation
    m_bf16_consts = ov::npuw::s11n::get_bf16_consts(model);
    ov::pass::ConvertPrecision(ov::element::bf16, ov::element::f16).run_on_model(kvcache_model);

    bool shared_head_enabled = m_cfg.get<::intel_npu::NPUW_LLM_SHARED_HEAD>();
    std::shared_ptr<ov::Model> lm_head_model = nullptr;
    if (shared_head_enabled) {
        LOG_DEBUG("Trying to separate Vocabulary matrix multiplication op into additional model...");
        lm_head_model = cut_lm_head(kvcache_model);
        if (lm_head_model) {
            LOG_INFO("Three-model pipeline will be created: LM head will be shared between prefill and generate.");
        } else {
            LOG_WARN("Three-model pipeline is requested, but LM head cutting is failed,"
                     " two-model pipeline will be created!");
        }
    } else {
        LOG_INFO("Two-model pipeline will be created.");
    }

    if (!m_is_whisper) {
        LOG_DEBUG("Try patch Phi-3 sliding window mask, if it exists.");
        patch_phi3_sliding_mask(kvcache_model);
    }

    LOG_DEBUG("Creating prefill model as clone of transformed kvcache one.");
    auto prefill_model = kvcache_model->clone();
    prefill_model->set_friendly_name(kvcache_model->get_friendly_name() + "_prefill");

    m_kvcache_desc =
        KVCacheDesc{max_prompt_len, max_prompt_len + min_response_len, 0u, seq_len_dim, max_generation_token_len};

    uint32_t whisper_lhs_seq_size = 0;  // Not applicable for LLMs/VLMs
    if (m_is_whisper) {
        axes = KVAxesPosition{whisper_batch_dim, whisper_seq_len_dim};
        m_kvcache_desc = KVCacheDesc{whisper_max_prompt_size, whisper_kvcache_size, 0u, whisper_seq_len_dim, 1u};
        whisper_lhs_seq_size =
            static_cast<uint32_t>(prefill_model->input("encoder_hidden_states").get_partial_shape()[1].get_length());

        ov::npuw::util::prepare_whisper_prefill_model(prefill_model,
                                                      m_kvcache_desc.max_prompt_size,
                                                      whisper_lhs_seq_size);  // Whisper decoder model
        ov::npuw::util::prepare_whisper_kvcache_model(kvcache_model);         // Whisper decoder_with_past model
    }

    LOG_DEBUG("Make prefill model with static shapes");
    m_max_lora_rank = m_cfg.get<::intel_npu::NPUW_LLM_MAX_LORA_RANK>();
    if (m_use_chunk_prefill) {
        reshape_to_static(prefill_model,
                          static_cast<uint32_t>(m_prefill_chunk_size),
                          m_kvcache_desc.max_prompt_size,
                          axes,
                          m_max_lora_rank);
    } else {
        reshape_to_static(prefill_model,
                          m_kvcache_desc.max_prompt_size,
                          m_kvcache_desc.max_prompt_size,
                          axes,
                          m_max_lora_rank,
                          whisper_lhs_seq_size);
    }
    LOG_DEBUG("Make kvcache model with static shapes");

    // Create generate model variants with different sizes
    auto generate_model_variants = create_generate_model_variants(kvcache_model, axes, whisper_lhs_seq_size);

    if (lm_head_model) {
        LOG_DEBUG("Shared LM head: slice the prefill output");
        // KVCache model is already reshaped to [1, max_generation_token_len, embed size],
        // so only apply slice to the Prefill model:
        slice_out_embeds(prefill_model, axes.batch, m_kvcache_desc.max_generation_token_len);
        LOG_DEBUG("Make LM head model with static shapes");
        reshape_sliced_head_to_static(lm_head_model, axes.batch, m_kvcache_desc.max_generation_token_len);
    }

    LOG_DEBUG("5.1, decompose GroupQueryAttention OP");
    decompose_GQA(prefill_model, true);
    for (auto& model_variant : generate_model_variants) {
        decompose_GQA(model_variant, false);
    }

    const auto prefill_attn_hint = m_cfg.get<::intel_npu::NPUW_LLM_PREFILL_ATTENTION_HINT>();
    const auto generate_attn_hint = m_cfg.get<::intel_npu::NPUW_LLM_GENERATE_ATTENTION_HINT>();
    const bool prefill_attn_dyn = prefill_attn_hint == ::intel_npu::npuw::llm::AttentionHint::DYNAMIC;
    const bool generate_attn_dyn = generate_attn_hint == ::intel_npu::npuw::llm::AttentionHint::DYNAMIC;

    const bool prefill_attn_pyramid = prefill_attn_hint == ::intel_npu::npuw::llm::AttentionHint::PYRAMID;
    const bool generate_attn_pyramid = generate_attn_hint == ::intel_npu::npuw::llm::AttentionHint::PYRAMID;

    const bool prefill_attn_hfa = prefill_attn_hint == ::intel_npu::npuw::llm::AttentionHint::HFA;
    const bool generate_attn_hfa = generate_attn_hint == ::intel_npu::npuw::llm::AttentionHint::HFA;

    const bool optimize_v_tensors = m_cfg.get<::intel_npu::NPUW_LLM_OPTIMIZE_V_TENSORS>();
    if (optimize_v_tensors) {
        LOG_DEBUG("Check and apply opt layout");
        LOG_BLOCK();
        // Only optimize V tensors for static attention types
        if (!generate_attn_dyn) {
            // Apply optimization to all variants and track results
            size_t optimized_count = 0;
            for (auto& model_variant : generate_model_variants) {
                if (ov::npuw::util::optimize_value_tensors(model_variant, false)) {
                    ++optimized_count;
                }
            }

            // Check consistency: either all or none should be optimized
            if (optimized_count == generate_model_variants.size()) {
                LOG_DEBUG("V-tensors transposed in generate model variants");
                m_kvcache_desc.v_tensors_transposed_gen = true;
            } else if (optimized_count == 0) {
                LOG_DEBUG("No V-tensors were optimized");
                m_kvcache_desc.v_tensors_transposed_gen = false;
            } else {
                // Partial optimization is not allowed
                OPENVINO_ASSERT(false,
                                "Partial optimization detected: ",
                                optimized_count,
                                " out of ",
                                generate_model_variants.size(),
                                " variants were optimized, which is not allowed.");
            }
        }
        if (!prefill_attn_dyn && ov::npuw::util::optimize_value_tensors(prefill_model, true)) {
            LOG_DEBUG("V-tensors tranposed in prefill model");
            m_kvcache_desc.v_tensors_transposed_pre = true;
        }
    } else {
        LOG_DEBUG("Check and apply opt layout --- SKIPPED");
    }

    if (!m_is_embedding) {
        if (!m_use_chunk_prefill) {
            // TODO: sometimes it is ok if we cannot find any empty inputs or not?
            NPUW_ASSERT(remove_empty_kv_inputs(prefill_model));
        } else {
            LOG_DEBUG("Don't remove input key/values from prefill model.");
            LOG_DEBUG("Ask prefill model to output key/values for prefill chunk size tokens.");
            prefill_model = redirect_new_kv_to_output(prefill_model);
        }

        LOG_DEBUG("Optimize generate model to output key/values for new token.");
        for (size_t i = 0; i < generate_model_variants.size(); ++i) {
            generate_model_variants[i] = redirect_new_kv_to_output(generate_model_variants[i]);
        }
    }

    LOG_DEBUG("Converting KV-cache in generate model to FP16.");
    for (size_t i = 0; i < generate_model_variants.size(); ++i) {
        generate_model_variants[i] = cvt_kvcache_to_low_precision(generate_model_variants[i], kv_kache_storage_type);
    }
    LOG_DEBUG("Converting KV-cache in prefill model to FP16.");
    prefill_model = cvt_kvcache_to_low_precision(prefill_model, kv_kache_storage_type);

    auto prefill_config =
        prefill_config_opt.value_or(get_default_prefill_config(prefill_model, npudesc)).as<ov::AnyMap>();

    // NB: GENERATE_HINT is only applicable for default generate config!
    if (generate_config_opt.has_value() && npuw_llm_props.count(ov::intel_npu::npuw::llm::generate_hint.name())) {
        OPENVINO_THROW("GENERATE_HINT only works with default generate config!");
    }
    const ::intel_npu::npuw::llm::GenerateHint generate_hint = m_cfg.get<::intel_npu::NPUW_LLM_GENERATE_HINT>();
    auto generate_config =
        generate_config_opt.value_or(get_default_generate_config(npudesc, generate_hint)).as<ov::AnyMap>();

    auto prefill_config_addition_value =
        prefill_config_addition.has_value() ? prefill_config_addition.value().as<ov::AnyMap>() : ov::AnyMap{};
    auto generate_config_addition_value =
        generate_config_addition.has_value() ? generate_config_addition.value().as<ov::AnyMap>() : ov::AnyMap{};

    merge_config_with(prefill_config, other_props);
    merge_config_with(generate_config, other_props);
    merge_config_with(prefill_config, prefill_config_addition_value);
    merge_config_with(generate_config, generate_config_addition_value);

    // Convert LLM-specific attention hints to NPUW_ATTN
    if (npuw_llm_props.count("NPUW_LLM_PREFILL_ATTENTION_HINT")) {
        prefill_config["NPUW_ATTN"] = npuw_llm_props["NPUW_LLM_PREFILL_ATTENTION_HINT"];
    }
    if (npuw_llm_props.count("NPUW_LLM_GENERATE_ATTENTION_HINT")) {
        generate_config["NPUW_ATTN"] = npuw_llm_props["NPUW_LLM_GENERATE_ATTENTION_HINT"];
    }

    // Generate a random weights bank name unique to this LLMCompiledModel object
    auto weights_bank_name = ov::npuw::util::generate_random_string();
    LOG_VERB("Generated a unique weights bank name: " << weights_bank_name);
    apply_weights_bank_name(prefill_config, weights_bank_name);
    apply_weights_bank_name(generate_config, weights_bank_name);

    // Handle attention hints. FIXME: Maybe it makes sense to make those
    // mutually exclusive with the precise configuration sections as well
    const ov::AnyMap dyn_attn_opts = {
        {"NPUW_ONLINE_PIPELINE", "REP"},
        {"NPUW_ONLINE_ISOLATE", "ATTN"},
        {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "4"},
        {"NPUW_UNFOLD_IREQS", "NO"},
    };
    if (prefill_attn_dyn || prefill_attn_pyramid || prefill_attn_hfa) {
        merge_config_with(prefill_config, dyn_attn_opts);
    }
    if (generate_attn_dyn || generate_attn_pyramid || generate_attn_hfa) {
        merge_config_with(generate_config, dyn_attn_opts);
    }

    // Auto-detect MoE model by scanning for router/expert nodes
    const bool is_moe = is_moe_model(kvcache_model);
    if (is_moe) {
        // Apply MoE optimizations for prefill stage
        const auto prefill_moe_hint = m_cfg.get<::intel_npu::NPUW_LLM_PREFILL_MOE_HINT>();
        apply_moe_optimizations(prefill_config, prefill_moe_hint, "PREFILL");

        // Apply MoE optimizations for generate stage
        const auto generate_moe_hint = m_cfg.get<::intel_npu::NPUW_LLM_GENERATE_MOE_HINT>();
        apply_moe_optimizations(generate_config, generate_moe_hint, "GENERATE");
    }

    // Note: with dynamic attention in EITHER STAGE, we have to
    // explicitly disable the run-time fallback to so extra ov::Model
    // references won't be held by the npuw::CompiledModel, resulting
    // in a higher memory consumption. This behavior should be reworked!
    // The reason here is that NPUW_DEVICES may come as a global setting,
    // impacting all the stages.
    if (prefill_attn_dyn || generate_attn_dyn) {
        const ov::AnyMap no_runtime_fallback = {{"NPUW_FALLBACK_EXEC", "NO"}};
        merge_config_with(prefill_config, no_runtime_fallback);
        merge_config_with(generate_config, no_runtime_fallback);
    }

    if (m_is_whisper) {
        update_config_for_whisper(prefill_config);
    }

    if (m_is_embedding) {
        update_config_for_text_embed(prefill_config);
    }

    if (m_cfg.get<::intel_npu::NPUW_LLM_CACHE_ROPE>()) {
        LOG_DEBUG("Caching preROPE ");
        const uint32_t CACHE_ROPE_START = 2048;
        const bool is_best = (generate_hint == ::intel_npu::npuw::llm::GenerateHint::BEST_PERF);

        if (!is_best || (max_prompt_len >= CACHE_ROPE_START)) {
            LOG_DEBUG("Enable RoPE Cache for prefill");
            ov::npuw::patterns::pre_compute::RopeCache rope_prefill_cacher(max_prompt_len);
            rope_prefill_cacher.run_on_model(prefill_model);
        }

        // Apply RoPE Cache to all generate variant models
        for (size_t i = 0; i < generate_model_variants.size(); ++i) {
            const uint32_t kv_size = m_kvcache_sizes[i];
            if (!is_best || (kv_size >= CACHE_ROPE_START)) {
                LOG_DEBUG("Enable RoPE Cache for generate variant with size: " << kv_size);
                ov::npuw::patterns::pre_compute::RopeCache rope_cacher(kv_size);
                rope_cacher.run_on_model(generate_model_variants[i]);
            }
        }
    }

    // Regularize models for the better partitioning assuming it is a transformer
    // Apply these transformations to all variant models
    {
        ov::pass::GraphRewrite rewr;
        rewr.add_matcher<ov::npuw::patterns::regularize::AttentionBroadcast>();
        rewr.add_matcher<ov::npuw::patterns::regularize::AttentionBroadcast2>();
        if (generate_attn_dyn || generate_attn_pyramid || generate_attn_hfa) {
            for (auto& model_variant : generate_model_variants) {
                rewr.run_on_model(model_variant);
            }
        }
        if (prefill_attn_dyn || prefill_attn_pyramid || prefill_attn_hfa) {
            rewr.run_on_model(prefill_model);
        }

        // FIXME: generally all these patterns are supposed to improve the partitioning - thus
        // the performance. However, ShapeOfParameter seems to be working fine for all known case,
        // while AttentionBroadcast patterns might break the partitioning (related to F16IC).
        ov::pass::GraphRewrite rewr2;
        rewr2.add_matcher<ov::npuw::patterns::regularize::ShapeOfParameter>();
        for (auto& model_variant : generate_model_variants) {
            rewr2.run_on_model(model_variant);
        }
        rewr2.run_on_model(prefill_model);
    }

    // Compile multiple generate model variants with different sizes
    compile_generate_model_variants(generate_model_variants, plugin, generate_config);

    m_prefill_compiled = std::dynamic_pointer_cast<ov::npuw::CompiledModel>(
        ov::npuw::ICompiledModel::create(prefill_model, plugin, prefill_config));
    NPUW_ASSERT(m_prefill_compiled && "Can't create ov::npuw::CompiledModel for passed prefill "
                                      "model and its config, please check passed config.");
    if (lm_head_model) {
        auto lm_head_config = get_default_lm_head_config(npudesc);
        merge_config_with(lm_head_config, other_props);
        auto lm_head_config_addition_value = lm_head_config_addition.value_or(ov::AnyMap{}).as<ov::AnyMap>();
        merge_config_with(lm_head_config, lm_head_config_addition_value);

        apply_weights_bank_name(lm_head_config, weights_bank_name);

        m_lm_head_compiled = std::dynamic_pointer_cast<ov::npuw::CompiledModel>(
            ov::npuw::ICompiledModel::create(lm_head_model, plugin, lm_head_config));
        NPUW_ASSERT(m_lm_head_compiled);
    }

    implement_properties();
    LOG_DEBUG("Done");
}

ov::npuw::LLMCompiledModel::LLMCompiledModel(const std::shared_ptr<ov::Model>& model,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             const bool serialized)
    : ov::npuw::ICompiledModel(model, plugin),
      m_name(model->get_friendly_name()),
      m_options_desc(std::make_shared<::intel_npu::OptionsDesc>()),
      m_cfg(m_options_desc) {
    NPUW_ASSERT(serialized && "This constructor should only be utilized during deserialization!");
    ::intel_npu::registerNPUWLLMOptions(*m_options_desc);
    LOG_DEBUG("LLMCompiledModel is being deserialized, skipping the full constructor flow...");
}

void ov::npuw::LLMCompiledModel::export_model(std::ostream& stream) const {
    using namespace ov::npuw::s11n;

    // Identify encryption flow
    bool encryption_required = false;
    EncryptionCallbacks enc_callbacks;
    if (auto it = m_non_llm_props.find(ov::cache_encryption_callbacks.name());
        it != m_non_llm_props.end() && it->second.as<EncryptionCallbacks>().encrypt) {
        LOG_INFO("Encryption will be done via the function provided.");
        encryption_required = true;
        enc_callbacks.encrypt = it->second.as<EncryptionCallbacks>().encrypt;
    }

    // Identify either full flow or weightless
    bool is_weightless = true;
    if (auto it = m_non_llm_props.find(ov::cache_mode.name());
        it != m_non_llm_props.end() && it->second.as<CacheMode>() == CacheMode::OPTIMIZE_SPEED) {
        LOG_INFO("Serialization will be done via flow with weights.");
        is_weightless = false;
    }

    // Write header regardless of encryption requirement - to identify NPUW serializated blobs
    // Serialize magic number first
    write(stream, NPUW_SERIALIZATION_INDICATOR);
    // Serilize LLMCompiledModel identifier
    write(stream, NPUW_LLM_COMPILED_MODEL_INDICATOR);
    // Serialize general meta info
    write(stream, OPENVINO_VERSION_MAJOR);
    write(stream, OPENVINO_VERSION_MINOR);
    write(stream, OPENVINO_VERSION_PATCH);
    write(stream, std::string(NPUW_SERIALIZATION_VERSION));
    // Serialize encrypted flag
    write(stream, encryption_required);
    // Write flow identifier
    write(stream, is_weightless);

    if (!encryption_required) {
        CompiledContext ctx(false, nullptr, nullptr);
        return serialize(stream, ctx);
    }

    // In case of weightless flow the whole blob will be encrypted on NPUW side.
    std::stringstream non_encrypted_stream;
    if (is_weightless) {
        non_encrypted_stream.copyfmt(stream);
        CompiledContext ctx(false, nullptr, nullptr);
        serialize(non_encrypted_stream, ctx);
        std::string encrypted = enc_callbacks.encrypt(non_encrypted_stream.str());
        write(stream, encrypted);
    } else {
        // In case of blob with weights only encrypt XML part of the model
        CompiledContext ctx(true, enc_callbacks.encrypt, nullptr);
        serialize(stream, ctx);
    }
}

void ov::npuw::LLMCompiledModel::serialize(std::ostream& stream, const ov::npuw::s11n::CompiledContext& ctx) const {
    LOG_INFO("Serializing LLMCompiledModel...");
    LOG_BLOCK();

    using namespace ov::npuw::s11n;

    // Identify either full flow or weightless
    bool is_weightless = true;
    if (auto it = m_non_llm_props.find(ov::cache_mode.name());
        it != m_non_llm_props.end() && it->second.as<CacheMode>() == CacheMode::OPTIMIZE_SPEED) {
        LOG_INFO("Serialization will be done via flow with weights.");
        is_weightless = false;
    }

    auto write_model_meta = [&](std::ostream& model_stream) {
        // Serialize name
        write(model_stream, m_name);

        // Serialize inputs and outputs
        write(model_stream, inputs());
        write(model_stream, outputs());

        // Serialize LLMCompiledModel-specific data
        write(model_stream, m_kvcache_desc.max_prompt_size);
        write(model_stream, m_kvcache_desc.total_size);
        write(model_stream, m_kvcache_desc.num_stored_tokens);
        write(model_stream, m_kvcache_desc.dim);
        write(model_stream, m_kvcache_desc.max_generation_token_len);
        write(model_stream, m_kvcache_desc.v_tensors_transposed_pre);
        write(model_stream, m_kvcache_desc.v_tensors_transposed_gen);
        write(model_stream, m_prefill_chunk_size);
        write(model_stream, m_use_chunk_prefill);
        write(model_stream, m_max_lora_rank);
        write(model_stream, m_enable_prefix_caching);
        write(model_stream, m_prefix_caching_block_size);
        write(model_stream, m_prefix_caching_max_num_blocks);
        write(model_stream, m_is_whisper);
        write(model_stream, m_is_eagle);
        write(model_stream, m_is_embedding);

        // Write config
        write(model_stream, m_cfg);

        // Serialize KV cache model variants
        write(model_stream, m_kvcache_sizes);
        write(model_stream, static_cast<uint32_t>(m_generate_compiled_variants.size()));

        // Serialize CompiledModels
        // Note: no need to pass any encryption here as it's done in export_model()
        CompiledContext enc_ctx(false, nullptr, nullptr, m_bf16_consts);

        // Serialize all generate variants
        for (const auto& compiled_variant : m_generate_compiled_variants) {
            compiled_variant->serialize(model_stream, enc_ctx);
        }

        m_prefill_compiled->serialize(model_stream, enc_ctx);
        const bool is_shared_lm_head = m_lm_head_compiled != nullptr;
        write(model_stream, is_shared_lm_head);
        if (is_shared_lm_head) {
            m_lm_head_compiled->serialize(model_stream, enc_ctx);
        }
    };

    std::stringstream non_encrypted_stream;
    if (ctx.encrypted) {
        NPUW_ASSERT(ctx.encrypt && "Encryption function isn't provided!");
        non_encrypted_stream.copyfmt(stream);
        write_model_meta(non_encrypted_stream);
        std::string encrypted_str = ctx.encrypt(non_encrypted_stream.str());
        write(stream, encrypted_str);
    } else {
        write_model_meta(stream);
    }

    // Serialize bank name
    const auto& kv_bank = m_kvcache_compiled->m_weights_bank;
    const auto& p_bank = m_prefill_compiled->m_weights_bank;
    NPUW_ASSERT(kv_bank && p_bank && kv_bank == p_bank && "Prefill and KVCache models' weight bank should be shared!");
    write(stream, kv_bank->get_name());

    if (!is_weightless) {
        // Serialize weights bank
        // Note: no need to encrypt weights in full flow
        kv_bank->serialize(stream);
    }

    LOG_INFO("Done.");
}

std::shared_ptr<ov::npuw::LLMCompiledModel> ov::npuw::LLMCompiledModel::import_model(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    LOG_INFO("Deserializing LLMCompiledModel...");
    LOG_BLOCK();

    using namespace ov::npuw::s11n;

    // Sanity check magic number
    ov::npuw::s11n::IndicatorType serialization_indicator;
    read(stream, serialization_indicator);
    NPUW_ASSERT(serialization_indicator == NPUW_SERIALIZATION_INDICATOR && "This blob wasn't serialized via NPUW!");

    ov::npuw::s11n::IndicatorType llm_compiled_indicator;
    read(stream, llm_compiled_indicator);
    NPUW_ASSERT(llm_compiled_indicator == NPUW_LLM_COMPILED_MODEL_INDICATOR &&
                "This blob wasn't serialized via LLMCompiledModel!");

    // Deserialize general meta info
    int vmajor, vminor, vpatch;
    std::string s11n_version;
    read(stream, vmajor);
    read(stream, vminor);
    read(stream, vpatch);
    read(stream, s11n_version);

    if (vmajor != OPENVINO_VERSION_MAJOR || vminor != OPENVINO_VERSION_MINOR || vpatch != OPENVINO_VERSION_PATCH ||
        s11n_version != std::string(NPUW_SERIALIZATION_VERSION)) {
        OPENVINO_THROW("This blobs was serialized with different OV version!",
                       "\nSerialized by OV ",
                       vmajor,
                       '.',
                       vminor,
                       '.',
                       vpatch,
                       "\nCurrent OV version ",
                       OPENVINO_VERSION_MAJOR,
                       '.',
                       OPENVINO_VERSION_MINOR,
                       '.',
                       OPENVINO_VERSION_PATCH,
                       "\nNPUW serialized by version ",
                       s11n_version,
                       "\nNPUW current serialization version ",
                       NPUW_SERIALIZATION_VERSION);
    }

    bool encrypted = false;
    read(stream, encrypted);
    bool is_weightless = true;
    read(stream, is_weightless);

    auto read_and_finalize_banks = [&](std::istream& model_stream,
                                       const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        // Deserialize weights bank name
        std::string bank_name;
        read(model_stream, bank_name);

        if (is_weightless) {
            auto bank = ov::npuw::weights::bank(bank_name, compiled->get_plugin()->get_core(), "");

            for (const auto& compiled_variant : compiled->m_generate_compiled_variants) {
                compiled_variant->m_weights_bank = bank;
                compiled_variant->finalize_weights_bank();
            }

            compiled->m_prefill_compiled->m_weights_bank = bank;
            compiled->m_prefill_compiled->finalize_weights_bank();

            if (compiled->m_lm_head_compiled) {
                compiled->m_lm_head_compiled->m_weights_bank = bank;
                compiled->m_lm_head_compiled->finalize_weights_bank();
            }
        } else {
            auto bank =
                ov::npuw::weights::Bank::deserialize(model_stream, compiled->get_plugin()->get_core(), bank_name);

            compiled->m_kvcache_compiled->m_weights_bank = bank;
            for (const auto& compiled_variant : compiled->m_generate_compiled_variants) {
                compiled_variant->m_weights_bank = bank;
                compiled_variant->reconstruct_closure();
            }

            compiled->m_prefill_compiled->m_weights_bank = bank;
            compiled->m_prefill_compiled->reconstruct_closure();

            if (compiled->m_lm_head_compiled) {
                compiled->m_lm_head_compiled->m_weights_bank = bank;
                compiled->m_lm_head_compiled->reconstruct_closure();
            }
        }
    };

    if (!encrypted) {
        CompiledContext ctx(false, nullptr, nullptr);
        auto compiled_model = ov::npuw::LLMCompiledModel::deserialize(stream, plugin, properties, ctx);
        NPUW_ASSERT(compiled_model && "Couldn't import NPUW compiled model!");
        read_and_finalize_banks(stream, compiled_model);
        LOG_INFO("Done.");
        return compiled_model;
    }

    EncryptionCallbacks enc_callbacks;
    NPUW_ASSERT(properties.count(ov::cache_encryption_callbacks.name()) &&
                properties.at(ov::cache_encryption_callbacks.name()).as<EncryptionCallbacks>().decrypt &&
                "Model is encrypted but no decrypt function was provided!");
    enc_callbacks.decrypt = properties.at(ov::cache_encryption_callbacks.name()).as<EncryptionCallbacks>().decrypt;

    LOG_INFO("Decryption will be done via the function provided.");

    std::shared_ptr<ov::npuw::LLMCompiledModel> compiled_model = nullptr;

    // Model is encrypted
    if (is_weightless) {
        std::string encrypted_str;
        read(stream, encrypted_str);
        std::istringstream decrypted_stream(std::move(enc_callbacks.decrypt(encrypted_str)));
        CompiledContext ctx(false, nullptr, nullptr);
        compiled_model = ov::npuw::LLMCompiledModel::deserialize(decrypted_stream, plugin, properties, ctx);
    } else {
        CompiledContext ctx(true, nullptr, enc_callbacks.decrypt);
        compiled_model = ov::npuw::LLMCompiledModel::deserialize(stream, plugin, properties, ctx);
    }

    NPUW_ASSERT(compiled_model && "Couldn't import NPUW compiled model!");
    read_and_finalize_banks(stream, compiled_model);

    LOG_INFO("Done.");

    return compiled_model;
}

std::shared_ptr<ov::npuw::LLMCompiledModel> ov::npuw::LLMCompiledModel::deserialize(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties,
    const ov::npuw::s11n::CompiledContext& ctx) {
    using namespace ov::npuw::s11n;

    auto read_model_meta = [&](std::istream& model_stream) {
        // Deserialize model name first
        std::string model_name;
        read(model_stream, model_name);

        // Create a dummy CompiledModel with an empty ov::Model - this will skip the constructor flow
        // to continue deserialization
        ov::ParameterVector parameters;
        ov::NodeVector results;

        read(model_stream, parameters);
        read(model_stream, results);

        auto ov_model = std::make_shared<ov::Model>(ov::as_output_vector(results), parameters, model_name);

        auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(ov_model, plugin, true);

        // Deserialize LLMCompiledModel-specific data
        read(model_stream, compiled->m_kvcache_desc.max_prompt_size);
        read(model_stream, compiled->m_kvcache_desc.total_size);
        read(model_stream, compiled->m_kvcache_desc.num_stored_tokens);
        read(model_stream, compiled->m_kvcache_desc.dim);
        read(model_stream, compiled->m_kvcache_desc.max_generation_token_len);
        read(model_stream, compiled->m_kvcache_desc.v_tensors_transposed_pre);
        read(model_stream, compiled->m_kvcache_desc.v_tensors_transposed_gen);
        read(model_stream, compiled->m_prefill_chunk_size);
        read(model_stream, compiled->m_use_chunk_prefill);
        read(model_stream, compiled->m_max_lora_rank);
        read(model_stream, compiled->m_enable_prefix_caching);
        read(model_stream, compiled->m_prefix_caching_block_size);
        read(model_stream, compiled->m_prefix_caching_max_num_blocks);
        read(model_stream, compiled->m_is_whisper);
        read(model_stream, compiled->m_is_eagle);
        read(model_stream, compiled->m_is_embedding);

        // Deserialize config
        read(model_stream, compiled->m_cfg);
        compiled->implement_properties();

        // Deserialize KV cache model variants
        read(model_stream, compiled->m_kvcache_sizes);
        uint32_t num_variants = 0;
        read(model_stream, num_variants);

        compiled->m_generate_compiled_variants.reserve(num_variants);

        // Deserialize CompiledModels
        // Note: no need to pass any encryption here as it's done in import_model()
        CompiledContext enc_ctx(false, nullptr, nullptr);

        // Deserialize all generate variants
        for (uint32_t i = 0; i < num_variants; ++i) {
            auto compiled_variant = ov::npuw::CompiledModel::deserialize(model_stream, plugin, properties, enc_ctx);
            compiled->m_generate_compiled_variants.push_back(compiled_variant);
        }

        // Set the main kvcache_compiled to the largest variant for backward compatibility
        if (!compiled->m_generate_compiled_variants.empty()) {
            compiled->m_kvcache_compiled = compiled->m_generate_compiled_variants.back();
        }

        compiled->m_prefill_compiled = ov::npuw::CompiledModel::deserialize(model_stream, plugin, properties, enc_ctx);
        bool is_shared_lm_head = false;
        read(model_stream, is_shared_lm_head);
        if (is_shared_lm_head) {
            compiled->m_lm_head_compiled =
                ov::npuw::CompiledModel::deserialize(model_stream, plugin, properties, enc_ctx);
        }

        return compiled;
    };

    std::shared_ptr<ov::npuw::LLMCompiledModel> compiled = nullptr;
    if (ctx.encrypted) {
        std::string encrypted_string;
        read(stream, encrypted_string);
        std::istringstream decrypted_stream(std::move(ctx.decrypt(encrypted_string)));
        compiled = read_model_meta(decrypted_stream);
    } else {
        compiled = read_model_meta(stream);
    }

    NPUW_ASSERT(compiled && "Couldn't create NPUW compiled model!");

    return compiled;
}

std::shared_ptr<const ov::Model> ov::npuw::LLMCompiledModel::get_runtime_model() const {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::npuw::LLMCompiledModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any ov::npuw::LLMCompiledModel::get_property(const std::string& name) const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (name == ov::intel_npu::npuw::llm::prefill_config.name() ||
        name == ov::intel_npu::npuw::llm::generate_config.name()) {
        OPENVINO_THROW(name, " is write-only option!");
    }

    auto&& configIterator = m_prop_to_opt.find(name);
    if (configIterator != m_prop_to_opt.cend()) {
        return std::get<1>(configIterator->second)(m_cfg);
    } else {
        return m_prefill_compiled->get_property(name);
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::LLMCompiledModel::create_sync_infer_request() const {
    auto* non_const_this = const_cast<ov::npuw::LLMCompiledModel*>(this);  // because of const in API
    if (m_is_whisper) {
        return non_const_this->create_whisper_infer_request();
    } else if (m_is_embedding) {
        return non_const_this->create_embedding_infer_request();
    } else {
        return non_const_this->create_llm_infer_request();
    }
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::LLMCompiledModel::create_llm_infer_request() {
    auto this_sptr = std::static_pointer_cast<ov::npuw::LLMCompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::LLMInferRequest>(this_sptr);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::LLMCompiledModel::create_whisper_infer_request() {
    auto this_sptr = std::static_pointer_cast<ov::npuw::LLMCompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::WhisperInferRequest>(this_sptr);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::LLMCompiledModel::create_embedding_infer_request() {
    auto this_sptr = std::static_pointer_cast<ov::npuw::LLMCompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::EmbeddingInferRequest>(this_sptr);
}

void ov::npuw::LLMCompiledModel::implement_properties() {
#define BIND(N, T, GETTER)                                                                 \
    {                                                                                      \
        ov::intel_npu::N.name(), {                                                         \
            ov::PropertyMutability::RW, [](const ::intel_npu::Config& config) -> ov::Any { \
                return config.GETTER<::intel_npu::T>();                                    \
            }                                                                              \
        }                                                                                  \
    }

    m_prop_to_opt.insert({BIND(npuw::llm::enabled, NPUW_LLM, get),
                          BIND(npuw::llm::batch_dim, NPUW_LLM_BATCH_DIM, get),
                          BIND(npuw::llm::seq_len_dim, NPUW_LLM_SEQ_LEN_DIM, get),
                          BIND(npuw::llm::max_prompt_len, NPUW_LLM_MAX_PROMPT_LEN, get),
                          BIND(npuw::llm::min_response_len, NPUW_LLM_MIN_RESPONSE_LEN, get),
                          BIND(npuw::llm::optimize_v_tensors, NPUW_LLM_OPTIMIZE_V_TENSORS, get),
                          BIND(npuw::llm::optimize_fp8, NPUW_LLM_OPTIMIZE_FP8, get),
                          BIND(npuw::llm::cache_rope, NPUW_LLM_CACHE_ROPE, get),
                          BIND(npuw::llm::prefill_moe_hint, NPUW_LLM_PREFILL_MOE_HINT, get),
                          BIND(npuw::llm::generate_moe_hint, NPUW_LLM_GENERATE_MOE_HINT, get),
                          BIND(npuw::llm::generate_pyramid, NPUW_LLM_GENERATE_PYRAMID, get),
                          BIND(npuw::llm::prefill_chunk_size, NPUW_LLM_PREFILL_CHUNK_SIZE, get),
                          BIND(npuw::llm::prefill_hint, NPUW_LLM_PREFILL_HINT, getString),
                          BIND(npuw::llm::generate_hint, NPUW_LLM_GENERATE_HINT, getString),
                          BIND(npuw::llm::prefill_attn_hint, NPUW_LLM_PREFILL_ATTENTION_HINT, getString),
                          BIND(npuw::llm::generate_attn_hint, NPUW_LLM_GENERATE_ATTENTION_HINT, getString),
                          BIND(npuw::llm::shared_lm_head, NPUW_LLM_SHARED_HEAD, get),
                          BIND(npuw::whisper::enabled, NPUW_WHISPER, get),
                          BIND(npuw::eagle::enabled, NPUW_EAGLE, get),
                          BIND(npuw::text_embed::enabled, NPUW_TEXT_EMBED, get)});
#undef BIND
}
