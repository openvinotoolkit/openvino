// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "llm_compiled_model.hpp"

#include "llm_infer_request.hpp"
#include "whisper_infer_request.hpp"
#include "logging.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "openvino/pass/validate.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "serialization.hpp"
#include "transformations/convert_precision.hpp"
#include "util.hpp"

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

        auto callback = [=](ov::pass::pattern::Matcher& m) {
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
        auto pattern_node = ov::pass::pattern::wrap_type<ov::op::internal::GroupQueryAttention>();

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
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

        auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_node, "GroupQueryAttentionDecomposition");
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
        auto cos_cache = node->input_value(6);
        auto sin_cache = node->input_value(7);

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

namespace {
uint32_t align_to(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

std::shared_ptr<ov::Model> cvt_kvcache_to_fp16(const std::shared_ptr<ov::Model>& model) {
    ov::preprocess::PrePostProcessor ppp(model);

    for (const auto& tensor : model->inputs()) {
        if (tensor.get_any_name().find("past_key") != std::string::npos) {
            ppp.input(tensor.get_any_name()).tensor().set_element_type(ov::element::Type_t::f16);
        }
    }

    for (const auto& tensor : model->outputs()) {
        if (tensor.get_any_name().find("present") != std::string::npos) {
            ppp.output(tensor.get_any_name()).tensor().set_element_type(ov::element::Type_t::f16);
        }
    }

    return ppp.build();
}

std::shared_ptr<ov::Model> redirect_new_kv_to_output(const std::shared_ptr<ov::Model>& model) {
    const auto kStartOutputKVCacheLayers = 1u;
    for (std::size_t i = kStartOutputKVCacheLayers; i < model->outputs().size(); ++i) {
        auto kvout = model->output(i);
        auto kvrslt = kvout.get_node();
        auto kvcat = kvrslt->inputs()[0].get_source_output().get_node();
        auto kvval = kvcat->inputs()[1].get_source_output();
        kvval.set_names({kvout.get_any_name()});
        kvrslt->inputs()[0].replace_source_output(kvval);
    }
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
}  // namespace


namespace {
struct KVAxesPosition {
    uint32_t batch;
    uint32_t seq_len;
};
}  // anonymous namespace

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

        auto last_op = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{matmul->output(0),
                                                                                    matmul_add->output(0),
                                                                                    matmul_transpose->output(0),
                                                                                    matmul_convert->output(0),
                                                                                    matmul_multiply->output(0)});
        auto res = opp::wrap_type<ov::op::v0::Result>({last_op->output(0)});

        auto callback = [=, &lm_head_model](ov::pass::pattern::Matcher& m) {
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
            lm_head_model = std::make_shared<ov::Model>(ov::OutputVector{new_result->output(0)},
                                                        ov::ParameterVector{new_param},
                                                        "NPUW_LMHead");

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
    model->validate_nodes_and_infer_types();

    return lm_head_model;
}

void reshape_to_static(std::shared_ptr<ov::Model> model,
                       const uint32_t input_size,
                       const uint32_t kvcache_size,
                       const KVAxesPosition& kv_axes_position,
                       const uint32_t lhs_seq_size = 0) {
    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : model->inputs()) {
        const auto& input_name = input.get_any_name();
        ov::PartialShape new_shape;
        if (input_name.find("input_ids") != std::string::npos) {
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
                new_shape = ov::PartialShape({1, kvcache_size+1});
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
            new_shape[0] = 1;            // batch_dim
        } else {
            const auto& partial_shape = input.get_partial_shape();
            new_shape = partial_shape;
            new_shape[kv_axes_position.batch] = 1;
            if (lhs_seq_size) { // Whisper model
                new_shape[kv_axes_position.seq_len] = (input_name.find(".decoder") != std::string::npos)
                                                      ? kvcache_size - input_size // kv_size for decoder
                                                      : lhs_seq_size;             // sequence size for encoder hidden states
            } else { // LLM/VLM
                new_shape[kv_axes_position.seq_len] = kvcache_size - input_size;
            }
        }
        new_shapes.emplace(input_name, new_shape);
    }
    model->reshape(new_shapes);
}

void reshape_sliced_head_to_static(std::shared_ptr<ov::Model> lm_head_model, const uint32_t& batch_dim) {
    // We have only one input with dynamic shapes: output of Slice operation, and this output
    // should have "1" for dimension representing number of embeddings to send to the matmul.
    // Batch size should be also equal "1" for NPU.
    const auto& input = lm_head_model->input(0);
    const auto& partial_shape = input.get_partial_shape();
    NPUW_ASSERT(partial_shape.size() == 3);

    ov::PartialShape new_shape = partial_shape;
    new_shape[batch_dim] = 1;
    // Left dynamic axis will be for number of embeddings
    for (auto i = 0; i < new_shape.rank().get_length(); i++) {
        if (new_shape[i].is_dynamic()) {
            new_shape[i] = 1;
            // Sanity check that only one left dimension is dynamic, as
            // another one should contain embedding space rank
            break;
        }
    }

    lm_head_model->reshape(new_shape);
}

void slice_out_embeds(std::shared_ptr<ov::Model> model, const uint32_t& batch_dim) {
    std::shared_ptr<ov::Node> embed_result;
    for (auto&& output : model->outputs()) {
        if (output.get_any_name() == ov::npuw::LLMCompiledModel::output_embeds) {
            embed_result = output.get_node_shared_ptr();
        }
    }

    if (embed_result) {
        auto shape = embed_result->input(0).get_shape();
        // If shape.size() is 3, then last axis should be the Vocab size.
        // But 1st and 2nd axis can mean different things.
        // 1st axis can represent the batch size, while 2nd - the number of embeddings,
        // or vice-versa (in chatglm)
        if (shape.size() == 3) {
            uint32_t num_embeds_dim = 1 - batch_dim;
            if (shape[num_embeds_dim] > 1) {
                std::vector<int32_t> start_pos{static_cast<int32_t>(batch_dim * (shape[num_embeds_dim] - 1)),
                                               static_cast<int32_t>(num_embeds_dim * (shape[num_embeds_dim] - 1)),
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

ov::AnyMap get_baseline_common_config(const std::optional<NPUDesc>& npudesc) {
    ov::AnyMap config = {
        {"NPU_COMPILATION_MODE_PARAMS", "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm"},
        {"NPUW_DEVICES", "NPU"},
        {"NPU_USE_NPUW", "YES"},
        {"NPUW_FOLD", "YES"},
        {"NPUW_DCOFF_TYPE", "f16"},
        {"NPUW_DCOFF_SCALE", "YES"},
        {"NPUW_WEIGHTS_BANK", "shared"},
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

std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        result.emplace(value.first, value.second.as<std::string>());
    }
    return result;
}

bool check_if_whisper_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ops()) {
        if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(node) && node->inputs().size() == 3u) {
            // Found cross-attention -> whisper model
            LOG_DEBUG("Whisper model was found");
            return true;
        }
    }
    return false;
}
}  // namespace

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

    m_is_whisper = check_if_whisper_model(model);

    const auto npudesc = extract_npu_descriptor(plugin);

    ov::AnyMap npuw_llm_props;
    ov::AnyMap other_props;
    split_llm_properties(properties, npuw_llm_props, other_props);
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

    LOG_DEBUG("Creating kvcache model as clone of passed one.");
    auto kvcache_model = model->clone();
    LOG_DEBUG("Transform kvcache model from stateful to stateless.");
    ov::pass::StatefulToStateless().run_on_model(kvcache_model);
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

    LOG_DEBUG("Creating prefill model as clone of transformed kvcache one.");
    auto prefill_model = kvcache_model->clone();
    prefill_model->set_friendly_name(kvcache_model->get_friendly_name() + "_prefill");

    const uint32_t batch_dim = m_cfg.get<::intel_npu::NPUW_LLM_BATCH_DIM>();
    const uint32_t seq_len_dim = m_cfg.get<::intel_npu::NPUW_LLM_SEQ_LEN_DIM>();
    KVAxesPosition axes{batch_dim, seq_len_dim};
    const uint32_t max_prompt_len = align_to(m_cfg.get<::intel_npu::NPUW_LLM_MAX_PROMPT_LEN>(), 64u);
    const uint32_t min_response_len = align_to(m_cfg.get<::intel_npu::NPUW_LLM_MIN_RESPONSE_LEN>(), 64u);

    // NB: PREFILL_HINT is now compatible with the PREFILL_CONFIG section, unlike for
    // the generate model they're not mutually exclusive
    const ::intel_npu::npuw::llm::PrefillHint prefill_hint = m_cfg.get<::intel_npu::NPUW_LLM_PREFILL_HINT>();

    m_prefill_chunk_size = m_cfg.get<::intel_npu::NPUW_LLM_PREFILL_CHUNK_SIZE>();
    m_use_chunk_prefill = (prefill_hint == ::intel_npu::npuw::llm::PrefillHint::DYNAMIC && m_prefill_chunk_size > 0);
    LOG_VERB("Enabled prefill chunking: " << m_use_chunk_prefill);
    LOG_VERB("Prefill chunk size: " << m_prefill_chunk_size);
    LOG_VERB("Maximum prompt length: " << max_prompt_len);

    auto is_power_of_two = [](uint64_t n) {
        return n > 0 && (n & (n - 1)) == 0;
    };
    if (m_use_chunk_prefill) {
        if (!is_power_of_two(m_prefill_chunk_size)) {
            OPENVINO_THROW("Configuration Error: chunk size (",
                           m_prefill_chunk_size,
                           ") is not power of 2. Please adjust NPUW_LLM_PREFILL_CHUNK_SIZE.");
        }

        if (max_prompt_len % m_prefill_chunk_size) {
            OPENVINO_THROW("Configuration Error: The maximum prompt length (",
                           max_prompt_len,
                           ") is not a multiple of chunk size (",
                           m_prefill_chunk_size,
                           "). Please adjust NPUW_LLM_MAX_PROMPT_LEN to be a multiple of NPUW_LLM_PREFILL_CHUNK_SIZE.");
        }
    }

    m_kvcache_desc = KVCacheDesc{max_prompt_len, max_prompt_len + min_response_len, 0u, seq_len_dim};

    uint32_t whisper_lhs_seq_size = 0; // Not applicable for LLMs/VLMs
    if (m_is_whisper) {
        axes = KVAxesPosition{whisper_batch_dim, whisper_seq_len_dim};
        m_kvcache_desc = KVCacheDesc{whisper_max_prompt_size, whisper_kvcache_size, 0u, whisper_seq_len_dim};
        whisper_lhs_seq_size = static_cast<uint32_t>(prefill_model->input("encoder_hidden_states").get_partial_shape()[1].get_length());

        ov::npuw::util::prepare_whisper_prefill_model(prefill_model, m_kvcache_desc.max_prompt_size, whisper_lhs_seq_size); // Whisper decoder model
        ov::npuw::util::prepare_whisper_kvcache_model(kvcache_model); // Whisper decoder_with_past model
    }

    LOG_DEBUG("Make prefill model with static shapes");
    if (m_use_chunk_prefill) {
        reshape_to_static(prefill_model,
                          static_cast<uint32_t>(m_prefill_chunk_size),
                          m_kvcache_desc.max_prompt_size,
                          axes);
    } else {
        reshape_to_static(prefill_model, m_kvcache_desc.max_prompt_size, m_kvcache_desc.max_prompt_size, axes, whisper_lhs_seq_size);
    }
    LOG_DEBUG("Make kvcache model with static shapes");
    reshape_to_static(kvcache_model, 1u, m_kvcache_desc.total_size, axes, whisper_lhs_seq_size);
    if (lm_head_model) {
        LOG_DEBUG("Shared LM head: slice the prefill output");
        // KVCache model is already reshaped to [1, 1, embed size], so only apply slice to
        // the Prefill model:
        slice_out_embeds(prefill_model, axes.batch);
        LOG_DEBUG("Make LM head model with static shapes");
        reshape_sliced_head_to_static(lm_head_model, axes.batch);
    }

    LOG_DEBUG("5.1, decompose GroupQueryAttention OP");
    decompose_GQA(prefill_model, true);
    decompose_GQA(kvcache_model, false);

    const bool optimize_v_tensors = m_cfg.get<::intel_npu::NPUW_LLM_OPTIMIZE_V_TENSORS>();
    if (optimize_v_tensors) {
        LOG_DEBUG("Check and apply opt layout");
        LOG_BLOCK();
        if (ov::npuw::util::optimize_value_tensors(kvcache_model, false)) {
            NPUW_ASSERT(ov::npuw::util::optimize_value_tensors(prefill_model, true));
            m_kvcache_desc.v_tensors_transposed = true;
        } else {
            LOG_DEBUG("vtensors optimisation not applied");
        }
    } else {
        LOG_DEBUG("Check and apply opt layout --- SKIPPED");
    }

    if (!m_use_chunk_prefill) {
        NPUW_ASSERT(remove_empty_kv_inputs(prefill_model));
    } else {
        LOG_DEBUG("Don't remove input key/values from prefill model.");
        LOG_DEBUG("Ask prefill model to output key/values for prefill chunk size tokens.");
        prefill_model = redirect_new_kv_to_output(prefill_model);
    }

    LOG_DEBUG("Optimize kvcache model to output key/values for new token.");
    kvcache_model = redirect_new_kv_to_output(kvcache_model);
    LOG_DEBUG("Converting KV-cache in kvcache model to FP16.");
    kvcache_model = cvt_kvcache_to_fp16(kvcache_model);
    LOG_DEBUG("Converting KV-cache in prefill model to FP16.");
    prefill_model = cvt_kvcache_to_fp16(prefill_model);

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

    if (m_is_whisper) {
        update_config_for_whisper(prefill_config);
    }

    m_kvcache_compiled = std::dynamic_pointer_cast<ov::npuw::CompiledModel>(
        ov::npuw::ICompiledModel::create(kvcache_model, plugin, generate_config));
    NPUW_ASSERT(m_kvcache_compiled && "Can't create ov::npuw::CompiledModel for passed kvcache "
                                      "model and its config, please check passed config.");
    m_prefill_compiled = std::dynamic_pointer_cast<ov::npuw::CompiledModel>(
        ov::npuw::ICompiledModel::create(prefill_model, plugin, prefill_config));
    NPUW_ASSERT(m_prefill_compiled && "Can't create ov::npuw::CompiledModel for passed prefill "
                                      "model and its config, please check passed config.");
    if (lm_head_model) {
        auto lm_head_config = get_default_lm_head_config(npudesc);
        merge_config_with(lm_head_config, other_props);
        auto lm_head_config_addition_value = lm_head_config_addition.value_or(ov::AnyMap{}).as<ov::AnyMap>();

        merge_config_with(lm_head_config, lm_head_config_addition_value);
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
        write(model_stream, m_kvcache_desc.v_tensors_transposed);
        write(model_stream, m_prefill_chunk_size);
        write(model_stream, m_use_chunk_prefill);

        // Write config
        write(model_stream, m_cfg);

        // Serialize CompiledModels
        // Note: no need to pass any encryption here as it's done in export_model()
        CompiledContext enc_ctx(false, nullptr, nullptr, m_bf16_consts);
        m_kvcache_compiled->serialize(model_stream, enc_ctx);
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

            compiled->m_kvcache_compiled->m_weights_bank = bank;
            compiled->m_prefill_compiled->m_weights_bank = bank;

            compiled->m_kvcache_compiled->finalize_weights_bank();
            compiled->m_kvcache_compiled->m_import_weights_ctx.reset();
            compiled->m_prefill_compiled->finalize_weights_bank();
            compiled->m_prefill_compiled->m_import_weights_ctx.reset();

            if (compiled->m_lm_head_compiled) {
                compiled->m_lm_head_compiled->m_weights_bank = bank;

                compiled->m_lm_head_compiled->finalize_weights_bank();
                compiled->m_lm_head_compiled->m_import_weights_ctx.reset();
            }
        } else {
            auto bank =
                ov::npuw::weights::Bank::deserialize(model_stream, compiled->get_plugin()->get_core(), bank_name);

            compiled->m_kvcache_compiled->m_weights_bank = bank;
            compiled->m_prefill_compiled->m_weights_bank = bank;

            compiled->m_kvcache_compiled->reconstruct_closure();
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
        read(model_stream, compiled->m_kvcache_desc.v_tensors_transposed);
        read(model_stream, compiled->m_prefill_chunk_size);
        read(model_stream, compiled->m_use_chunk_prefill);

        // Deserialize config
        read(model_stream, compiled->m_cfg);
        compiled->implement_properties();

        // Deserialize CompiledModels
        // Note: no need to pass any encryption here as it's done in import_model()
        CompiledContext enc_ctx(false, nullptr, nullptr);
        compiled->m_kvcache_compiled = ov::npuw::CompiledModel::deserialize(model_stream, plugin, properties, enc_ctx);
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
    return m_is_whisper ? non_const_this->create_whisper_infer_request()
                        : non_const_this->create_llm_infer_request();
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::LLMCompiledModel::create_llm_infer_request() {
    auto this_sptr = std::static_pointer_cast<ov::npuw::LLMCompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::LLMInferRequest>(this_sptr);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::LLMCompiledModel::create_whisper_infer_request() {
    auto this_sptr = std::static_pointer_cast<ov::npuw::LLMCompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::WhisperInferRequest>(this_sptr);
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
                          BIND(npuw::llm::prefill_chunk_size, NPUW_LLM_PREFILL_CHUNK_SIZE, get),
                          BIND(npuw::llm::prefill_hint, NPUW_LLM_PREFILL_HINT, getString),
                          BIND(npuw::llm::generate_hint, NPUW_LLM_GENERATE_HINT, getString),
                          BIND(npuw::llm::shared_lm_head, NPUW_LLM_SHARED_HEAD, get)});
#undef BIND
}
