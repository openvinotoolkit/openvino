// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prepare_whisper_model.hpp"

#include <regex>

#include "../llm_compiled_model_utils.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"

namespace opp = ov::pass::pattern;

namespace {

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

class AttentionMaskInputPast : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::AttentionMaskInputPast");

    AttentionMaskInputPast(std::shared_ptr<ov::Model> model) {
        auto range = opp::wrap_type<ov::op::v4::Range>();
        auto convert1 = opp::wrap_type<ov::op::v0::Convert>({range});
        auto greater = opp::wrap_type<ov::op::v1::Greater>({convert1, opp::any_input()});
        auto convert2 = opp::wrap_type<ov::op::v0::Convert>({greater});

        register_matcher(std::make_shared<opp::Matcher>(convert2, this->get_type_info().name),
                         [model](opp::Matcher& m) {
                             auto node = m.get_match_root();
                             auto attention_mask =
                                 std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
                             attention_mask->get_output_tensor(0).set_names({"attention_mask"});
                             model->add_parameters({attention_mask});

                             auto cvt =
                                 std::make_shared<ov::op::v0::Convert>(attention_mask->output(0), ov::element::f32);
                             ov::replace_node(node, cvt);
                             return false;
                         });
    }
};

class AttentionMaskInputPast_2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::AttentionMaskInputPast_2");

    AttentionMaskInputPast_2(std::shared_ptr<ov::Model> model) {
        auto range = opp::wrap_type<ov::op::v4::Range>();
        auto unsqueeze1 = opp::wrap_type<ov::op::v0::Unsqueeze>({range, opp::any_input()});
        auto unsqueeze2 = opp::wrap_type<ov::op::v0::Unsqueeze>({unsqueeze1, opp::any_input()});
        auto unsqueeze3 = opp::wrap_type<ov::op::v0::Unsqueeze>({unsqueeze2, opp::any_input()});
        auto opt_convert = opp::optional<ov::op::v0::Convert>({unsqueeze3->output(0)});
        auto lessequal = opp::wrap_type<ov::op::v1::LessEqual>({opt_convert, opp::any_input()});

        register_matcher(
            std::make_shared<opp::Matcher>(lessequal, this->get_type_info().name),
            [model](opp::Matcher& m) {
                auto node = m.get_match_root();
                auto attention_mask =
                    std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
                attention_mask->get_output_tensor(0).set_names({"attention_mask"});
                model->add_parameters({attention_mask});

                auto cst_0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 0);
                auto cst_1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 1);
                auto cst_2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 2);

                auto attn_mask_shape =
                    std::make_shared<ov::op::v3::ShapeOf>(attention_mask, ov::element::i64)->output(0);
                auto gather = std::make_shared<ov::op::v8::Gather>(attn_mask_shape, cst_1, cst_0)->output(0);
                auto attn_mask_size_minus_one = std::make_shared<ov::op::v1::Subtract>(gather, cst_1)->output(0);
                auto slice = std::make_shared<ov::op::v8::Slice>(attention_mask->output(0),
                                                                 cst_0,
                                                                 attn_mask_size_minus_one,
                                                                 cst_1,
                                                                 cst_1);

                auto unsqueeze_1 = std::make_shared<ov::op::v0::Unsqueeze>(slice->output(0), cst_1->output(0));
                auto unsqueeze_2 = std::make_shared<ov::op::v0::Unsqueeze>(unsqueeze_1->output(0), cst_2->output(0));

                auto equal = std::make_shared<ov::op::v1::Equal>(unsqueeze_2->output(0), cst_0->output(0));

                ov::replace_node(node, equal);
                return false;
            });
    }
};

class AttentionMaskInput : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::AttentionMaskInput");

    AttentionMaskInput(std::shared_ptr<ov::Model> model,
                       const uint32_t& max_prompt_len,
                       const uint32_t& lhs_seq_size,
                       bool transform_cross_attn) {
        std::vector<std::shared_ptr<ov::Node>> self_attn_nodes;
        std::vector<std::shared_ptr<ov::Node>> cross_attn_nodes;
        const auto kAttnMaskPort = 3;
        for (auto node : model->get_ops()) {
            if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(node)) {
                if (node->inputs().size() > kAttnMaskPort &&
                    (ov::is_type<ov::op::v8::Slice>(node->input(kAttnMaskPort).get_source_output().get_node()) ||
                     ov::is_type<ov::op::v1::Select>(node->input(kAttnMaskPort).get_source_output().get_node()))) {
                    self_attn_nodes.push_back(node);
                } else {
                    cross_attn_nodes.push_back(node);
                }
            }
        }

        // Self-attention
        OPENVINO_ASSERT(!self_attn_nodes.empty());

        auto attention_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
        attention_mask->get_output_tensor(0).set_names({"attention_mask"});
        model->add_parameters({attention_mask});

        auto cst_ninf = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                               ov::Shape{1},
                                                               std::vector<float>{-std::numeric_limits<float>::max()});
        auto cst_1 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{1});
        auto cst_0 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0});

        auto slice = self_attn_nodes[0]->input(kAttnMaskPort).get_source_output().get_node_shared_ptr();
        std::shared_ptr<ov::Node> slice_f32;
        if (slice->get_element_type() == ov::element::boolean) {
            slice_f32 = std::make_shared<ov::op::v1::Select>(slice->output(0), cst_0->output(0), cst_ninf->output(0));
        } else {
            slice_f32 = slice;
        }
        auto cvt = std::make_shared<ov::op::v0::Convert>(attention_mask->output(0), ov::element::f32);
        auto add = std::make_shared<ov::op::v1::Add>(slice_f32->output(0), cvt->output(0));

        auto trps = std::make_shared<ov::op::v1::Transpose>(
            cvt->output(0),
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int>{1, 0}));
        auto mtpl = std::make_shared<ov::op::v1::Multiply>(trps->output(0), add->output(0));

        auto equal = std::make_shared<ov::op::v1::Equal>(mtpl->output(0), cst_1->output(0));
        auto select = std::make_shared<ov::op::v1::Select>(equal->output(0), cst_0->output(0), cst_ninf->output(0));

        for (auto self_attn : self_attn_nodes) {
            self_attn->input(3).replace_source_output(select->output(0));
        }

        if (transform_cross_attn) {
            // Cross attn
            OPENVINO_ASSERT(!cross_attn_nodes.empty());
            // FIXME: Should be taken from topology - don't hardcode!!!
            auto shape_cst =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{2},
                                                       std::vector<float>{static_cast<float>(max_prompt_len), 1});
            auto target_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64,
                ov::Shape{2},
                std::vector<float>{static_cast<float>(max_prompt_len), static_cast<float>(lhs_seq_size)});
            // FIXME: Must be transpose if batch present
            auto reshape = std::make_shared<ov::op::v1::Reshape>(cvt->output(0), shape_cst->output(0), false);
            auto equal = std::make_shared<ov::op::v1::Equal>(reshape->output(0), cst_1->output(0));
            auto select = std::make_shared<ov::op::v1::Select>(equal->output(0), cst_0->output(0), cst_ninf->output(0));
            auto broadcast = std::make_shared<ov::op::v3::Broadcast>(select->output(0), target_shape->output(0));
            auto unsq1 = std::make_shared<ov::op::v0::Unsqueeze>(broadcast->output(0), cst_0->output(0));
            auto unsq2 = std::make_shared<ov::op::v0::Unsqueeze>(unsq1->output(0), cst_1->output(0));
            for (auto cross_attn_node : cross_attn_nodes) {
                if (cross_attn_node->inputs().size() == 3) {
                    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
                        cross_attn_node->input(0).get_source_output(),
                        cross_attn_node->input(1).get_source_output(),
                        cross_attn_node->input(2).get_source_output(),
                        unsq2->output(0),
                        false);
                    ov::replace_node(cross_attn_node, sdpa);
                } else {
                    cross_attn_node->input(3).replace_source_output(unsq2->output(0));
                }
            }
        }
    }
};

class CachePositionInput : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::CachePositionInput");

    CachePositionInput(std::shared_ptr<ov::Model> model) {
        auto gather = opp::wrap_type<ov::op::v8::Gather>({opp::any_input(), opp::any_input(), opp::any_input()});
        auto add = opp::wrap_type<ov::op::v1::Add>({gather, opp::any_input()});
        auto range = opp::wrap_type<ov::op::v4::Range>({gather, add, opp::any_input()});
        auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({range, opp::any_input()});
        auto tile = opp::wrap_type<ov::op::v0::Tile>({unsqueeze, opp::any_input()});

        register_matcher(
            std::make_shared<opp::Matcher>(tile, this->get_type_info().name),
            [model, unsqueeze](opp::Matcher& m) {
                auto& node_to_output = m.get_pattern_value_map();
                auto unsqueeze_node = node_to_output.at(unsqueeze).get_node_shared_ptr();
                auto matched_unsqueeze = std::static_pointer_cast<ov::op::v0::Unsqueeze>(unsqueeze_node);

                auto cache_position = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
                cache_position->get_output_tensor(0).set_names({"cache_position"});
                cache_position->set_friendly_name("cache_position");
                model->add_parameters({cache_position});
                std::shared_ptr<ov::Node> cache_pos_unsqueeze_arg;
                if (matched_unsqueeze->input(0).get_element_type() == ov::element::f32) {
                    cache_pos_unsqueeze_arg = std::make_shared<ov::op::v0::Convert>(cache_position, ov::element::f32);
                } else {
                    cache_pos_unsqueeze_arg = cache_position;
                }

                matched_unsqueeze->input(0).replace_source_output(cache_pos_unsqueeze_arg->output(0));
                return false;
            });
    }
};

bool can_move_scale_after_matmul(const ov::Output<ov::Node>& query,
                                 const ov::Output<ov::Node>& kT,
                                 const ov::Output<ov::Node>& scale) {
    const auto& scale_pshape = scale.get_partial_shape();
    const auto& query_pshape = query.get_partial_shape();
    if (scale_pshape.is_dynamic() || query_pshape.is_dynamic()) {
        return false;
    }

    // According to the ov SDPA specification, the scale input have to be 1d with 1 element
    // or scalar.
    if (ov::shape_size(scale_pshape.to_shape()) != 1) {
        return false;
    }

    // using the original implementation to calculate the shapes.
    // we need to move the scale after MatMul only if the tensor after MatMul is smaller.
    auto q_scaled = std::make_shared<ov::op::v1::Multiply>(query, scale);
    auto scaled_attn = std::make_shared<ov::op::v0::MatMul>(q_scaled, kT);
    const auto& scaled_attn_pshape = scaled_attn->output(0).get_partial_shape();
    if (scaled_attn_pshape.is_static()) {
        return ov::shape_size(query_pshape.to_shape()) > ov::shape_size(scaled_attn_pshape.to_shape());
    }
    return false;
}

// FIXME: Whisper Decompose SDPA
class WhisperScaledDotProductAttentionDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("WhisperScaledDotProductAttentionDecomposition");
    WhisperScaledDotProductAttentionDecomposition() {
        auto pattern_node = ov::pass::pattern::wrap_type<ov::op::v13::ScaledDotProductAttention>();

        ov::matcher_pass_callback callback = [this, pattern_node](ov::pass::pattern::Matcher& m) {
            auto& pattern_to_output = m.get_pattern_value_map();

            auto node = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(
                pattern_to_output.at(pattern_node).get_node_shared_ptr());

            if (node == nullptr || transformation_callback(node)) {
                return false;
            }

            const std::string& node_name = node->get_friendly_name();
            if (node_name.find("encoder_attn") == std::string::npos) {
                // This pass is only for encoder-decoder cross-attention layers
                return false;
            }

            auto new_output_node = decompose(node);
            ov::replace_node(node, new_output_node);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_node,
                                                              "WhisperScaledDotProductAttentionDecompositionMatcher");
        register_matcher(m, callback);
    }

    std::shared_ptr<ov::Node> decompose(std::shared_ptr<ov::op::v13::ScaledDotProductAttention> node) {
        using namespace ov::op;
        using namespace ov;
        auto query = node->input_value(0);
        auto key = node->input_value(1);
        auto value = node->input_value(2);
        auto q_shape = register_new_node<v3::ShapeOf>(query, element::i32);
        auto k_shape = register_new_node<v3::ShapeOf>(key, element::i32);
        auto minus_one = register_new_node(v0::Constant::create(element::i32, Shape{}, {-1}));
        auto minus_two = register_new_node(v0::Constant::create(element::i32, Shape{}, {-2}));
        auto zero_i = register_new_node(v0::Constant::create(element::i32, Shape{}, {0}));
        auto one_i = register_new_node(v0::Constant::create(element::i32, Shape{}, {1}));
        auto one_f = register_new_node<v1::ConvertLike>(one_i, query);
        auto zero_f = register_new_node<v1::ConvertLike>(zero_i, query);

        auto build_extract_dim_subgraph = [this, &zero_i](const std::shared_ptr<v3::ShapeOf>& shape_of,
                                                          const int64_t idx) -> std::shared_ptr<ov::Node> {
            const auto dim_to_extract_const = v0::Constant::create(element::i32, Shape{}, {idx});
            const auto gather = std::make_shared<v8::Gather>(shape_of, dim_to_extract_const, zero_i);

            register_new_node(dim_to_extract_const);
            return register_new_node(gather);
        };

        Output<Node> scale;
        Output<Node> sink;
        bool has_sink = false;
        if (node->get_input_size() < 5) {
            scale = build_extract_dim_subgraph(q_shape, -1);
            scale = register_new_node<v1::ConvertLike>(scale, query);
            auto sqrt_scale = register_new_node<v0::Sqrt>(scale);
            scale = register_new_node<v1::Divide>(one_f, sqrt_scale);
        } else {
            scale = node->input_value(4);
            if (node->get_input_size() == 6) {
                sink = node->input_value(5);
                has_sink = true;
            }
        }

        auto k_rank = register_new_node<v3::ShapeOf>(k_shape, element::i32)->output(0);
        auto k_last_dim = register_new_node<v1::Add>(k_rank, minus_one);
        auto k_next_dim = register_new_node<v1::Add>(k_rank, minus_two)->output(0);
        k_rank = register_new_node<v0::Squeeze>(k_rank, zero_i);
        auto minus_inf =
            register_new_node(v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()}))
                ->output(0);
        auto keep_dim_last = register_new_node<v0::Squeeze>(k_next_dim, zero_i);
        auto k_dims_before_transpose = register_new_node<v4::Range>(zero_i, keep_dim_last, one_i, element::i32);

        auto transpose_dims =
            register_new_node<v0::Concat>(OutputVector{k_dims_before_transpose, k_last_dim, k_next_dim}, 0);
        auto k_transposed = register_new_node<v1::Transpose>(key, transpose_dims);

        ov::Output<Node> scaled_atten;
        if (can_move_scale_after_matmul(query, k_transposed, scale)) {
            auto atten = register_new_node<v0::MatMul>(query, k_transposed)->output(0);
            scaled_atten = register_new_node<v1::Multiply>(atten, scale)->output(0);
        } else {
            auto q_scaled = register_new_node<v1::Multiply>(query, scale);
            scaled_atten = register_new_node<v0::MatMul>(q_scaled, k_transposed)->output(0);
        }

        minus_inf = register_new_node<v1::ConvertLike>(minus_inf, scaled_atten);

        if (node->get_causal() || node->get_input_size() > 3) {
            Output<Node> mask;
            Output<Node> atten_mask;
            if (!node->get_causal()) {
                mask = node->input_value(3);

                // two types of masks are supported. A boolean mask where a value of True indicates that the element
                // should take part in attention. A float mask of the same type as query, key, value that is added to
                // the attention score.
                if (mask.get_element_type() == element::boolean) {
                    atten_mask = register_new_node<v1::Select>(mask, zero_f, minus_inf);
                } else {
                    atten_mask = mask;
                }
            } else {
                auto target_s_len = build_extract_dim_subgraph(q_shape, -2);
                auto source_s_len = build_extract_dim_subgraph(k_shape, -2);
                auto ssl = register_new_node<v0::Unsqueeze>(source_s_len, zero_i);
                auto tsl = register_new_node<v0::Unsqueeze>(target_s_len, zero_i);
                auto mask_shape = register_new_node<v0::Concat>(OutputVector{tsl, ssl}, 0);
                mask = register_new_node<v1::Broadcast>(minus_inf, mask_shape);
                auto horizontal_range =
                    register_new_node<v4::Range>(zero_i, source_s_len, one_i, element::i32)->output(0);
                horizontal_range = register_new_node<v0::Unsqueeze>(horizontal_range, zero_i);
                auto stop = register_new_node<v1::Add>(target_s_len, one_i);
                auto vertical_range = register_new_node<v4::Range>(one_i, stop, one_i, element::i32)->output(0);
                vertical_range = register_new_node<v0::Unsqueeze>(vertical_range, one_i);
                auto triu = register_new_node<v1::GreaterEqual>(horizontal_range, vertical_range);
                atten_mask = register_new_node<v1::Select>(triu, mask, zero_f);
            }
            scaled_atten = register_new_node<v1::Add>(scaled_atten, atten_mask);
        }

        scaled_atten.add_names({"cross_attention_qk_scaled_scores"});

        if (has_sink) {
            auto minus_two = register_new_node(v0::Constant::create(element::i32, Shape{1}, {-2}));
            auto minus_one = register_new_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
            auto zero_i = register_new_node(v0::Constant::create(element::i32, Shape{1}, {0}));
            auto one_i = register_new_node(v0::Constant::create(element::i32, Shape{1}, {1}));

            auto q_last_but_one_dim = register_new_node<v1::Subtract>(register_new_node<v0::ShapeOf>(q_shape),
                                                                      v0::Constant::create(element::i64, Shape{}, {1}));
            auto sink_target_shape_1 = register_new_node<v8::Slice>(q_shape, zero_i, q_last_but_one_dim, one_i);
            auto sink_target_shape = register_new_node<v0::Concat>(OutputVector{sink_target_shape_1, one_i}, 0);
            auto sink_broadcast = register_new_node<v1::Broadcast>(sink, sink_target_shape);

            auto scaled_attn_sink = register_new_node<v0::Concat>(OutputVector{scaled_atten, sink_broadcast}, -1);
            scaled_atten = register_new_node<v8::Softmax>(scaled_attn_sink, -1);

            auto prev_seq_len = register_new_node<v8::Gather>(k_shape, minus_two, zero_i);
            scaled_atten = register_new_node<v8::Slice>(scaled_atten, zero_i, prev_seq_len, one_i, minus_one);
        } else {
            scaled_atten = register_new_node<v8::Softmax>(scaled_atten, -1);
        }

        auto result = register_new_node<v0::MatMul>(scaled_atten, value);
        result->set_friendly_name(node->get_friendly_name());
        copy_runtime_info(node, get_new_nodes());
        return result;
    }
};

auto remove_encoder_attn_read_value(const std::shared_ptr<ov::Node>& rv_node,
                                    const ov::Output<ov::Node>& kv_out,
                                    const ov::Input<ov::Node>& sdpa_in) {
    // Find Assign node
    OPENVINO_ASSERT(rv_node->outputs().size() == 1);
    auto rv_out = rv_node->outputs()[0];
    ov::NodeVector rv_readers;
    for (const auto& target_in : rv_out.get_target_inputs()) {
        rv_readers.push_back(target_in.get_node()->shared_from_this());
    }
    // Assign and SDPA
    OPENVINO_ASSERT(rv_readers.size() == 2);
    auto assign_node = (strstr(rv_readers[0]->get_type_name(), "Assign") != nullptr) ? rv_readers[0] : rv_readers[1];
    OPENVINO_ASSERT(strstr(assign_node->get_type_name(), "Assign") != nullptr);
    // Redirect KV-cache tensor to SDPA
    sdpa_in.replace_source_output(kv_out);
    return std::make_pair(std::make_shared<ov::op::v0::Result>(kv_out),
                          ov::as_type_ptr<ov::op::v6::Assign>(assign_node));
}

std::string transform_key_value_name(std::string input_string,
                                     std::string prefix,
                                     std::string enc_or_dec,
                                     std::string key_or_value) {
    std::regex pattern("[0-9]+");
    std::smatch match;
    std::regex_search(input_string, match, pattern);

    if (match.empty())
        OPENVINO_THROW("Input string does not match the expected pattern");

    auto number = std::string(match[0]);
    return prefix + "." + number + enc_or_dec + key_or_value;
}

void set_name(std::shared_ptr<ov::Node> result, const std::string& name) {
    result->set_friendly_name(name);
    result->get_output_tensor(0).set_names({name});
}

bool is_fake_cvt_to_key_tensor(const ov::Input<ov::Node>& reader) {
    auto fc_reader = reader.get_node()->outputs()[0].get_target_inputs();
    // FakeConvert node has only 1 consumer
    OPENVINO_ASSERT(fc_reader.size() == 1);
    // FakeConvert -> SDPA : 'key' tensor is input with index 1 to SDPA
    return fc_reader.begin()->get_index() == 1;
}

void expose_runtime_states_as_outputs(const std::shared_ptr<ov::Model>& model) {
    // Find all ReadValue nodes
    ov::NodeVector read_value_nodes;
    for (const auto& op : model->get_ops()) {
        if (strstr(op->get_type_name(), "ReadValue") != nullptr) {
            read_value_nodes.push_back(op);
        }
    }

    // Holds result layers for cross-attn KV-cache tensors
    ov::ResultVector results;
    ov::SinkVector assigns;

    // Go through all ReadValue nodes and remove them
    for (const auto& rv_node : read_value_nodes) {
        OPENVINO_ASSERT(rv_node->inputs().size() == 1);
        OPENVINO_ASSERT(rv_node->outputs().size() == 1);
        auto rv_in = rv_node->inputs()[0];
        auto x = rv_in.get_source_output();
        auto rv_out = rv_node->outputs()[0];
        // Gather all nodes that read from ReadValue, there must be SDPA and Assign
        auto rv_readers = rv_out.get_target_inputs();
        OPENVINO_ASSERT(rv_readers.size() == 2);
        // Input port for SDPA node
        for (const auto& reader : rv_readers) {
            bool is_fake_cvt = strstr(reader.get_node()->get_type_name(), "FakeConvert") != nullptr;
            if (strstr(reader.get_node()->get_type_name(), "ScaledDotProductAttention") != nullptr || is_fake_cvt) {
                auto sdpa_in = reader;

                // In case there's additional FakeConvert node(fp8): ReadValue -> FakeConvert -> SDPA
                auto is_fc_key_tensor = is_fake_cvt ? is_fake_cvt_to_key_tensor(reader) : false;

                // Remove ReadValue, store new Result and Assign
                auto key_or_value = (sdpa_in.get_index() == 1 || is_fc_key_tensor) ? "key" : "value";
                auto [result, assign] = remove_encoder_attn_read_value(rv_node, rv_in.get_source_output(), sdpa_in);
                auto normalized_name =
                    transform_key_value_name(rv_node->inputs()[0].get_source_output().get_node()->get_friendly_name(),
                                             "present",
                                             ".encoder.",
                                             key_or_value);
                set_name(result, normalized_name);
                results.push_back(result);
                assigns.push_back(assign);
            }
        }
    }

    // Add, remove, validate
    model->add_results(results);
    for (const auto& assign : assigns) {
        model->remove_sink(assign);
    }
    model->validate_nodes_and_infer_types();
}

void remove_cache_position(const std::shared_ptr<ov::Model>& model) {
    // Build subgraph that will replace cache_pos
    auto input_ids = model->input("input_ids").get_node();
    auto shape_of_node = std::make_shared<ov::op::v3::ShapeOf>(input_ids->outputs()[0]);

    std::vector<int> v_0{0};
    std::vector<int> v_1{1};

    auto indices = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, v_1);
    indices->set_friendly_name("indices");
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, v_0);
    axis->set_friendly_name("axis");

    auto gather_node = std::make_shared<ov::op::v8::Gather>(shape_of_node->outputs()[0], indices, axis);

    auto cst_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, v_0);
    auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, v_1);
    step->set_friendly_name("step");
    auto range_node = std::make_shared<ov::op::v4::Range>(cst_node->outputs()[0],
                                                          gather_node->outputs()[0],
                                                          step->outputs()[0],
                                                          ov::element::i64);
    // Replace cache_position
    auto cache_pos =
        ov::as_type_ptr<ov::op::v0::Parameter>(model->input("cache_position").get_node()->shared_from_this());
    for (const auto& target_input : cache_pos->outputs()[0].get_target_inputs()) {
        target_input.replace_source_output(range_node->outputs()[0]);
    }

    model->remove_parameter(cache_pos);
    model->validate_nodes_and_infer_types();
}

void expose_runtime_states_as_inputs(const std::shared_ptr<ov::Model>& model) {
    // Store Assign nodes to perform remove_sink later on
    ov::SinkVector assigns;
    // To add new Params to the model
    ov::ParameterVector params;

    ov::NodeVector read_value_nodes;
    for (const auto& op : model->get_ops()) {
        if (strstr(op->get_type_name(), "ReadValue") != nullptr) {
            read_value_nodes.push_back(op);
        }
    }

    for (const auto& rv_node : read_value_nodes) {
        auto rv_out = rv_node->outputs()[0];
        auto rv_readers = rv_out.get_target_inputs();
        for (auto rv_reader : rv_readers) {
            bool is_fake_cvt = strstr(rv_reader.get_node()->get_type_name(), "FakeConvert") != nullptr;
            if (strstr(rv_reader.get_node()->get_type_name(), "Assign") != nullptr) {
                auto assign_node = ov::as_type_ptr<ov::op::v6::Assign>(rv_reader.get_node()->shared_from_this());
                assigns.push_back(assign_node);
            } else if (strstr(rv_reader.get_node()->get_type_name(), "ScaledDotProductAttention") != nullptr ||
                       is_fake_cvt) {
                auto sdpa_in = rv_reader;

                auto shape = rv_node->get_output_partial_shape(0);
                auto new_param = std::make_shared<ov::op::v0::Parameter>(rv_node->get_output_element_type(0), shape);

                // In case there's additional FakeConvert node(fp8): ReadValue -> FakeConvert -> SDPA
                auto is_fc_key_tensor = is_fake_cvt ? is_fake_cvt_to_key_tensor(rv_reader) : false;

                auto key_or_value = (sdpa_in.get_index() == 1 || is_fc_key_tensor) ? "key" : "value";
                auto normalized_name = transform_key_value_name(sdpa_in.get_node()->get_friendly_name(),
                                                                "past_key_values",
                                                                ".encoder.",
                                                                key_or_value);
                set_name(new_param, normalized_name);

                params.push_back(new_param);
                sdpa_in.replace_source_output(new_param->outputs()[0]);
            }
        }
    }

    // Remove sinks and add new params
    model->add_parameters(params);
    for (const auto& assign : assigns) {
        model->remove_sink(assign);
    }
}

void normalize_input_key_value_names(const std::shared_ptr<ov::Model>& model) {
    ov::ResultVector new_results, old_results;
    for (const auto& in : model->inputs()) {
        if (in.get_any_name().find("decoder") == std::string::npos) {
            continue;
        }

        auto key_or_value = (in.get_any_name().find(".key") != std::string::npos) ? "key" : "value";
        auto normalized_name =
            transform_key_value_name(in.get_any_name(), "past_key_values", ".decoder.", key_or_value);
        set_name(in.get_node_shared_ptr(), normalized_name);
    }

    model->validate_nodes_and_infer_types();
}

void normalize_output_key_value_names(const std::shared_ptr<ov::Model>& model) {
    ov::ResultVector new_results, old_results;
    for (const auto& out : model->outputs()) {
        if (out.get_any_name().find("decoder") == std::string::npos) {
            continue;
        }

        auto key_or_value = (out.get_any_name().find(".key") != std::string::npos) ? "key" : "value";
        auto normalized_name = transform_key_value_name(out.get_any_name(), "present", ".decoder.", key_or_value);
        set_name(out.get_node_shared_ptr(), normalized_name);
    }

    model->validate_nodes_and_infer_types();
}

void add_attention_mask_input(const std::shared_ptr<ov::Model>& model,
                              const uint32_t& max_prompt_size = 0,
                              const uint32_t& lhs_seq_size = 0,
                              bool transform_cross_attn = false) {
    ov::pass::GraphRewrite rewr;
    if (transform_cross_attn) {
        rewr.add_matcher<AttentionMaskInput>(model, max_prompt_size, lhs_seq_size, transform_cross_attn);
    } else {
        rewr.add_matcher<AttentionMaskInputPast>(model);
        rewr.add_matcher<AttentionMaskInputPast_2>(model);  // transformers>=4.53
    }

    rewr.run_on_model(model);

    ov::pass::Validate().run_on_model(model);
}

void add_cache_position_input(const std::shared_ptr<ov::Model>& model) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<CachePositionInput>(model);
    rewr.run_on_model(model);

    ov::pass::Validate().run_on_model(model);
}

// FIXME: Whisper Decompose SDPA
void decompose_scaled_dot_product_attention_for_whisper(std::shared_ptr<ov::Model> model) {
    ov::pass::Manager manager;
    manager.register_pass<WhisperScaledDotProductAttentionDecomposition>();
    manager.run_passes(model);
}

// FIXME: Whisper Decompose SDPA
size_t add_cross_attention_qk_scaled_scores_outputs_for_whisper(std::shared_ptr<ov::Model> model) {
    size_t idx = 0;
    for (auto& op : model->get_ordered_ops()) {
        if (op->get_type_info().name != std::string("Add")) {
            continue;
        }

        bool should_skip_op = true;

        for (const auto& output : op->outputs()) {
            for (const auto& name : output.get_names()) {
                if (name.find("cross_attention_qk_scaled_scores") != std::string::npos) {
                    should_skip_op = false;
                    break;
                }
            }

            // output found, exit outputs loop
            if (!should_skip_op) {
                break;
            }
        }

        if (should_skip_op) {
            continue;
        }

        model->add_output(op->output(0)).set_names({"cross_attention_qk_scaled_scores_" + std::to_string(idx)});
        idx++;
    }

    return idx;
}

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // namespace

bool ov::npuw::util::PrepareWhisperPrefillModel::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // 2) Remove all non-runtime states from inputs (they empty on first iteration)
    // remove_input_kv_tensors(model); -> Done for LLM also
    // 3) Expose all states that requires initialization on the first run as outputs
    expose_runtime_states_as_outputs(model);
    // 4) Remove cache_position input if it exists
    if (has_input(model, "cache_position")) {
        remove_cache_position(model);
    }
    // 5) Normalize output names - should be done in stateful_to_stateless_transformation
    normalize_output_key_value_names(model);

    add_attention_mask_input(model, m_max_prompt_size, m_lhs_seq_size, true);

    // FIXME: Whisper Decompose SDPA
    if (m_decompose_sdpa) {
        decompose_scaled_dot_product_attention_for_whisper(model);
        m_decomposed_layers_size = add_cross_attention_qk_scaled_scores_outputs_for_whisper(model);
    }

    model->validate_nodes_and_infer_types();

    return true;
}

bool ov::npuw::util::PrepareWhisperKVCacheModel::run_on_model(const std::shared_ptr<ov::Model>& model) {
    normalize_input_key_value_names(model);
    normalize_output_key_value_names(model);
    expose_runtime_states_as_inputs(model);

    if (!has_input(model, "cache_position")) {
        add_cache_position_input(model);
    }

    add_attention_mask_input(model);

    model->reshape({{"input_ids", ov::PartialShape({-1, 1})}});

    model->validate_nodes_and_infer_types();

    return true;
}
