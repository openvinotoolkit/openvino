// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_compiled_model_utils.hpp"

#include <regex>

#include "logging.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"

namespace opp = ov::pass::pattern;

class TransposeValueTensors : public ov::pass::MatcherPass {
public:
    struct Context {
        using Ref = std::reference_wrapper<Context>;
        bool bTransposed = false;
    };

protected:
    // generic part of matchers, to transpose v-tensors, and concat, and update matmul args
    void transpose_matmul_b(Context::Ref ctx,
                            const std::shared_ptr<ov::op::v0::Parameter>& matched_param,
                            const std::shared_ptr<ov::op::v0::Concat>& matched_concat,
                            const std::shared_ptr<ov::op::v1::Transpose>& matched_transpose,
                            const std::shared_ptr<ov::op::v0::MatMul>& matched_matmul) {
        auto param_shape = matched_param->get_partial_shape();
        NPUW_ASSERT(param_shape.size() == 4u);
        // NB: Transpose Parameter that correspond to V-tensor it will
        // speed-up its multiplication with attention scores
        std::swap(param_shape[2], param_shape[3]);

        matched_param->set_partial_shape(param_shape);

        auto order_cst = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});

        matched_transpose->set_argument(1, order_cst);
        matched_concat->set_axis(3u);
        matched_matmul->set_transpose_b(true);
        ctx.get().bTransposed = true;
    }

    void transpose_matmul_b(Context::Ref ctx,
                            const std::shared_ptr<ov::Node>& node_param,
                            const std::shared_ptr<ov::Node>& node_concat,
                            const std::shared_ptr<ov::Node>& node_transpose,
                            const std::shared_ptr<ov::Node>& node_matmul) {
        auto matched_param = std::static_pointer_cast<ov::op::v0::Parameter>(node_param);
        auto matched_concat = std::static_pointer_cast<ov::op::v0::Concat>(node_concat);
        auto matched_transpose = std::static_pointer_cast<ov::op::v1::Transpose>(node_transpose);
        auto matched_matmul = std::static_pointer_cast<ov::op::v0::MatMul>(node_matmul);

        transpose_matmul_b(ctx, matched_param, matched_concat, matched_transpose, matched_matmul);
    }
};

// llama2 pattern for value tensor concate
class TransposeValueTensors_llama2 : public TransposeValueTensors {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::TransposeValueTensors_llama2");
    TransposeValueTensors_llama2(Context::Ref ctx) {
        register_matcher_llama2(ctx);
    }

private:
    void register_matcher_llama2(Context::Ref ctx) {
        auto param = opp::wrap_type<ov::op::v0::Parameter>();
        auto transpose = opp::wrap_type<ov::op::v1::Transpose>({opp::any_input(), opp::any_input()});
        auto convert = opp::optional<ov::op::v0::Convert>({param->output(0)});
        auto concat = opp::wrap_type<ov::op::v0::Concat>({convert, transpose});
        auto softmax = opp::wrap_type<ov::op::v8::Softmax>({opp::any_input()});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({softmax, concat});

        auto callback = [=](ov::pass::pattern::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            auto matched_node_param = node_to_output.at(param).get_node_shared_ptr();
            auto matched_node_concat = node_to_output.at(concat).get_node_shared_ptr();
            auto matched_node_transpose = node_to_output.at(transpose).get_node_shared_ptr();
            auto matched_node_matmul = node_to_output.at(matmul).get_node_shared_ptr();

            transpose_matmul_b(ctx,
                               matched_node_param,
                               matched_node_concat,
                               matched_node_transpose,
                               matched_node_matmul);
            LOG_DEBUG("vtensors transposed: LLama2 pattern");
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(matmul, "TransposeValueTensors_llama2"), std::move(callback));
    }
};

// llama3, phi3, mistral, etc, concate value tensors with broadcasting
class TransposeValueTensors_llama3 : public TransposeValueTensors {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::TransposeValueTensors_llama3");
    TransposeValueTensors_llama3(Context::Ref ctx) {
        register_matcher_llama3(ctx);
    }

private:
    void register_matcher_llama3(Context::Ref ctx) {
        auto param = opp::wrap_type<ov::op::v0::Parameter>();
        auto transpose = opp::wrap_type<ov::op::v1::Transpose>({opp::any_input(), opp::any_input()});
        auto convert = opp::optional<ov::op::v0::Convert>({param->output(0)});
        auto concat = opp::wrap_type<ov::op::v0::Concat>({convert, transpose});

        // only difference is that broadcast wrapped into unsquese/reshape, while transposed tensor didn't change
        const auto unsqueeze_axes = opp::wrap_type<ov::op::v0::Constant>();
        auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({concat, unsqueeze_axes});
        auto broadcast = opp::wrap_type<ov::op::v1::Broadcast, ov::op::v3::Broadcast>({unsqueeze, opp::any_input()});
        auto reshape = opp::wrap_type<ov::op::v1::Reshape>({broadcast, opp::any_input()});

        // v8 softmax? what? can be other softmaxes
        auto softmax = opp::wrap_type<ov::op::v8::Softmax>({opp::any_input()});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({softmax, reshape});

        auto callback = [=](ov::pass::pattern::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            auto matched_node_param = node_to_output.at(param).get_node_shared_ptr();
            auto matched_node_concat = node_to_output.at(concat).get_node_shared_ptr();
            auto matched_node_transpose = node_to_output.at(transpose).get_node_shared_ptr();
            auto matched_node_matmul = node_to_output.at(matmul).get_node_shared_ptr();
            auto matched_node_unsqueeze = node_to_output.at(unsqueeze).get_node_shared_ptr();
            auto matched_node_unsqueeze_axes = node_to_output.at(unsqueeze_axes).get_node_shared_ptr();
            auto matched_node_broadcast = node_to_output.at(broadcast).get_node_shared_ptr();
            auto matched_node_reshape = node_to_output.at(reshape).get_node_shared_ptr();

            auto matched_param = std::static_pointer_cast<ov::op::v0::Parameter>(matched_node_param);
            auto matched_concat = std::static_pointer_cast<ov::op::v0::Concat>(matched_node_concat);
            auto matched_transpose = std::static_pointer_cast<ov::op::v1::Transpose>(matched_node_transpose);
            auto matched_matmul = std::static_pointer_cast<ov::op::v0::MatMul>(matched_node_matmul);
            auto matched_unsqueeze = std::static_pointer_cast<ov::op::v0::Unsqueeze>(matched_node_unsqueeze);
            auto matched_broadcast = std::static_pointer_cast<ov::op::v3::Broadcast>(matched_node_broadcast);
            auto matched_reshape = std::static_pointer_cast<ov::op::v1::Reshape>(matched_node_reshape);

            auto shape_broadcast = matched_broadcast->get_output_shape(0);
            NPUW_ASSERT(shape_broadcast.size() == 5u);
            std::swap(shape_broadcast[3], shape_broadcast[4]);

            LOG_DEBUG("shape_broadcast for: " << matched_broadcast->get_friendly_name()
                                              << ", shape=" << shape_broadcast);

            const auto broadcast_axes_node =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{5}, shape_broadcast);
            broadcast_axes_node->set_friendly_name(matched_broadcast->get_friendly_name() + "/new_broadcast_shape");
            matched_broadcast->input(1).replace_source_output(broadcast_axes_node);

            auto shape_reshape = matched_reshape->get_output_shape(0);
            NPUW_ASSERT(shape_reshape.size() == 4u);
            std::swap(shape_reshape[2], shape_reshape[3]);

            LOG_DEBUG("shape_reshape for: " << matched_reshape->get_friendly_name() << ", shape=" << shape_reshape);

            const auto reshape_axes_node =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, shape_reshape);
            reshape_axes_node->set_friendly_name(matched_reshape->get_friendly_name() + "/new_reshape_shape");
            matched_reshape->input(1).replace_source_output(reshape_axes_node);

            transpose_matmul_b(ctx, matched_param, matched_concat, matched_transpose, matched_matmul);
            LOG_DEBUG("vtensors transposed: LLama3 pattern");
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(matmul, "TransposeValueTensors_llama3"), std::move(callback));
    }
};

class ScaledDotProductAttentionDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::ScaledDotProductAttentionDecomposition");
    explicit ScaledDotProductAttentionDecomposition(bool use_high_precision_on_add) {
        auto pattern_node = ov::pass::pattern::wrap_type<ov::op::v13::ScaledDotProductAttention>();

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            auto& pattern_to_output = m.get_pattern_value_map();
            auto node = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(
                pattern_to_output.at(pattern_node).get_node_shared_ptr());

            if (node == nullptr || transformation_callback(node)) {
                return false;
            }

            auto new_output_node = decompose(node, use_high_precision_on_add);
            ov::replace_node(node, new_output_node);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_node, "ScaledDotProductAttentionDecomposition");
        register_matcher(m, std::move(callback));
    }
    std::shared_ptr<ov::Node> decompose(std::shared_ptr<ov::op::v13::ScaledDotProductAttention> node,
                                        bool use_high_precision_on_add) {
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

        Output<Node> scale;
        if (node->get_input_size() < 5) {
            scale = register_new_node<v8::Gather>(q_shape, minus_one, zero_i)->output(0);
            scale = register_new_node<v1::ConvertLike>(scale, query);
            auto sqrt_scale = register_new_node<v0::Sqrt>(scale);
            scale = register_new_node<v1::Divide>(one_f, sqrt_scale);
        } else {
            scale = node->input_value(4);
        }

        auto q_scaled = register_new_node<v1::Multiply>(query, scale);
        auto k_rank = register_new_node<v3::ShapeOf>(k_shape, element::i32)->output(0);
        auto k_last_dim = register_new_node<v1::Add>(k_rank, minus_one);
        auto k_next_dim = register_new_node<v1::Add>(k_rank, minus_two)->output(0);
        k_rank = register_new_node<v0::Squeeze>(k_rank, zero_i);
        auto minus_inf =
            register_new_node(v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()}))
                ->output(0);
        auto keep_dim_last = register_new_node<v0::Squeeze>(k_next_dim, zero_i);
        auto k_dims_before_transpose = register_new_node<v4::Range>(zero_i, keep_dim_last, one_i, element::i32);

        auto scaled_atten = register_new_node<v0::MatMul>(q_scaled, key, false, true)->output(0);
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
                    atten_mask = register_new_node<v1::ConvertLike>(mask, scaled_atten);
                    auto inv_mask = register_new_node<v1::LogicalNot>(mask);
                    atten_mask = register_new_node<v1::Select>(inv_mask, atten_mask, minus_inf);
                } else {
                    atten_mask = mask;
                }
            } else {
                auto target_s_len = register_new_node<v8::Gather>(q_shape, minus_two, zero_i);
                auto source_s_len = register_new_node<v8::Gather>(k_shape, minus_two, zero_i);
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
            if (use_high_precision_on_add) {
                npuw::util::HighPrecisionAttr attr_hp;
                attr_hp.compute_precision_type = ov::element::f32;
                atten_mask.get_rt_info()[npuw::util::HighPrecisionAttr::get_type_info_static()] = attr_hp;
            }

            scaled_atten = register_new_node<v1::Add>(scaled_atten, atten_mask);
        }

        scaled_atten = register_new_node<v8::Softmax>(scaled_atten, -1);
        auto result = register_new_node<v0::MatMul>(scaled_atten, value);
        result->set_friendly_name(node->get_friendly_name());
        copy_runtime_info(node, get_new_nodes());
        return result;
    }
};

class AttentionMaskInputPast : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::AttentionMaskInputPast");

    AttentionMaskInputPast(std::shared_ptr<ov::Model> model) {
        auto range = opp::wrap_type<ov::op::v4::Range>();
        auto convert1 = opp::wrap_type<ov::op::v0::Convert>({range});
        auto greater = opp::wrap_type<ov::op::v1::Greater>({convert1, opp::any_input()});
        auto convert2 = opp::wrap_type<ov::op::v0::Convert>({greater});

        register_matcher(std::make_shared<opp::Matcher>(convert2, this->get_type_info().name), [model](opp::Matcher& m) {
            auto node = m.get_match_root();
            auto attention_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
            attention_mask->get_output_tensor(0).set_names({"attention_mask"});
            model->add_parameters({attention_mask});

            auto cvt = std::make_shared<ov::op::v0::Convert>(attention_mask->output(0), ov::element::f32);
            ov::replace_node(node, cvt);
            return false;
        });
    }
};

class AttentionMaskInput : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::AttentionMaskInput");

    AttentionMaskInput(std::shared_ptr<ov::Model> model, const uint32_t& max_prompt_len, const uint32_t& lhs_seq_size, bool transform_cross_attn) {
        std::vector<std::shared_ptr<ov::Node>> self_attn_nodes;
        std::vector<std::shared_ptr<ov::Node>> cross_attn_nodes;
        for (auto node : model->get_ops()) {
            if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(node)) {
                if (node->inputs().size() == 4u) {
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

        const auto kAttnMaskPort = 3;
        auto slice = self_attn_nodes[0]->input(kAttnMaskPort).get_source_output().get_node();
        auto cvt = std::make_shared<ov::op::v0::Convert>(attention_mask->output(0), ov::element::f32);
        auto add = std::make_shared<ov::op::v1::Add>(slice->output(0), cvt->output(0));

        auto trps = std::make_shared<ov::op::v1::Transpose>(cvt->output(0), ov::op::v0::Constant::create(ov::element::i32,
                                                                                                         ov::Shape{2},
                                                                                                         std::vector<int>{1, 0}));
        auto mtpl = std::make_shared<ov::op::v1::Multiply>(trps->output(0), add->output(0));

        auto cst_ninf = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1},
                                                               std::vector<float>{-std::numeric_limits<float>::max()}
                                                               );
        auto cst_1 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{1});
        auto cst_0 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0});

        auto equal = std::make_shared<ov::op::v1::Equal>(mtpl->output(0), cst_1->output(0));
        auto select = std::make_shared<ov::op::v1::Select>(equal->output(0), cst_0->output(0), cst_ninf->output(0));

        for (auto self_attn : self_attn_nodes) {
            self_attn->input(3).replace_source_output(select->output(0));
        }

        if (transform_cross_attn) {
            // Cross attn
            OPENVINO_ASSERT(!cross_attn_nodes.empty());
            // FIXME: Should be taken from topology - don't hardcode!!!
            auto shape_cst = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64,
                ov::Shape{2},
                std::vector<float>{static_cast<float>(max_prompt_len), 1}
            );
            auto target_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64,
                ov::Shape{2},
                std::vector<float>{static_cast<float>(max_prompt_len), static_cast<float>(lhs_seq_size)}
            );
            // FIXME: Must be transpose if batch present
            auto reshape = std::make_shared<ov::op::v1::Reshape>(cvt->output(0), shape_cst->output(0), false);
            auto equal = std::make_shared<ov::op::v1::Equal>(reshape->output(0), cst_1->output(0));
            auto select = std::make_shared<ov::op::v1::Select>(
                equal->output(0), cst_0->output(0), cst_ninf->output(0)
            );
            auto broadcast = std::make_shared<ov::op::v3::Broadcast>(select->output(0), target_shape->output(0));
            auto unsq1 = std::make_shared<ov::op::v0::Unsqueeze>(broadcast->output(0), cst_0->output(0));
            auto unsq2 = std::make_shared<ov::op::v0::Unsqueeze>(unsq1->output(0), cst_1->output(0));
            for (auto cross_attn_node : cross_attn_nodes) {
                auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
                    cross_attn_node->input(0).get_source_output(),
                    cross_attn_node->input(1).get_source_output(),
                    cross_attn_node->input(2).get_source_output(),
                    unsq2->output(0),
                    false
                );
                ov::replace_node(cross_attn_node, sdpa);
            }
        }
    }
};


namespace {
auto remove_encoder_attn_read_value(const std::shared_ptr<ov::Node>& rv_node,
                                    const ov::Output<ov::Node>& kv_out,
                                    const ov::Input<ov::Node>& sdpa_in) {
    // Find Assign node
    OPENVINO_ASSERT(rv_node->outputs().size() == 1);
    auto rv_out = rv_node->outputs()[0];
    ov::NodeVector rv_readers;
    for (const auto& target_in: rv_out.get_target_inputs()) {
        rv_readers.push_back(target_in.get_node()->shared_from_this());
    }
    // Assign and SDPA
    OPENVINO_ASSERT(rv_readers.size() == 2);
    auto assign_node = (strstr(rv_readers[0]->get_type_name(), "Assign") != nullptr) ? rv_readers[0] : rv_readers[1];
    OPENVINO_ASSERT(strstr(assign_node->get_type_name(), "Assign") != nullptr);
    // Redirect KV-cache tensor to SDPA
    sdpa_in.replace_source_output(kv_out);
    return std::make_pair(std::make_shared<ov::op::v0::Result>(kv_out), ov::as_type_ptr<ov::op::v6::Assign>(assign_node));
}

std::string transform_key_value_name(std::string input_string, std::string prefix, std::string enc_or_dec, std::string key_or_value) {
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

void expose_runtime_states_as_outputs(std::shared_ptr<ov::Model>& model) {
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
        auto rv_in  = rv_node->inputs()[0];
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
                auto normalized_name = transform_key_value_name(
                    rv_node->inputs()[0].get_source_output().get_node()->get_friendly_name(),
                    "present",
                    ".encoder.",
                    key_or_value
                );
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

void remove_cache_position(std::shared_ptr<ov::Model>& model) {
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
    auto step     = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, v_1);
    step->set_friendly_name("step");
    auto range_node = std::make_shared<ov::op::v4::Range>(cst_node->outputs()[0], gather_node->outputs()[0], step->outputs()[0], ov::element::i64);
    // Replace cache_position
    auto cache_pos = ov::as_type_ptr<ov::op::v0::Parameter>(model->input("cache_position").get_node()->shared_from_this());
    for (const auto& target_input : cache_pos->outputs()[0].get_target_inputs()) {
        target_input.replace_source_output(range_node->outputs()[0]);
    }

    model->remove_parameter(cache_pos);
    model->validate_nodes_and_infer_types();
}

void expose_runtime_states_as_inputs(std::shared_ptr<ov::Model>& model) {
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
        for (auto rv_reader: rv_readers) {
            bool is_fake_cvt = strstr(rv_reader.get_node()->get_type_name(), "FakeConvert") != nullptr;
            if (strstr(rv_reader.get_node()->get_type_name(), "Assign") != nullptr) {
                auto assign_node = ov::as_type_ptr<ov::op::v6::Assign>(rv_reader.get_node()->shared_from_this());
                assigns.push_back(assign_node);
            } else if (strstr(rv_reader.get_node()->get_type_name(), "ScaledDotProductAttention") != nullptr || is_fake_cvt) {
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
    for (const auto& assign: assigns) {
        model->remove_sink(assign);
    }
}

void normalize_input_key_value_names(std::shared_ptr<ov::Model>& model) {
    ov::ResultVector new_results, old_results;
    for (const auto& in : model->inputs()) {
        if (in.get_any_name().find("decoder") == std::string::npos) {
            continue;
        }

        auto key_or_value = (in.get_any_name().find(".key") != std::string::npos) ? "key" : "value";
        auto normalized_name = transform_key_value_name(in.get_any_name(), "past_key_values", ".decoder.", key_or_value);
        set_name(in.get_node_shared_ptr(), normalized_name);
    }

    model->validate_nodes_and_infer_types();
}

void normalize_output_key_value_names(std::shared_ptr<ov::Model>& model) {
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

void add_attention_mask_input(std::shared_ptr<ov::Model> model,
                              const uint32_t& max_prompt_size = 0,
                              const uint32_t& lhs_seq_size = 0,
                              bool transform_cross_attn = false) {
    ov::pass::GraphRewrite rewr;
    if (transform_cross_attn) {
        rewr.add_matcher<AttentionMaskInput>(model, max_prompt_size, lhs_seq_size, transform_cross_attn);
    } else {
        rewr.add_matcher<AttentionMaskInputPast>(model);
    }

    rewr.run_on_model(model);

    ov::pass::Validate().run_on_model(model);
}
}  // namespace

bool ov::npuw::util::optimize_value_tensors(std::shared_ptr<ov::Model> model, bool isPrefill) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ScaledDotProductAttentionDecomposition>(isPrefill);
    TransposeValueTensors::Context ctx;
    rewr.add_matcher<TransposeValueTensors_llama2>(std::ref(ctx));
    rewr.add_matcher<TransposeValueTensors_llama3>(std::ref(ctx));
    rewr.run_on_model(model);

    ov::pass::Validate().run_on_model(model);

    // NB: matmul parameters gets transposed, if pass applied
    return ctx.bTransposed;
}

std::shared_ptr<ov::Model> ov::npuw::util::prepare_whisper_prefill_model(std::shared_ptr<ov::Model>& model,
                                                                         const uint32_t& max_prompt_size,
                                                                         const uint32_t& lhs_seq_size) {
    // 2) Remove all non-runtime states from inputs (they empty on first iteration)
    // remove_input_kv_tensors(model); -> Done for LLM also 
    // 3) Expose all states that requires initialization on the first run as outputs
    expose_runtime_states_as_outputs(model);
    // 4) Remove cache_position input
    remove_cache_position(model);
    // 5) Normalize output names - should be done in stateful_to_stateless_transformation
    normalize_output_key_value_names(model);

    add_attention_mask_input(model, max_prompt_size, lhs_seq_size, true);

    model->validate_nodes_and_infer_types();
    return model;
}

std::shared_ptr<ov::Model> ov::npuw::util::prepare_whisper_kvcache_model(std::shared_ptr<ov::Model>& model) {
    // FIXME: normalization should be done inside stateful_to_stateless_transformation
    normalize_input_key_value_names(model);
    normalize_output_key_value_names(model);
    expose_runtime_states_as_inputs(model);

    add_attention_mask_input(model);

    model->reshape({{"input_ids", ov::PartialShape({-1, 1})}});

    model->validate_nodes_and_infer_types();
    return model;
}
