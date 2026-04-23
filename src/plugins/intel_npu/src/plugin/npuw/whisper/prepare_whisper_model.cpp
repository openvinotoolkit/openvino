// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prepare_whisper_model.hpp"

#include <regex>

#include "../llm_compiled_model_utils.hpp"
#include "../util.hpp"
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
                    ov::is_type<ov::op::v8::Slice>(node->input(kAttnMaskPort).get_source_output().get_node())) {
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
                                             ov::npuw::util::constants::present,
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
                                                                ov::npuw::util::constants::past_key_values,
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

        const auto key_idx = ov::npuw::util::isPastKeyValuesKey(in.get_any_name());
        const auto value_idx = ov::npuw::util::isPastKeyValuesValue(in.get_any_name());
        const auto normalized_name = [&]() {
            if (key_idx.has_value() || value_idx.has_value()) {
                const auto idx = key_idx.has_value() ? key_idx.value() : value_idx.value();
                const auto key_or_value = key_idx.has_value() ? "key" : "value";
                return std::string(ov::npuw::util::constants::past_key_values) + "." + std::to_string(idx) +
                       ".decoder." + key_or_value;
            }

            const auto key_or_value = (in.get_any_name().find(".key") != std::string::npos) ? "key" : "value";
            return transform_key_value_name(in.get_any_name(),
                                            ov::npuw::util::constants::past_key_values,
                                            ".decoder.",
                                            key_or_value);
        }();
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

        const auto key_idx = ov::npuw::util::isPresentKeyValuesKey(out.get_any_name());
        const auto value_idx = ov::npuw::util::isPresentKeyValuesValue(out.get_any_name());
        const auto normalized_name = [&]() {
            if (key_idx.has_value() || value_idx.has_value()) {
                const auto idx = key_idx.has_value() ? key_idx.value() : value_idx.value();
                const auto key_or_value = key_idx.has_value() ? "key" : "value";
                return std::string(ov::npuw::util::constants::present) + "." + std::to_string(idx) + ".decoder." +
                       key_or_value;
            }

            const auto key_or_value = (out.get_any_name().find(".key") != std::string::npos) ? "key" : "value";
            return transform_key_value_name(out.get_any_name(),
                                            ov::npuw::util::constants::present,
                                            ".decoder.",
                                            key_or_value);
        }();
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
