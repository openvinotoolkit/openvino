// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_model_utils.hpp"

#include <cfloat>
#include <regex>

#include "logging.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"
#include "transformations/utils/utils.hpp"

namespace opp = ov::pass::pattern;

namespace {

std::string combine_key_value_name(std::string prefix, std::string layer_id, std::string key_or_value) {
    return prefix + "." + layer_id + "." + key_or_value;
}

void set_node_name(std::shared_ptr<ov::Node> node, const std::string& name) {
    node->set_friendly_name(name);
    node->get_output_tensor(0).set_names({name});
}

// Generate Qwen3 LLM mask matrix using ngraph ops
// Qwen3 uses a 4D attention mask with shape (batch, 1, seq_len, seq_len + cache_len)
// The mask combines causal attention pattern with input attention_mask values
// Inputs:
//   - input_ids: shape [batch, seq_len]
//   - attention_mask: shape [batch, seq_len + cache_len], values are 0 (masked) or 1 (valid)
// Output: 4D mask tensor of shape (batch, 1, seq_len, seq_len + cache_len)
//   - Values are 0.0 for positions that can be attended to
//   - Values are -inf for masked positions
ov::Output<ov::Node> create_new_mask(std::shared_ptr<ov::Model> model,
                                     ov::element::Type element_type = ov::element::f32) {
    using namespace ov::op;

    auto input_ids = model->input("input_ids");
    auto attention_mask = model->input("attention_mask");

    // Constants
    auto zero_i = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 0);
    auto one_i = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 1);
    auto two_i = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 2);
    auto minus_one = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, -1);
    auto minus_two = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, -2);
    auto zero_f = std::make_shared<v0::Constant>(element_type, ov::Shape{}, 0.0f);
    auto one_f = std::make_shared<v0::Constant>(element_type, ov::Shape{}, 1.0f);
    auto minus_inf = std::make_shared<v0::Constant>(element_type, ov::Shape{}, -FLT_MAX);

    // Extract shapes
    // input_ids: [batch, seq_len]
    auto input_ids_shape = std::make_shared<v3::ShapeOf>(input_ids, ov::element::i64);
    auto batch_size = std::make_shared<v8::Gather>(input_ids_shape, zero_i, zero_i);
    auto seq_len = std::make_shared<v8::Gather>(input_ids_shape, one_i, zero_i);

    // attention_mask: [batch, total_seq_len] where total_seq_len = cache_len + seq_len
    auto attn_mask_shape = std::make_shared<v3::ShapeOf>(attention_mask, ov::element::i64);
    auto total_seq_len = std::make_shared<v8::Gather>(attn_mask_shape, one_i, zero_i);

    // Calculate cache_len
    auto cache_len = std::make_shared<v1::Subtract>(total_seq_len, seq_len);

    auto query_positions = std::make_shared<v4::Range>(zero_i, seq_len, one_i, ov::element::i64);
    auto query_pos_unsqueeze = std::make_shared<v0::Unsqueeze>(query_positions, one_i);

    // Create key positions: [0, 1, 2, ..., total_seq_len-1] with shape (1, total_seq_len)
    auto key_positions = std::make_shared<v4::Range>(zero_i, total_seq_len, one_i, ov::element::i64);
    auto key_pos_unsqueeze = std::make_shared<v0::Unsqueeze>(key_positions, zero_i);

    auto causal_threshold = std::make_shared<v1::Add>(cache_len, query_pos_unsqueeze);
    auto causal_condition = std::make_shared<v1::LessEqual>(key_pos_unsqueeze, causal_threshold);
    auto attn_mask_bool = std::make_shared<v0::Convert>(attention_mask, ov::element::boolean);

    // Expand attention_mask from [batch, total_seq_len] to [batch, 1, total_seq_len]
    auto attn_mask_expanded = std::make_shared<v0::Unsqueeze>(attn_mask_bool, one_i);
    // (1, seq_len, total_seq_len)
    auto causal_expanded = std::make_shared<v0::Unsqueeze>(causal_condition, zero_i);

    auto combined_condition = std::make_shared<v1::LogicalAnd>(causal_expanded, attn_mask_expanded);
    auto float_mask = std::make_shared<v1::Select>(combined_condition, zero_f, minus_inf);
    // (batch, 1, seq_len, total_seq_len)
    auto final_mask = std::make_shared<v0::Unsqueeze>(float_mask, one_i);

    return final_mask;
}

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

class AddKVCacheNodes : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::AddKVCacheNodes");
    explicit AddKVCacheNodes(std::shared_ptr<ov::Model> model, uint32_t seq_len_dim) {
        const auto unsqueeze_axes = opp::wrap_type<ov::op::v0::Constant>();
        const auto transpose_order = opp::wrap_type<ov::op::v0::Constant>();
        const std::regex layer_id_convertion = std::regex(R"(layers\.(\d+)\.self_attn)");

        auto shape_concat = opp::wrap_type<ov::op::v0::Concat>(
            {opp::any_input(), opp::any_input(), opp::any_input(), opp::any_input()});

        auto k_add = opp::wrap_type<ov::op::v1::Add>({opp::any_input(), opp::any_input()});
        auto k_unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({k_add, unsqueeze_axes});
        auto k_broadcast = opp::wrap_type<ov::op::v1::Broadcast, ov::op::v3::Broadcast>({k_unsqueeze, shape_concat});
        auto k_reshape = opp::wrap_type<ov::op::v1::Reshape>({k_broadcast, opp::any_input()});

        auto v_transpose = opp::wrap_type<ov::op::v1::Transpose>({opp::any_input(), transpose_order});
        auto v_unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({v_transpose, unsqueeze_axes});
        auto v_broadcast = opp::wrap_type<ov::op::v1::Broadcast, ov::op::v3::Broadcast>({v_unsqueeze, shape_concat});
        auto v_reshape = opp::wrap_type<ov::op::v1::Reshape>({v_broadcast, opp::any_input()});

        auto sdpa = opp::wrap_type<ov::op::v13::ScaledDotProductAttention>(
            {opp::any_input(), k_reshape, v_reshape, opp::any_input(), opp::any_input()});

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            auto& pattern_to_output = m.get_pattern_value_map();
            auto sdpa_node = pattern_to_output.at(sdpa).get_node_shared_ptr();
            auto k_add_node = pattern_to_output.at(k_add).get_node_shared_ptr();
            auto k_unsqueeze_node = pattern_to_output.at(k_unsqueeze).get_node_shared_ptr();
            auto v_transpose_node = pattern_to_output.at(v_transpose).get_node_shared_ptr();
            auto v_unsqueeze_node = pattern_to_output.at(v_unsqueeze).get_node_shared_ptr();

            auto k_broadcast_node = pattern_to_output.at(k_broadcast).get_node_shared_ptr();
            auto v_broadcast_node = pattern_to_output.at(v_broadcast).get_node_shared_ptr();

            std::smatch match;
            if (!std::regex_search(sdpa_node->get_friendly_name(), match, layer_id_convertion)) {
                return false;
            }

            auto layer_id = std::string(match[1]);
            auto k_add_out_shape = k_add_node->get_output_partial_shape(0);
            auto k_add_type = k_add_node->get_output_element_type(0);
            auto k_cache = std::make_shared<ov::op::v0::Parameter>(k_add_type, k_add_out_shape);
            set_node_name(k_cache, combine_key_value_name("past_key_values", layer_id, "key"));

            auto k_concat = register_new_node<ov::op::v0::Concat>(ov::OutputVector{k_cache, k_add_node}, seq_len_dim);
            set_node_name(k_concat, combine_key_value_name("concat", layer_id, "key"));

            k_unsqueeze_node->input(0).replace_source_output(k_concat);

            auto k_cache_out = std::make_shared<ov::op::v0::Result>(k_add_node);
            set_node_name(k_cache_out, combine_key_value_name("present", layer_id, "key"));

            auto v_transpose_out_shape = v_transpose_node->get_output_partial_shape(0);
            auto v_transpose_type = v_transpose_node->get_output_element_type(0);
            auto v_cache = std::make_shared<ov::op::v0::Parameter>(v_transpose_type, v_transpose_out_shape);
            set_node_name(v_cache, combine_key_value_name("past_key_values", layer_id, "value"));

            auto v_concat =
                register_new_node<ov::op::v0::Concat>(ov::OutputVector{v_cache, v_transpose_node}, seq_len_dim);
            set_node_name(v_concat, combine_key_value_name("concat", layer_id, "value"));

            v_unsqueeze_node->input(0).replace_source_output(v_concat);

            auto v_cache_out = std::make_shared<ov::op::v0::Result>(v_transpose_node);
            set_node_name(v_cache_out, combine_key_value_name("present", layer_id, "value"));

            model->add_parameters(ov::ParameterVector{k_cache, v_cache});
            model->add_results(ov::ResultVector{k_cache_out, v_cache_out});
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa, "AddKVCacheNodes");
        register_matcher(m, std::move(callback));
    }
};

class AddPositionIdsNode : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::AddPositionIdsNode");
    explicit AddPositionIdsNode(std::shared_ptr<ov::Model> model) {
        auto range = opp::wrap_type<ov::op::v4::Range>();
        auto unsqueeze_axes = opp::wrap_type<ov::op::v0::Constant>();
        auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({range, unsqueeze_axes});

        auto unsqueeze1_axes = opp::wrap_type<ov::op::v0::Constant>();
        auto unsqueeze1 = opp::wrap_type<ov::op::v0::Unsqueeze>({unsqueeze, unsqueeze1_axes});

        auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze1});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), convert});
        auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});

        auto concat = opp::wrap_type<ov::op::v0::Concat>({transpose, transpose});
        auto sin = opp::wrap_type<ov::op::v0::Sin>(concat);
        auto cos = opp::wrap_type<ov::op::v0::Cos>(concat);

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            auto& pattern_to_output = m.get_pattern_value_map();

            auto unsqueeze1_node = pattern_to_output.at(unsqueeze1).get_node_shared_ptr();

            auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
            set_node_name(position_ids, "position_ids");

            unsqueeze1_node->input(0).replace_source_output(position_ids);
            model->add_parameters(ov::ParameterVector{position_ids});
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(cos, "AddPositionIdsNode");
        register_matcher(m, std::move(callback));
    }
};

class OPENVINO_API ReConstructEmbeddingModel : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ReConstructEmbeddingModel");

    explicit ReConstructEmbeddingModel(uint32_t seq_len_dim) : m_seq_len_dim(seq_len_dim), m_sdpa(nullptr) {}

    bool replace_mask_node(const std::shared_ptr<ov::Model>& model) {
        auto new_mask = create_new_mask(model);
        for (const auto& op : model->get_ops()) {
            if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(op)) {
                auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op);
                if (m_sdpa == nullptr) {
                    m_sdpa = sdpa;
                }
                sdpa->input(3).replace_source_output(new_mask);
            }
        }
        return true;
    }

    bool update_kv_concat_shape(std::shared_ptr<ov::Model> model) {
        if (m_sdpa == nullptr) {
            return false;
        }

        auto reshape_node = m_sdpa->input(1).get_source_output().get_node();
        if (strstr(reshape_node->get_type_name(), "Reshape") == nullptr) {
            return false;
        }

        auto broadcast_node = reshape_node->input(0).get_source_output().get_node();
        if (strstr(broadcast_node->get_type_name(), "Broadcast") == nullptr) {
            return false;
        }

        auto concat_node = broadcast_node->input(1).get_source_output().get_node();
        if (strstr(concat_node->get_type_name(), "Concat") == nullptr) {
            return false;
        }

        auto attention_mask = model->input("attention_mask");

        auto minus_one =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
        auto zero_i = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
        auto one_i = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

        auto mask_shapeof = std::make_shared<ov::op::v3::ShapeOf>(attention_mask, ov::element::i64);
        auto s_len = std::make_shared<ov::op::v8::Gather>(mask_shapeof, minus_one, zero_i);
        auto s_len_reshape = std::make_shared<ov::op::v1::Reshape>(s_len, one_i, false);
        concat_node->input(2).replace_source_output(s_len_reshape);

        return true;
    }

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        OPENVINO_ASSERT(ov::op::util::has_op_with_type<ov::op::v13::ScaledDotProductAttention>(model),
                        "No ScaledDotProductAttention operation observed in the graph");

        ov::pass::Manager manager("construct-embedding");
        manager.set_per_pass_validation(true);
        manager.register_pass<AddPositionIdsNode>(model);
        manager.register_pass<AddKVCacheNodes>(model, m_seq_len_dim);
        manager.run_passes(model);

        replace_mask_node(model);
        OPENVINO_ASSERT(update_kv_concat_shape(model), "Fail to re-construct text-embedding model");
        model->validate_nodes_and_infer_types();
        return true;
    }

private:
    uint32_t m_seq_len_dim;
    std::shared_ptr<ov::Node> m_sdpa;
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // namespace

void ov::npuw::util::prepare_text_embedding_model(std::shared_ptr<ov::Model> model, uint32_t seq_len_dim) {
    ov::pass::Manager manager("prepare-embedding");
    manager.set_per_pass_validation(true);
    manager.register_pass<ReConstructEmbeddingModel>(seq_len_dim);
    manager.run_passes(model);
}
