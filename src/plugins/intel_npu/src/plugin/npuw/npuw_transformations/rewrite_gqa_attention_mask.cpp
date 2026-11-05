// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rewrite_gqa_attention_mask.hpp"

#include <numeric>
#include <optional>

#include "../llm_infer_base_request.hpp"
#include "../logging.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

constexpr const char* kGQANameMarker = "GroupQueryAttention";

// Query branch of the mask: CumSum -> Add -> Broadcast -> Transpose
struct QueryBranch {
    std::shared_ptr<ov::Node> cumsum;
    std::shared_ptr<ov::Node> add;
    std::shared_ptr<ov::Node> broadcast;
    std::shared_ptr<ov::Node> transpose;
};

std::optional<QueryBranch> match_query_branch(const ov::Output<ov::Node>& value) {
    auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(value.get_node_shared_ptr());
    if (!transpose) {
        return std::nullopt;
    }
    auto broadcast = transpose->input_value(0).get_node_shared_ptr();
    if (!ov::is_type_any_of<ov::op::v1::Broadcast, ov::op::v3::Broadcast>(broadcast)) {
        return std::nullopt;
    }
    auto add = ov::as_type_ptr<ov::op::v1::Add>(broadcast->input_value(0).get_node_shared_ptr());
    if (!add) {
        return std::nullopt;
    }
    for (size_t i = 0; i < add->get_input_size(); ++i) {
        auto cumsum = add->input_value(i).get_node_shared_ptr();
        if (ov::is_type<ov::op::v0::CumSum>(cumsum)) {
            return QueryBranch{cumsum, add, broadcast, transpose};
        }
    }
    return std::nullopt;
}

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

class RewriteGQAAttentionMaskImpl : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::RewriteGQAAttentionMaskImpl");

    explicit RewriteGQAAttentionMaskImpl(const std::shared_ptr<ov::Node>& attention_mask_bool) {
        auto less = opp::wrap_type<ov::op::v1::Less>({opp::any_input(), opp::any_input()});
        auto select = opp::wrap_type<ov::op::v1::Select>(
            {less, opp::any_input(), opp::any_input()},
            [](const ov::Output<ov::Node>& output) {
                return output.get_node()->get_friendly_name().find(kGQANameMarker) != std::string::npos;
            });
        auto add_mask = opp::wrap_type<ov::op::v1::Add>({select, opp::any_input()});
        // GroupQueryAttentionDecomposition inserts a Maximum clamp for f16/bf16 logits.
        auto clamp = opp::optional<ov::op::v1::Maximum>({add_mask, opp::any_input()});
        auto softmax = opp::wrap_type<ov::op::v1::Softmax, ov::op::v8::Softmax>({clamp});

        auto callback = [=](opp::Matcher& m) {
            const auto& node_to_output = m.get_pattern_value_map();
            auto matched_less = node_to_output.at(less).get_node_shared_ptr();
            auto matched_select = node_to_output.at(select).get_node_shared_ptr();

            // Less can take the key CumSum and the query branch in either order; keep the original order.
            std::optional<QueryBranch> query;
            size_t query_idx = 0;
            for (size_t i = 0; i < 2 && !query; ++i) {
                const auto other = matched_less->input_value(1 - i).get_node_shared_ptr();
                if (!ov::is_type<ov::op::v0::CumSum>(other)) {
                    continue;
                }
                query = match_query_branch(matched_less->input_value(i));
                query_idx = i;
            }
            if (!query) {
                return false;
            }

            const auto& mask_shape = matched_less->get_output_partial_shape(0);
            if (mask_shape.rank().is_dynamic() || mask_shape.rank().get_length() < 2) {
                LOG_WARN("RewriteGQAAttentionMask: unexpected mask rank for " << matched_less->get_friendly_name()
                                                                              << ", skipping");
                return false;
            }
            const auto& query_length_dim = mask_shape[mask_shape.rank().get_length() - 2];
            OPENVINO_ASSERT(query_length_dim.is_static() && query_length_dim.get_length() > 0,
                            "RewriteGQAAttentionMask: query length must be static and positive");
            const bool is_kvcache_model = query_length_dim.get_length() == 1;
            ov::NodeVector new_nodes{attention_mask_bool};
            std::shared_ptr<ov::Node> attn_mask = attention_mask_bool;
            const auto rank = mask_shape.rank().get_length();
            if (rank > 2) {
                std::vector<int64_t> axes(rank - 2);
                std::iota(axes.begin(), axes.end(), 1);
                auto axes_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
                attn_mask = std::make_shared<ov::op::v0::Unsqueeze>(attention_mask_bool, axes_const);
                new_nodes.insert(new_nodes.end(), {axes_const, attn_mask});
            }
            if (is_kvcache_model) {
                ov::copy_runtime_info({matched_less, query->add, query->broadcast, query->transpose}, new_nodes);
                matched_select->input(0).replace_source_output(attn_mask);
                return true;
            }

            // Clone Broadcast/Transpose instead of editing them in place: they may be shared by other masks.
            auto bcast_inputs = query->broadcast->input_values();
            bcast_inputs[0] = query->cumsum;
            auto new_broadcast = query->broadcast->clone_with_new_inputs(bcast_inputs);
            auto new_transpose =
                query->transpose->clone_with_new_inputs({new_broadcast, query->transpose->input_value(1)});

            ov::OutputVector less_inputs = matched_less->input_values();
            less_inputs[query_idx] = new_transpose;
            auto less_equal = std::make_shared<ov::op::v1::LessEqual>(less_inputs[0], less_inputs[1]);

            auto logical_and = std::make_shared<ov::op::v1::LogicalAnd>(less_equal, attn_mask);

            new_broadcast->set_friendly_name(query->broadcast->get_friendly_name() + "/npuw_gqa_mask");
            new_transpose->set_friendly_name(query->transpose->get_friendly_name() + "/npuw_gqa_mask");
            less_equal->set_friendly_name(matched_less->get_friendly_name() + "/npuw_less_equal");
            logical_and->set_friendly_name(matched_less->get_friendly_name() + "/npuw_and_attention_mask");
            new_nodes.insert(new_nodes.end(), {new_broadcast, new_transpose, less_equal, logical_and});
            ov::copy_runtime_info({matched_less, query->add, query->broadcast, query->transpose}, new_nodes);

            matched_select->input(0).replace_source_output(logical_and);
            LOG_DEBUG("RewriteGQAAttentionMask: rewrote mask for " << matched_select->get_friendly_name());
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(softmax, "RewriteGQAAttentionMaskImpl"), std::move(callback));
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // namespace

namespace ov::npuw {

bool RewriteGQAAttentionMask::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::Node> attention_mask = nullptr;
    for (const auto& input : model->inputs()) {
        if (input.get_names().count(LLMInferBaseRequest::layer_names::attention_mask) != 0) {
            attention_mask = input.get_node_shared_ptr();
            break;
        }
    }
    if (!attention_mask) {
        return false;
    }

    auto attention_mask_bool = std::make_shared<ov::op::v0::Convert>(attention_mask, ov::element::boolean);
    attention_mask_bool->set_friendly_name("attention_mask/npuw_gqa_bool");

    ov::pass::Manager manager("rewrite-gqa-attention-mask");
    manager.set_per_pass_validation(true);
    manager.register_pass<RewriteGQAAttentionMaskImpl>(attention_mask_bool);
    return manager.run_passes(model);
}

}  // namespace ov::npuw
