// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "replace_deepstack_scatter_with_add.hpp"

#include <string>
#include <unordered_set>
#include <vector>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"

namespace opp = ov::pass::pattern;

namespace {

bool is_named(const std::shared_ptr<ov::Node>& node, const std::string& needle) {
    if (!node) {
        return false;
    }
    if (node->get_friendly_name().find(needle) != std::string::npos) {
        return true;
    }
    for (const auto& name : node->output(0).get_names()) {
        if (name.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// Returns the unique consumer of `node` whose op type name matches `type_name`,
// or nullptr if there is none or more than one.
std::shared_ptr<ov::Node> single_consumer_of_type(const std::shared_ptr<ov::Node>& node,
                                                   const std::string& type_name) {
    std::shared_ptr<ov::Node> found;
    for (const auto& target : node->output(0).get_target_inputs()) {
        auto* consumer = target.get_node();
        if (std::string(consumer->get_type_name()) == type_name) {
            if (found) {
                return nullptr;  // ambiguous - bail out
            }
            found = consumer->shared_from_this();
        }
    }
    return found;
}

// True if `target` is backward-reachable from any Result/Sink of the model (i.e. still alive).
bool is_reachable_from_outputs(const std::shared_ptr<ov::Model>& model, const ov::Node* target) {
    std::unordered_set<const ov::Node*> visited;
    std::vector<ov::Node*> stack;
    for (const auto& result : model->get_results()) {
        stack.push_back(result.get());
    }
    for (const auto& sink : model->get_sinks()) {
        stack.push_back(sink.get());
    }
    while (!stack.empty()) {
        auto* node = stack.back();
        stack.pop_back();
        if (node == target) {
            return true;
        }
        if (!visited.insert(node).second) {
            continue;
        }
        for (const auto& input : node->inputs()) {
            stack.push_back(input.get_source_output().get_node());
        }
    }
    return false;
}

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

// Matches a single Qwen3-VL DeepStack injection
//     ScatterNDUpdate(hidden, pos, Add(GatherND(hidden, pos), Gather(deepstack, L)))
// and replaces it (plus the optional SliceAssign identity wrapper that follows) with a
// plain residual add: hidden + Gather(deepstack, L).
class DeepstackScatterToAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::DeepstackScatterToAdd");
    explicit DeepstackScatterToAdd(bool& matched) {
        auto deepstack = opp::wrap_type<ov::op::v0::Parameter>();
        auto select = opp::wrap_type<ov::op::v8::Gather, ov::op::v7::Gather, ov::op::v1::Gather>(
            {deepstack, opp::any_input(), opp::any_input()});
        auto gathered = opp::wrap_type<ov::op::v8::GatherND, ov::op::v5::GatherND>({opp::any_input(), opp::any_input()});
        auto add = opp::wrap_type<ov::op::v1::Add>({gathered, select});
        auto scatter = opp::wrap_type<ov::op::v3::ScatterNDUpdate, ov::op::v15::ScatterNDUpdate>(
            {opp::any_input(), opp::any_input(), add});

        auto callback = [=, &matched](opp::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();

            // The Add/GatherND/ScatterNDUpdate shape is generic; only accept it when the gathered
            // parameter is really deepstack_visual_embeds.
            if (!is_named(pattern_map.at(deepstack).get_node_shared_ptr(), "deepstack_visual_embeds")) {
                return false;
            }

            auto select_node = pattern_map.at(select).get_node_shared_ptr();
            auto scatter_node = pattern_map.at(scatter).get_node_shared_ptr();
            auto base = scatter_node->input_value(0);

            // The scatter result is wrapped by a SliceAssign (Reshape -> ScatterNDUpdate -> Reshape)
            // identity that writes the whole tensor back. Bypass it as well when present.
            ov::Output<ov::Node> cluster_out = scatter_node->output(0);
            if (auto reshape_in = single_consumer_of_type(scatter_node, "Reshape")) {
                if (auto slice_scatter = single_consumer_of_type(reshape_in, "ScatterNDUpdate")) {
                    if (auto reshape_out = single_consumer_of_type(slice_scatter, "Reshape")) {
                        cluster_out = reshape_out->output(0);
                    }
                }
            }

            // hidden + Gather(deepstack, L). The host pre-scatters the deepstack values into their
            // visual-token positions (zeros elsewhere), so a plain broadcasted add reproduces the
            // original per-position injection.
            auto residual = std::make_shared<ov::op::v1::Add>(base, select_node->output(0));
            residual->set_friendly_name(select_node->get_friendly_name() + "/deepstack_residual_add");
            ov::copy_runtime_info(cluster_out.get_node_shared_ptr(), residual);

            for (auto& target : cluster_out.get_target_inputs()) {
                target.replace_source_output(residual->output(0));
            }
            register_new_node(residual);
            matched = true;
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(scatter, "ov::npuw::DeepstackScatterToAdd"),
                         std::move(callback));
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // namespace

namespace ov::npuw {

bool ReplaceDeepstackScatterWithAdd::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool matched = false;
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<DeepstackScatterToAdd>(matched);
    rewr.run_on_model(model);

    if (!matched) {
        return false;
    }

    // The visual_pos_masks input now only feeds the dead gather/scatter cluster; drop it.
    for (const auto& param : model->get_parameters()) {
        if (is_named(param, "visual_pos_masks") && !is_reachable_from_outputs(model, param.get())) {
            model->remove_parameter(param);
            break;
        }
    }

    ov::pass::Validate().run_on_model(model);
    return true;
}

}  // namespace ov::npuw
