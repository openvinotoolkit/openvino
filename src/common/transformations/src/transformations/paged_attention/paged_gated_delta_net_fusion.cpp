// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_gated_delta_net_fusion.hpp"

#include <map>
#include <memory>
#include <string>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/assign_base.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;

namespace {

constexpr const char* MARKED_FOR_PAGED_EXTENSIONS_CLEANUP = "marked_for_paged_extensions_cleanup";
constexpr const char* GATED_DELTA_STATE_TABLE_PREFIX = "gated_delta_state_table.";

struct SharedRuntimeInputs {
    std::shared_ptr<v0::Parameter> subsequence_begins;
    std::shared_ptr<v0::Parameter> block_indices;
    std::shared_ptr<v0::Parameter> block_indices_begins;
    std::shared_ptr<v0::Parameter> past_lens;
    std::shared_ptr<v0::Parameter> cache_interval;
};

struct ParameterCreationResult {
    std::shared_ptr<v0::Parameter> parameter;
    bool created = false;
};

void mark_for_paged_extensions_cleanup(const std::shared_ptr<ov::Node>& node) {
    node->get_rt_info()[MARKED_FOR_PAGED_EXTENSIONS_CLEANUP] = true;
}

ParameterCreationResult create_or_get_named_parameter(const std::shared_ptr<ov::Model>& model,
                                                      const std::string& name,
                                                      const ov::element::Type& type,
                                                      const ov::PartialShape& pshape) {
    for (const auto& input : model->inputs()) {
        if (!input.get_names().count(name)) {
            continue;
        }
        const auto parameter = ov::as_type_ptr<v0::Parameter>(input.get_node_shared_ptr());
        OPENVINO_ASSERT(parameter,
                        "The model is in an inconsistent state. Found input '",
                        name,
                        "', but it is not a v0::Parameter.");
        return {parameter, false};
    }

    const auto parameter = std::make_shared<v0::Parameter>(type, pshape);
    parameter->set_friendly_name(name);
    parameter->get_output_tensor(0).set_names({name});
    model->add_parameters({parameter});
    return {parameter, true};
}

std::string make_gated_delta_state_table_name(const size_t layer_index) {
    return std::string(GATED_DELTA_STATE_TABLE_PREFIX) + std::to_string(layer_index);
}

ov::PartialShape make_gated_delta_state_table_shape(const ov::PartialShape& state_shape) {
    if (state_shape.rank().is_static() && state_shape.rank().get_length() == 4) {
        return ov::PartialShape{ov::Dimension::dynamic(), state_shape[1], state_shape[2], state_shape[3]};
    }
    return ov::PartialShape::dynamic(4);
}

// Walks upstream through single-input shape-only ops (Reshape, Squeeze, Unsqueeze)
// to find the underlying ReadValue or Parameter that provides the GDN recurrent state.
// Returns nullptr if no such source is found within a limited number of hops.
std::shared_ptr<ov::Node> find_upstream_gdn_state_source(const std::shared_ptr<ov::Node>& node) {
    constexpr int max_hops = 5;
    std::shared_ptr<ov::Node> cur = node;
    for (int i = 0; i < max_hops; ++i) {
        if (ov::as_type_ptr<ov::op::util::ReadValueBase>(cur) || ov::as_type_ptr<v0::Parameter>(cur)) {
            return cur;
        }
          // Only traverse single-input, shape-only ops that do not change tensor semantics.
          if ((ov::as_type_ptr<ov::op::v1::Reshape>(cur) || ov::as_type_ptr<ov::op::v0::Unsqueeze>(cur) ||
               ov::as_type_ptr<ov::op::v1::Transpose>(cur) || ov::as_type_ptr<ov::op::v8::Slice>(cur) ||
               ov::as_type_ptr<ov::op::v0::Convert>(cur)) &&
            cur->get_input_size() > 0) {
            cur = cur->get_input_node_shared_ptr(0);
        } else {
            break;
        }
    }
    return nullptr;
}

bool is_allowed_state_passthrough(const std::shared_ptr<ov::Node>& node) {
    return ov::as_type_ptr<ov::op::v1::Reshape>(node) || ov::as_type_ptr<ov::op::v0::Unsqueeze>(node) ||
           ov::as_type_ptr<ov::op::v1::Transpose>(node) || ov::as_type_ptr<ov::op::v8::Slice>(node) ||
           ov::as_type_ptr<ov::op::v0::Convert>(node);
}

ov::Output<ov::Node> flatten_blhd_to_thd(const ov::Output<ov::Node>& input, ov::NodeVector& created_nodes) {
    const auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    const auto idx_hd = v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
    const auto axis_0 = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    const auto dims_hd = std::make_shared<ov::op::v8::Gather>(shape_of, idx_hd, axis_0);
    const auto flat_dim = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    const auto flat_shape = std::make_shared<v0::Concat>(ov::OutputVector{flat_dim, dims_hd}, 0);
    const auto reshaped = std::make_shared<ov::op::v1::Reshape>(input, flat_shape, false);
    created_nodes.insert(created_nodes.end(), {shape_of, idx_hd, axis_0, dims_hd, flat_dim, flat_shape, reshaped});
    return reshaped;
}

ov::Output<ov::Node> flatten_blh_to_th(const ov::Output<ov::Node>& input, ov::NodeVector& created_nodes) {
    const auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    const auto idx_h = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    const auto axis_0 = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    const auto dim_h = std::make_shared<ov::op::v8::Gather>(shape_of, idx_h, axis_0);
    const auto flat_dim = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    const auto flat_shape = std::make_shared<v0::Concat>(ov::OutputVector{flat_dim, dim_h}, 0);
    const auto reshaped = std::make_shared<ov::op::v1::Reshape>(input, flat_shape, false);
    created_nodes.insert(created_nodes.end(), {shape_of, idx_h, axis_0, dim_h, flat_dim, flat_shape, reshaped});
    return reshaped;
}

bool is_state_writeback_branch(const std::shared_ptr<ov::Node>& node,
                               std::set<std::shared_ptr<ov::Node>>& visited,
                               const int depth = 0) {
    if (!visited.insert(node).second) {
        return true;
    }
    if (ov::as_type_ptr<v0::Result>(node) || ov::as_type_ptr<ov::op::util::AssignBase>(node)) {
        return true;
    }
    if (!is_allowed_state_passthrough(node)) {
        return false;
    }
    if (depth > 8) {
        return false;
    }
    bool has_downstream = false;
    for (const auto& output : node->outputs()) {
        for (const auto& target_input : output.get_target_inputs()) {
            has_downstream = true;
            if (!is_state_writeback_branch(target_input.get_node()->shared_from_this(), visited, depth + 1)) {
                return false;
            }
        }
    }
    return has_downstream;
}

void collect_state_writeback_sinks(const std::shared_ptr<ov::Node>& node,
                                   std::set<std::shared_ptr<ov::Node>>& visited,
                                   ov::NodeVector& sinks_to_mark,
                                   const int depth = 0) {
    if (!visited.insert(node).second || depth > 8) {
        return;
    }
    if (ov::as_type_ptr<v0::Result>(node) || ov::as_type_ptr<ov::op::util::AssignBase>(node)) {
        sinks_to_mark.push_back(node);
        return;
    }
    for (const auto& output : node->outputs()) {
        for (const auto& target_input : output.get_target_inputs()) {
            collect_state_writeback_sinks(target_input.get_node()->shared_from_this(), visited, sinks_to_mark, depth + 1);
        }
    }
}

class PagedGatedDeltaNetFusionMatcher : public ov::pass::MatcherPass {
public:
    PagedGatedDeltaNetFusionMatcher(const SharedRuntimeInputs& shared_inputs, const std::shared_ptr<ov::Model>& model)
        : m_shared_inputs(shared_inputs),
          m_model(model) {
        auto query = any_input();
        auto key = any_input();
        auto value = any_input();
        auto recurrent_state = any_input();
        auto gate = any_input();
        auto beta = any_input();

        auto gdn = wrap_type<ov::op::internal::GatedDeltaNet>({query, key, value, recurrent_state, gate, beta});

        ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            if (transformation_callback(m.get_match_root())) {
                return false;
            }

            const auto& pm = m.get_pattern_value_map();
            const auto gdn_node = ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(pm.at(gdn).get_node_shared_ptr());
            if (!gdn_node || gdn_node->get_output_size() != 2) {
                return false;
            }

            const auto state_source = gdn_node->input_value(3).get_node_shared_ptr();
            const auto state_origin = find_upstream_gdn_state_source(state_source);
            const auto state_table_source = state_origin ? state_origin : state_source;

            const auto state_consumers = gdn_node->output(1).get_target_inputs();
            for (const auto& state_consumer : state_consumers) {
                const auto consumer = state_consumer.get_node()->shared_from_this();
                std::set<std::shared_ptr<ov::Node>> visited;
                if (!is_state_writeback_branch(consumer, visited)) {
                    return false;
                }
            }

            if (!m_state_to_state_table.count(state_table_source)) {
                const auto state_table = create_or_get_named_parameter(m_model,
                                                                       make_gated_delta_state_table_name(
                                                                           m_state_to_state_table.size()),
                                                                       state_table_source->get_output_element_type(0),
                                                                       make_gated_delta_state_table_shape(
                                                                           state_table_source->get_output_partial_shape(0)));
                m_state_to_state_table[state_table_source] = state_table.parameter;
            }

            const auto state_table = m_state_to_state_table.at(state_table_source);

            ov::NodeVector reshape_nodes;
            reshape_nodes.reserve(24);
            const auto query_flat = flatten_blhd_to_thd(pm.at(query), reshape_nodes);
            const auto key_flat = flatten_blhd_to_thd(pm.at(key), reshape_nodes);
            const auto value_flat = flatten_blhd_to_thd(pm.at(value), reshape_nodes);
            const auto gate_flat = flatten_blh_to_th(pm.at(gate), reshape_nodes);
            const auto beta_flat = flatten_blh_to_th(pm.at(beta), reshape_nodes);

            // Inputs 0-5 are flattened from matched GatedDeltaNet [B,L,H,*] to PagedGDN [B*L,H,*].
            const auto paged_gdn = std::make_shared<ov::op::internal::PagedGatedDeltaNet>(query_flat,
                                                                                            key_flat,
                                                                                            value_flat,
                                                                                            state_table,
                                                                                            gate_flat,
                                                                                            beta_flat,
                                                                                            m_shared_inputs.subsequence_begins,
                                                                                            m_shared_inputs.block_indices,
                                                                                            m_shared_inputs.block_indices_begins,
                                                                                            m_shared_inputs.past_lens,
                                                                                            m_shared_inputs.cache_interval,
                                                                                            gdn_node->get_fuse_qk_l2norm(),
                                                                                            gdn_node->get_q_l2_norm_eps(),
                                                                                            gdn_node->get_k_l2_norm_eps());

            paged_gdn->set_friendly_name(gdn_node->get_friendly_name() + "/PagedGatedDeltaNet");
            const auto query_shape = std::make_shared<ov::op::v3::ShapeOf>(pm.at(query), ov::element::i64);
            const auto value_shape = std::make_shared<ov::op::v3::ShapeOf>(pm.at(value), ov::element::i64);
            const auto axis_0 = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
            const auto idx_q = v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 2});
            const auto idx_v = v0::Constant::create(ov::element::i64, ov::Shape{1}, {3});
            const auto q_dims = std::make_shared<ov::op::v8::Gather>(query_shape, idx_q, axis_0);
            const auto v_dim = std::make_shared<ov::op::v8::Gather>(value_shape, idx_v, axis_0);
            const auto out0_shape = std::make_shared<v0::Concat>(ov::OutputVector{q_dims, v_dim}, 0);
            const auto paged_gdn_out = std::make_shared<ov::op::v1::Reshape>(paged_gdn, out0_shape, false);
            paged_gdn_out->set_friendly_name(gdn_node->get_friendly_name());
            reshape_nodes.push_back(paged_gdn);
            reshape_nodes.push_back(query_shape);
            reshape_nodes.push_back(value_shape);
            reshape_nodes.push_back(axis_0);
            reshape_nodes.push_back(idx_q);
            reshape_nodes.push_back(idx_v);
            reshape_nodes.push_back(q_dims);
            reshape_nodes.push_back(v_dim);
            reshape_nodes.push_back(out0_shape);
            reshape_nodes.push_back(paged_gdn_out);
            ov::copy_runtime_info(gdn_node, reshape_nodes);

            // Disconnect GDN state output consumers and mark state write-back branches for cleanup.
            for (const auto& state_consumer : state_consumers) {
                const auto consumer = state_consumer.get_node()->shared_from_this();
                // Reconnect consumer to the original state source so it becomes a dead branch.
                state_consumer.replace_source_output(gdn_node->input_value(3));
                std::set<std::shared_ptr<ov::Node>> visited;
                ov::NodeVector sinks_to_mark;
                collect_state_writeback_sinks(consumer, visited, sinks_to_mark);
                for (const auto& sink_to_mark : sinks_to_mark) {
                    mark_for_paged_extensions_cleanup(sink_to_mark);
                }
            }

            if (!ov::replace_output_update_name(gdn_node->output(0), paged_gdn_out->output(0))) {
                gdn_node->output(0).replace(paged_gdn_out->output(0));
            }

            register_new_node(paged_gdn_out);
            register_new_node(paged_gdn);
            return true;
        };

        const auto matcher = std::make_shared<ov::pass::pattern::Matcher>(gdn, "PagedGatedDeltaNetFusion");
        register_matcher(matcher, callback);
    }

private:
    SharedRuntimeInputs m_shared_inputs;
    std::shared_ptr<ov::Model> m_model;
    std::map<std::shared_ptr<ov::Node>, std::shared_ptr<v0::Parameter>> m_state_to_state_table;
};

}  // namespace

namespace ov::pass {

PagedGatedDeltaNetFusion::PagedGatedDeltaNetFusion() {
    set_property(ov::pass::PassProperty::REQUIRE_STATIC_SHAPE, false);
}

bool PagedGatedDeltaNetFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    SharedRuntimeInputs shared_inputs{
        create_or_get_named_parameter(model, "subsequence_begins", ov::element::i32, ov::PartialShape{-1}).parameter,
        create_or_get_named_parameter(model, "paged_gdn.block_indices", ov::element::i32, ov::PartialShape{-1})
            .parameter,
        create_or_get_named_parameter(model,
                                      "paged_gdn.block_indices_begins",
                                      ov::element::i32,
                                      ov::PartialShape{-1})
            .parameter,
        create_or_get_named_parameter(model, "paged_gdn.past_lens", ov::element::i32, ov::PartialShape{-1})
            .parameter,
        create_or_get_named_parameter(model,
                                      "paged_gdn.cache_interval",
                                      ov::element::i32,
                                      ov::PartialShape{-1})
            .parameter};

    ov::pass::Manager manager(get_pass_config(), "PagedGatedDeltaNetFusion");
    manager.set_per_pass_validation(false);
    manager.register_pass<PagedGatedDeltaNetFusionMatcher>(shared_inputs, model);
    return manager.run_passes(model);
}

}  // namespace ov::pass
