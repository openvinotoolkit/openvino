// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_gated_delta_net_fusion.hpp"

#include <memory>
#include <string>
#include <unordered_set>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/fuse_gated_delta_net.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;

namespace {

constexpr const char* GATED_DELTA_STATE_TABLE_PREFIX = "gated_delta_state_table.";

std::string make_gated_delta_state_table_name(const size_t layer_index) {
    return std::string(GATED_DELTA_STATE_TABLE_PREFIX) + std::to_string(layer_index);
}

ov::PartialShape make_gated_delta_state_table_shape(const ov::PartialShape& state_shape) {
    // GDN input state shape is [B, H, D_k, D_v].
    // PagedGatedDeltaNet state table expects [?, H, D_v, D_k] — swap last two dims.
    if (state_shape.rank().is_static() && state_shape.rank().get_length() == 4) {
        return ov::PartialShape{ov::Dimension::dynamic(), state_shape[1], state_shape[3], state_shape[2]};
    }
    return ov::PartialShape::dynamic(4);
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

class PagedGatedDeltaNetFusionMatcher : public ov::pass::MatcherPass {
public:
    PagedGatedDeltaNetFusionMatcher(ov::pass::paged_attention::PaParams& pa_params,
                                    std::unordered_set<std::string>& var_ids_to_remove)
        : m_pa_params(pa_params),
          m_var_ids_to_remove(var_ids_to_remove) {
        auto query = any_input();
        auto key = any_input();
        auto value = any_input();
        auto gate = any_input();
        auto beta = any_input();

        // recurrent_state must come from ReadValue(cache_param).
        auto cache_param = any_input();
        auto read_value = wrap_type<ov::op::util::ReadValueBase>({cache_param});
        auto gdn = wrap_type<ov::op::internal::GatedDeltaNet>({query, key, value, read_value, gate, beta});

        ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            if (transformation_callback(m.get_match_root())) {
                return false;
            }

            const auto& pm = m.get_pattern_value_map();
            const auto gdn_node = ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(pm.at(gdn).get_node_shared_ptr());
            if (!gdn_node || gdn_node->get_output_size() != 2) {
                return false;
            }

            const auto state_consumers = gdn_node->output(1).get_target_inputs();

            const auto& state_out = pm.at(read_value);

            const auto state_table_param =
                m_pa_params.add(make_gated_delta_state_table_name(m_layer_index++),
                                state_out.get_element_type(),
                                make_gated_delta_state_table_shape(state_out.get_partial_shape()));
            enable_keep_const_precision(state_table_param);

            const auto rv = ov::as_type_ptr<ov::op::util::ReadValueBase>(pm.at(read_value).get_node_shared_ptr());
            OPENVINO_ASSERT(rv, "Matched cache node is expected to be ReadValue");
            m_var_ids_to_remove.insert(rv->get_variable_id());

            ov::NodeVector reshape_nodes;
            reshape_nodes.reserve(24);
            const auto query_flat = flatten_blhd_to_thd(pm.at(query), reshape_nodes);
            const auto key_flat = flatten_blhd_to_thd(pm.at(key), reshape_nodes);
            const auto value_flat = flatten_blhd_to_thd(pm.at(value), reshape_nodes);
            const auto gate_flat = flatten_blh_to_th(pm.at(gate), reshape_nodes);
            const auto beta_flat = flatten_blh_to_th(pm.at(beta), reshape_nodes);

            // Inputs 0-5 are flattened from matched GatedDeltaNet [B,L,H,*] to PagedGDN [B*L,H,*].
            const auto paged_gdn =
                std::make_shared<ov::op::internal::PagedGatedDeltaNet>(query_flat,
                                                                       key_flat,
                                                                       value_flat,
                                                                       state_table_param->output(0),
                                                                       gate_flat,
                                                                       beta_flat,
                                                                       m_pa_params["subsequence_begins"],
                                                                       m_pa_params["la.block_indices"],
                                                                       m_pa_params["la.block_indices_begins"],
                                                                       m_pa_params["la.past_lens"],
                                                                       m_pa_params["la.cache_interval"],
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

            // Disconnect GDN state output consumers; cleanup is driven by ReadValue variable ids.
            for (const auto& state_consumer : state_consumers) {
                // Reconnect consumer to the original state source so it becomes a dead branch.
                state_consumer.replace_source_output(gdn_node->input_value(3));
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
    ov::pass::paged_attention::PaParams& m_pa_params;
    std::unordered_set<std::string>& m_var_ids_to_remove;
    size_t m_layer_index;
};

}  // namespace

namespace ov::pass {

PagedGatedDeltaNetFusion::PagedGatedDeltaNetFusion(ov::pass::paged_attention::PaParams& pa_params,
                                                   const ov::pass::paged_attention::Options& options,
                                                   std::unordered_set<std::string>& var_ids_to_remove)
    : m_params(pa_params),
      m_options(options),
      m_var_ids_to_remove(var_ids_to_remove) {
    set_property(ov::pass::PassProperty::REQUIRE_STATIC_SHAPE, false);
}

bool PagedGatedDeltaNetFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(PagedGatedDeltaNetFusion);

    m_params.add("subsequence_begins", ov::element::i32, ov::PartialShape{-1});
    m_params.add("la.block_indices", ov::element::i32, ov::PartialShape{-1});
    m_params.add("la.block_indices_begins", ov::element::i32, ov::PartialShape{-1});
    m_params.add("la.past_lens", ov::element::i32, ov::PartialShape{-1});
    m_params.add("la.cache_interval", ov::element::i32, ov::PartialShape{-1});

    bool is_model_changed = false;

    {
        ov::pass::Manager manager(get_pass_config(), "GatedDeltaNetFusion");
        manager.set_per_pass_validation(false);
        manager.register_pass<ov::pass::GatedDeltaNetFusion>();
        manager.run_passes(model);
    }

    {
        ov::pass::Manager manager(get_pass_config(), "PagedGatedDeltaNetFusion");
        manager.set_per_pass_validation(false);
        manager.register_pass<PagedGatedDeltaNetFusionMatcher>(m_params, m_var_ids_to_remove);
        is_model_changed = manager.run_passes(model);
    }

    if (is_model_changed) {
        model->add_parameters(m_params.items());
    }

    return is_model_changed;
}

}  // namespace ov::pass
