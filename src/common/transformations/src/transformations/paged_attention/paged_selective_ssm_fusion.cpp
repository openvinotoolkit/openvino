// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_selective_ssm_fusion.hpp"

#include <memory>
#include <string>
#include <unordered_set>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::optional;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;

namespace {

constexpr const char* SELECTIVE_SSM_STATE_TABLE_PREFIX = "selective_ssm_state_table.";

std::string make_selective_ssm_state_table_name(const size_t layer_index) {
    return std::string(SELECTIVE_SSM_STATE_TABLE_PREFIX) + std::to_string(layer_index);
}

ov::PartialShape make_selective_ssm_state_table_shape(const ov::PartialShape& state_shape) {
    // SelectiveSSM recurrent_state shape is [B, H, D, S].
    // PagedSelectiveSSM state table keeps the same head/head_dim/state_size layout but
    // replaces the batch dimension with the (dynamic) number of physical blocks.
    if (state_shape.rank().is_static() && state_shape.rank().get_length() == 4) {
        return ov::PartialShape{ov::Dimension::dynamic(), state_shape[1], state_shape[2], state_shape[3]};
    }
    return ov::PartialShape::dynamic(4);
}

// Flattens [B, L, ...tail] to [B*L, ...tail] via a runtime shape subgraph, keeping every dim past batch and length.
// The [-1, ...tail] target shape is assembled from ShapeOf so dynamic trailing dims are handled; it constant-folds
// to a literal shape when the trailing dims are static.
ov::Output<ov::Node> flatten_batch_length(const ov::Output<ov::Node>& input) {
    const auto rank = input.get_partial_shape().size();
    std::vector<int64_t> tail_dim_indices;
    tail_dim_indices.reserve(rank - 2);
    for (size_t i = 2; i < rank; ++i) {
        tail_dim_indices.push_back(static_cast<int64_t>(i));
    }

    const auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    const auto tail_idx = v0::Constant::create(ov::element::i64, ov::Shape{tail_dim_indices.size()}, tail_dim_indices);
    const auto axis_0 = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    const auto tail_dims = std::make_shared<ov::op::v8::Gather>(shape_of, tail_idx, axis_0);
    const auto flat_dim = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    const auto flat_shape = std::make_shared<v0::Concat>(ov::OutputVector{flat_dim, tail_dims}, 0);
    const auto reshaped = std::make_shared<ov::op::v1::Reshape>(input, flat_shape, false);

    ov::copy_runtime_info(input.get_node_shared_ptr(), {shape_of, tail_dims, flat_shape, reshaped});
    return reshaped;
}
}  // namespace

namespace ov::pass {

PagedSelectiveSSMFusion::PagedSelectiveSSMFusion(ov::pass::paged_attention::PaParams& pa_params,
                                                 std::unordered_set<std::string>& var_ids_to_remove) {
    // SelectiveSSM inputs: A, dt, B, x, C, recurrent_state (state at index 5).
    auto a = any_input(ov::pass::pattern::rank_equals(1));
    auto dt = any_input(ov::pass::pattern::rank_equals(3));
    auto b = any_input(ov::pass::pattern::rank_equals(4));
    auto x = any_input(ov::pass::pattern::rank_equals(4));
    auto c = any_input(ov::pass::pattern::rank_equals(4));

    auto cache_param = any_input();
    auto read_value = wrap_type<ov::op::util::ReadValueBase>({cache_param});
    auto gathered_state = optional<ov::op::util::GatherBase>({read_value, any_input(), any_input()});
    auto ssm = wrap_type<ov::op::internal::SelectiveSSM>({a, dt, b, x, c, gathered_state});

    ov::matcher_pass_callback callback =
        [OV_CAPTURE_CPY_AND_THIS, &pa_params, &var_ids_to_remove](ov::pass::pattern::Matcher& m) {
            if (transformation_callback(m.get_match_root())) {
                return false;
            }

            const auto& pm = m.get_pattern_value_map();
            const auto ssm_node = ov::as_type_ptr<ov::op::internal::SelectiveSSM>(pm.at(ssm).get_node_shared_ptr());
            if (!ssm_node || ssm_node->get_output_size() != 2) {
                return false;
            }

            pa_params.add("subsequence_begins", ov::element::i32, ov::PartialShape{-1});
            pa_params.add("la.block_indices", ov::element::i32, ov::PartialShape{-1});
            pa_params.add("la.block_indices_begins", ov::element::i32, ov::PartialShape{-1});
            pa_params.add("la.past_lens", ov::element::i32, ov::PartialShape{-1});
            pa_params.add("la.cache_interval", ov::element::i32, ov::PartialShape{-1});

            const auto state_consumers = ssm_node->output(1).get_target_inputs();
            const auto& state_out = pm.at(read_value);

            const auto state_table_param =
                pa_params.add(make_selective_ssm_state_table_name(m_layer_index++),
                              ov::element::dynamic,
                              make_selective_ssm_state_table_shape(state_out.get_partial_shape()));
            enable_keep_const_precision(state_table_param);

            const auto rv = ov::as_type_ptr<ov::op::util::ReadValueBase>(pm.at(read_value).get_node_shared_ptr());
            OPENVINO_ASSERT(rv, "Matched cache node is expected to be ReadValue");
            var_ids_to_remove.insert(rv->get_variable_id());

            // Flatten [B, L, ...] inputs to [B*L, ...]. A carries no batch/length dims and is passed through.
            const auto dt_flat = flatten_batch_length(pm.at(dt));
            const auto b_flat = flatten_batch_length(pm.at(b));
            const auto x_flat = flatten_batch_length(pm.at(x));
            const auto c_flat = flatten_batch_length(pm.at(c));

            const auto paged_ssm =
                std::make_shared<ov::op::internal::PagedSelectiveSSM>(pm.at(a),
                                                                      dt_flat,
                                                                      b_flat,
                                                                      x_flat,
                                                                      c_flat,
                                                                      state_table_param->output(0),
                                                                      pa_params["subsequence_begins"],
                                                                      pa_params["la.block_indices"],
                                                                      pa_params["la.block_indices_begins"],
                                                                      pa_params["la.past_lens"],
                                                                      pa_params["la.cache_interval"]);

            paged_ssm->set_friendly_name(ssm_node->get_friendly_name() + "/PagedSelectiveSSM");

            // PagedSelectiveSSM output is [B*L, H, D]; reshape back to the matched SelectiveSSM output [B, L, H, D].
            const auto x_shape = std::make_shared<ov::op::v3::ShapeOf>(pm.at(x), ov::element::i64);
            const auto paged_ssm_out = std::make_shared<ov::op::v1::Reshape>(paged_ssm, x_shape, false);
            paged_ssm_out->set_friendly_name(ssm_node->get_friendly_name());

            ov::copy_runtime_info(ssm_node, {paged_ssm, x_shape, paged_ssm_out});

            // Reconnect consumer to the original state source so it becomes a dead branch.
            for (const auto& state_consumer : state_consumers) {
                state_consumer.replace_source_output(ssm_node->input_value(5));
            }

            if (!ov::replace_output_update_name(ssm_node->output(0), paged_ssm_out->output(0))) {
                ssm_node->output(0).replace(paged_ssm_out->output(0));
            }

            register_new_node(paged_ssm_out);
            register_new_node(paged_ssm);
            return true;
        };

    const auto matcher = std::make_shared<ov::pass::pattern::Matcher>(ssm, "PagedSelectiveSSMFusion");
    register_matcher(matcher, callback);
}

}  // namespace ov::pass
