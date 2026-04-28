// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_gated_delta_net_fusion.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <unordered_set>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/runtime/core.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v8 = ov::op::v8;
namespace internal = ov::op::internal;

std::shared_ptr<v0::Parameter> make_f32_param(const std::string& name, const Shape& shape) {
    auto p = std::make_shared<v0::Parameter>(element::f32, shape);
    p->set_friendly_name(name);
    p->get_output_tensor(0).set_names({name});
    return p;
}

std::shared_ptr<ov::Model> build_fusable_model() {
    auto query = make_f32_param("query", Shape{2, 3, 4, 8});
    auto key = make_f32_param("key", Shape{2, 3, 4, 8});
    auto value = make_f32_param("value", Shape{2, 3, 4, 6});

    auto recurrent_state = make_f32_param("past_recurrent_state", Shape{2, 4, 8, 6});
    recurrent_state->get_output_tensor(0).set_names({"cache_params.past.recurrent_state.0"});
    auto read_value = std::make_shared<ov::op::v3::ReadValue>(recurrent_state->output(0), "cache_param_0");

    auto gate = make_f32_param("gate", Shape{2, 3, 4});
    auto beta = make_f32_param("beta", Shape{2, 3, 4});

    auto gdn = std::make_shared<internal::GatedDeltaNet>(query, key, value, read_value, gate, beta);

    auto out = std::make_shared<v0::Result>(gdn->output(0));
    auto present_state = std::make_shared<v0::Result>(gdn->output(1));
    present_state->get_output_tensor(0).set_names({"cache_params.present.recurrent_state.0"});

    ParameterVector params{query, key, value, recurrent_state, gate, beta};

    return std::make_shared<ov::Model>(ResultVector{out, present_state}, params);
}

std::shared_ptr<ov::Model> build_non_fusable_model() {
    auto query = make_f32_param("query", Shape{2, 3, 4, 8});
    auto key = make_f32_param("key", Shape{2, 3, 4, 8});
    auto value = make_f32_param("value", Shape{2, 3, 4, 6});

    auto recurrent_state = make_f32_param("past_recurrent_state", Shape{2, 4, 8, 6});
    recurrent_state->get_output_tensor(0).set_names({"cache_params.past.recurrent_state.0"});
    auto read_value = std::make_shared<ov::op::v3::ReadValue>(recurrent_state->output(0), "cache_param_0");

    auto gate = make_f32_param("gate", Shape{2, 3, 4});
    auto beta = make_f32_param("beta", Shape{2, 3, 4});

    auto gdn = std::make_shared<internal::GatedDeltaNet>(query, key, value, read_value, gate, beta);
    auto add_rhs = make_f32_param("state_add_rhs", Shape{2, 4, 8, 6});
    auto state_add = std::make_shared<v1::Add>(gdn->output(1), add_rhs);

    auto out = std::make_shared<v0::Result>(gdn->output(0));
    auto present_state = std::make_shared<v0::Result>(state_add);
    present_state->get_output_tensor(0).set_names({"cache_params.present.recurrent_state.0"});

    ParameterVector params{query, key, value, recurrent_state, gate, beta, add_rhs};
    return std::make_shared<ov::Model>(ResultVector{out, present_state}, params);
}

std::shared_ptr<ov::Model> build_fusable_model_with_gathered_state() {
    auto query = make_f32_param("query", Shape{2, 3, 4, 8});
    auto key = make_f32_param("key", Shape{2, 3, 4, 8});
    auto value = make_f32_param("value", Shape{2, 3, 4, 6});

    auto recurrent_state = make_f32_param("past_recurrent_state", Shape{2, 4, 8, 6});
    recurrent_state->get_output_tensor(0).set_names({"cache_params.past.recurrent_state.0"});
    auto read_value = std::make_shared<ov::op::v3::ReadValue>(recurrent_state->output(0), "cache_param_0");
    auto beam_idx = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    beam_idx->set_friendly_name("beam_idx");
    beam_idx->get_output_tensor(0).set_names({"beam_idx"});
    auto gather_axis = v0::Constant::create(element::i64, Shape{}, {0});
    auto gathered_state = std::make_shared<ov::op::v8::Gather>(read_value, beam_idx, gather_axis);

    auto gate = make_f32_param("gate", Shape{2, 3, 4});
    auto beta = make_f32_param("beta", Shape{2, 3, 4});

    auto gdn = std::make_shared<internal::GatedDeltaNet>(query, key, value, gathered_state, gate, beta);

    auto out = std::make_shared<v0::Result>(gdn->output(0));
    auto present_state = std::make_shared<v0::Result>(gdn->output(1));
    present_state->get_output_tensor(0).set_names({"cache_params.present.recurrent_state.0"});

    ParameterVector params{query, key, value, recurrent_state, beam_idx, gate, beta};
    return std::make_shared<ov::Model>(ResultVector{out, present_state}, params);
}

std::shared_ptr<v0::Parameter> make_pa_param(const std::string& name,
                                             ov::element::Type et,
                                             const ov::PartialShape& shape) {
    auto p = std::make_shared<v0::Parameter>(et, shape);
    p->set_friendly_name(name);
    p->get_output_tensor(0).set_names({name});
    return p;
}

// Mirrors flatten_blhd_to_thd from paged_gated_delta_net_fusion.cpp.
ov::Output<ov::Node> ref_flatten_blhd_to_thd(const ov::Output<ov::Node>& input) {
    const auto shape_of = std::make_shared<v3::ShapeOf>(input, element::i64);
    const auto idx_hd = v0::Constant::create(element::i64, Shape{2}, {2, 3});
    const auto axis_0 = v0::Constant::create(element::i64, Shape{}, {0});
    const auto dims_hd = std::make_shared<v8::Gather>(shape_of, idx_hd, axis_0);
    const auto flat_dim = v0::Constant::create(element::i64, Shape{1}, {-1});
    const auto flat_shape = std::make_shared<v0::Concat>(OutputVector{flat_dim, dims_hd}, 0);
    return std::make_shared<v1::Reshape>(input, flat_shape, false);
}

// Mirrors flatten_blh_to_th from paged_gated_delta_net_fusion.cpp.
ov::Output<ov::Node> ref_flatten_blh_to_th(const ov::Output<ov::Node>& input) {
    const auto shape_of = std::make_shared<v3::ShapeOf>(input, element::i64);
    const auto idx_h = v0::Constant::create(element::i64, Shape{1}, {2});
    const auto axis_0 = v0::Constant::create(element::i64, Shape{}, {0});
    const auto dim_h = std::make_shared<v8::Gather>(shape_of, idx_h, axis_0);
    const auto flat_dim = v0::Constant::create(element::i64, Shape{1}, {-1});
    const auto flat_shape = std::make_shared<v0::Concat>(OutputVector{flat_dim, dim_h}, 0);
    return std::make_shared<v1::Reshape>(input, flat_shape, false);
}

// Builds the PagedGDN block + output reshape that replaces GDN in the fused graph.
// Returns {paged_gdn_out, paged_gdn} where paged_gdn_out is the final reshaped output.
ov::Output<ov::Node> build_paged_gdn_block(const std::shared_ptr<v0::Parameter>& query,
                                           const std::shared_ptr<v0::Parameter>& key,
                                           const std::shared_ptr<v0::Parameter>& value,
                                           const std::shared_ptr<v0::Parameter>& gate,
                                           const std::shared_ptr<v0::Parameter>& beta,
                                           const std::shared_ptr<v0::Parameter>& state_table,
                                           const std::shared_ptr<v0::Parameter>& subseq_begins,
                                           const std::shared_ptr<v0::Parameter>& block_indices,
                                           const std::shared_ptr<v0::Parameter>& block_indices_begins,
                                           const std::shared_ptr<v0::Parameter>& past_lens,
                                           const std::shared_ptr<v0::Parameter>& cache_interval,
                                           const std::string& gdn_friendly_name) {
    const auto query_flat = ref_flatten_blhd_to_thd(query);
    const auto key_flat = ref_flatten_blhd_to_thd(key);
    const auto value_flat = ref_flatten_blhd_to_thd(value);
    const auto gate_flat = ref_flatten_blh_to_th(gate);
    const auto beta_flat = ref_flatten_blh_to_th(beta);

    auto paged_gdn = std::make_shared<internal::PagedGatedDeltaNet>(query_flat,
                                                                    key_flat,
                                                                    value_flat,
                                                                    state_table->output(0),
                                                                    gate_flat,
                                                                    beta_flat,
                                                                    subseq_begins->output(0),
                                                                    block_indices->output(0),
                                                                    block_indices_begins->output(0),
                                                                    past_lens->output(0),
                                                                    cache_interval->output(0));
    paged_gdn->set_friendly_name(gdn_friendly_name + "/PagedGatedDeltaNet");

    const auto q_shape = std::make_shared<v3::ShapeOf>(query, element::i64);
    const auto v_shape = std::make_shared<v3::ShapeOf>(value, element::i64);
    const auto axis_0 = v0::Constant::create(element::i64, Shape{}, {0});
    const auto idx_q = v0::Constant::create(element::i64, Shape{3}, {0, 1, 2});
    const auto idx_v = v0::Constant::create(element::i64, Shape{1}, {3});
    const auto q_dims = std::make_shared<v8::Gather>(q_shape, idx_q, axis_0);
    const auto v_dim = std::make_shared<v8::Gather>(v_shape, idx_v, axis_0);
    const auto out_shape = std::make_shared<v0::Concat>(OutputVector{q_dims, v_dim}, 0);
    auto paged_gdn_out = std::make_shared<v1::Reshape>(paged_gdn, out_shape, false);
    paged_gdn_out->set_friendly_name(gdn_friendly_name);
    return paged_gdn_out->output(0);
}

// Reference graph for build_fusable_model() after PagedGatedDeltaNetFusion.
// GDN is replaced by PagedGDN; state Result reconnected to ReadValue.
std::shared_ptr<ov::Model> build_reference_fused_model() {
    auto query = make_f32_param("query", Shape{2, 3, 4, 8});
    auto key = make_f32_param("key", Shape{2, 3, 4, 8});
    auto value = make_f32_param("value", Shape{2, 3, 4, 6});
    auto recurrent_state = make_f32_param("past_recurrent_state", Shape{2, 4, 8, 6});
    recurrent_state->get_output_tensor(0).set_names({"cache_params.past.recurrent_state.0"});
    auto gate = make_f32_param("gate", Shape{2, 3, 4});
    auto beta = make_f32_param("beta", Shape{2, 3, 4});

    // ReadValue remains as the reconnected source for the state Result (dead branch).
    const auto read_value = std::make_shared<v3::ReadValue>(recurrent_state->output(0), "cache_param_0");

    // New PA params added by the pass (in creation order).
    // State input shape [2,4,8,6] = [B,H,D_k,D_v] → table shape [?,H,D_v,D_k] = [?,4,6,8].
    auto subseq_begins = make_pa_param("subsequence_begins", element::i32, PartialShape{-1});
    auto block_indices = make_pa_param("la.block_indices", element::i32, PartialShape{-1});
    auto block_indices_begins = make_pa_param("la.block_indices_begins", element::i32, PartialShape{-1});
    auto past_lens = make_pa_param("la.past_lens", element::i32, PartialShape{-1});
    auto cache_interval = make_pa_param("la.cache_interval", element::i32, PartialShape{-1});
    auto state_table =
        make_pa_param("gated_delta_state_table.0", element::f32, PartialShape{Dimension::dynamic(), 4, 6, 8});

    const auto paged_gdn_out = build_paged_gdn_block(query,
                                                     key,
                                                     value,
                                                     gate,
                                                     beta,
                                                     state_table,
                                                     subseq_begins,
                                                     block_indices,
                                                     block_indices_begins,
                                                     past_lens,
                                                     cache_interval,
                                                     "GatedDeltaNet");

    auto out = std::make_shared<v0::Result>(paged_gdn_out);
    auto present_state = std::make_shared<v0::Result>(read_value->output(0));
    present_state->get_output_tensor(0).set_names({"cache_params.present.recurrent_state.0"});

    ParameterVector params{query,
                           key,
                           value,
                           recurrent_state,
                           gate,
                           beta,
                           subseq_begins,
                           block_indices,
                           block_indices_begins,
                           past_lens,
                           cache_interval,
                           state_table};
    return std::make_shared<ov::Model>(ResultVector{out, present_state}, params);
}

// Reference graph for build_non_fusable_model() after PagedGatedDeltaNetFusion.
// GDN is replaced by PagedGDN; the Add consumer of the state output is reconnected to ReadValue.
std::shared_ptr<ov::Model> build_reference_fused_non_fusable_model() {
    auto query = make_f32_param("query", Shape{2, 3, 4, 8});
    auto key = make_f32_param("key", Shape{2, 3, 4, 8});
    auto value = make_f32_param("value", Shape{2, 3, 4, 6});
    auto recurrent_state = make_f32_param("past_recurrent_state", Shape{2, 4, 8, 6});
    recurrent_state->get_output_tensor(0).set_names({"cache_params.past.recurrent_state.0"});
    auto gate = make_f32_param("gate", Shape{2, 3, 4});
    auto beta = make_f32_param("beta", Shape{2, 3, 4});
    auto add_rhs = make_f32_param("state_add_rhs", Shape{2, 4, 8, 6});

    const auto read_value = std::make_shared<v3::ReadValue>(recurrent_state->output(0), "cache_param_0");

    auto subseq_begins = make_pa_param("subsequence_begins", element::i32, PartialShape{-1});
    auto block_indices = make_pa_param("la.block_indices", element::i32, PartialShape{-1});
    auto block_indices_begins = make_pa_param("la.block_indices_begins", element::i32, PartialShape{-1});
    auto past_lens = make_pa_param("la.past_lens", element::i32, PartialShape{-1});
    auto cache_interval = make_pa_param("la.cache_interval", element::i32, PartialShape{-1});
    auto state_table =
        make_pa_param("gated_delta_state_table.0", element::f32, PartialShape{Dimension::dynamic(), 4, 6, 8});

    const auto paged_gdn_out = build_paged_gdn_block(query,
                                                     key,
                                                     value,
                                                     gate,
                                                     beta,
                                                     state_table,
                                                     subseq_begins,
                                                     block_indices,
                                                     block_indices_begins,
                                                     past_lens,
                                                     cache_interval,
                                                     "GatedDeltaNet");

    // The Add consumer of GDN state output is reconnected to ReadValue.
    const auto state_add = std::make_shared<v1::Add>(read_value->output(0), add_rhs);
    auto out = std::make_shared<v0::Result>(paged_gdn_out);
    auto present_state = std::make_shared<v0::Result>(state_add);
    present_state->get_output_tensor(0).set_names({"cache_params.present.recurrent_state.0"});

    ParameterVector params{query,
                           key,
                           value,
                           recurrent_state,
                           gate,
                           beta,
                           add_rhs,
                           subseq_begins,
                           block_indices,
                           block_indices_begins,
                           past_lens,
                           cache_interval,
                           state_table};
    return std::make_shared<ov::Model>(ResultVector{out, present_state}, params);
}

// Reference graph for build_fusable_model_with_gathered_state() after PagedGatedDeltaNetFusion.
// GDN is replaced by PagedGDN; state Result reconnected to Gather(ReadValue, beam_idx, axis).
std::shared_ptr<ov::Model> build_reference_fused_model_with_gathered_state() {
    auto query = make_f32_param("query", Shape{2, 3, 4, 8});
    auto key = make_f32_param("key", Shape{2, 3, 4, 8});
    auto value = make_f32_param("value", Shape{2, 3, 4, 6});
    auto recurrent_state = make_f32_param("past_recurrent_state", Shape{2, 4, 8, 6});
    recurrent_state->get_output_tensor(0).set_names({"cache_params.past.recurrent_state.0"});
    auto beam_idx = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    beam_idx->set_friendly_name("beam_idx");
    beam_idx->get_output_tensor(0).set_names({"beam_idx"});
    auto gate = make_f32_param("gate", Shape{2, 3, 4});
    auto beta = make_f32_param("beta", Shape{2, 3, 4});

    const auto read_value = std::make_shared<v3::ReadValue>(recurrent_state->output(0), "cache_param_0");
    const auto gather_axis = v0::Constant::create(element::i64, Shape{}, {0});
    const auto gathered_state = std::make_shared<v8::Gather>(read_value, beam_idx, gather_axis);

    auto subseq_begins = make_pa_param("subsequence_begins", element::i32, PartialShape{-1});
    auto block_indices = make_pa_param("la.block_indices", element::i32, PartialShape{-1});
    auto block_indices_begins = make_pa_param("la.block_indices_begins", element::i32, PartialShape{-1});
    auto past_lens = make_pa_param("la.past_lens", element::i32, PartialShape{-1});
    auto cache_interval = make_pa_param("la.cache_interval", element::i32, PartialShape{-1});
    // The pattern matches on read_value (not gathered_state), so state shape comes from ReadValue output: [2,4,8,6].
    auto state_table =
        make_pa_param("gated_delta_state_table.0", element::f32, PartialShape{Dimension::dynamic(), 4, 6, 8});

    const auto paged_gdn_out = build_paged_gdn_block(query,
                                                     key,
                                                     value,
                                                     gate,
                                                     beta,
                                                     state_table,
                                                     subseq_begins,
                                                     block_indices,
                                                     block_indices_begins,
                                                     past_lens,
                                                     cache_interval,
                                                     "GatedDeltaNet");

    auto out = std::make_shared<v0::Result>(paged_gdn_out);
    // State Result reconnected to gathered_state (gdn_node->input_value(3)).
    auto present_state = std::make_shared<v0::Result>(gathered_state->output(0));
    present_state->get_output_tensor(0).set_names({"cache_params.present.recurrent_state.0"});

    ParameterVector params{query,
                           key,
                           value,
                           recurrent_state,
                           beam_idx,
                           gate,
                           beta,
                           subseq_begins,
                           block_indices,
                           block_indices_begins,
                           past_lens,
                           cache_interval,
                           state_table};
    return std::make_shared<ov::Model>(ResultVector{out, present_state}, params);
}

}  // namespace

class PagedGatedDeltaNetFusionTest : public ::TransformationTestsF {};

void run_paged_gated_delta_net_fusion(const std::shared_ptr<ov::Model>& model) {
    ov::pass::paged_attention::PaParams pa_params{model->get_parameters()};
    std::unordered_set<std::string> var_ids_to_remove;

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::PagedGatedDeltaNetFusion>(pa_params, var_ids_to_remove);
    manager.run_passes(model);
    model->add_parameters(pa_params.items());
    model->validate_nodes_and_infer_types();
}

TEST_F(PagedGatedDeltaNetFusionTest, FusesWhenStateOutputIsOnlyResultConsumer) {
    model = build_fusable_model();
    model_ref = build_reference_fused_model();
    disable_rt_info_check();
    comparator.disable(FunctionsComparator::PRECISIONS);
    comparator.disable(FunctionsComparator::TENSOR_NAMES);

    run_paged_gated_delta_net_fusion(model);
}

TEST_F(PagedGatedDeltaNetFusionTest, FusesWhenStateOutputHasNonResultConsumer) {
    model = build_non_fusable_model();
    model_ref = build_reference_fused_non_fusable_model();

    run_paged_gated_delta_net_fusion(model);
}

TEST_F(PagedGatedDeltaNetFusionTest, FusesWhenStateInputIsGatherFromReadValue) {
    model = build_fusable_model_with_gathered_state();
    model_ref = build_reference_fused_model_with_gathered_state();
    run_paged_gated_delta_net_fusion(model);
}
