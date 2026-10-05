// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_selective_ssm_fusion.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

namespace {

using namespace ov;
using ov::pass::paged_attention::PaParams;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v6 = ov::op::v6;
namespace v8 = ov::op::v8;
namespace internal = ov::op::internal;

// SelectiveSSM dimensions: batch B, seq_len L, num_heads H, num_groups G, head_dim P, state_size N.
constexpr int64_t B = 2;
constexpr int64_t L = 3;
constexpr int64_t H = 4;
constexpr int64_t G = 2;
constexpr int64_t P = 8;
constexpr int64_t N = 6;

std::shared_ptr<v0::Parameter> make_param(const std::string& name,
                                          ov::element::Type et,
                                          const ov::PartialShape& shape) {
    auto p = std::make_shared<v0::Parameter>(et, shape);
    p->set_friendly_name(name);
    p->get_output_tensor(0).set_names({name});
    return p;
}

struct SSMInputs {
    std::shared_ptr<v0::Parameter> a;
    std::shared_ptr<v0::Parameter> dt;
    std::shared_ptr<v0::Parameter> b;
    std::shared_ptr<v0::Parameter> x;
    std::shared_ptr<v0::Parameter> c;
    std::shared_ptr<v0::Parameter> recurrent_state;
};

SSMInputs make_ssm_inputs() {
    SSMInputs in;
    in.a = make_param("A", element::f32, Shape{static_cast<size_t>(H)});
    in.dt = make_param("dt", element::f32, Shape{B, L, H});
    in.b = make_param("B", element::f32, Shape{B, L, G, N});
    in.x = make_param("x", element::f32, Shape{B, L, H, P});
    in.c = make_param("C", element::f32, Shape{B, L, G, N});
    in.recurrent_state = make_param("past_ssm_state", element::f32, Shape{B, H, P, N});
    in.recurrent_state->get_output_tensor(0).set_names({"cache_params.past.ssm_state.0"});
    return in;
}

struct GatheredState {
    std::shared_ptr<v0::Parameter> beam_idx;
    std::shared_ptr<ov::op::util::Variable> variable;
    std::shared_ptr<v6::ReadValue> read_value;
    std::shared_ptr<v8::Gather> gathered_state;
};

GatheredState make_gathered_state(const SSMInputs& in) {
    GatheredState s;
    s.beam_idx = make_param("beam_idx", element::i32, PartialShape{-1});
    s.variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{PartialShape{B, H, P, N}, element::f32, "ssm_var_0"});
    s.read_value = std::make_shared<v6::ReadValue>(in.recurrent_state->output(0), s.variable);
    const auto gather_axis = v0::Constant::create(element::i64, Shape{}, {0});
    s.gathered_state = std::make_shared<v8::Gather>(s.read_value, s.beam_idx, gather_axis);
    return s;
}

// Mirrors flatten_batch_length from paged_selective_ssm_fusion.cpp: [B, L, ...tail] -> [B*L, ...tail]
ov::Output<ov::Node> ref_flatten_batch_length(const ov::Output<ov::Node>& input) {
    const auto rank = input.get_partial_shape().size();
    std::vector<int64_t> tail_dim_indices;
    tail_dim_indices.reserve(rank - 2);
    for (size_t i = 2; i < rank; ++i) {
        tail_dim_indices.push_back(static_cast<int64_t>(i));
    }
    const auto shape_of = std::make_shared<v3::ShapeOf>(input, element::i64);
    const auto tail_idx = v0::Constant::create(element::i64, Shape{tail_dim_indices.size()}, tail_dim_indices);
    const auto axis_0 = v0::Constant::create(element::i64, Shape{}, {0});
    const auto tail_dims = std::make_shared<v8::Gather>(shape_of, tail_idx, axis_0);
    const auto flat_dim = v0::Constant::create(element::i64, Shape{1}, {-1});
    const auto flat_shape = std::make_shared<v0::Concat>(OutputVector{flat_dim, tail_dims}, 0);
    return std::make_shared<v1::Reshape>(input, flat_shape, false);
}

ov::Output<ov::Node> build_paged_ssm_block(const SSMInputs& in,
                                           const std::shared_ptr<v0::Parameter>& state_table,
                                           const std::shared_ptr<v0::Parameter>& subseq_begins,
                                           const std::shared_ptr<v0::Parameter>& block_indices,
                                           const std::shared_ptr<v0::Parameter>& block_indices_begins,
                                           const std::shared_ptr<v0::Parameter>& past_lens,
                                           const std::shared_ptr<v0::Parameter>& cache_interval,
                                           const std::string& ssm_friendly_name) {
    const auto dt_flat = ref_flatten_batch_length(in.dt);
    const auto b_flat = ref_flatten_batch_length(in.b);
    const auto x_flat = ref_flatten_batch_length(in.x);
    const auto c_flat = ref_flatten_batch_length(in.c);

    auto paged_ssm = std::make_shared<internal::PagedSelectiveSSM>(in.a,
                                                                   dt_flat,
                                                                   b_flat,
                                                                   x_flat,
                                                                   c_flat,
                                                                   state_table->output(0),
                                                                   subseq_begins->output(0),
                                                                   block_indices->output(0),
                                                                   block_indices_begins->output(0),
                                                                   past_lens->output(0),
                                                                   cache_interval->output(0));
    paged_ssm->set_friendly_name(ssm_friendly_name + "/PagedSelectiveSSM");

    const auto x_shape = std::make_shared<v3::ShapeOf>(in.x, element::i64);
    auto paged_ssm_out = std::make_shared<v1::Reshape>(paged_ssm, x_shape, false);
    paged_ssm_out->set_friendly_name(ssm_friendly_name);
    return paged_ssm_out->output(0);
}

struct PagedParams {
    std::shared_ptr<v0::Parameter> subseq_begins;
    std::shared_ptr<v0::Parameter> block_indices;
    std::shared_ptr<v0::Parameter> block_indices_begins;
    std::shared_ptr<v0::Parameter> past_lens;
    std::shared_ptr<v0::Parameter> cache_interval;
};

PagedParams make_paged_params() {
    PagedParams p;
    p.subseq_begins = make_param("subsequence_begins", element::i32, PartialShape{-1});
    p.block_indices = make_param("la.block_indices", element::i32, PartialShape{-1});
    p.block_indices_begins = make_param("la.block_indices_begins", element::i32, PartialShape{-1});
    p.past_lens = make_param("la.past_lens", element::i32, PartialShape{-1});
    p.cache_interval = make_param("la.cache_interval", element::i32, PartialShape{-1});
    return p;
}

std::shared_ptr<ov::Model> build_model_stateful() {
    auto in = make_ssm_inputs();
    auto st = make_gathered_state(in);

    auto ssm = std::make_shared<internal::SelectiveSSM>(in.a, in.dt, in.b, in.x, in.c, st.gathered_state);

    auto out = std::make_shared<v0::Result>(ssm->output(0));
    auto assign = std::make_shared<v6::Assign>(ssm->output(1), st.variable);

    ParameterVector params{in.a, in.dt, in.b, in.x, in.c, in.recurrent_state, st.beam_idx};
    return std::make_shared<ov::Model>(ResultVector{out}, SinkVector{assign}, params);
}

std::shared_ptr<ov::Model> build_reference_model() {
    auto in = make_ssm_inputs();
    auto pp = make_paged_params();
    // SelectiveSSM recurrent_state [B,H,P,N] -> state table [num_blocks,H,P,N].
    auto state_table =
        make_param("selective_ssm_state_table.0", element::dynamic, PartialShape{Dimension::dynamic(), H, P, N});

    const auto paged_ssm_out = build_paged_ssm_block(in,
                                                     state_table,
                                                     pp.subseq_begins,
                                                     pp.block_indices,
                                                     pp.block_indices_begins,
                                                     pp.past_lens,
                                                     pp.cache_interval,
                                                     "SelectiveSSM");

    auto out = std::make_shared<v0::Result>(paged_ssm_out);

    // The Assign is reconnected to Gather(ReadValue) as a dead branch.
    auto st = make_gathered_state(in);
    auto assign = std::make_shared<v6::Assign>(st.gathered_state->output(0), st.variable);

    ParameterVector params{in.a,
                           in.dt,
                           in.b,
                           in.x,
                           in.c,
                           in.recurrent_state,
                           st.beam_idx,
                           pp.subseq_begins,
                           pp.block_indices,
                           pp.block_indices_begins,
                           pp.past_lens,
                           pp.cache_interval,
                           state_table};
    return std::make_shared<ov::Model>(ResultVector{out}, SinkVector{assign}, params);
}

}  // namespace

class PagedSelectiveSSMFusionTest : public ::TransformationTestsF {};

std::unordered_set<std::string> run_paged_selective_ssm_fusion(const std::shared_ptr<ov::Model>& model) {
    PaParams pa_params{model->get_parameters()};
    std::unordered_set<std::string> var_ids_to_remove;

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::PagedSelectiveSSMFusion>(pa_params, var_ids_to_remove);
    manager.run_passes(model);
    model->add_parameters(pa_params.items());
    model->validate_nodes_and_infer_types();
    return var_ids_to_remove;
}

TEST_F(PagedSelectiveSSMFusionTest, FusePagedSSM) {
    model = build_model_stateful();
    model_ref = build_reference_model();

    const auto var_ids_to_remove = run_paged_selective_ssm_fusion(model);
    EXPECT_EQ(var_ids_to_remove.count("ssm_var_0"), 1u);

    const auto& parameters = model->get_parameters();
    const auto state_table =
        std::find_if(parameters.begin(), parameters.end(), [](const std::shared_ptr<v0::Parameter>& parameter) {
            return parameter->get_friendly_name() == "selective_ssm_state_table.0";
        });
    ASSERT_NE(state_table, parameters.end());
    EXPECT_TRUE(ov::is_keep_const_precision(*state_table));
    EXPECT_EQ((*state_table)->get_output_element_type(0), ov::element::dynamic);
}

TEST(PagedSelectiveSSMFusionCount, ReportsConvertedCount) {
    auto model = build_model_stateful();
    PaParams pa_params{model->get_parameters()};
    std::unordered_set<std::string> var_ids_to_remove;

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    auto pass = manager.register_pass<ov::pass::PagedSelectiveSSMFusion>(pa_params, var_ids_to_remove);
    manager.run_passes(model);

    EXPECT_EQ(pass->get_fused_count(), 1u);
}
