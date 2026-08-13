// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_selective_ssm_fusion.hpp"

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v3 = ov::op::v3;
namespace internal = ov::op::internal;

std::shared_ptr<v0::Parameter> make_f32_param(const std::string& name, const PartialShape& shape) {
    auto p = std::make_shared<v0::Parameter>(element::f32, shape);
    p->set_friendly_name(name);
    p->get_output_tensor(0).set_names({name});
    return p;
}

std::shared_ptr<ov::Model> build_fusable_model(bool dynamic_shapes = false) {
    const PartialShape dt_shape = dynamic_shapes ? PartialShape{-1, -1, 4} : PartialShape{2, 3, 4};
    const PartialShape projection_shape =
        dynamic_shapes ? PartialShape{-1, -1, 2, 8} : PartialShape{2, 3, 2, 8};
    const PartialShape x_shape = dynamic_shapes ? PartialShape{-1, -1, 4, 6} : PartialShape{2, 3, 4, 6};
    const PartialShape state_shape = dynamic_shapes ? PartialShape{-1, 4, 6, 8} : PartialShape{2, 4, 6, 8};

    auto A = make_f32_param("A", PartialShape{4});
    auto dt = make_f32_param("dt", dt_shape);
    auto B = make_f32_param("B", projection_shape);
    auto x = make_f32_param("x", x_shape);
    auto C = make_f32_param("C", projection_shape);
    auto recurrent_state = make_f32_param("past_recurrent_state", state_shape);
    recurrent_state->get_output_tensor(0).set_names({"cache_params.past.recurrent_state.0"});
    auto read_value = std::make_shared<v3::ReadValue>(recurrent_state->output(0), "cache_param_0");
    auto ssm = std::make_shared<internal::SelectiveSSM>(A, dt, B, x, C, read_value);
    auto out = std::make_shared<v0::Result>(ssm->output(0));
    auto present_state = std::make_shared<v0::Result>(ssm->output(1));
    present_state->get_output_tensor(0).set_names({"cache_params.present.recurrent_state.0"});
    return std::make_shared<ov::Model>(ResultVector{out, present_state},
                                       ParameterVector{A, dt, B, x, C, recurrent_state});
}

std::shared_ptr<internal::SelectiveSSM> find_selective_ssm(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ops()) {
        if (const auto ssm = ov::as_type_ptr<internal::SelectiveSSM>(node)) {
            return ssm;
        }
    }
    return nullptr;
}

std::shared_ptr<internal::PagedSelectiveSSM> find_paged_selective_ssm(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ops()) {
        if (const auto paged_ssm = ov::as_type_ptr<internal::PagedSelectiveSSM>(node)) {
            return paged_ssm;
        }
    }
    return nullptr;
}

void run_paged_fusion(const std::shared_ptr<ov::Model>& model, std::unordered_set<std::string>& ids) {
    ov::pass::paged_attention::PaParams pa_params(model->get_parameters());
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::PagedSelectiveSSMFusion>(pa_params, ids);
    manager.run_passes(model);
    model->add_parameters(pa_params.items());
    model->validate_nodes_and_infer_types();
}

}  // namespace

TEST(TransformationTests, PagedSelectiveSSMFusion_Positive) {
    auto model = build_fusable_model();
    std::unordered_set<std::string> ids;
    run_paged_fusion(model, ids);

    const auto paged_ssm = find_paged_selective_ssm(model);
    ASSERT_NE(paged_ssm, nullptr);
    EXPECT_EQ(find_selective_ssm(model), nullptr);
    EXPECT_EQ(ids.count("cache_param_0"), 1u);

    EXPECT_EQ(paged_ssm->get_input_partial_shape(0), PartialShape({4}));
    EXPECT_EQ(paged_ssm->get_input_partial_shape(1), PartialShape({6, 4}));
    EXPECT_EQ(paged_ssm->get_input_partial_shape(2), PartialShape({6, 2, 8}));
    EXPECT_EQ(paged_ssm->get_input_partial_shape(3), PartialShape({6, 4, 6}));
    EXPECT_EQ(paged_ssm->get_input_partial_shape(4), PartialShape({6, 2, 8}));
    EXPECT_EQ(paged_ssm->get_output_partial_shape(0), PartialShape({6, 4, 6}));

    const auto state_table = ov::as_type_ptr<v0::Parameter>(paged_ssm->get_input_node_shared_ptr(5));
    ASSERT_NE(state_table, nullptr);
    EXPECT_EQ(state_table->get_friendly_name(), "selective_ssm_state_table.0");
    EXPECT_EQ(state_table->get_element_type(), element::dynamic);
    EXPECT_EQ(state_table->get_partial_shape(), PartialShape({Dimension::dynamic(), 4, 6, 8}));

    const std::vector<std::string> expected_paged_inputs = {"subsequence_begins",
                                                            "la.block_indices",
                                                            "la.block_indices_begins",
                                                            "la.past_lens",
                                                            "la.cache_interval"};
    for (size_t i = 0; i < expected_paged_inputs.size(); ++i) {
        const auto parameter = ov::as_type_ptr<v0::Parameter>(paged_ssm->get_input_node_shared_ptr(i + 6));
        ASSERT_NE(parameter, nullptr);
        EXPECT_EQ(parameter->get_friendly_name(), expected_paged_inputs[i]);
        EXPECT_EQ(parameter->get_element_type(), element::i32);
        EXPECT_EQ(parameter->get_partial_shape(), PartialShape({Dimension::dynamic()}));
    }

    ASSERT_EQ(model->get_results().size(), 2u);
    EXPECT_TRUE(ov::is_type<v3::ReadValue>(model->get_results()[1]->get_input_node_shared_ptr(0)));
    EXPECT_EQ(model->get_results()[0]->get_input_partial_shape(0), PartialShape({2, 3, 4, 6}));
}

TEST(TransformationTests, PagedSelectiveSSMFusion_DynamicBatchAndSequence) {
    auto model = build_fusable_model(true);
    std::unordered_set<std::string> ids;
    run_paged_fusion(model, ids);

    const auto paged_ssm = find_paged_selective_ssm(model);
    ASSERT_NE(paged_ssm, nullptr);
    EXPECT_EQ(find_selective_ssm(model), nullptr);
    EXPECT_EQ(ids.count("cache_param_0"), 1u);

    EXPECT_EQ(paged_ssm->get_input_partial_shape(1), PartialShape({-1, 4}));
    EXPECT_EQ(paged_ssm->get_input_partial_shape(2), PartialShape({-1, 2, 8}));
    EXPECT_EQ(paged_ssm->get_input_partial_shape(3), PartialShape({-1, 4, 6}));
    EXPECT_EQ(paged_ssm->get_input_partial_shape(4), PartialShape({-1, 2, 8}));
    EXPECT_EQ(paged_ssm->get_output_partial_shape(0), PartialShape({-1, 4, 6}));

    const auto state_table = ov::as_type_ptr<v0::Parameter>(paged_ssm->get_input_node_shared_ptr(5));
    ASSERT_NE(state_table, nullptr);
    EXPECT_EQ(state_table->get_partial_shape(), PartialShape({-1, 4, 6, 8}));
    EXPECT_EQ(model->get_results()[0]->get_input_partial_shape(0), PartialShape({-1, -1, 4, 6}));
}

TEST(TransformationTests, PagedSelectiveSSMFusion_GatheredState) {
    auto model = build_fusable_model();
    const auto ssm = find_selective_ssm(model);
    ASSERT_NE(ssm, nullptr);
    const auto read_value = ssm->get_input_node_shared_ptr(5);
    auto beam_idx = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    beam_idx->set_friendly_name("beam_idx");
    const auto axis = v0::Constant::create(element::i64, Shape{}, {0});
    const auto gathered_state = std::make_shared<ov::op::v8::Gather>(read_value, beam_idx, axis);
    ssm->input(5).replace_source_output(gathered_state);
    model->add_parameters({beam_idx});
    model->validate_nodes_and_infer_types();

    std::unordered_set<std::string> ids;
    run_paged_fusion(model, ids);

    EXPECT_NE(find_paged_selective_ssm(model), nullptr);
    EXPECT_EQ(ids.count("cache_param_0"), 1u);
    ASSERT_EQ(model->get_results().size(), 2u);
    EXPECT_TRUE(ov::is_type<ov::op::v8::Gather>(model->get_results()[1]->get_input_node_shared_ptr(0)));
}

TEST(TransformationTests, PagedSelectiveSSMFusion_DoesNotFuseWithoutReadValue) {
    auto model = build_fusable_model();
    const auto ssm = find_selective_ssm(model);
    ASSERT_NE(ssm, nullptr);
    ssm->input(5).replace_source_output(model->get_parameters().back());
    model->validate_nodes_and_infer_types();

    std::unordered_set<std::string> ids;
    run_paged_fusion(model, ids);

    EXPECT_EQ(find_paged_selective_ssm(model), nullptr);
    EXPECT_NE(find_selective_ssm(model), nullptr);
    EXPECT_TRUE(ids.empty());
}
