// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_selective_ssm_fusion.hpp"

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>

#include "common_test_utils/ov_test_utils.hpp"
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

std::shared_ptr<v0::Parameter> make_f32_param(const std::string& name, const Shape& shape) {
    auto p = std::make_shared<v0::Parameter>(element::f32, shape);
    p->set_friendly_name(name);
    p->get_output_tensor(0).set_names({name});
    return p;
}

std::shared_ptr<ov::Model> build_fusable_model() {
    auto A = make_f32_param("A", Shape{4});
    auto dt = make_f32_param("dt", Shape{2, 3, 4});
    auto B = make_f32_param("B", Shape{2, 3, 2, 8});
    auto x = make_f32_param("x", Shape{2, 3, 4, 6});
    auto C = make_f32_param("C", Shape{2, 3, 2, 8});
    auto recurrent_state = make_f32_param("past_recurrent_state", Shape{2, 4, 6, 8});
    recurrent_state->get_output_tensor(0).set_names({"cache_params.past.recurrent_state.0"});
    auto read_value = std::make_shared<v3::ReadValue>(recurrent_state->output(0), "cache_param_0");
    auto ssm = std::make_shared<internal::SelectiveSSM>(A, dt, B, x, C, read_value);
    auto out = std::make_shared<v0::Result>(ssm->output(0));
    auto present_state = std::make_shared<v0::Result>(ssm->output(1));
    present_state->get_output_tensor(0).set_names({"cache_params.present.recurrent_state.0"});
    return std::make_shared<ov::Model>(ResultVector{out, present_state}, ParameterVector{A, dt, B, x, C, recurrent_state});
}

}  // namespace

TEST(TransformationTests, PagedSelectiveSSMFusion_Positive) {
    auto model = build_fusable_model();
    ov::pass::paged_attention::PaParams pa_params(model->get_parameters());
    std::unordered_set<std::string> ids;
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::PagedSelectiveSSMFusion>(pa_params, ids);
    manager.run_passes(model);
    model->add_parameters(pa_params.items());
    model->validate_nodes_and_infer_types();

    size_t paged_count = 0;
    for (const auto& node : model->get_ops()) {
        if (node->get_type_name() == std::string("PagedSelectiveSSM")) {
            ++paged_count;
        }
    }
    EXPECT_EQ(paged_count, 1u);
    EXPECT_FALSE(ids.empty());
}
