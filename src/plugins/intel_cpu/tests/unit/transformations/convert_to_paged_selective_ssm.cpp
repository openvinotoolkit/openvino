// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/convert_to_paged_selective_ssm.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "nodes/paged_selective_ssm.h"
#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/validate.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/paged_attention/convert_pagedattn_inputs.hpp"

namespace ov::intel_cpu::test {
namespace {

struct PagedSSMModel {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::op::v0::Parameter> state;
};

PagedSSMModel make_paged_ssm_model() {
    const auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    const auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    const auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 8});
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4, 3});
    const auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 8});
    const auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4, 3, 8});
    const auto subsequences = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto blocks = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto block_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto processed = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    const auto intervals = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    const auto ssm = std::make_shared<ov::op::internal::PagedSelectiveSSM>(A,
                                                                           dt,
                                                                           B,
                                                                           x,
                                                                           C,
                                                                           state,
                                                                           subsequences,
                                                                           blocks,
                                                                           block_begins,
                                                                           processed,
                                                                           intervals);
    return {std::make_shared<ov::Model>(
                ov::OutputVector{ssm},
                ov::ParameterVector{A, dt, B, x, C, state, subsequences, blocks, block_begins, processed, intervals}),
            state};
}

TEST(ConvertToPagedSelectiveSSMTest, KeepsComputationAndStatePrecisionsIndependent) {
    auto [model, state] = make_paged_ssm_model();

    ov::pass::ConvertPagedAttnInputs::KVCacheConfig cache_config;
    cache_config.inferencePrecision = ov::element::f16;
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::ConvertPagedAttnInputs>(cache_config, nullptr, nullptr);
    manager.register_pass<ConvertToPagedSelectiveSSM>();
    manager.register_pass<ov::pass::Validate>();
    EXPECT_TRUE(manager.run_passes(model));

    const auto ssm = model->get_results().front()->input_value(0).get_node_shared_ptr();
    EXPECT_NE(std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(ssm), nullptr);
    EXPECT_TRUE(ov::is_type<ov::op::internal::PagedSelectiveSSM>(ssm));
    EXPECT_EQ(state->get_element_type(), ov::element::f16);
    EXPECT_EQ(ssm->get_input_element_type(input_port_index(PagedSelectiveSSMInputPort::State)), ov::element::f16);
    EXPECT_EQ(ssm->get_output_element_type(0), ov::element::f32);
}

TEST(ConvertToPagedSelectiveSSMTest, LeavesAlignedPrecisionUnchanged) {
    auto [model, state] = make_paged_ssm_model();
    const auto original_ssm = model->get_results().front()->input_value(0).get_node_shared_ptr();

    ov::pass::Manager manager;
    manager.register_pass<ConvertToPagedSelectiveSSM>();
    EXPECT_FALSE(manager.run_passes(model));

    EXPECT_EQ(model->get_results().front()->input_value(0).get_node_shared_ptr(), original_ssm);
    EXPECT_EQ(state->get_element_type(), ov::element::f32);
}

}  // namespace
}  // namespace ov::intel_cpu::test
