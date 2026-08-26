// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/preserve_standalone_selective_ssm_precision.hpp"

#include <gtest/gtest.h>

#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_gpu::test {
namespace {

std::shared_ptr<ov::op::internal::PagedSelectiveSSM> make_paged_ssm(ov::ParameterVector& parameters, ov::element::Type data_type = ov::element::f32) {
    const auto parameter = [&parameters](ov::element::Type type, const ov::PartialShape& shape) {
        auto value = std::make_shared<ov::op::v0::Parameter>(type, shape);
        parameters.push_back(value);
        return value;
    };

    const auto A = parameter(data_type, {4});
    const auto dt = parameter(data_type, {2, 4});
    const auto B = parameter(data_type, {2, 2, 5});
    const auto x = parameter(data_type, {2, 4, 3});
    const auto C = parameter(data_type, {2, 2, 5});
    const auto state = parameter(data_type, {2, 4, 3, 5});
    const auto subsequences = parameter(ov::element::i64, {2});
    const auto blocks = parameter(ov::element::i64, {2});
    const auto block_begins = parameter(ov::element::i64, {2});
    const auto processed = parameter(ov::element::i64, {1});
    const auto intervals = parameter(ov::element::i64, {1});
    return std::make_shared<ov::op::internal::PagedSelectiveSSM>(A, dt, B, x, C, state, subsequences, blocks, block_begins, processed, intervals);
}

TEST(PreserveStandaloneSelectiveSSMPrecisionTest, MarksDataAndMetadataInputs) {
    ov::ParameterVector parameters;
    const auto paged_ssm = make_paged_ssm(parameters);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{paged_ssm}, parameters);

    PreserveStandaloneSelectiveSSMPrecision pass;
    EXPECT_FALSE(pass.run_on_model(model));
    EXPECT_TRUE(ov::is_conversion_disabled(paged_ssm, ov::element::f32, ov::element::f16));
    for (size_t input_idx = 0; input_idx < 6; ++input_idx) {
        EXPECT_TRUE(ov::is_conversion_disabled(paged_ssm->get_input_node_shared_ptr(input_idx), ov::element::f32, ov::element::f16));
    }
    for (size_t input_idx = 6; input_idx < 11; ++input_idx) {
        EXPECT_TRUE(ov::is_conversion_disabled(paged_ssm->get_input_node_shared_ptr(input_idx), ov::element::i64, ov::element::i32));
    }
}

TEST(PreserveStandaloneSelectiveSSMPrecisionTest, MarksNonPagedSelectiveSSM) {
    const auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    const auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});
    const auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 2, 5});
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4, 3});
    const auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 2, 5});
    const auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 3, 5});
    const ov::ParameterVector parameters{A, dt, B, x, C, state};
    const auto ssm = std::make_shared<ov::op::internal::SelectiveSSM>(A, dt, B, x, C, state);
    const auto model = std::make_shared<ov::Model>(ssm->outputs(), parameters);

    PreserveStandaloneSelectiveSSMPrecision pass;
    EXPECT_FALSE(pass.run_on_model(model));
    EXPECT_TRUE(ov::is_conversion_disabled(ssm, ov::element::f32, ov::element::f16));
    for (const auto& parameter : parameters) {
        EXPECT_TRUE(ov::is_conversion_disabled(parameter, ov::element::f32, ov::element::f16));
    }
}

TEST(PreserveStandaloneSelectiveSSMPrecisionTest, RestoresPagedStateTablePrecision) {
    ov::ParameterVector parameters;
    const auto paged_ssm = make_paged_ssm(parameters);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{paged_ssm}, parameters);
    const auto state = parameters.at(5);
    state->set_element_type(ov::element::f16);
    state->validate_and_infer_types();

    RestoreStandalonePagedSelectiveSSMStatePrecision pass;
    EXPECT_TRUE(pass.run_on_model(model));
    EXPECT_EQ(state->get_element_type(), ov::element::f32);
    EXPECT_NO_THROW(paged_ssm->validate_and_infer_types());
}

TEST(PreserveStandaloneSelectiveSSMPrecisionTest, KeepsNativeHalfStateTablePrecision) {
    ov::ParameterVector parameters;
    const auto paged_ssm = make_paged_ssm(parameters, ov::element::f16);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{paged_ssm}, parameters);
    const auto state = parameters.at(5);

    RestoreStandalonePagedSelectiveSSMStatePrecision pass;
    EXPECT_FALSE(pass.run_on_model(model));
    EXPECT_EQ(state->get_element_type(), ov::element::f16);
    EXPECT_NO_THROW(paged_ssm->validate_and_infer_types());
}

}  // namespace
}  // namespace ov::intel_gpu::test
