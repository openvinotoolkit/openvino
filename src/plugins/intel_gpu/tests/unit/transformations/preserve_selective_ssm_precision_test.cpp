// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/preserve_selective_ssm_precision.hpp"

#include <gtest/gtest.h>

#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_gpu::test {
namespace {

void expect_conversion_disabled(const std::shared_ptr<ov::Node>& node) {
    EXPECT_TRUE(ov::is_conversion_disabled(node, ov::element::dynamic, ov::element::dynamic));
}

TEST(PreserveSelectiveSSMPrecisionTest, MarksSelectiveSSMAndDataInputs) {
    const auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    const auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});
    const auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 2, 3});
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4, 5});
    const auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 2, 3});
    const auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 5, 3});
    const ov::ParameterVector parameters{A, dt, B, x, C, state};
    const auto ssm = std::make_shared<ov::op::internal::SelectiveSSM>(A, dt, B, x, C, state);
    const auto model = std::make_shared<ov::Model>(ssm->outputs(), parameters);

    PreserveSelectiveSSMPrecision pass;
    EXPECT_FALSE(pass.run_on_model(model));

    expect_conversion_disabled(ssm);
    for (const auto& parameter : parameters) {
        expect_conversion_disabled(parameter);
    }
}

TEST(PreserveSelectiveSSMPrecisionTest, MarksPagedOperationAndAllInputs) {
    const auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    const auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    const auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 3});
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4, 5});
    const auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 3});
    const auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4, 5, 3});
    const auto subsequences = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto blocks = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto block_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto processed = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    const auto intervals = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    const ov::ParameterVector data_parameters{A, dt, B, x, C, state};
    const ov::ParameterVector metadata_parameters{subsequences, blocks, block_begins, processed, intervals};
    ov::ParameterVector parameters = data_parameters;
    parameters.insert(parameters.end(), metadata_parameters.begin(), metadata_parameters.end());
    const auto ssm = std::make_shared<ov::op::internal::PagedSelectiveSSM>(A, dt, B, x, C, state, subsequences, blocks, block_begins, processed, intervals);
    const auto model = std::make_shared<ov::Model>(ssm->outputs(), parameters);

    PreserveSelectiveSSMPrecision pass;
    EXPECT_FALSE(pass.run_on_model(model));

    expect_conversion_disabled(ssm);
    for (const auto& parameter : data_parameters) {
        expect_conversion_disabled(parameter);
    }
    for (const auto& parameter : metadata_parameters) {
        expect_conversion_disabled(parameter);
    }
}

}  // namespace
}  // namespace ov::intel_gpu::test
