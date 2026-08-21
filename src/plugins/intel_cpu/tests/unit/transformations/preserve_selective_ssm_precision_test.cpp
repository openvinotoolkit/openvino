// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/transformation_pipeline.h"

namespace ov::intel_cpu::test {
namespace {

void expect_conversion_disabled(const std::shared_ptr<ov::Node>& node) {
    EXPECT_TRUE(ov::is_conversion_disabled(node, ov::element::dynamic, ov::element::dynamic));
}

TEST(PreserveSelectiveSSMPrecisionTest, MarksSelectiveSSMAndSamePrecisionInputs) {
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

TEST(PreserveSelectiveSSMPrecisionTest, PreservesPagedSelectiveSSMI64MetadataThroughConvertPrecision) {
    const auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    const auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    const auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 3});
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4, 5});
    const auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 3});
    const auto state_table = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4, 5, 3});
    const auto subsequence_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});
    const auto block_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});
    const auto block_indices_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});
    const auto num_processed_tokens = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    const auto cache_interval = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    const ov::ParameterVector data_parameters{A, dt, B, x, C, state_table};
    const ov::ParameterVector metadata_parameters{
        subsequence_begins,
        block_indices,
        block_indices_begins,
        num_processed_tokens,
        cache_interval,
    };
    ov::ParameterVector parameters = data_parameters;
    parameters.insert(parameters.end(), metadata_parameters.begin(), metadata_parameters.end());
    const auto ssm = std::make_shared<ov::op::internal::PagedSelectiveSSM>(A,
                                                                           dt,
                                                                           B,
                                                                           x,
                                                                           C,
                                                                           state_table,
                                                                           subsequence_begins,
                                                                           block_indices,
                                                                           block_indices_begins,
                                                                           num_processed_tokens,
                                                                           cache_interval);
    const auto model = std::make_shared<ov::Model>(ssm->outputs(), parameters);

    ov::pass::Manager manager;
    manager.register_pass<PreserveSelectiveSSMPrecision>();
    manager.register_pass<ov::pass::ConvertPrecision>(ov::element::i64,
                                                      ov::element::i32,
                                                      type_to_fuse_map{},
                                                      false,
                                                      false);
    manager.run_passes(model);

    expect_conversion_disabled(ssm);
    for (const auto& parameter : data_parameters) {
        expect_conversion_disabled(parameter);
    }
    for (const auto& parameter : metadata_parameters) {
        expect_conversion_disabled(parameter);
        EXPECT_EQ(parameter->get_element_type(), ov::element::i64);
    }
    for (size_t input = 6; input < ssm->get_input_size(); ++input) {
        EXPECT_EQ(ssm->get_input_element_type(input), ov::element::i64);
    }
}

}  // namespace
}  // namespace ov::intel_cpu::test
