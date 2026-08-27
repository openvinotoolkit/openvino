// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/preserve_paged_selective_ssm_metadata_width.hpp"

#include <gtest/gtest.h>

#include <array>

#include "openvino/op/convert.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"

namespace ov::intel_gpu::test {
namespace {

std::shared_ptr<ov::op::internal::PagedSelectiveSSM> make_paged_ssm(const std::array<ov::Output<ov::Node>, 5>& metadata_inputs,
                                                                    ov::ParameterVector& parameters) {
    const auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2});
    const auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    const auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 1, 8});
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 1});
    const auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 1, 8});
    const auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 1, 8});
    parameters.insert(parameters.end(), {A, dt, B, x, C, state});
    return std::make_shared<ov::op::internal::PagedSelectiveSSM>(A,
                                                                 dt,
                                                                 B,
                                                                 x,
                                                                 C,
                                                                 state,
                                                                 metadata_inputs[0],
                                                                 metadata_inputs[1],
                                                                 metadata_inputs[2],
                                                                 metadata_inputs[3],
                                                                 metadata_inputs[4]);
}

TEST(PreservePagedSelectiveSSMMetadataWidthTest, BypassesOnlyPagedMetadataConversions) {
    ov::ParameterVector parameters;
    std::array<std::shared_ptr<ov::op::v0::Parameter>, 5> metadata_parameters;
    std::array<std::shared_ptr<ov::op::v0::Convert>, 5> metadata_converts;
    std::array<ov::Output<ov::Node>, 5> metadata_inputs;
    const std::array<ov::Shape, 5> metadata_shapes{{{2}, {2}, {2}, {1}, {1}}};
    for (size_t i = 0; i < metadata_inputs.size(); ++i) {
        metadata_parameters[i] = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, metadata_shapes[i]);
        metadata_converts[i] = std::make_shared<ov::op::v0::Convert>(metadata_parameters[i], ov::element::i32);
        metadata_inputs[i] = metadata_converts[i];
        parameters.push_back(metadata_parameters[i]);
    }

    const auto paged_ssm = make_paged_ssm(metadata_inputs, parameters);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{paged_ssm, metadata_converts[0]}, parameters);

    PreservePagedSelectiveSSMMetadataWidth pass;
    EXPECT_TRUE(pass.run_on_model(model));
    for (size_t i = 0; i < metadata_inputs.size(); ++i) {
        EXPECT_EQ(paged_ssm->input_value(i + 6), metadata_parameters[i]->output(0));
        EXPECT_EQ(paged_ssm->get_input_element_type(i + 6), ov::element::i64);
    }
    EXPECT_EQ(model->get_results()[1]->input_value(0), metadata_converts[0]->output(0));
    EXPECT_EQ(model->get_results()[1]->get_input_element_type(0), ov::element::i32);
}

TEST(PreservePagedSelectiveSSMMetadataWidthTest, LeavesNativeI32MetadataUnchanged) {
    ov::ParameterVector parameters;
    std::array<ov::Output<ov::Node>, 5> metadata_inputs;
    const std::array<ov::Shape, 5> metadata_shapes{{{2}, {2}, {2}, {1}, {1}}};
    for (size_t i = 0; i < metadata_inputs.size(); ++i) {
        const auto metadata = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, metadata_shapes[i]);
        metadata_inputs[i] = metadata;
        parameters.push_back(metadata);
    }

    const auto paged_ssm = make_paged_ssm(metadata_inputs, parameters);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{paged_ssm}, parameters);

    PreservePagedSelectiveSSMMetadataWidth pass;
    EXPECT_FALSE(pass.run_on_model(model));
    for (size_t i = 0; i < metadata_inputs.size(); ++i) {
        EXPECT_EQ(paged_ssm->input_value(i + 6), metadata_inputs[i]);
        EXPECT_EQ(paged_ssm->get_input_element_type(i + 6), ov::element::i32);
    }
}

}  // namespace
}  // namespace ov::intel_gpu::test
