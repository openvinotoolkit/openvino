// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/preserve_paged_selective_ssm_metadata_width.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "intel_gpu/primitives/paged_selective_ssm.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/convert_precision.hpp"

namespace ov::intel_gpu::test {
namespace {

constexpr size_t metadata_first_input = cldnn::paged_selective_ssm::SUBSEQUENCE_BEGINS;
constexpr size_t metadata_input_count = cldnn::paged_selective_ssm::CACHE_INTERVAL - metadata_first_input + 1;
constexpr size_t test_token_count = 2;
constexpr size_t test_sequence_count = 1;
constexpr size_t test_block_count = 2;
constexpr size_t test_num_heads = 2;
constexpr size_t test_num_groups = 1;
constexpr size_t test_head_dim = 1;
constexpr size_t test_state_size = 8;

const std::array<ov::Shape, metadata_input_count> metadata_shapes{{
    {test_sequence_count + 1},
    {test_block_count},
    {test_sequence_count + 1},
    {test_sequence_count},
    {test_sequence_count},
}};

std::shared_ptr<ov::op::internal::PagedSelectiveSSM> make_paged_ssm(const std::array<ov::Output<ov::Node>, metadata_input_count>& metadata_inputs,
                                                                    ov::ParameterVector& parameters) {
    const auto metadata_input = [&](cldnn::paged_selective_ssm::PagedSelectiveSSMInputIdx input_index) -> const ov::Output<ov::Node>& {
        return metadata_inputs[static_cast<size_t>(input_index) - metadata_first_input];
    };
    const auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{test_num_heads});
    const auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{test_token_count, test_num_heads});
    const auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{test_token_count, test_num_groups, test_state_size});
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{test_token_count, test_num_heads, test_head_dim});
    const auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{test_token_count, test_num_groups, test_state_size});
    const auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{test_block_count, test_num_heads, test_head_dim, test_state_size});
    parameters.insert(parameters.end(), {A, dt, B, x, C, state});
    return std::make_shared<ov::op::internal::PagedSelectiveSSM>(A,
                                                                 dt,
                                                                 B,
                                                                 x,
                                                                 C,
                                                                 state,
                                                                 metadata_input(cldnn::paged_selective_ssm::SUBSEQUENCE_BEGINS),
                                                                 metadata_input(cldnn::paged_selective_ssm::BLOCK_INDICES),
                                                                 metadata_input(cldnn::paged_selective_ssm::BLOCK_INDICES_BEGINS),
                                                                 metadata_input(cldnn::paged_selective_ssm::NUM_PROCESSED_TOKENS),
                                                                 metadata_input(cldnn::paged_selective_ssm::CACHE_INTERVAL));
}

void run_metadata_precision_pipeline(const std::shared_ptr<ov::Model>& model) {
    constexpr bool keep_precision_sensitive_in_fp32 = true;
    constexpr bool convert_input_output_precision = false;

    ov::pass::Manager manager;
    manager.register_pass<RecordPagedSelectiveSSMMetadataInputs>();
    manager.register_pass<ov::pass::ConvertPrecision>(ov::element::i64,
                                                      ov::element::i32,
                                                      type_to_fuse_map{},
                                                      keep_precision_sensitive_in_fp32,
                                                      convert_input_output_precision);
    manager.register_pass<PreservePagedSelectiveSSMMetadataWidth>();
    manager.run_passes(model);
}

TEST(PreservePagedSelectiveSSMMetadataWidthTest, PreservesNativeI64ConstantsOnlyOnPagedSSMEdges) {
    constexpr int64_t metadata_value = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;

    ov::ParameterVector parameters;
    std::array<std::shared_ptr<ov::op::v0::Constant>, metadata_input_count> metadata_constants;
    std::array<ov::Output<ov::Node>, metadata_input_count> metadata_inputs;
    for (size_t i = 0; i < metadata_inputs.size(); ++i) {
        const std::vector<int64_t> values(metadata_shapes[i].front(), metadata_value);
        metadata_constants[i] = ov::op::v0::Constant::create(ov::element::i64, metadata_shapes[i], values);
        metadata_inputs[i] = metadata_constants[i];
    }

    const auto paged_ssm = make_paged_ssm(metadata_inputs, parameters);
    const auto other_consumer = std::make_shared<ov::op::v0::Abs>(metadata_inputs.front());
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{paged_ssm, other_consumer}, parameters);

    run_metadata_precision_pipeline(model);

    PreservePagedSelectiveSSMMetadataWidth preserve_pass;
    EXPECT_FALSE(preserve_pass.run_on_model(model));

    for (size_t i = 0; i < metadata_inputs.size(); ++i) {
        EXPECT_EQ(paged_ssm->input_value(i + metadata_first_input), metadata_inputs[i]);
        EXPECT_EQ(paged_ssm->get_input_element_type(i + metadata_first_input), ov::element::i64);
        const auto preserved_constant = ov::as_type_ptr<ov::op::v0::Constant>(paged_ssm->get_input_node_shared_ptr(i + metadata_first_input));
        ASSERT_NE(preserved_constant, nullptr);
        EXPECT_EQ(preserved_constant->cast_vector<int64_t>(), std::vector<int64_t>(metadata_shapes[i].front(), metadata_value));
    }
    EXPECT_EQ(other_consumer->get_input_element_type(0), ov::element::i32);
}

TEST(PreservePagedSelectiveSSMMetadataWidthTest, KeepsExplicitMetadataConversions) {
    ov::ParameterVector parameters;
    std::array<std::shared_ptr<ov::op::v0::Convert>, metadata_input_count> metadata_converts;
    std::array<ov::Output<ov::Node>, metadata_input_count> metadata_inputs;
    for (size_t i = 0; i < metadata_inputs.size(); ++i) {
        const auto metadata = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, metadata_shapes[i]);
        metadata_converts[i] = std::make_shared<ov::op::v0::Convert>(metadata, ov::element::i32);
        metadata_inputs[i] = metadata_converts[i];
        parameters.push_back(metadata);
    }

    const auto paged_ssm = make_paged_ssm(metadata_inputs, parameters);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{paged_ssm}, parameters);

    run_metadata_precision_pipeline(model);

    PreservePagedSelectiveSSMMetadataWidth preserve_pass;
    EXPECT_FALSE(preserve_pass.run_on_model(model));
    for (size_t i = 0; i < metadata_inputs.size(); ++i) {
        EXPECT_EQ(paged_ssm->input_value(i + metadata_first_input), metadata_converts[i]->output(0));
        EXPECT_EQ(paged_ssm->get_input_element_type(i + metadata_first_input), ov::element::i32);
    }
}

TEST(PreservePagedSelectiveSSMMetadataWidthTest, RestoresConversionsIntroducedByConvertPrecision) {
    ov::ParameterVector parameters;
    std::array<ov::Output<ov::Node>, metadata_input_count> metadata_inputs;
    for (size_t i = 0; i < metadata_inputs.size(); ++i) {
        const auto metadata = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, metadata_shapes[i]);
        metadata_inputs[i] = metadata;
        parameters.push_back(metadata);
    }

    const auto paged_ssm = make_paged_ssm(metadata_inputs, parameters);
    const auto other_consumer = std::make_shared<ov::op::v0::Abs>(metadata_inputs.front());
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{paged_ssm, other_consumer}, parameters);

    run_metadata_precision_pipeline(model);

    for (size_t i = 0; i < metadata_inputs.size(); ++i) {
        EXPECT_EQ(paged_ssm->input_value(i + metadata_first_input), metadata_inputs[i]);
        EXPECT_EQ(paged_ssm->get_input_element_type(i + metadata_first_input), ov::element::i64);
    }
    EXPECT_EQ(other_consumer->get_input_element_type(0), ov::element::i32);
    EXPECT_NE(ov::as_type_ptr<ov::op::v0::Convert>(other_consumer->get_input_node_shared_ptr(0)), nullptr);
}

TEST(PreservePagedSelectiveSSMMetadataWidthTest, LeavesNativeI32MetadataUnchanged) {
    ov::ParameterVector parameters;
    std::array<ov::Output<ov::Node>, metadata_input_count> metadata_inputs;
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
        EXPECT_EQ(paged_ssm->input_value(i + metadata_first_input), metadata_inputs[i]);
        EXPECT_EQ(paged_ssm->get_input_element_type(i + metadata_first_input), ov::element::i32);
    }
}

}  // namespace
}  // namespace ov::intel_gpu::test
