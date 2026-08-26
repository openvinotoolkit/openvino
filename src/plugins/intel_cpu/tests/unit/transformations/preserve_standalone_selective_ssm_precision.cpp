// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/preserve_standalone_selective_ssm_precision.hpp"

#include <gtest/gtest.h>

#include <array>
#include <utility>

#include "openvino/op/paged_attention.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_cpu::test {
namespace {

struct PagedSSMGraph {
    ov::ParameterVector parameters;
    std::shared_ptr<ov::op::internal::PagedSelectiveSSM> op;
    std::shared_ptr<ov::Model> model;
};

PagedSSMGraph make_paged_ssm(ov::element::Type data_precision = ov::element::f32,
                             bool is_paged_attention_model = false) {
    PagedSSMGraph graph;
    const auto parameter = [&graph](ov::element::Type precision, const ov::PartialShape& shape) {
        auto value = std::make_shared<ov::op::v0::Parameter>(precision, shape);
        graph.parameters.push_back(value);
        return value;
    };

    const auto A = parameter(data_precision, {4});
    const auto dt = parameter(data_precision, {2, 4});
    const auto B = parameter(data_precision, {2, 2, 5});
    const auto x = parameter(data_precision, {2, 4, 3});
    const auto C = parameter(data_precision, {2, 2, 5});
    const auto state = parameter(data_precision, {2, 4, 3, 5});
    const auto subsequences = parameter(ov::element::i64, {2});
    const auto blocks = parameter(ov::element::i64, {2});
    const auto block_begins = parameter(ov::element::i64, {2});
    const auto processed = parameter(ov::element::i64, {1});
    const auto intervals = parameter(ov::element::i64, {1});

    graph.op = std::make_shared<ov::op::internal::PagedSelectiveSSM>(A,
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
    if (is_paged_attention_model) {
        graph.op->add_control_dependency(std::make_shared<ov::op::PagedAttentionExtension>());
    }
    graph.model = std::make_shared<ov::Model>(ov::OutputVector{graph.op}, graph.parameters);
    return graph;
}

TEST(PreserveStandaloneSelectiveSSMPrecisionTest, MarksPagedDataAndMetadata) {
    const auto graph = make_paged_ssm();

    PreserveStandaloneSelectiveSSMPrecision pass;
    EXPECT_FALSE(pass.run_on_model(graph.model));
    EXPECT_TRUE(ov::is_conversion_disabled(graph.op, ov::element::f32, ov::element::f16));
    for (size_t input_idx = 0; input_idx < 6; ++input_idx) {
        EXPECT_TRUE(ov::is_conversion_disabled(graph.op->get_input_node_shared_ptr(input_idx),
                                               ov::element::f32,
                                               ov::element::f16));
    }
    for (size_t input_idx = 6; input_idx < 11; ++input_idx) {
        EXPECT_TRUE(ov::is_conversion_disabled(graph.op->get_input_node_shared_ptr(input_idx),
                                               ov::element::i64,
                                               ov::element::i32));
    }
}

TEST(PreserveStandaloneSelectiveSSMPrecisionTest, MarksNonPagedData) {
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

TEST(PreserveStandaloneSelectiveSSMPrecisionTest, SkipsPagedAttentionModel) {
    const auto graph = make_paged_ssm(ov::element::f32, true);

    PreserveStandaloneSelectiveSSMPrecision pass;
    EXPECT_FALSE(pass.run_on_model(graph.model));
    EXPECT_FALSE(ov::is_conversion_disabled(graph.op, ov::element::f32, ov::element::f16));
    for (const auto& parameter : graph.parameters) {
        EXPECT_FALSE(ov::is_conversion_disabled(parameter, ov::element::dynamic, ov::element::dynamic));
    }
}

TEST(RestoreStandalonePagedSelectiveSSMStatePrecisionTest, RestoresDataPrecision) {
    const std::array<std::pair<ov::element::Type, ov::element::Type>, 3> precision_pairs = {
        std::make_pair(ov::element::f32, ov::element::bf16),
        std::make_pair(ov::element::f16, ov::element::f32),
        std::make_pair(ov::element::bf16, ov::element::f16),
    };

    for (const auto& [data_precision, converted_state_precision] : precision_pairs) {
        const auto graph = make_paged_ssm(data_precision);
        const auto state = ov::as_type_ptr<ov::op::v0::Parameter>(graph.parameters.at(5));
        state->set_element_type(converted_state_precision);
        state->validate_and_infer_types();

        RestoreStandalonePagedSelectiveSSMStatePrecision pass;
        EXPECT_TRUE(pass.run_on_model(graph.model));
        EXPECT_EQ(state->get_element_type(), data_precision);
        EXPECT_NO_THROW(graph.op->validate_and_infer_types());
    }
}

TEST(RestoreStandalonePagedSelectiveSSMStatePrecisionTest, KeepsMatchingPrecision) {
    for (const auto precision : {ov::element::f32, ov::element::f16, ov::element::bf16}) {
        const auto graph = make_paged_ssm(precision);
        const auto state = graph.parameters.at(5);

        RestoreStandalonePagedSelectiveSSMStatePrecision pass;
        EXPECT_FALSE(pass.run_on_model(graph.model));
        EXPECT_EQ(state->get_element_type(), precision);
        EXPECT_NO_THROW(graph.op->validate_and_infer_types());
    }
}

TEST(RestoreStandalonePagedSelectiveSSMStatePrecisionTest, SkipsPagedAttentionModel) {
    const auto graph = make_paged_ssm(ov::element::f32, true);
    const auto state = ov::as_type_ptr<ov::op::v0::Parameter>(graph.parameters.at(5));
    state->set_element_type(ov::element::bf16);
    state->validate_and_infer_types();

    RestoreStandalonePagedSelectiveSSMStatePrecision pass;
    EXPECT_FALSE(pass.run_on_model(graph.model));
    EXPECT_EQ(state->get_element_type(), ov::element::bf16);
}

}  // namespace
}  // namespace ov::intel_cpu::test
