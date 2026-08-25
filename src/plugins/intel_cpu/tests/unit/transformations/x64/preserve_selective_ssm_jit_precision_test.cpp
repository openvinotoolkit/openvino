// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/x64/pass/preserve_selective_ssm_jit_precision.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>

#include "nodes/kernels/x64/selective_ssm_jit_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/convert_precision.hpp"

namespace ov::intel_cpu::test {
namespace {

struct SelectiveSSMModel {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::Node> operation;
};

SelectiveSSMModel make_selective_ssm(const ov::element::Type& data_precision,
                                     const ov::Dimension& state_size,
                                     bool paged,
                                     const ov::element::Type& index_precision = ov::element::i64) {
    const auto A = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{4});
    const auto state = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{2, 4, 5, state_size});

    ov::ParameterVector parameters;
    std::shared_ptr<ov::Node> operation;
    if (paged) {
        const auto dt = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{-1, 4});
        const auto B = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{-1, 2, state_size});
        const auto x = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{-1, 4, 5});
        const auto C = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{-1, 2, state_size});
        const auto subsequences =
            std::make_shared<ov::op::v0::Parameter>(index_precision, ov::PartialShape::dynamic(1));
        const auto blocks = std::make_shared<ov::op::v0::Parameter>(index_precision, ov::PartialShape::dynamic(1));
        const auto block_begins =
            std::make_shared<ov::op::v0::Parameter>(index_precision, ov::PartialShape::dynamic(1));
        const auto processed = std::make_shared<ov::op::v0::Parameter>(index_precision, ov::PartialShape::dynamic(1));
        const auto intervals = std::make_shared<ov::op::v0::Parameter>(index_precision, ov::PartialShape::dynamic(1));
        parameters = {A, dt, B, x, C, state, subsequences, blocks, block_begins, processed, intervals};
        operation = std::make_shared<ov::op::internal::PagedSelectiveSSM>(parameters[0],
                                                                          parameters[1],
                                                                          parameters[2],
                                                                          parameters[3],
                                                                          parameters[4],
                                                                          parameters[5],
                                                                          parameters[6],
                                                                          parameters[7],
                                                                          parameters[8],
                                                                          parameters[9],
                                                                          parameters[10]);
    } else {
        const auto dt = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{-1, -1, 4});
        const auto B = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{-1, -1, 2, state_size});
        const auto x = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{-1, -1, 4, 5});
        const auto C = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{-1, -1, 2, state_size});
        parameters = {A, dt, B, x, C, state};
        operation = std::make_shared<ov::op::internal::SelectiveSSM>(parameters[0],
                                                                     parameters[1],
                                                                     parameters[2],
                                                                     parameters[3],
                                                                     parameters[4],
                                                                     parameters[5]);
    }

    return {std::make_shared<ov::Model>(operation->outputs(), parameters), operation};
}

void run_precision_conversion(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<PreserveSelectiveSSMJitPrecision>();
    manager.register_pass<ov::pass::ConvertPrecision>(precisions_map{{ov::element::f16, ov::element::f32},
                                                                     {ov::element::bf16, ov::element::f32},
                                                                     {ov::element::i64, ov::element::i32}},
                                                      type_to_fuse_map{},
                                                      false,
                                                      false);
    manager.run_passes(model);
}

class PreserveSelectiveSSMJitPrecisionTest : public testing::TestWithParam<ov::element::Type> {};

TEST_P(PreserveSelectiveSSMJitPrecisionTest, PreservesEligibleSelectiveDataPrecision) {
    auto test_model = make_selective_ssm(GetParam(), 16, false);

    run_precision_conversion(test_model.model);

    for (size_t input_index = 0; input_index < 6; ++input_index) {
        EXPECT_EQ(test_model.operation->get_input_element_type(input_index), GetParam());
    }
    EXPECT_EQ(test_model.operation->get_output_element_type(0), GetParam());
    EXPECT_EQ(test_model.operation->get_output_element_type(1), GetParam());
}

TEST_P(PreserveSelectiveSSMJitPrecisionTest, PreservesEligiblePagedDataButConvertsMetadata) {
    auto test_model = make_selective_ssm(GetParam(), 16, true);

    run_precision_conversion(test_model.model);

    for (size_t input_index = 0; input_index < 6; ++input_index) {
        EXPECT_EQ(test_model.operation->get_input_element_type(input_index), GetParam());
    }
    for (size_t input_index = 6; input_index < 11; ++input_index) {
        EXPECT_EQ(test_model.operation->get_input_element_type(input_index), ov::element::i32);
    }
    EXPECT_EQ(test_model.operation->get_output_element_type(0), GetParam());
}

TEST_P(PreserveSelectiveSSMJitPrecisionTest, DoesNotPreserveUnsupportedStateSize) {
    const auto unsupported_size = static_cast<int64_t>(kernel::max_selective_ssm_jit_state_size + 1);
    auto test_model = make_selective_ssm(GetParam(), unsupported_size, false);

    run_precision_conversion(test_model.model);

    for (size_t input_index = 0; input_index < 6; ++input_index) {
        EXPECT_EQ(test_model.operation->get_input_element_type(input_index), ov::element::f32);
    }
    EXPECT_EQ(test_model.operation->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(test_model.operation->get_output_element_type(1), ov::element::f32);
}

TEST_P(PreserveSelectiveSSMJitPrecisionTest, DoesNotPreserveDynamicStateSize) {
    auto test_model = make_selective_ssm(GetParam(), ov::Dimension::dynamic(), true);

    run_precision_conversion(test_model.model);

    for (size_t input_index = 0; input_index < 6; ++input_index) {
        EXPECT_EQ(test_model.operation->get_input_element_type(input_index), ov::element::f32);
    }
    EXPECT_EQ(test_model.operation->get_output_element_type(0), ov::element::f32);
}

INSTANTIATE_TEST_SUITE_P(LowPrecision,
                         PreserveSelectiveSSMJitPrecisionTest,
                         testing::Values(ov::element::f16, ov::element::bf16));

}  // namespace
}  // namespace ov::intel_cpu::test
