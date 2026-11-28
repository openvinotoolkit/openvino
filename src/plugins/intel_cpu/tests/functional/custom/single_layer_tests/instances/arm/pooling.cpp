// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/pooling.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "gtest/gtest.h"
#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "shared_test_classes/single_op/pooling.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Pooling {

// Alternate shapes whose last CEIL window starts inside and outside the trailing padding.
const std::vector<poolSpecificParams> paramsAvgV14CeilTorchDynamic = {
    {utils::PoolingTypes::AVG,
     {2, 2},
     {2, 2},
     {1, 1},
     {1, 1},
     ov::op::RoundingType::CEIL_TORCH,
     ov::op::PadType::EXPLICIT,
     false},
    {utils::PoolingTypes::AVG,
     {2, 2},
     {2, 2},
     {1, 1},
     {1, 1},
     ov::op::RoundingType::CEIL_TORCH,
     ov::op::PadType::EXPLICIT,
     true},
    {utils::PoolingTypes::AVG,
     {1, 1},
     {2, 2},
     {0, 0},
     {0, 0},
     ov::op::RoundingType::CEIL_TORCH,
     ov::op::PadType::EXPLICIT,
     true},
    {utils::PoolingTypes::AVG,
     {3, 2},
     {2, 2},
     {1, 1},
     {1, 1},
     ov::op::RoundingType::CEIL_TORCH,
     ov::op::PadType::EXPLICIT,
     true},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_AvgPoolV14CeilTorchDynamic_CPU_4D,
    AvgPoolingV14LayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(paramsAvgV14CeilTorchDynamic),
                       ::testing::Values(InputShape{
                           {1, 3, -1, -1},
                           {{1, 3, 4, 4}, {1, 3, 5, 5}, {1, 3, 6, 6}, {1, 3, 7, 7}, {1, 3, 5, 5}, {1, 3, 4, 4}}}),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),
                       // The divisor rewrite adds an auxiliary ACL pool alongside the fallback pool.
                       ::testing::Values(CPUSpecificParams{{}, {}, {}, "(ref_any|acl)"}),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(ov::AnyMap{}, ov::AnyMap{ov::hint::inference_precision(ov::element::f32)})),
    AvgPoolingV14LayerCPUTest::getTestCaseName);

const std::vector<poolSpecificParams> paramsAvgV14CeilTorchDynamicAcl = {
    {utils::PoolingTypes::AVG,
     {2, 2},
     {2, 2},
     {0, 0},
     {0, 0},
     ov::op::RoundingType::CEIL_TORCH,
     ov::op::PadType::EXPLICIT,
     false},
    {utils::PoolingTypes::AVG,
     {2, 2},
     {2, 2},
     {0, 0},
     {0, 0},
     ov::op::RoundingType::CEIL_TORCH,
     ov::op::PadType::EXPLICIT,
     true},
    {utils::PoolingTypes::AVG,
     {3, 3},
     {2, 2},
     {1, 1},
     {1, 1},
     ov::op::RoundingType::CEIL_TORCH,
     ov::op::PadType::EXPLICIT,
     true},
    {utils::PoolingTypes::AVG,
     {3, 3},
     {1, 1},
     {0, 0},
     {0, 0},
     ov::op::RoundingType::CEIL_TORCH,
     ov::op::PadType::EXPLICIT,
     true},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_AvgPoolV14CeilTorchDynamicAcl_CPU_4D,
    AvgPoolingV14LayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(paramsAvgV14CeilTorchDynamicAcl),
                       ::testing::Values(InputShape{
                           {1, 3, -1, -1},
                           {{1, 3, 4, 4}, {1, 3, 5, 5}, {1, 3, 5, 6}, {1, 3, 6, 5}, {1, 3, 7, 7}, {1, 3, 4, 4}}}),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),
                       ::testing::Values(CPUSpecificParams{{}, {}, {}, "acl"}),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(ov::AnyMap{}, ov::AnyMap{ov::hint::inference_precision(ov::element::f32)})),
    AvgPoolingV14LayerCPUTest::getTestCaseName);

// Use an explicit expected value because the shared reference counts ceil overhang in the divisor.
TEST(AvgPoolingV14DynamicCPUTest, IncludePaddingDivisor) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, -1});
    auto pool = std::make_shared<ov::op::v14::AvgPool>(input,
                                                       ov::Strides{2, 2},
                                                       ov::Shape{1, 1},
                                                       ov::Shape{1, 1},
                                                       ov::Shape{3, 3},
                                                       false,
                                                       ov::op::RoundingType::CEIL_TORCH,
                                                       ov::op::PadType::EXPLICIT);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{pool}, ov::ParameterVector{input});
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU", {ov::hint::inference_precision(ov::element::f32)});
    auto request = compiled.create_infer_request();
    for (std::size_t size : {4, 5, 6, 7, 4}) {
        SCOPED_TRACE(size);
        ov::Tensor tensor(ov::element::f32, ov::Shape{1, 1, size, size});
        std::fill_n(tensor.data<float>(), tensor.get_size(), 1.0F);
        request.set_input_tensor(tensor);
        request.infer();
        const auto output = request.get_output_tensor();
        EXPECT_EQ(output.get_shape(), (ov::Shape{1, 1, size / 2 + 1, size / 2 + 1}));
        // Even sizes leave one input value and three explicit padding positions in the last window.
        const float expected = size % 2 == 0 ? 1.0F / 4.0F : 4.0F / 9.0F;
        EXPECT_FLOAT_EQ(output.data<const float>()[output.get_size() - 1], expected);
    }
}

// Use an explicit expected value because the shared reference counts ceil overhang in the divisor.
TEST(AvgPoolingV14DynamicCPUTest, Stride3IncludePaddingDivisor) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, -1});
    auto pool = std::make_shared<ov::op::v14::AvgPool>(input,
                                                       ov::Strides{3, 3},
                                                       ov::Shape{1, 1},
                                                       ov::Shape{1, 1},
                                                       ov::Shape{3, 3},
                                                       false,
                                                       ov::op::RoundingType::CEIL_TORCH,
                                                       ov::op::PadType::EXPLICIT);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{pool}, ov::ParameterVector{input});
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU", {ov::hint::inference_precision(ov::element::f32)});
    auto request = compiled.create_infer_request();
    for (std::size_t size : {6, 8, 6}) {
        SCOPED_TRACE(size);
        ov::Tensor tensor(ov::element::f32, ov::Shape{1, 1, size, size});
        std::fill_n(tensor.data<float>(), tensor.get_size(), 1.0F);
        request.set_input_tensor(tensor);
        request.infer();
        const auto output = request.get_output_tensor();
        EXPECT_EQ(output.get_shape(), (ov::Shape{1, 1, 3, 3}));
        // At size 6 the last window contains one input value and three explicit padding positions.
        const float expected = size == 6 ? 1.0F / 4.0F : 1.0F;
        EXPECT_FLOAT_EQ(output.data<const float>()[output.get_size() - 1], expected);
    }
}

const std::vector<maxPoolV8SpecificParams>& paramsMaxV144D_2x2kernel = {
        maxPoolV8SpecificParams{ {2, 2}, {2, 2}, {1, 1}, {0, 0}, {0, 0},
                                                          ov::element::Type_t::i32, 0,
                                                          ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::SAME_UPPER },
        maxPoolV8SpecificParams{ {2, 2}, {2, 2}, {1, 1}, {0, 0}, {0, 0},
                                                          ov::element::Type_t::i32, 0,
                                                          ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::SAME_LOWER }
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV14_CPU_4D_2x2Kernel, MaxPoolingV14LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV144D_2x2kernel),
                                 ::testing::ValuesIn(inputShapes4D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::ValuesIn(filterCPUInfo(vecCpuConfigsFusing_4D())),
                                 ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MaxPoolingV14LayerCPUTest::getTestCaseName);

const std::vector<maxPoolV8SpecificParams>& paramsMaxV144D_non2x2kernel = {
            maxPoolV8SpecificParams{ {11, 7}, {2, 2}, {1, 1}, {2, 2}, {2, 2},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT},
};

//The test checks that fallback to nGraph works for ACL non-supported cases
INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV14_CPU_4D_non2x2Kernel_ref, MaxPoolingV14LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV144D_non2x2kernel),
                                 ::testing::ValuesIn(inputShapes4D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"}),
                                 ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MaxPoolingV14LayerCPUTest::getTestCaseName);

}  // namespace Pooling
}  // namespace test
}  // namespace ov
