// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/softmax.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace SoftMax {
const auto notOptimizedCPUSpec = CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"};

const std::vector<SoftMaxConfig> optimizedConfigsFP32 = {
    // Static shapes
    {ov::test::InputShape{ov::PartialShape{1, 100}, {ov::Shape{1, 100}}}, 1},
    {ov::test::InputShape{ov::PartialShape{10, 10}, {ov::Shape{10, 10}}}, 1},
    {ov::test::InputShape{ov::PartialShape{100, 1}, {ov::Shape{100, 1}}}, 0},
    {ov::test::InputShape{ov::PartialShape{100, 1}, {ov::Shape{100, 1}}}, 1},
    {ov::test::InputShape{ov::PartialShape{5, 5, 1}, {ov::Shape{5, 5, 1}}}, 1},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5}, {ov::Shape{5, 5, 5}}}, 2},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5}}}, 0},
    {ov::test::InputShape{ov::PartialShape{5, 5, 1, 1}, {ov::Shape{5, 5, 1, 1}}}, 1},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5}}}, 1},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 1}, {ov::Shape{5, 5, 5, 1}}}, 2},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5}}}, 2},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5}}}, 3},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5}}}, 0},
    {ov::test::InputShape{ov::PartialShape{5, 5, 1, 1, 1}, {ov::Shape{5, 5, 1, 1, 1}}}, 1},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5}}}, 1},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 1, 1}, {ov::Shape{5, 5, 5, 1, 1}}}, 2},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5}}}, 2},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 1, 1}, {ov::Shape{5, 5, 5, 1, 1}}}, 3},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5}}}, 3},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 1}, {ov::Shape{5, 5, 5, 5, 1}}}, 4},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5}}}, 4},
    // Dynamic shapes
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{-1, -1},
                          {// target static shapes
                           ov::Shape{10, 10},
                           ov::Shape{15, 15},
                           ov::Shape{10, 10},
                           ov::Shape{10, 5}}},
     1},
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{-1, -1, 1, 1, 1},
                          {// target static shapes
                           ov::Shape{5, 5, 1, 1, 1},
                           ov::Shape{10, 7, 1, 1, 1},
                           ov::Shape{5, 5, 1, 1, 1}}},
     1},
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{{1, 10}, 10},
                          {// target static shapes
                           ov::Shape{10, 10},
                           ov::Shape{5, 10}}},
     1},
};

const std::vector<SoftMaxConfig> notOptimizedConfigsFP32{
    // Static shapes
    {ov::test::InputShape{ov::PartialShape{1, 100}, {ov::Shape{1, 100}}}, 0},
    {ov::test::InputShape{ov::PartialShape{10, 10}, {ov::Shape{10, 10}}}, 0},
    {ov::test::InputShape{ov::PartialShape{10, 10, 10}, {ov::Shape{10, 10, 10}}}, 0},
    {ov::test::InputShape{ov::PartialShape{10, 10, 10}, {ov::Shape{10, 10, 10}}}, 1},
    // Dynamic shapes
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{-1, -1},
                          {// target static shapes
                           ov::Shape{10, 1},
                           ov::Shape{15, 15},
                           ov::Shape{10, 5},
                           ov::Shape{15, 15}}},
     0},
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{ov::Dimension{1, 100}, ov::Dimension{1, 100}, -1},
                          {// target static shapes
                           ov::Shape{10, 10, 10},
                           ov::Shape{10, 10, 1},
                           ov::Shape{10, 5, 10},
                           ov::Shape{10, 10, 1}}},
     1},
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{{1, 10}, 10},
                          {// target static shapes
                           ov::Shape{7, 10},
                           ov::Shape{10, 10}}},
     0},
};

const std::vector<SoftMaxConfig> unsupportedConfigsFP32{
    // Static shapes
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5, 5}}}, 0},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5, 5}}}, 1},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5, 5}}}, 2},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5, 5}}}, 3},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5, 5}}}, 4},
    {ov::test::InputShape{ov::PartialShape{5, 5, 5, 5, 5, 5}, {ov::Shape{5, 5, 5, 5, 5, 5}}}, 5},
    // Dynamic shapes
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{-1, -1, -1, -1, -1, -1},
                          {// target static shapes
                           ov::Shape{5, 5, 5, 5, 5, 5},
                           ov::Shape{7, 7, 7, 7, 7, 7},
                           ov::Shape{5, 5, 5, 5, 5, 5}}},
     4},
    {ov::test::InputShape{// dynamic shape
                          ov::PartialShape{{1, 10}, 5, 5, 5, 5, 5},
                          {// target static shapes
                           ov::Shape{5, 5, 5, 5, 5, 5},
                           ov::Shape{7, 5, 5, 5, 5, 5}}},
     4},
};

const auto OptimizedParams = testing::Combine(testing::Values(ElementType::f32, ElementType::bf16),
                                              testing::ValuesIn(optimizedConfigsFP32),
                                              testing::Values(ov::test::utils::DEVICE_CPU),
                                              testing::Values(notOptimizedCPUSpec),
                                              testing::Values(CPUTestUtils::empty_plugin_config));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax_Optimized_CPU,
                         SoftMaxLayerCPUTest,
                         OptimizedParams,
                         SoftMaxLayerCPUTest::getTestCaseName);

const auto NotOptimizedParams = testing::Combine(testing::Values(ElementType::f32, ElementType::bf16),
                                                 testing::ValuesIn(notOptimizedConfigsFP32),
                                                 testing::Values(ov::test::utils::DEVICE_CPU),
                                                 testing::Values(notOptimizedCPUSpec),
                                                 testing::Values(CPUTestUtils::empty_plugin_config));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax_CPU,
                         SoftMaxLayerCPUTest,
                         NotOptimizedParams,
                         SoftMaxLayerCPUTest::getTestCaseName);

const auto UnsupportedParams = testing::Combine(testing::Values(ElementType::f32, ElementType::bf16),
                                                testing::ValuesIn(unsupportedConfigsFP32),
                                                testing::Values(ov::test::utils::DEVICE_CPU),
                                                testing::Values(notOptimizedCPUSpec),
                                                testing::Values(CPUTestUtils::empty_plugin_config));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax_Unsupported_CPU,
                         SoftMaxLayerCPUTest,
                         UnsupportedParams,
                         SoftMaxLayerCPUTest::getTestCaseName);
}  // namespace SoftMax
}  // namespace test
}  // namespace ov
