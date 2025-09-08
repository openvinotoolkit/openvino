// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/softmax.hpp"
#include <vector>
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include <vector>

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace SoftMax {
namespace {
const auto optimizedCPUSpec = []()-> std::vector<CPUSpecificParams>{
    const auto avx512 = CPUSpecificParams{{}, {}, {"jit"}, "jit_avx512"};
    const auto avx2 = CPUSpecificParams{{}, {}, {"jit"}, "jit_avx2"};
    const std::vector<CPUSpecificParams> vecCpuConfigs = {avx512, avx2};
    auto supportConfigure = CPUTestUtils::filterCPUInfoForDevice(vecCpuConfigs);
    // only the MAX ISA of vecCpuConfigs will be tested
    if (supportConfigure.size() > 0) {
         return std::vector<CPUSpecificParams>{supportConfigure[0]};
     } else {
         return std::vector<CPUSpecificParams>{};
     }
};

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

const auto OptimizedParams = testing::Combine(testing::Values(ElementType::f32, ElementType::bf16),
                                              testing::ValuesIn(optimizedConfigsFP32),
                                              testing::Values(ov::test::utils::DEVICE_CPU),
                                              testing::ValuesIn(optimizedCPUSpec()),
                                              testing::Values(CPUTestUtils::empty_plugin_config));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax_Optimized_CPU,
                         SoftMaxLayerCPUTest,
                         OptimizedParams,
                         SoftMaxLayerCPUTest::getTestCaseName);

//TODO CVS-143812

}  // namespace
}  // namespace SoftMax
}  // namespace test
}  // namespace ov
