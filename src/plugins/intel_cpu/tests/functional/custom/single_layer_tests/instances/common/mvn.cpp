// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/mvn.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace MVN {

const std::vector<bool> normalizeVariance = {
       true
};

const std::vector<CPUSpecificParams> cpuParams_4D_ncsp = {
    CPUSpecificParams({nchw}, {nchw}, {}, {})
};

const std::vector<CPUSpecificParams> cpuParams_5D_ncsp = {
    CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})
};

std::vector<fusingSpecificParams> fusingParamsSetStaticShape {
        emptyFusingSpec,
};

const auto Mvn3D = ::testing::Combine(
       ::testing::Combine(
           ::testing::ValuesIn(inputShapes_3D()),
           ::testing::Values(ElementType::f32),
           ::testing::ValuesIn(emptyReductionAxes()),
           ::testing::ValuesIn(acrossChannels()),
           ::testing::ValuesIn(normalizeVariance),
           ::testing::ValuesIn(epsilon())),
       ::testing::Values(emptyCPUSpec),
       ::testing::ValuesIn(fusingParamsSet()),
       ::testing::ValuesIn(inpPrc()),
       ::testing::ValuesIn(outPrc()),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn3D, MvnLayerCPUTest, Mvn3D, MvnLayerCPUTest::getTestCaseName);

const auto Mvn4D_across_channels = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_4D()),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes()),
               ::testing::Values(true),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon())),
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
       ::testing::ValuesIn(fusingParamsSet()),
       ::testing::ValuesIn(inpPrc()),
       ::testing::ValuesIn(outPrc()),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn4D_across_channels, MvnLayerCPUTest, Mvn4D_across_channels, MvnLayerCPUTest::getTestCaseName);

const auto Mvn4D_no_across_channels_ncsp = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_4D()),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes()),
               ::testing::Values(false),
               ::testing::Values(true),
               ::testing::ValuesIn(epsilon())),
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_ncsp)),
       ::testing::ValuesIn(fusingParamsSet()),
       ::testing::ValuesIn(inpPrc()),
       ::testing::ValuesIn(outPrc()),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn4D_no_across_channels_ncsp, MvnLayerCPUTest, Mvn4D_no_across_channels_ncsp, MvnLayerCPUTest::getTestCaseName);

const auto Mvn5D_across_channels = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_5D()),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes()),
               ::testing::Values(true),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon())),
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D())),
       ::testing::ValuesIn(fusingParamsSet()),
       ::testing::ValuesIn(inpPrc()),
       ::testing::ValuesIn(outPrc()),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn5D_across_channels, MvnLayerCPUTest, Mvn5D_across_channels, MvnLayerCPUTest::getTestCaseName);

const auto Mvn5D_no_across_channels_ncsp = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_5D()),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes()),
               ::testing::Values(false),
               ::testing::Values(true),
               ::testing::ValuesIn(epsilon())),
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_ncsp)),
       ::testing::ValuesIn(fusingParamsSet()),
       ::testing::ValuesIn(inpPrc()),
       ::testing::ValuesIn(outPrc()),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn5D_no_across_channels_ncsp, MvnLayerCPUTest, Mvn5D_no_across_channels_ncsp, MvnLayerCPUTest::getTestCaseName);

// 1D 2D case
std::vector<fusingSpecificParams> fusingUnaryEltwiseParamsSet {
    emptyFusingSpec,
};

const auto Mvn1D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_1D()),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes()),
               ::testing::ValuesIn(acrossChannels()),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon())),
       ::testing::Values(emptyCPUSpec),
       ::testing::ValuesIn(fusingUnaryEltwiseParamsSet),
       ::testing::ValuesIn(inpPrc()),
       ::testing::ValuesIn(outPrc()),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn1D, MvnLayerCPUTest, Mvn1D, MvnLayerCPUTest::getTestCaseName);

// 2D no transformed
const auto Mvn2D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_2D()),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes()),
               ::testing::Values(false),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon())),
       ::testing::Values(emptyCPUSpec),
       ::testing::ValuesIn(fusingParamsSet()),
       ::testing::ValuesIn(inpPrc()),
       ::testing::ValuesIn(outPrc()),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn2D, MvnLayerCPUTest, Mvn2D, MvnLayerCPUTest::getTestCaseName);

// 2d transformed
const auto Mvn2DTrans = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_2D()),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes()),
               ::testing::Values(true),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon())),
       ::testing::Values(emptyCPUSpec),
       ::testing::ValuesIn(fusingUnaryEltwiseParamsSet),
       ::testing::ValuesIn(inpPrc()),
       ::testing::ValuesIn(outPrc()),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn2DTrans, MvnLayerCPUTest, Mvn2DTrans, MvnLayerCPUTest::getTestCaseName);

const auto Mvn2DStatic = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapesStatic_2D()),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes()),
               ::testing::Values(false),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon())),
       ::testing::Values(emptyCPUSpec),
       ::testing::ValuesIn(fusingParamsSetStaticShape),
       ::testing::ValuesIn(inpPrc()),
       ::testing::ValuesIn(outPrc()),
       ::testing::ValuesIn(additionalConfig()));

const auto Mvn3DStatic = ::testing::Combine(
       ::testing::Combine(
           ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesStatic_3D())),
           ::testing::Values(ElementType::f32),
           ::testing::ValuesIn(emptyReductionAxes()),
           ::testing::ValuesIn(acrossChannels()),
           ::testing::ValuesIn(normalizeVariance),
           ::testing::ValuesIn(epsilon())),
       ::testing::Values(emptyCPUSpec),
       ::testing::ValuesIn(fusingParamsSetStaticShape),
       ::testing::ValuesIn(inpPrc()),
       ::testing::ValuesIn(outPrc()),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn3D_Static, MvnLayerCPUTest, Mvn3DStatic, MvnLayerCPUTest::getTestCaseName);

const auto Mvn4DStatic = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesStatic_4D())),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes()),
               ::testing::Values(true),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon())),
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
       ::testing::ValuesIn(fusingParamsSetStaticShape),
       ::testing::ValuesIn(inpPrc()),
       ::testing::ValuesIn(outPrc()),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn4D_Static, MvnLayerCPUTest, Mvn4DStatic, MvnLayerCPUTest::getTestCaseName);

const auto Mvn5DStatic = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesStatic_5D())),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes()),
               ::testing::Values(true),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon())),
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D())),
       ::testing::ValuesIn(fusingParamsSetStaticShape),
       ::testing::ValuesIn(inpPrc()),
       ::testing::ValuesIn(outPrc()),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn5D_Static, MvnLayerCPUTest, Mvn5DStatic, MvnLayerCPUTest::getTestCaseName);

}  // namespace MVN
}  // namespace test
}  // namespace ov