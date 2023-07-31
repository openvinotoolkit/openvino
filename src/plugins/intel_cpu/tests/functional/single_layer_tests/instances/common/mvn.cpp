// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/mvn.hpp"
#include "shared_test_classes/single_layer/mvn.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace MVN {

const std::vector<bool> normalizeVariance = {
       true
};

std::vector<ElementType> inpPrc = {
        ElementType::i8,
        ElementType::f32,
};
std::vector<ElementType> outPrc = {
        ElementType::f32,
};

std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nchw}, {nchw}, {}, {}),
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
};

std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {}),
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
};

std::vector<fusingSpecificParams> fusingParamsSet {
        emptyFusingSpec,
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
       ::testing::ValuesIn(fusingParamsSet),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn3D, MvnLayerCPUTest, Mvn3D, MvnLayerCPUTest::getTestCaseName);

const auto Mvn4D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_4D()),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes()),
               ::testing::ValuesIn(acrossChannels()),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon())),
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
       ::testing::ValuesIn(fusingParamsSet),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn4D, MvnLayerCPUTest, Mvn4D, MvnLayerCPUTest::getTestCaseName);

const auto Mvn5D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_5D()),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes()),
               ::testing::ValuesIn(acrossChannels()),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon())),
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
       ::testing::ValuesIn(fusingParamsSet),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn5D, MvnLayerCPUTest, Mvn5D, MvnLayerCPUTest::getTestCaseName);

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
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc),
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
       ::testing::ValuesIn(fusingParamsSet),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc),
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
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc),
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
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc),
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
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc),
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
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
       ::testing::ValuesIn(fusingParamsSetStaticShape),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc),
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
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
       ::testing::ValuesIn(fusingParamsSetStaticShape),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn5D_Static, MvnLayerCPUTest, Mvn5DStatic, MvnLayerCPUTest::getTestCaseName);

} // namespace MVN
} // namespace CPULayerTestsDefinitions