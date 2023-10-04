// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/mvn.hpp"
#include "shared_test_classes/single_layer/mvn.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include <ov_models/builders.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;


namespace CPULayerTestsDefinitions {
namespace MVN {
namespace {

const std::vector<bool> normalizeVariance = {
       false
};

std::vector<ElementType> inpPrc = {
        ElementType::bf16
};
std::vector<ElementType> outPrc = {
        ElementType::bf16
};

std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {})
};

std::vector<fusingSpecificParams> fusingParamsSet {
        /* activations */
        fusingRelu,
        fusingElu,
        fusingTanh,
        fusingSwish,
        /* FQ */
        fusingFakeQuantizePerTensorRelu,
        /* another patterns */
        fusingAddPerTensor
};

std::vector<fusingSpecificParams> fusingParamsSetStaticShape {
       /* FQ */
       fusingFakeQuantizePerChannel,
       fusingFakeQuantizePerChannelRelu,
       /* another patterns */
       fusingScaleShift,
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

// 1D 2D case
std::vector<fusingSpecificParams> fusingUnaryEltwiseParamsSet {
       /* activations */
       fusingRelu,
       fusingElu,
       fusingTanh,
       fusingSwish,
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
               ::testing::Values(false),
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
               ::testing::Values(false),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon())),
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
       ::testing::ValuesIn(fusingParamsSetStaticShape),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn5D_Static, MvnLayerCPUTest, Mvn5DStatic, MvnLayerCPUTest::getTestCaseName);

// no transformed with small spatial dim and i8 data and no fusion to cover model use case
const std::vector<InputShape> inputShapesSmallSpatial = {
       { {}, {{4, 1}}},
       { {}, {{2, 2}}},
       { {}, {{1, 2, 1}}},
       { {}, {{3, 1, 1, 1}}},
};

const auto MvnSmallSpatial = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapesSmallSpatial),
               ::testing::Values(ElementType::i8),
               ::testing::ValuesIn(emptyReductionAxes()),
               ::testing::Values(false),
               ::testing::Values(false),
               ::testing::ValuesIn(epsilon())),
       ::testing::Values(emptyCPUSpec),
       ::testing::Values(emptyFusingSpec),
       ::testing::Values(ElementType::i8),
       ::testing::Values(ElementType::f32),
       ::testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_MvnSmallSpatial, MvnLayerCPUTest, MvnSmallSpatial, MvnLayerCPUTest::getTestCaseName);

} // namespace
} // namespace MVN
} // namespace CPULayerTestsDefinitions