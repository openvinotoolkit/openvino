// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/mvn.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using basicCpuMvnParams = std::tuple<
       InputShape, // Input shapes
       ElementType, // Input precision
       ngraph::AxisSet, // Reduction axes
       bool, // Across channels
       bool, // Normalize variance
       double>; // Epsilon

using MvnLayerCPUTestParamSet = std::tuple<
       basicCpuMvnParams,
       CPUSpecificParams,
       fusingSpecificParams,
       ElementType, // CNNNetwork input precision
       ElementType>; // CNNNetwork output precision

class MvnLayerCPUTest : public testing::WithParamInterface<MvnLayerCPUTestParamSet>,
                       virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
   static std::string getTestCaseName(testing::TestParamInfo<MvnLayerCPUTestParamSet> obj) {
       basicCpuMvnParams basicParamsSet;
       CPUSpecificParams cpuParams;
       fusingSpecificParams fusingParams;
       ElementType inputPrecision, outputPrecision;
       std::tie(basicParamsSet, cpuParams, fusingParams, inputPrecision, outputPrecision) = obj.param;

       InputShape inputShapes;
       ElementType netPrecision;
       ngraph::AxisSet axes;
       bool acrossChanels, normalizeVariance;
       double eps;
       std::tie(inputShapes, netPrecision, axes, acrossChanels, normalizeVariance, eps) = basicParamsSet;

       std::ostringstream result;
       result << "IS=" << CommonTestUtils::partialShape2str({inputShapes.first}) << "_";
       result << "TS=";
       for (const auto& shape : inputShapes.second) {
           result << "(" << CommonTestUtils::vec2str(shape) << ")_";
       }
       result << "Precision=" << netPrecision << "_";
       if (!axes.empty()) {
           result << "ReductionAccess=" << CommonTestUtils::vec2str(axes.to_vector()) << "_";
       } else {
           result << "AcrossChannels=" << (acrossChanels ? "TRUE" : "FALSE") << "_";
       }
       result << "NormalizeVariance=" << (normalizeVariance ? "TRUE" : "FALSE") << "_";
       result << "Epsilon=" << eps;
       result << "_" << "CNNInpPrc=" << inputPrecision;
       result << "_" << "CNNOutPrc=" << outputPrecision;

       result << CPUTestsBase::getTestCaseName(cpuParams);

       result << CpuTestWithFusing::getTestCaseName(fusingParams);

       return result.str();
   }
protected:
   void SetUp() override {
       targetDevice = CommonTestUtils::DEVICE_CPU;

       basicCpuMvnParams basicParamsSet;
       CPUSpecificParams cpuParams;
       fusingSpecificParams fusingParams;
       ElementType inPrc;
       ElementType outPrc;
       std::tie(basicParamsSet, cpuParams, fusingParams, inPrc, outPrc) = this->GetParam();

       std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
       std::tie(postOpMgrPtr, fusedOps) = fusingParams;

       InputShape inputShapes;
       ElementType netPrecision;
       ngraph::AxisSet axes;
       bool acrossChanels, normalizeVariance;
       double eps;
       std::tie(inputShapes, netPrecision, axes, acrossChanels, normalizeVariance, eps) = basicParamsSet;

       init_input_shapes({inputShapes});

       auto param = ngraph::builder::makeDynamicParams(netPrecision, inputDynamicShapes);
       auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(param));
       auto mvn = ngraph::builder::makeMVN(paramOuts[0], acrossChanels, normalizeVariance, eps);
       if (!axes.empty()) {
            mvn = ngraph::builder::makeMVN(paramOuts[0], axes, normalizeVariance, eps);
       }

       selectedType = getPrimitiveType();
       selectedType = makeSelectedTypeStr(selectedType, netPrecision);

       rel_threshold = 0.015f;
       function = makeNgraphFunction(netPrecision, param, mvn, "mvn");
   }
};

TEST_P(MvnLayerCPUTest, CompareWithRefs) {
   run();
   CheckPluginRelatedResults(compiledModel, "MVN");
}

namespace {

const std::vector<InputShape> inputShapes_1D = {
       { {}, {{5}}},
       { {}, {{16}}},
       {
           // dynamic
           {-1},
           // target
           {
               {2},
               {16},
               {1},
               {2}
           }
       },
       {
           // dynamic
           {{1, 20}},
           // target
           {
               {1},
               {16},
               {4},
               {16}
           }
       }
};

const std::vector<InputShape> inputShapes_2D = {
       { {}, {{1, 32}}},
       { {}, {{16, 64}}},

       {
           // dynamic
           {-1, -1},
           // target
           {
               {2, 16},
               {4, 16},
               {1, 16},
               {4, 16}
           }
       },
       {
           // dynamic
           {{1, 5}, {1, 20}},
           // target
           {
               {1, 1},
               {2, 16},
               {4, 16},
               {2, 16}
           }
       }
};

const std::vector<InputShape> inputShapes_3D = {
       { {}, {{1, 32, 17}}},
       { {}, {{1, 37, 9}}},
       { {}, {{1, 16, 4}}},
       {
           // dynamic
           {-1, -1, -1},
           // target
           {
               {2, 16, 6},
               {4, 16, 2},
               {2, 16, 6},
               {4, 16, 2}
           }
       },
       {
           // dynamic
           {{1, 5}, {1, 20}, {1, 7}},
           // target
           {
               {1, 1, 1},
               {2, 16, 6},
               {4, 16, 2},
               {2, 16, 6}
           }
       }
};

const std::vector<InputShape> inputShapes_4D = {
       { {}, {{1, 16, 5, 8}}},
       { {}, {{2, 19, 5, 10}}},
       { {}, {{7, 32, 2, 8}}},
       { {}, {{5, 8, 3, 5}}},
       { {}, {{1, 2, 7, 5}}},
       { {}, {{1, 4, 5, 5}}},
       { {}, {{1, 7, 3, 5}}},
       { {}, {{1, 15, 9, 5}}},
       { {}, {{4, 41, 6, 9}}},
       {
           // dynamic
           {-1, -1, -1, -1},
           // target
           {
               {2, 16, 10, 6},
               {4, 16, 2, 2},
               {2, 16, 10, 6},
               {4, 16, 2, 2}
           }
       },
       {
           // dynamic
           {{1, 5}, {1, 20}, {1, 10}, {1, 7}},
           // target
           {
               {1, 1, 1, 1},
               {2, 16, 10, 6},
               {4, 16, 2, 2},
               {2, 16, 10, 6}
           }
       }
};

const std::vector<InputShape> inputShapes_5D = {
       { {}, {{1, 32, 8, 1, 6}}},
       { {}, {{1, 9, 1, 15, 9}}},
       { {}, {{6, 64, 6, 1, 18}}},
       { {}, {{2, 31, 2, 9, 1}}},
       { {}, {{10, 16, 5, 10, 6}}},
       {
           // dynamic
           {-1, -1, -1, -1, -1},
           // target
           {
               {2, 16, 5, 10, 6},
               {4, 16, 7, 2, 2},
               {2, 16, 5, 10, 6},
               {4, 16, 7, 2, 2}
           }
       },
       {
           // dynamic
           {{1, 5}, {1, 20}, {1, 7}, {1, 10}, {1, 7}},
           // target
           {
               {1, 1, 1, 1, 1},
               {2, 16, 5, 10, 6},
               {4, 16, 7, 2, 2},
               {2, 16, 5, 10, 6}
           }
       }
};

const std::vector<bool> acrossChannels = {
       true,
       false
};

const std::vector<bool> normalizeVariance = {
       true,
       false
};

const std::vector<double> epsilon = {
       0.000000001
};

const std::vector<ngraph::AxisSet> emptyReductionAxes = {{}};

std::vector<ElementType> inpPrc = {ElementType::i8, ElementType::bf16, ElementType::f32};
std::vector<ElementType> outPrc = {ElementType::bf16, ElementType::f32};

std::vector<CPUSpecificParams> cpuParams_4D = {
       CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
       CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
       CPUSpecificParams({nchw}, {nchw}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_5D = {
       CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
       CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
       CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})
};

std::vector<fusingSpecificParams> fusingParamsSet {
       emptyFusingSpec,
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
           ::testing::ValuesIn(inputShapes_3D),
           ::testing::Values(ElementType::f32),
           ::testing::ValuesIn(emptyReductionAxes),
           ::testing::ValuesIn(acrossChannels),
           ::testing::ValuesIn(normalizeVariance),
           ::testing::ValuesIn(epsilon)),
       ::testing::Values(emptyCPUSpec),
       ::testing::ValuesIn(fusingParamsSet),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn3D, MvnLayerCPUTest, Mvn3D, MvnLayerCPUTest::getTestCaseName);

const auto Mvn4D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_4D),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes),
               ::testing::ValuesIn(acrossChannels),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
       ::testing::ValuesIn(fusingParamsSet),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn4D, MvnLayerCPUTest, Mvn4D, MvnLayerCPUTest::getTestCaseName);

const auto Mvn5D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_5D),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes),
               ::testing::ValuesIn(acrossChannels),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
       ::testing::ValuesIn(fusingParamsSet),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn5D, MvnLayerCPUTest, Mvn5D, MvnLayerCPUTest::getTestCaseName);

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
               ::testing::ValuesIn(inputShapes_1D),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes),
               ::testing::ValuesIn(acrossChannels),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::Values(emptyCPUSpec),
       ::testing::ValuesIn(fusingUnaryEltwiseParamsSet),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn1D, MvnLayerCPUTest, Mvn1D, MvnLayerCPUTest::getTestCaseName);

// 2D no transformed
const auto Mvn2D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_2D),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes),
               ::testing::Values(false),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::Values(emptyCPUSpec),
       ::testing::ValuesIn(fusingParamsSet),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn2D, MvnLayerCPUTest, Mvn2D, MvnLayerCPUTest::getTestCaseName);

// 2d transformed
const auto Mvn2DTrans = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_2D),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes),
               ::testing::Values(true),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::Values(emptyCPUSpec),
       ::testing::ValuesIn(fusingUnaryEltwiseParamsSet),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn2DTrans, MvnLayerCPUTest, Mvn2DTrans, MvnLayerCPUTest::getTestCaseName);

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
               ::testing::ValuesIn(emptyReductionAxes),
               ::testing::Values(false),
               ::testing::Values(false),
               ::testing::ValuesIn(epsilon)),
       ::testing::Values(emptyCPUSpec),
       ::testing::Values(emptyFusingSpec),
       ::testing::Values(ElementType::i8),
       ::testing::Values(ElementType::f32));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_MvnSmallSpatial, MvnLayerCPUTest, MvnSmallSpatial, MvnLayerCPUTest::getTestCaseName);

// Static shape test for some specific fusing parameters in fusingParamsSetStaticShape

const std::vector<ov::Shape> inputShapesStatic_2D = {
        {1},
        {16},
        {4}
};

const std::vector<ov::Shape> inputShapesStatic_3D = {
        {2, 16, 6},
        {4, 16, 2},
        {1, 16, 4}
};

const std::vector<ov::Shape> inputShapesStatic_4D = {
        {1, 7, 3, 5},
        {1, 15, 9, 5},
        {4, 41, 6, 9},
        // cover channel case 4*16*2+16+3=147
        {1, 147, 2, 2}
};

const std::vector<ov::Shape> inputShapesStatic_5D = {
        {1, 32, 8, 1, 6},
        {1, 9, 1, 15, 9},
        {6, 64, 6, 1, 18},
        // cover channel case 4*16*2+16+9=153
        {6, 153, 2, 2, 2}
};

const auto Mvn2DStatic = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapesStatic_2D),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes),
               ::testing::Values(false),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::Values(emptyCPUSpec),
       ::testing::ValuesIn(fusingParamsSetStaticShape),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

const auto Mvn3DStatic = ::testing::Combine(
       ::testing::Combine(
           ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesStatic_3D)),
           ::testing::Values(ElementType::f32),
           ::testing::ValuesIn(emptyReductionAxes),
           ::testing::ValuesIn(acrossChannels),
           ::testing::ValuesIn(normalizeVariance),
           ::testing::ValuesIn(epsilon)),
       ::testing::Values(emptyCPUSpec),
       ::testing::ValuesIn(fusingParamsSetStaticShape),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn3D_Static, MvnLayerCPUTest, Mvn3DStatic, MvnLayerCPUTest::getTestCaseName);

const auto Mvn4DStatic = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesStatic_4D)),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes),
               ::testing::ValuesIn(acrossChannels),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
       ::testing::ValuesIn(fusingParamsSetStaticShape),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn4D_Static, MvnLayerCPUTest, Mvn4DStatic, MvnLayerCPUTest::getTestCaseName);

const auto Mvn5DStatic = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesStatic_5D)),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes),
               ::testing::ValuesIn(acrossChannels),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
       ::testing::ValuesIn(fusingParamsSetStaticShape),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn5D_Static, MvnLayerCPUTest, Mvn5DStatic, MvnLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
