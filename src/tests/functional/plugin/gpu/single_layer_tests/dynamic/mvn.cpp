// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/mvn.hpp>
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

using basicGPUMvnParams = std::tuple<
       InputShape,      // Input shapes
       ElementType,     // Input precision
       ngraph::AxisSet, // Reduction axes
       bool,            // Across channels
       bool,            // Normalize variance
       double>;         // Epsilon

using MvnLayerGPUTestParamSet = std::tuple<
       basicGPUMvnParams,
       ElementType, // CNNNetwork input precision
       ElementType>; // CNNNetwork output precision

class MvnLayerGPUTest : public testing::WithParamInterface<MvnLayerGPUTestParamSet>,
                       virtual public SubgraphBaseTest {
public:
   static std::string getTestCaseName(testing::TestParamInfo<MvnLayerGPUTestParamSet> obj) {
       basicGPUMvnParams basicParamsSet;
       ElementType inputPrecision, outputPrecision;
       std::tie(basicParamsSet, inputPrecision, outputPrecision) = obj.param;

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

       return result.str();
   }
protected:
   void SetUp() override {
       targetDevice = CommonTestUtils::DEVICE_GPU;

       basicGPUMvnParams basicParamsSet;
       ElementType inPrc;
       ElementType outPrc;
       std::tie(basicParamsSet, inPrc, outPrc) = this->GetParam();

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

       rel_threshold = 0.015f;

       ngraph::ResultVector results;
       for (int i = 0; i < mvn->get_output_size(); ++i) {
           results.push_back(std::make_shared<ngraph::opset1::Result>(mvn->output(i)));
       }
       function = std::make_shared<ngraph::Function>(results, param, "Pad");
   }
};

TEST_P(MvnLayerGPUTest, CompareWithRefs) {
   SKIP_IF_CURRENT_TEST_IS_DISABLED()
   run();
}

namespace {

const std::vector<InputShape> inputShapes_1D = {
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

std::vector<ElementType> inpPrc = {ElementType::i8, ElementType::f16, ElementType::f32};
std::vector<ElementType> outPrc = {ElementType::f16, ElementType::f32};

const auto Mvn3D = ::testing::Combine(
       ::testing::Combine(
           ::testing::ValuesIn(inputShapes_3D),
           ::testing::Values(ElementType::f32),
           ::testing::ValuesIn(emptyReductionAxes),
           ::testing::ValuesIn(acrossChannels),
           ::testing::ValuesIn(normalizeVariance),
           ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn3D, MvnLayerGPUTest, Mvn3D, MvnLayerGPUTest::getTestCaseName);

const auto Mvn4D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_4D),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes),
               ::testing::ValuesIn(acrossChannels),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn4D, MvnLayerGPUTest, Mvn4D, MvnLayerGPUTest::getTestCaseName);

const auto Mvn5D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_5D),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes),
               ::testing::ValuesIn(acrossChannels),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn5D, MvnLayerGPUTest, Mvn5D, MvnLayerGPUTest::getTestCaseName);

const auto Mvn1D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_1D),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes),
               ::testing::ValuesIn(acrossChannels),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn1D, MvnLayerGPUTest, Mvn1D, MvnLayerGPUTest::getTestCaseName);

// 2D no transformed
const auto Mvn2D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_2D),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes),
               ::testing::Values(false),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn2D, MvnLayerGPUTest, Mvn2D, MvnLayerGPUTest::getTestCaseName);

// 2d transformed
const auto Mvn2DTrans = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_2D),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn(emptyReductionAxes),
               ::testing::Values(true),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(inpPrc),
       ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn2DTrans, MvnLayerGPUTest, Mvn2DTrans, MvnLayerGPUTest::getTestCaseName);

} // namespace
} // namespace GPULayerTestsDefinitions
