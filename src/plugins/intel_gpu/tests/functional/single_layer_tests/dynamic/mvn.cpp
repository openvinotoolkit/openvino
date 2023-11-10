// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/mvn.hpp>
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

using basicGPUMvnParams = std::tuple<
       InputShape,        // Input shapes
       ElementType,       // Input precision
       std::vector<int>,  // Reduction axes
       bool,              // Normalize variance
       double>;           // Epsilon

using MvnLayerGPUTestParamSet = std::tuple<
       basicGPUMvnParams,
       ElementType>; // CNNNetwork input precision

class MvnLayerGPUTest : public testing::WithParamInterface<MvnLayerGPUTestParamSet>,
                       virtual public SubgraphBaseTest {
public:
   static std::string getTestCaseName(testing::TestParamInfo<MvnLayerGPUTestParamSet> obj) {
       basicGPUMvnParams basicParamsSet;
       ElementType inputPrecision;
       std::tie(basicParamsSet, inputPrecision) = obj.param;

       InputShape inputShapes;
       ElementType netPrecision;
       std::vector<int> axes;
       bool normalizeVariance;
       double eps;
       std::tie(inputShapes, netPrecision, axes, normalizeVariance, eps) = basicParamsSet;

       std::ostringstream result;
       result << "IS=" << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
       result << "TS=";
       for (const auto& shape : inputShapes.second) {
           result << "(" << ov::test::utils::vec2str(shape) << ")_";
       }
       result << "Precision=" << netPrecision << "_";
       result << "ReductionAxes=" << ov::test::utils::vec2str(axes) << "_";
       result << "NormalizeVariance=" << (normalizeVariance ? "TRUE" : "FALSE") << "_";
       result << "Epsilon=" << eps;
       result << "_" << "CNNInpPrc=" << inputPrecision;

       return result.str();
   }
protected:
   void SetUp() override {
       targetDevice = ov::test::utils::DEVICE_GPU;

       basicGPUMvnParams basicParamsSet;
       ElementType inPrc;
       std::tie(basicParamsSet, inPrc) = this->GetParam();

       InputShape inputShapes;
       ElementType netPrecision;
       std::vector<int> axes;
       bool normalizeVariance;
       double eps;
       std::tie(inputShapes, netPrecision, axes, normalizeVariance, eps) = basicParamsSet;

       init_input_shapes({inputShapes});

       auto axesType = ov::element::i64;
       std::string eps_mode = "inside_sqrt";

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));

       auto axesNode = ngraph::builder::makeConstant(axesType, ngraph::Shape{axes.size()}, axes);
       auto mvn = ngraph::builder::makeMVN6(params[0], axesNode, normalizeVariance, eps, eps_mode);

       rel_threshold = 0.015f;

       ngraph::ResultVector results;
       for (size_t i = 0; i < mvn->get_output_size(); ++i) {
           results.push_back(std::make_shared<ngraph::opset1::Result>(mvn->output(i)));
       }
       function = std::make_shared<ngraph::Function>(results, params, "MVN");
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


const std::vector<bool> normalizeVariance = {
       true,
       false
};

const std::vector<double> epsilon = {
       0.000000001
};

const std::vector<int> reduction_axes_1234 = {1, 2, 3, 4};
const std::vector<int> reduction_axes_234 = {2, 3, 4};
const std::vector<int> reduction_axes_123 = {1, 2, 3};
const std::vector<int> reduction_axes_23 = {2, 3};
const std::vector<int> reduction_axes_12 = {1, 2};
const std::vector<int> reduction_axes_3 = {3};
const std::vector<int> reduction_axes_2 = {2};

std::vector<ElementType> inpPrc = {ElementType::i8, ElementType::f16, ElementType::f32};

const auto Mvn3D = ::testing::Combine(
       ::testing::Combine(
           ::testing::ValuesIn(inputShapes_3D),
           ::testing::Values(ElementType::f32),
           ::testing::ValuesIn({reduction_axes_12, reduction_axes_2}),
           ::testing::ValuesIn(normalizeVariance),
           ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(inpPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn3D, MvnLayerGPUTest, Mvn3D, MvnLayerGPUTest::getTestCaseName);

const auto Mvn4D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_4D),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn({reduction_axes_2, reduction_axes_3, reduction_axes_12, reduction_axes_23, reduction_axes_123}),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(inpPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn4D, MvnLayerGPUTest, Mvn4D, MvnLayerGPUTest::getTestCaseName);

const auto Mvn5D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_5D),
               ::testing::Values(ElementType::f32),
               ::testing::ValuesIn({reduction_axes_3, reduction_axes_23, reduction_axes_123, reduction_axes_1234}),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(inpPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn5D, MvnLayerGPUTest, Mvn5D, MvnLayerGPUTest::getTestCaseName);

} // namespace
} // namespace GPULayerTestsDefinitions
