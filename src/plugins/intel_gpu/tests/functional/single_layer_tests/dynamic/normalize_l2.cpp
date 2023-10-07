// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/normalize_l2.hpp>
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

using NormalizeL2LayerGPUTestParams = std::tuple<
       InputShape,              // Input shapes
       ElementType,             // Input precision
       std::vector<int64_t>,    // Reduction axes
       ngraph::op::EpsMode,     // EpsMode
       float>;                  // Epsilon

class NormalizeL2LayerGPUTest : public testing::WithParamInterface<NormalizeL2LayerGPUTestParams>,
                       virtual public SubgraphBaseTest {
public:
   static std::string getTestCaseName(testing::TestParamInfo<NormalizeL2LayerGPUTestParams> obj) {
       InputShape inputShapes;
       ElementType netPrecision;
       std::vector<int64_t> axes;
       ngraph::op::EpsMode epsMode;
       float eps;
       std::tie(inputShapes, netPrecision, axes, epsMode, eps) = obj.param;

       std::ostringstream result;
       result << "IS=" << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
       result << "TS=";
       for (const auto& shape : inputShapes.second) {
           result << "(" << ov::test::utils::vec2str(shape) << ")_";
       }
       result << "Precision=" << netPrecision << "_";
       result << "ReductionAxes=" << ov::test::utils::vec2str(axes) << "_";
       result << "EpsMode=" << epsMode << "_";
       result << "Epsilon=" << eps;

       return result.str();
   }
protected:
   void SetUp() override {
       targetDevice = ov::test::utils::DEVICE_GPU;

       InputShape inputShapes;
       ElementType netPrecision;
       std::vector<int64_t> axes;
       ngraph::op::EpsMode epsMode;
       float eps;
       std::tie(inputShapes, netPrecision, axes, epsMode, eps) = this->GetParam();

       init_input_shapes({inputShapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));
        }
       auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
       auto normalize = ngraph::builder::makeNormalizeL2(paramOuts[0], axes, eps, epsMode);

       ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(normalize)};
       function = std::make_shared<ngraph::Function>(results, params, "NormalizeL2");
   }
};

TEST_P(NormalizeL2LayerGPUTest, CompareWithRefs) {
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

const std::vector<ngraph::op::EpsMode> epsMode = {
       ngraph::op::EpsMode::ADD, ngraph::op::EpsMode::MAX
};

const std::vector<float> epsilon = {
       0.000000001
};

const std::vector<int64_t> reduction_axes_1234 = {1, 2, 3, 4};
const std::vector<int64_t> reduction_axes_234 = {2, 3, 4};
const std::vector<int64_t> reduction_axes_123 = {1, 2, 3};
const std::vector<int64_t> reduction_axes_23 = {2, 3};
const std::vector<int64_t> reduction_axes_12 = {1, 2};
const std::vector<int64_t> reduction_axes_3 = {3};
const std::vector<int64_t> reduction_axes_2 = {2};

std::vector<ElementType> nrtPrecision = {ElementType::f16, ElementType::f32};

const auto NormalizeL2_3D = ::testing::Combine(
           ::testing::ValuesIn(inputShapes_3D),
           ::testing::ValuesIn(nrtPrecision),
           ::testing::ValuesIn({reduction_axes_12, reduction_axes_2}),
           ::testing::ValuesIn(epsMode),
           ::testing::ValuesIn(epsilon));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_NormalizeL2_3D, NormalizeL2LayerGPUTest, NormalizeL2_3D, NormalizeL2LayerGPUTest::getTestCaseName);

const auto NormalizeL2_4D = ::testing::Combine(
               ::testing::ValuesIn(inputShapes_4D),
               ::testing::ValuesIn(nrtPrecision),
               ::testing::ValuesIn({reduction_axes_2, reduction_axes_3, reduction_axes_12, reduction_axes_23, reduction_axes_123}),
               ::testing::ValuesIn(epsMode),
               ::testing::ValuesIn(epsilon));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_NormalizeL2_4D, NormalizeL2LayerGPUTest, NormalizeL2_4D, NormalizeL2LayerGPUTest::getTestCaseName);

const auto NormalizeL2_5D = ::testing::Combine(
               ::testing::ValuesIn(inputShapes_5D),
               ::testing::ValuesIn(nrtPrecision),
               ::testing::ValuesIn({reduction_axes_3, reduction_axes_23, reduction_axes_123, reduction_axes_1234}),
               ::testing::ValuesIn(epsMode),
               ::testing::ValuesIn(epsilon));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_NormalizeL2_5D, NormalizeL2LayerGPUTest, NormalizeL2_5D, NormalizeL2LayerGPUTest::getTestCaseName);

} // namespace
} // namespace GPULayerTestsDefinitions
