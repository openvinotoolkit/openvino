// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/mvn.hpp"

namespace {
using ov::test::InputShape;

using basicGPUMvnParams = std::tuple<
       InputShape,        // Input shapes
       ov::element::Type, // Input precision
       std::vector<int>,  // Reduction axes
       bool,              // Normalize variance
       double>;           // Epsilon

using MvnLayerGPUTestParamSet = std::tuple<
       basicGPUMvnParams,
       ov::element::Type>; // CNNNetwork input precision

class MvnLayerGPUTest : public testing::WithParamInterface<MvnLayerGPUTestParamSet>,
                        virtual public ov::test::SubgraphBaseTest {
public:
   static std::string getTestCaseName(const testing::TestParamInfo<MvnLayerGPUTestParamSet>& obj) {
       const auto& [basicParamsSet, inputPrecision] = obj.param;

       const auto& [inputShapes, netPrecision, axes, normalizeVariance, eps] = basicParamsSet;

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

       const auto& [basicParamsSet, inPrc] = this->GetParam();

       const auto& [inputShapes, netPrecision, axes, normalizeVariance, eps] = basicParamsSet;

       init_input_shapes({inputShapes});

       auto axesType = ov::element::i64;
       std::string eps_mode = "inside_sqrt";

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));

       auto axesNode = std::make_shared<ov::op::v0::Constant>(axesType, ov::Shape{axes.size()}, axes);
       ov::op::MVNEpsMode nEpsMode = ov::op::MVNEpsMode::INSIDE_SQRT;
       if (eps_mode == "outside_sqrt")
           nEpsMode = ov::op::MVNEpsMode::OUTSIDE_SQRT;
       auto mvn = std::make_shared<ov::op::v6::MVN>(params[0], axesNode, normalizeVariance, eps, nEpsMode);

       rel_threshold = 0.015f;

       ov::ResultVector results;
       for (size_t i = 0; i < mvn->get_output_size(); ++i) {
           results.push_back(std::make_shared<ov::op::v0::Result>(mvn->output(i)));
       }
       function = std::make_shared<ov::Model>(results, params, "MVN");
   }
};

TEST_P(MvnLayerGPUTest, Inference) {
   run();
}

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

std::vector<ov::element::Type> inpPrc = {ov::element::i8, ov::element::f16, ov::element::f32};

const auto Mvn3D = ::testing::Combine(
       ::testing::Combine(
           ::testing::ValuesIn(inputShapes_3D),
           ::testing::Values(ov::element::f32),
           ::testing::ValuesIn({reduction_axes_12, reduction_axes_2}),
           ::testing::ValuesIn(normalizeVariance),
           ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(inpPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn3D, MvnLayerGPUTest, Mvn3D, MvnLayerGPUTest::getTestCaseName);

const auto Mvn4D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_4D),
               ::testing::Values(ov::element::f32),
               ::testing::ValuesIn({reduction_axes_2, reduction_axes_3, reduction_axes_12, reduction_axes_23, reduction_axes_123}),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(inpPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn4D, MvnLayerGPUTest, Mvn4D, MvnLayerGPUTest::getTestCaseName);

const auto Mvn5D = ::testing::Combine(
       ::testing::Combine(
               ::testing::ValuesIn(inputShapes_5D),
               ::testing::Values(ov::element::f32),
               ::testing::ValuesIn({reduction_axes_3, reduction_axes_23, reduction_axes_123, reduction_axes_1234}),
               ::testing::ValuesIn(normalizeVariance),
               ::testing::ValuesIn(epsilon)),
       ::testing::ValuesIn(inpPrc));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Mvn5D, MvnLayerGPUTest, Mvn5D, MvnLayerGPUTest::getTestCaseName);
} // namespace
