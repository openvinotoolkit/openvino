// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/normalize_l2.hpp"

namespace {
using ov::test::InputShape;

using NormalizeL2LayerGPUTestParams = std::tuple<
       InputShape,              // Input shapes
       ov::element::Type,       // Input precision
       std::vector<int64_t>,    // Reduction axes
       ov::op::EpsMode,     // EpsMode
       float>;                  // Epsilon

class NormalizeL2LayerGPUTest : public testing::WithParamInterface<NormalizeL2LayerGPUTestParams>,
                                virtual public ov::test::SubgraphBaseTest {
public:
   static std::string getTestCaseName(const testing::TestParamInfo<NormalizeL2LayerGPUTestParams>& obj) {
       const auto& [inputShapes, netPrecision, axes, epsMode, eps] = obj.param;

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

       const auto& [inputShapes, netPrecision, axes, epsMode, eps] = this->GetParam();

       init_input_shapes({inputShapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));

       auto normAxes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);
       auto normalize = std::make_shared<ov::op::v0::NormalizeL2>(params[0], normAxes, eps, epsMode);

       ov::ResultVector results{std::make_shared<ov::op::v0::Result>(normalize)};
       function = std::make_shared<ov::Model>(results, params, "NormalizeL2");
   }
};

TEST_P(NormalizeL2LayerGPUTest, Inference) {
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

const std::vector<ov::op::EpsMode> epsMode = {
       ov::op::EpsMode::ADD, ov::op::EpsMode::MAX
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

std::vector<ov::element::Type> nrtPrecision = {ov::element::f16, ov::element::f32};

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
