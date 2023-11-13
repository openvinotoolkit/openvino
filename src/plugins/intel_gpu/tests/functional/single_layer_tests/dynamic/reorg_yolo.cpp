// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/reorg_yolo.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include <string>

using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
    InputShape,     // Input Shape
    size_t,         // Stride
    ElementType,    // Network precision
    TargetDevice    // Device
> ReorgYoloGPUTestParams;

class ReorgYoloLayerGPUTest : public testing::WithParamInterface<ReorgYoloGPUTestParams>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReorgYoloGPUTestParams> obj) {
        InputShape inputShape;
        size_t stride;
        ElementType netPrecision;
        TargetDevice targetDev;
        std::tie(inputShape, stride, netPrecision, targetDev) = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        for (const auto& item : inputShape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "stride=" << stride << "_";
        result << "netPRC=" << netPrecision << "_";
        result << "targetDevice=" << targetDev << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        InputShape inputShape;
        size_t stride;
        ElementType netPrecision;
        std::tie(inputShape, stride, netPrecision, targetDevice) = this->GetParam();

        init_input_shapes({inputShape});

        auto param = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, inputDynamicShapes[0]);
        auto reorg_yolo = std::make_shared<ngraph::op::v0::ReorgYolo>(param, stride);
        function = std::make_shared<ov::Model>(std::make_shared<ngraph::opset1::Result>(reorg_yolo),
                                               ngraph::ParameterVector{param},
                                               "ReorgYolo");
    }
};

TEST_P(ReorgYoloLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
};

namespace {

const std::vector<ov::test::InputShape> inShapesDynamic1 = {
    {{{1, 2}, -1, -1, -1}, {{1, 4, 4, 4}, {1, 8, 4, 4}, {2, 8, 4, 4}}}
};

const std::vector<size_t> strides = {2, 3};

const std::vector<ov::test::InputShape> inShapesDynamic2 = {
    {{{1, 2}, -1, -1, -1}, {{1, 9, 3, 3}}}
};

const auto testCase_stride1_Dynamic = ::testing::Combine(::testing::ValuesIn(inShapesDynamic1),
                                                         ::testing::Values(strides[0]),
                                                         ::testing::Values(ov::element::f32),
                                                         ::testing::Values(ov::test::utils::DEVICE_GPU));

const auto testCase_stride2_Dynamic = ::testing::Combine(::testing::ValuesIn(inShapesDynamic2),
                                                         ::testing::Values(strides[1]),
                                                         ::testing::Values(ov::element::f32),
                                                         ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_stride1_DynamicShape, ReorgYoloLayerGPUTest,
                         testCase_stride1_Dynamic,
                         ReorgYoloLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_stride2_DynamicShape, ReorgYoloLayerGPUTest,
                         testCase_stride2_Dynamic,
                         ReorgYoloLayerGPUTest::getTestCaseName);

} // namespace
} // namespace GPULayerTestsDefinitions
