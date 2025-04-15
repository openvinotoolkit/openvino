// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using ReorgYoloCPUParamsTuple = typename std::tuple<InputShape,     // Input Shape
                                                    size_t,         // stride
                                                    ElementType,    // Network precision
                                                    TargetDevice>;  // Device

class ReorgYoloLayerCPUTest : public testing::WithParamInterface<ReorgYoloCPUParamsTuple>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReorgYoloCPUParamsTuple>& obj) {
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

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        auto reorg_yolo = std::make_shared<ov::op::v0::ReorgYolo>(param, stride);
        function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(reorg_yolo),
                                               ov::ParameterVector{param},
                                               "ReorgYolo");
    }
};

TEST_P(ReorgYoloLayerCPUTest, CompareWithRefs) {
    run();
};

const std::vector<ov::test::InputShape> inShapesDynamic = {
    {{{1, 2}, -1, -1, -1}, {{1, 4, 4, 4}, {1, 8, 4, 4}, {2, 8, 4, 4}}}};

const std::vector<size_t> strides = {2, 3};

const std::vector<ov::test::InputShape> inShapesDynamic2 = {{{{1, 2}, -1, -1, -1}, {{1, 9, 3, 3}}}};

const auto testCase_stride2_Dynamic = ::testing::Combine(::testing::ValuesIn(inShapesDynamic),
                                                         ::testing::Values(strides[0]),
                                                         ::testing::Values(ov::element::f32),
                                                         ::testing::Values(ov::test::utils::DEVICE_CPU));

const auto testCase_stride3_Dynamic = ::testing::Combine(::testing::ValuesIn(inShapesDynamic2),
                                                         ::testing::Values(strides[1]),
                                                         ::testing::Values(ov::element::f32),
                                                         ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_stride2_DynamicShape,
                         ReorgYoloLayerCPUTest,
                         testCase_stride2_Dynamic,
                         ReorgYoloLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_stride3_DynamicShape,
                         ReorgYoloLayerCPUTest,
                         testCase_stride3_Dynamic,
                         ReorgYoloLayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
