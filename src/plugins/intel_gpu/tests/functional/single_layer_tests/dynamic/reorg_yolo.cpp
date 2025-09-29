// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reorg_yolo.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
    InputShape,          // Input Shape
    size_t,              // Stride
    ov::element::Type,   // Model type
    std::string          // Device
> ReorgYoloGPUTestParams;

class ReorgYoloLayerGPUTest : public testing::WithParamInterface<ReorgYoloGPUTestParams>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReorgYoloGPUTestParams>& obj) {
        const auto& [shapes, stride, model_type, targetDev] = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        for (const auto& item : shapes.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "stride=" << stride << "_";
        result << "modelPRC=" << model_type << "_";
        result << "targetDevice=" << targetDev << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [shapes, stride, model_type, _targetDevice] = this->GetParam();
        targetDevice = _targetDevice;

        init_input_shapes({shapes});

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        auto reorg_yolo = std::make_shared<ov::op::v0::ReorgYolo>(param, stride);
        function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(reorg_yolo),
                                               ov::ParameterVector{param},
                                               "ReorgYolo");
    }
};

TEST_P(ReorgYoloLayerGPUTest, Inference) {
    run();
};

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
