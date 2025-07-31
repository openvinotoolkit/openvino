// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "openvino/op/roll.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using RollCPUTestParams = typename std::tuple<
        InputShape,                  // Input shape
        ov::element::Type,           // Input precision
        std::vector<int64_t>,        // Shift
        std::vector<int64_t>,        // Axes
        std::string>;                // Device name

class RollLayerCPUTest : public testing::WithParamInterface<RollCPUTestParams>,
                         virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RollCPUTestParams> obj) {
        const auto& [inputShape, inputPrecision, shift, axes, targetDevice] = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        result << "TS=";
        for (const auto& item : inputShape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "Precision=" << inputPrecision.get_type_name() << "_";
        result << "Shift=" << ov::test::utils::vec2str(shift) << "_";
        result << "Axes=" << ov::test::utils::vec2str(axes) << "_";
        result << "TargetDevice=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [inputShape, inputPrecision, shift, axes, _targetDevice] = GetParam();
        targetDevice = _targetDevice;
        init_input_shapes({inputShape});

        ov::ParameterVector paramsIn;
        for (auto&& shape : inputDynamicShapes) {
            paramsIn.push_back(std::make_shared<ov::op::v0::Parameter>(inputPrecision, shape));
        }
        auto shiftNode =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{shift.size()}, shift)->output(0);
        auto axesNode =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes)->output(0);

        const auto roll = std::make_shared<ov::op::v7::Roll>(paramsIn[0], shiftNode, axesNode);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(roll)};
        function = std::make_shared<ov::Model>(results, paramsIn, "roll");
    }
};

TEST_P(RollLayerCPUTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<ov::element::Type> inputPrecisions = {
    ov::element::i8,
    ov::element::u8,
    ov::element::i16,
    ov::element::i32,
    ov::element::f32,
    ov::element::bf16
};

const std::vector<InputShape> data2DZeroShiftShapes = {{{}, {{17, 19}}}, {{-1, -1}, {{5, 17}, {10, 20}}}};
const std::vector<InputShape> data1DShapes = {{{}, {{12}}}, {{-1}, {{10}, {20}}}};
const std::vector<InputShape> data2DShapes = {{{}, {{100, 200}}}, {{{100, 500}, 450}, {{250, 450}, {120, 450}}}};
const std::vector<InputShape> data3DShapes = {{{}, {{2, 300, 320}}},
                                              {{2, {100, 500}, -1}, {{2, 320, 420}, {2, 500, 200}}}};
const std::vector<InputShape> data4DNegativeAxesShapes = {{{}, {{3, 11, 6, 4}}},
                                                          {{-1, -1, {5, 6}, -1}, {{5, 10, 6, 15}, {10, 20, 5, 7}}}};
const std::vector<InputShape> data5DRepeatingAxesNegativeShiftShapes = {{{}, {{2, 7, 32, 32, 5}}},
                                                                       {{2, -1, -1, -1, {2, 7}}, {{2, 5, 20, 17, 3}, {2, 10, 18, 40, 7}}}};

INSTANTIATE_TEST_SUITE_P(smoke_RollCPU_2DZeroShift, RollLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(data2DZeroShiftShapes),
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::Values(std::vector<int64_t>{0, 0}),  // Shift
                            ::testing::Values(std::vector<int64_t>{0, 1}),  // Axes
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        RollLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RollCPU_1D, RollLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(data1DShapes),
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::Values(std::vector<int64_t>{5}),     // Shift
                            ::testing::Values(std::vector<int64_t>{0}),     // Axes
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        RollLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RollCPU_2D, RollLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(data2DShapes),
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::Values(std::vector<int64_t>{50, 150}), // Shift
                            ::testing::Values(std::vector<int64_t>{0, 1}),    // Axes
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        RollLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RollCPU_3D, RollLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(data3DShapes),
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::Values(std::vector<int64_t>{160, 150}), // Shift
                            ::testing::Values(std::vector<int64_t>{1, 2}),     // Axes
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        RollLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RollCPU_4DNegativeAxes, RollLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(data4DNegativeAxesShapes),
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::Values(std::vector<int64_t>{7, 3}),   // Shift
                            ::testing::Values(std::vector<int64_t>{-3, -2}), // Axes
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        RollLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RollCPU_5DRepeatingAxesNegativeShift, RollLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(data5DRepeatingAxesNegativeShiftShapes),
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::Values(std::vector<int64_t>{4, -1, 7, 2, -5}),  // Shift
                            ::testing::Values(std::vector<int64_t>{-1, 0, 3, 3, 2}),   // Axes
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        RollLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
