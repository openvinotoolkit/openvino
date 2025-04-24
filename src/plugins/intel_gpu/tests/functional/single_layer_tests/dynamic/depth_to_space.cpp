// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/depth_to_space.hpp"

namespace {
using ov::test::InputShape;
using ov::op::v0::DepthToSpace;

typedef std::tuple<
    InputShape,                         // Input shape
    ov::element::Type,                  // Input element type
    DepthToSpace::DepthToSpaceMode,     // Mode
    std::size_t                         // Block size
> DepthToSpaceLayerGPUTestParams;

class DepthToSpaceLayerGPUTest : public testing::WithParamInterface<DepthToSpaceLayerGPUTestParams>,
                                 virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DepthToSpaceLayerGPUTestParams> obj) {
        InputShape shapes;
        ov::element::Type inType;
        DepthToSpace::DepthToSpaceMode mode;
        std::size_t blockSize;
        std::tie(shapes, inType, mode, blockSize) = obj.param;

        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "Prc=" << inType << "_";
        switch (mode) {
            case ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST:
                results << "BLOCKS_FIRST_";
                break;
            case ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST:
                results << "DEPTH_FIRST_";
                break;
            default:
                throw std::runtime_error("Unsupported DepthToSpaceMode");
        }
        results << "BS=" << blockSize;

        return results.str();
    }

protected:
    void SetUp() override {
        InputShape shapes;
        ov::op::v0::DepthToSpace::DepthToSpaceMode mode;
        std::size_t blockSize;
        std::tie(shapes, inType, mode, blockSize) = this->GetParam();

        targetDevice = ov::test::utils::DEVICE_GPU;
        init_input_shapes({shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));

        auto d2s = std::make_shared<ov::op::v0::DepthToSpace>(params[0], mode, blockSize);

        ov::ResultVector results;
        for (size_t i = 0; i < d2s->get_output_size(); i++)
            results.push_back(std::make_shared<ov::op::v0::Result>(d2s->output(i)));
        function = std::make_shared<ov::Model>(results, params, "DepthToSpace");
    }
};

TEST_P(DepthToSpaceLayerGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> input_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i8
};

const std::vector<ov::op::v0::DepthToSpace::DepthToSpaceMode> depthToSpaceModes = {
        ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST
};

// ======================== Static Shapes Tests ========================

namespace static_shapes {

const std::vector<ov::Shape> inputShapesBS2_4D = {
        {1, 64,  1, 1},
        {1, 64,  1, 3},
        {1, 128, 3, 3},
        {2, 128, 1, 1},
        {1, 192, 2, 2},
        {2, 256, 2, 3},
        {1, 512, 2, 1}
};

const std::vector<ov::Shape> inputShapesBS3_4D = {
        {1, 27, 1, 1},
        {1, 27, 2, 3},
        {1, 18, 2, 3},
        {3, 18, 1, 1},
        {2, 18, 3, 1}
};

INSTANTIATE_TEST_SUITE_P(smoke_GPUDepthToSpaceStaticBS2_4D, DepthToSpaceLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesBS2_4D)),
                                 testing::ValuesIn(input_types),
                                 testing::ValuesIn(depthToSpaceModes),
                                 testing::Values(1, 2)),
                         DepthToSpaceLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPUDepthToSpaceStaticBS3_4D, DepthToSpaceLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesBS3_4D)),
                                 testing::ValuesIn(input_types),
                                 testing::ValuesIn(depthToSpaceModes),
                                 testing::Values(1, 3)),
                         DepthToSpaceLayerGPUTest::getTestCaseName);

const std::vector<ov::Shape> inputShapesBS2_5D = {
        {1, 128, 1, 1, 1},
        {1, 128, 2, 1, 2},
        {1, 256, 2, 1, 3},
        {2, 256, 3, 1, 1},
        {1, 384, 1, 2, 2},
        {2, 512, 1, 2, 1}
};

const std::vector<ov::Shape> inputShapesBS3_5D = {
        {1, 54, 1, 1, 1},
        {1, 54, 2, 1, 2},
        {3, 54, 1, 1, 1},
        {2, 54, 3, 1, 2},
        {1, 54, 3, 2, 2}
};

INSTANTIATE_TEST_SUITE_P(smoke_GPUDepthToSpaceStaticBS2_5D, DepthToSpaceLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesBS2_5D)),
                                 testing::ValuesIn(input_types),
                                 testing::ValuesIn(depthToSpaceModes),
                                 testing::Values(1, 2)),
                         DepthToSpaceLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPUDepthToSpaceStaticBS3_5D, DepthToSpaceLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesBS3_5D)),
                                 testing::ValuesIn(input_types),
                                 testing::ValuesIn(depthToSpaceModes),
                                 testing::Values(1, 3)),
                         DepthToSpaceLayerGPUTest::getTestCaseName);

} // namespace static_shapes

//======================== Dynamic Shapes Tests ========================

const std::vector<InputShape> inputShapes4D = {
    {{-1, -1, -1, -1},                                               // dynamic
     {{2, 36, 1, 1}, {1, 36, 3, 1}, {2, 36, 1, 1}, {1, 36, 3, 1}}},  // target

    {{-1, 576, -1, -1},                                                  // dynamic
     {{1, 576, 1, 1}, {1, 576, 2, 2}, {3, 576, 4, 1}, {1, 576, 1, 1}}},  // target

    {{{1, 5}, {36, 72}, {1, 16}, {1, 16}},                               // dynamic
     {{3, 36, 4, 4}, {1, 36, 16, 12}, {3, 72, 8, 8}, {1, 36, 16, 12}}},  // target
};

const std::vector<InputShape> inputShapes5D = {
    {{-1, -1, -1, -1, -1},  // dynamic
     {{2, 216, 1, 1, 1},
      {1, 216, 3, 1, 2},
      {1, 432, 2, 3, 1},
      {2, 216, 1, 1, 1}}},  // target

    {{{1, 3}, {216, 432}, {1, 4}, {1, 4}, {1, 4}},                // dynamic
     {{3, 216, 2, 2, 2}, {1, 432, 1, 1, 1}, {3, 216, 2, 2, 2}}},  // target
};

INSTANTIATE_TEST_SUITE_P(smoke_GPUDepthToSpaceDynamic4D, DepthToSpaceLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(inputShapes4D),
                                 testing::ValuesIn(input_types),
                                 testing::ValuesIn(depthToSpaceModes),
                                 testing::Values(1, 2, 3)),
                         DepthToSpaceLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPUDepthToSpaceDynamic5D, DepthToSpaceLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(inputShapes5D),
                                 testing::ValuesIn(input_types),
                                 testing::ValuesIn(depthToSpaceModes),
                                 testing::Values(1, 2, 3)),
                         DepthToSpaceLayerGPUTest::getTestCaseName);

} // namespace
