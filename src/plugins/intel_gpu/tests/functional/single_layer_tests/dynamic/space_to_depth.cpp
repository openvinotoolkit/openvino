// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/space_to_depth.hpp"

namespace {
using ov::test::InputShape;
using ov::op::v0::SpaceToDepth;

typedef std::tuple<
    InputShape,                      // Input shape
    ov::element::Type,               // Input element type
    SpaceToDepth::SpaceToDepthMode,  // Mode
    std::size_t                      // Block size
> SpaceToDepthLayerGPUTestParams;

class SpaceToDepthLayerGPUTest : public testing::WithParamInterface<SpaceToDepthLayerGPUTestParams>,
                                 virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SpaceToDepthLayerGPUTestParams> obj) {
        const auto& [shapes, model_type, mode, block_size] = obj.param;

        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "Prc=" << model_type << "_";
        switch (mode) {
            case SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST:
                results << "BLOCKS_FIRST_";
                break;
            case SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST:
                results << "DEPTH_FIRST_";
                break;
            default:
                throw std::runtime_error("Unsupported SpaceToDepthMode");
        }
        results << "BS=" << block_size;

        return results.str();
    }

protected:
    void SetUp() override {
        const auto& [shapes, model_type, mode, block_size] = this->GetParam();

        targetDevice = ov::test::utils::DEVICE_GPU;
        init_input_shapes({shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));

        auto d2s = std::make_shared<ov::op::v0::SpaceToDepth>(params[0], mode, block_size);

        ov::ResultVector results;
        for (size_t i = 0; i < d2s->get_output_size(); i++)
            results.push_back(std::make_shared<ov::op::v0::Result>(d2s->output(i)));
        function = std::make_shared<ov::Model>(results, params, "SpaceToDepth");
    }
};

TEST_P(SpaceToDepthLayerGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i8
};

const std::vector<SpaceToDepth::SpaceToDepthMode> SpaceToDepthModes = {
        SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
        SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST
};

// ======================== Static Shapes Tests ========================

const std::vector<ov::Shape> inputShapesBS2_4D = {
        {1, 16,  8,  8},
        {1, 5,   8, 16},
        {2, 1,   8,  4},
        {1, 32,  4,  4},
        {2, 16, 64,  8}
};

const std::vector<ov::Shape> inputShapesBS3_4D = {
        {1,  1, 27, 27},
        {1,  4,  9, 18},
        {3, 18,  9,  9},
        {2, 18,  3,  3}
};

INSTANTIATE_TEST_SUITE_P(smoke_GPUSpaceToDepthStaticBS2_4D, SpaceToDepthLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesBS2_4D)),
                                 testing::ValuesIn(model_types),
                                 testing::ValuesIn(SpaceToDepthModes),
                                 testing::Values(1, 4)),
                         SpaceToDepthLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPUSpaceToDepthStaticBS3_4D, SpaceToDepthLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesBS3_4D)),
                                 testing::ValuesIn(model_types),
                                 testing::ValuesIn(SpaceToDepthModes),
                                 testing::Values(1, 3)),
                         SpaceToDepthLayerGPUTest::getTestCaseName);

const std::vector<ov::Shape> inputShapesBS2_5D = {
        {1,  1,  4, 8,   8},
        {2,  5,  4, 8,   4},
        {4,  1,  4, 4,   4},
        {2,  1,  4, 20,  4},
        {2, 16,  4, 32,  4}
};

const std::vector<ov::Shape> inputShapesBS3_5D = {
        {1,  3,  3,  6,  6},
        {1,  1,  9, 27, 27},
        {2,  1,  3,  3,  3},
        {1,  3, 27, 27, 27}
};

INSTANTIATE_TEST_SUITE_P(smoke_GPUSpaceToDepthStaticBS2_5D, SpaceToDepthLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesBS2_5D)),
                                 testing::ValuesIn(model_types),
                                 testing::ValuesIn(SpaceToDepthModes),
                                 testing::Values(1, 4)),
                         SpaceToDepthLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPUSpaceToDepthStaticBS3_5D, SpaceToDepthLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesBS3_5D)),
                                 testing::ValuesIn(model_types),
                                 testing::ValuesIn(SpaceToDepthModes),
                                 testing::Values(1, 3)),
                         SpaceToDepthLayerGPUTest::getTestCaseName);


//======================== Dynamic Shapes Tests ========================

const std::vector<InputShape> inputShapes4D = {
        {{-1, -1, -1, -1}, {{2, 3, 12, 24}}},
};

const std::vector<InputShape> inputShapes5D = {
        {{-1, -1, -1, -1, -1}, {{2, 3, 2, 4, 8}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_GPUSpaceToDepthDynamic4D, SpaceToDepthLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(inputShapes4D),
                                 testing::ValuesIn(model_types),
                                 testing::ValuesIn(SpaceToDepthModes),
                                 testing::Values(1, 2, 3)),
                         SpaceToDepthLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPUSpaceToDepthDynamic5D, SpaceToDepthLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(inputShapes5D),
                                 testing::ValuesIn(model_types),
                                 testing::ValuesIn(SpaceToDepthModes),
                                 testing::Values(1, 2)),
                         SpaceToDepthLayerGPUTest::getTestCaseName);

} // namespace
