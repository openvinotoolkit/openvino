// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/space_to_depth.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "ov_models/builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include <string>

using namespace ov::op::v0;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
    InputShape,                         // Input shape
    ElementType,                        // Input element type
    SpaceToDepth::SpaceToDepthMode,     // Mode
    std::size_t                         // Block size
> SpaceToDepthLayerGPUTestParams;

class SpaceToDepthLayerGPUTest : public testing::WithParamInterface<SpaceToDepthLayerGPUTestParams>,
                                 virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SpaceToDepthLayerGPUTestParams> obj) {
        InputShape shapes;
        ElementType inType;
        SpaceToDepth::SpaceToDepthMode mode;
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
            case SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST:
                results << "BLOCKS_FIRST_";
                break;
            case SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST:
                results << "DEPTH_FIRST_";
                break;
            default:
                throw std::runtime_error("Unsupported SpaceToDepthMode");
        }
        results << "BS=" << blockSize;

        return results.str();
    }

protected:
    void SetUp() override {
        InputShape shapes;
        SpaceToDepth::SpaceToDepthMode mode;
        std::size_t blockSize;
        std::tie(shapes, inType, mode, blockSize) = this->GetParam();

        targetDevice = ov::test::utils::DEVICE_GPU;
        init_input_shapes({shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));

        auto d2s = std::make_shared<ov::op::v0::SpaceToDepth>(params[0], mode, blockSize);

        ngraph::ResultVector results;
        for (size_t i = 0; i < d2s->get_output_size(); i++)
            results.push_back(std::make_shared<ngraph::opset1::Result>(d2s->output(i)));
        function = std::make_shared<ngraph::Function>(results, params, "SpaceToDepth");
    }
};

TEST_P(SpaceToDepthLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

namespace {

const std::vector<ElementType> inputElementType = {
        ElementType::f32,
        ElementType::f16,
        ElementType::i8
};

const std::vector<SpaceToDepth::SpaceToDepthMode> SpaceToDepthModes = {
        SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
        SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST
};

// ======================== Static Shapes Tests ========================
namespace static_shapes {

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
                                 testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS2_4D)),
                                 testing::ValuesIn(inputElementType),
                                 testing::ValuesIn(SpaceToDepthModes),
                                 testing::Values(1, 4)),
                         SpaceToDepthLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPUSpaceToDepthStaticBS3_4D, SpaceToDepthLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS3_4D)),
                                 testing::ValuesIn(inputElementType),
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
                                 testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS2_5D)),
                                 testing::ValuesIn(inputElementType),
                                 testing::ValuesIn(SpaceToDepthModes),
                                 testing::Values(1, 4)),
                         SpaceToDepthLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPUSpaceToDepthStaticBS3_5D, SpaceToDepthLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS3_5D)),
                                 testing::ValuesIn(inputElementType),
                                 testing::ValuesIn(SpaceToDepthModes),
                                 testing::Values(1, 3)),
                         SpaceToDepthLayerGPUTest::getTestCaseName);

} // namespace static_shapes

//======================== Dynamic Shapes Tests ========================
namespace dynamic_shapes {

const std::vector<InputShape> inputShapes4D = {
        {{-1, -1, -1, -1}, {{2, 3, 12, 24}}},
};

const std::vector<InputShape> inputShapes5D = {
        {{-1, -1, -1, -1, -1}, {{2, 3, 2, 4, 8}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_GPUSpaceToDepthDynamic4D, SpaceToDepthLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(inputShapes4D),
                                 testing::ValuesIn(inputElementType),
                                 testing::ValuesIn(SpaceToDepthModes),
                                 testing::Values(1, 2, 3)),
                         SpaceToDepthLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPUSpaceToDepthDynamic5D, SpaceToDepthLayerGPUTest,
                         testing::Combine(
                                 testing::ValuesIn(inputShapes5D),
                                 testing::ValuesIn(inputElementType),
                                 testing::ValuesIn(SpaceToDepthModes),
                                 testing::Values(1, 2)),
                         SpaceToDepthLayerGPUTest::getTestCaseName);

} // namespace dynamic_shapes

} // namespace
} // namespace GPULayerTestsDefinitions
