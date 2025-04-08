// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {

using DepthToSpaceLayerCPUTestParamSet = std::tuple<InputShape,                                  // Input shape
                                                    ElementType,                                 // Input element type
                                                    ov::op::v0::DepthToSpace::DepthToSpaceMode,  // Mode
                                                    std::size_t,                                 // Block size
                                                    CPUSpecificParams>;

class DepthToSpaceLayerCPUTest : public testing::WithParamInterface<DepthToSpaceLayerCPUTestParamSet>,
                                 virtual public ov::test::SubgraphBaseTest,
                                 public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DepthToSpaceLayerCPUTestParamSet> obj) {
        InputShape shapes;
        ElementType inType;
        ov::op::v0::DepthToSpace::DepthToSpaceMode mode;
        std::size_t blockSize;
        CPUSpecificParams cpuParams;
        std::tie(shapes, inType, mode, blockSize, cpuParams) = obj.param;

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
        results << "BS=" << blockSize << "_";
        results << CPUTestsBase::getTestCaseName(cpuParams);

        return results.str();
    }

protected:
    void SetUp() override {
        InputShape shapes;
        ov::op::v0::DepthToSpace::DepthToSpaceMode mode;
        std::size_t blockSize;
        CPUSpecificParams cpuParams;
        std::tie(shapes, inType, mode, blockSize, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        if (selectedType.empty()) {
            selectedType = getPrimitiveType();
        }
        selectedType = makeSelectedTypeStr(selectedType, inType);
        targetDevice = ov::test::utils::DEVICE_CPU;
        init_input_shapes({shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }
        auto d2s = std::make_shared<ov::op::v0::DepthToSpace>(params[0], mode, blockSize);
        function = makeNgraphFunction(inType, params, d2s, "DepthToSpace");
    }
};

TEST_P(DepthToSpaceLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "DepthToSpace");
}

namespace {

const auto cpuParams_nChw16c = CPUSpecificParams{{nChw16c}, {nChw16c}, {}, {}};
const auto cpuParams_nCdhw16c = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {}, {}};

const auto cpuParams_nChw8c = CPUSpecificParams{{nChw8c}, {nChw8c}, {}, {}};
const auto cpuParams_nCdhw8c = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {}, {}};

const auto cpuParams_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {}, {}};
const auto cpuParams_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {}, {}};

const auto cpuParams_nchw = CPUSpecificParams{{nchw}, {nchw}, {}, {}};
const auto cpuParams_ncdhw = CPUSpecificParams{{ncdhw}, {ncdhw}, {}, {}};

const std::vector<CPUSpecificParams> CPUParams4D = {cpuParams_nhwc, cpuParams_nchw};

const std::vector<CPUSpecificParams> CPUParamsBlocked4D = {cpuParams_nChw16c, cpuParams_nChw8c, cpuParams_nhwc};

const std::vector<CPUSpecificParams> CPUParams5D = {cpuParams_ndhwc, cpuParams_ncdhw};

const std::vector<CPUSpecificParams> CPUParamsBlocked5D = {cpuParams_nCdhw16c, cpuParams_nCdhw8c, cpuParams_ndhwc};

const std::vector<ElementType> inputElementType = {ElementType::f32, ElementType::bf16, ElementType::i8};

const std::vector<ov::op::v0::DepthToSpace::DepthToSpaceMode> depthToSpaceModes = {
    ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
    ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST};

/* *========================* Static Shapes Tests *========================* */

namespace static_shapes {

const std::vector<ov::Shape> inputShapesBS2_4D =
    {{1, 64, 1, 1}, {1, 64, 1, 3}, {1, 128, 3, 3}, {2, 128, 1, 1}, {1, 192, 2, 2}, {2, 256, 2, 3}, {1, 512, 2, 1}};

const std::vector<ov::Shape> inputShapesBS3_4D = {{1, 27, 1, 1},
                                                  {1, 27, 2, 3},
                                                  {1, 18, 2, 3},
                                                  {3, 18, 1, 1},
                                                  {2, 18, 3, 1}};

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceBS2_4D,
                         DepthToSpaceLayerCPUTest,
                         testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS2_4D)),
                                          testing::ValuesIn(inputElementType),
                                          testing::ValuesIn(depthToSpaceModes),
                                          testing::Values(1, 2),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBlocked4D))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceStaticBS3_4D,
                         DepthToSpaceLayerCPUTest,
                         testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS3_4D)),
                                          testing::ValuesIn(inputElementType),
                                          testing::ValuesIn(depthToSpaceModes),
                                          testing::Values(1, 3),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParams4D))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

const std::vector<ov::Shape> inputShapesBS2_5D =
    {{1, 128, 1, 1, 1}, {1, 128, 2, 1, 2}, {1, 256, 2, 1, 3}, {2, 256, 3, 1, 1}, {1, 384, 1, 2, 2}, {2, 512, 1, 2, 1}};

const std::vector<ov::Shape> inputShapesBS3_5D = {{1, 54, 1, 1, 1},
                                                  {1, 54, 2, 1, 2},
                                                  {3, 54, 1, 1, 1},
                                                  {2, 54, 3, 1, 2},
                                                  {1, 54, 3, 2, 2}};

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceStaticBS2_5D,
                         DepthToSpaceLayerCPUTest,
                         testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS2_5D)),
                                          testing::ValuesIn(inputElementType),
                                          testing::ValuesIn(depthToSpaceModes),
                                          testing::Values(1, 2),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBlocked5D))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceStaticBS3_5D,
                         DepthToSpaceLayerCPUTest,
                         testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS3_5D)),
                                          testing::ValuesIn(inputElementType),
                                          testing::ValuesIn(depthToSpaceModes),
                                          testing::Values(1, 3),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParams5D))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

}  // namespace static_shapes
/* *========================* *==================* *========================* */

/* *========================* Dynamic Shapes Tests *========================* */
namespace dynamic_shapes {

const std::vector<InputShape> inputShapes4D = {
    {{-1, -1, -1, -1},                                               // dynamic
     {{2, 36, 1, 1}, {1, 36, 3, 1}, {2, 36, 1, 1}, {1, 36, 3, 1}}},  // target

    {{-1, 576, -1, -1},                                                  // dynamic
     {{1, 576, 1, 1}, {1, 576, 2, 2}, {3, 576, 4, 1}, {1, 576, 1, 1}}},  // target

    {{{1, 5}, {36, 72}, {1, 16}, {1, 16}},                               // dynamic
     {{3, 36, 4, 4}, {1, 36, 16, 12}, {3, 72, 8, 8}, {1, 36, 16, 12}}},  // target
};

const std::vector<InputShape> inputShapes5D = {
    {{-1, -1, -1, -1, -1},                                                           // dynamic
     {{2, 216, 1, 1, 1}, {1, 216, 3, 1, 2}, {1, 432, 2, 3, 1}, {2, 216, 1, 1, 1}}},  // target

    {{{1, 3}, {216, 432}, {1, 4}, {1, 4}, {1, 4}},                // dynamic
     {{3, 216, 2, 2, 2}, {1, 432, 1, 1, 1}, {3, 216, 2, 2, 2}}},  // target
};

const std::vector<InputShape> inputShapesBlocked5D = {
    {{-1, 256, -1, -1, -1},                                                          // dynamic
     {{1, 256, 1, 1, 1}, {1, 256, 2, 1, 4}, {3, 256, 4, 1, 2}, {1, 256, 1, 1, 1}}},  // target

    {{{1, 3}, 256, {1, 3}, {1, 3}, {1, 3}},                                          // dynamic
     {{1, 256, 1, 1, 1}, {1, 256, 2, 1, 3}, {3, 256, 3, 1, 2}, {1, 256, 2, 1, 3}}},  // target
};

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceDynamic4D,
                         DepthToSpaceLayerCPUTest,
                         testing::Combine(testing::ValuesIn(inputShapes4D),
                                          testing::ValuesIn(inputElementType),
                                          testing::ValuesIn(depthToSpaceModes),
                                          testing::Values(1, 2, 3),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParams4D))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceDynamicBlocksFirstBlocked4D,
                         DepthToSpaceLayerCPUTest,
                         testing::Combine(testing::Values(inputShapes4D[1]),
                                          testing::ValuesIn(inputElementType),
                                          testing::Values(ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST),
                                          testing::Values(1, 2, 3),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBlocked4D))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceDynamicDepthFirstBlocked4D,
                         DepthToSpaceLayerCPUTest,
                         testing::Combine(testing::Values(inputShapes4D[1]),
                                          testing::ValuesIn(inputElementType),
                                          testing::Values(ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST),
                                          testing::Values(1, 2),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBlocked4D))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceDynamic5D,
                         DepthToSpaceLayerCPUTest,
                         testing::Combine(testing::ValuesIn(inputShapes5D),
                                          testing::ValuesIn(inputElementType),
                                          testing::ValuesIn(depthToSpaceModes),
                                          testing::Values(1, 2, 3),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParams5D))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceDynamicCPUSpecific5D,
                         DepthToSpaceLayerCPUTest,
                         testing::Combine(testing::ValuesIn(inputShapesBlocked5D),
                                          testing::ValuesIn(inputElementType),
                                          testing::ValuesIn(depthToSpaceModes),
                                          testing::Values(1, 2),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBlocked5D))),
                         DepthToSpaceLayerCPUTest::getTestCaseName);

}  // namespace dynamic_shapes
/* *========================* *==================* *========================* */

}  // namespace
}  // namespace test
}  // namespace ov
