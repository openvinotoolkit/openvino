// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

using SpaceToDepthLayerCPUTestParamSet = std::tuple<InputShape,                                  // Input shape
                                                    ElementType,                                 // Input element type
                                                    ov::op::v0::SpaceToDepth::SpaceToDepthMode,  // Mode
                                                    std::size_t,                                 // Block size
                                                    CPUSpecificParams>;

class SpaceToDepthLayerCPUTest : public testing::WithParamInterface<SpaceToDepthLayerCPUTestParamSet>,
                                 virtual public ov::test::SubgraphBaseTest,
                                 public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SpaceToDepthLayerCPUTestParamSet> obj) {
        InputShape shapes;
        ElementType inType;
        ov::op::v0::SpaceToDepth::SpaceToDepthMode mode;
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
        case ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST:
            results << "BLOCKS_FIRST_";
            break;
        case ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST:
            results << "DEPTH_FIRST_";
            break;
        default:
            throw std::runtime_error("Unsupported SpaceToDepthMode");
        }
        results << "BS=" << blockSize << "_";
        results << CPUTestsBase::getTestCaseName(cpuParams);

        return results.str();
    }

protected:
    void SetUp() override {
        InputShape shapes;
        ElementType inType;
        ov::op::v0::SpaceToDepth::SpaceToDepthMode mode;
        std::size_t blockSize;
        CPUSpecificParams cpuParams;
        std::tie(shapes, inType, mode, blockSize, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        if (selectedType.empty()) {
            selectedType = getPrimitiveType();
        }
        selectedType = selectedType + "_" + ov::element::Type(inType).get_type_name();
        targetDevice = ov::test::utils::DEVICE_CPU;
        init_input_shapes({shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }
        auto d2s = std::make_shared<ov::op::v0::SpaceToDepth>(params[0], mode, blockSize);
        function = makeNgraphFunction(inType, params, d2s, "SpaceToDepthCPU");
    }
};

TEST_P(SpaceToDepthLayerCPUTest, CompareWithRefs) {
    run();
    CPUTestsBase::CheckPluginRelatedResults(compiledModel, "SpaceToDepth");
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

const std::vector<ov::op::v0::SpaceToDepth::SpaceToDepthMode> SpaceToDepthModes = {
    ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
    ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST};

/* *========================* Static Shapes Tests *========================* */

namespace static_shapes {

const std::vector<ov::Shape> inputShapesBS2_4D =
    {{1, 16, 2, 2}, {1, 16, 4, 2}, {1, 32, 6, 8}, {2, 32, 4, 6}, {2, 48, 4, 4}, {2, 64, 8, 2}};

const std::vector<ov::Shape> inputShapesBS3_4D = {{1, 2, 3, 3}, {1, 3, 3, 6}, {1, 5, 6, 3}, {2, 5, 9, 3}, {3, 5, 6, 6}};

INSTANTIATE_TEST_SUITE_P(smoke_CPUSpaceToDepthBS2_4D,
                         SpaceToDepthLayerCPUTest,
                         testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS2_4D)),
                                          testing::ValuesIn(inputElementType),
                                          testing::ValuesIn(SpaceToDepthModes),
                                          testing::Values(1, 2),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBlocked4D))),
                         SpaceToDepthLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CPUSpaceToDepthStaticBS3_4D,
                         SpaceToDepthLayerCPUTest,
                         testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS3_4D)),
                                          testing::ValuesIn(inputElementType),
                                          testing::ValuesIn(SpaceToDepthModes),
                                          testing::Values(1, 3),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParams4D))),
                         SpaceToDepthLayerCPUTest::getTestCaseName);

const std::vector<ov::Shape> inputShapesBS2_5D =
    {{1, 16, 2, 2, 2}, {1, 16, 4, 4, 2}, {1, 32, 2, 6, 2}, {2, 32, 4, 2, 2}, {1, 48, 6, 2, 2}, {2, 64, 2, 2, 6}};

const std::vector<ov::Shape> inputShapesBS3_5D = {{1, 2, 3, 3, 3},
                                                  {1, 2, 3, 6, 9},
                                                  {1, 5, 6, 3, 3},
                                                  {2, 5, 3, 9, 3},
                                                  {3, 5, 3, 3, 6}};

INSTANTIATE_TEST_SUITE_P(smoke_CPUSpaceToDepthStaticBS2_5D,
                         SpaceToDepthLayerCPUTest,
                         testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS2_5D)),
                                          testing::ValuesIn(inputElementType),
                                          testing::ValuesIn(SpaceToDepthModes),
                                          testing::Values(1, 2),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBlocked5D))),
                         SpaceToDepthLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CPUSpaceToDepthStaticBS3_5D,
                         SpaceToDepthLayerCPUTest,
                         testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS3_5D)),
                                          testing::ValuesIn(inputElementType),
                                          testing::ValuesIn(SpaceToDepthModes),
                                          testing::Values(1, 3),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParams5D))),
                         SpaceToDepthLayerCPUTest::getTestCaseName);

}  // namespace static_shapes
/* *========================* *==================* *========================* */

/* *========================* Dynamic Shapes Tests *========================* */
namespace dynamic_shapes {

const std::vector<InputShape> inputShapes4D = {
    {{-1, -1, -1, -1},                                                   // dynamic
     {{2, 12, 6, 12}, {1, 2, 18, 6}, {1, 10, 12, 24}, {2, 12, 6, 12}}},  // target

    {{-1, 32, -1, -1},                                                     // dynamic
     {{1, 32, 6, 12}, {2, 32, 24, 24}, {3, 32, 30, 6}, {2, 32, 24, 24}}},  // target

    {{{2, 5}, {1, 50}, {1, 100}, {1, 100}},                            // dynamic
     {{3, 5, 12, 36}, {2, 3, 6, 12}, {5, 2, 12, 18}, {2, 3, 6, 12}}},  // target
};

const std::vector<InputShape> inputShapes5D = {
    {{-1, -1, -1, -1, -1},                                                            // dynamic
     {{2, 2, 6, 12, 24}, {1, 4, 24, 24, 36}, {1, 7, 6, 30, 18}, {2, 2, 6, 12, 24}}},  // target

    {{{1, 3}, {5, 16}, {1, 60}, {1, 60}, {1, 60}},                // dynamic
     {{3, 5, 12, 6, 24}, {1, 6, 24, 18, 6}, {3, 5, 12, 6, 24}}},  // target
};

const std::vector<InputShape> inputShapesBlocked5D = {
    {{-1, 16, -1, -1, -1},                                                            // dynamic
     {{1, 16, 4, 6, 10}, {1, 16, 12, 8, 2}, {3, 16, 2, 14, 24}, {1, 16, 12, 8, 2}}},  // target

    {{{1, 3}, 32, {1, 32}, {1, 32}, {1, 32}},                                            // dynamic
     {{1, 32, 4, 16, 10}, {1, 32, 18, 6, 14}, {3, 32, 2, 14, 12}, {1, 32, 18, 6, 14}}},  // target
};

INSTANTIATE_TEST_SUITE_P(smoke_CPUSpaceToDepthDynamic4D,
                         SpaceToDepthLayerCPUTest,
                         testing::Combine(testing::ValuesIn(inputShapes4D),
                                          testing::ValuesIn(inputElementType),
                                          testing::ValuesIn(SpaceToDepthModes),
                                          testing::Values(1, 2, 3),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParams4D))),
                         SpaceToDepthLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CPUSpaceToDepthDynamicBlocksFirstBlocked4D,
                         SpaceToDepthLayerCPUTest,
                         testing::Combine(testing::Values(inputShapes4D[1]),
                                          testing::ValuesIn(inputElementType),
                                          testing::Values(ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST),
                                          testing::Values(1, 2, 3),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBlocked4D))),
                         SpaceToDepthLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CPUSpaceToDepthDynamicDepthFirstBlocked4D,
                         SpaceToDepthLayerCPUTest,
                         testing::Combine(testing::Values(inputShapes4D[1]),
                                          testing::ValuesIn(inputElementType),
                                          testing::Values(ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST),
                                          testing::Values(1, 2),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBlocked4D))),
                         SpaceToDepthLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CPUSpaceToDepthDynamic5D,
                         SpaceToDepthLayerCPUTest,
                         testing::Combine(testing::ValuesIn(inputShapes5D),
                                          testing::ValuesIn(inputElementType),
                                          testing::ValuesIn(SpaceToDepthModes),
                                          testing::Values(1, 2, 3),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParams5D))),
                         SpaceToDepthLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CPUSpaceToDepthDynamicCPUSpecific5D,
                         SpaceToDepthLayerCPUTest,
                         testing::Combine(testing::ValuesIn(inputShapesBlocked5D),
                                          testing::ValuesIn(inputElementType),
                                          testing::ValuesIn(SpaceToDepthModes),
                                          testing::Values(1, 2),
                                          testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBlocked5D))),
                         SpaceToDepthLayerCPUTest::getTestCaseName);
}  // namespace dynamic_shapes
/* *========================* *==================* *========================* */

}  // namespace
