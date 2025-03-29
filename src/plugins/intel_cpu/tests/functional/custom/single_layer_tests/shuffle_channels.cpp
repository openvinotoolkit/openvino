// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/shuffle_channels.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {

using ShuffleChannelsLayerCPUTestParamsSet = std::tuple<InputShape,   // Input shape
                                                        ElementType,  // Input element type
                                                        ov::test::shuffleChannelsSpecificParams,
                                                        CPUSpecificParams>;

class ShuffleChannelsLayerCPUTest : public testing::WithParamInterface<ShuffleChannelsLayerCPUTestParamsSet>,
                                    virtual public SubgraphBaseTest,
                                    public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ShuffleChannelsLayerCPUTestParamsSet> obj) {
        InputShape shapes;
        ElementType inType;
        ov::test::shuffleChannelsSpecificParams shuffleChannelsParams;
        CPUSpecificParams cpuParams;
        std::tie(shapes, inType, shuffleChannelsParams, cpuParams) = obj.param;
        int axis, group;
        std::tie(axis, group) = shuffleChannelsParams;

        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "Prc=" << inType << "_";
        results << "Axis=" << std::to_string(axis) << "_";
        results << "Group=" << std::to_string(group) << "_";
        results << CPUTestsBase::getTestCaseName(cpuParams);

        return results.str();
    }

protected:
    void SetUp() override {
        InputShape shapes;
        ElementType inType;
        ov::test::shuffleChannelsSpecificParams shuffleChannelsParams;
        int axis, group;
        CPUSpecificParams cpuParams;
        std::tie(shapes, inType, shuffleChannelsParams, cpuParams) = this->GetParam();
        std::tie(axis, group) = shuffleChannelsParams;

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
        auto shuffleChannels = std::make_shared<ov::op::v0::ShuffleChannels>(params[0], axis, group);
        function = makeNgraphFunction(inType, params, shuffleChannels, "ShuffleChannels");
    }
};

TEST_P(ShuffleChannelsLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "ShuffleChannels");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice4D() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx512"}, "jit_avx512"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx2"}, "jit_avx2"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc}, {nhwc}, {"jit_sse42"}, "jit_sse42"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"ref"}, "ref"});
    }
    return resCPUParams;
}

std::vector<CPUSpecificParams> filterCPUInfoForDevice5D() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{ncdhw}, {ncdhw}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_avx512"}, "jit_avx512"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{ncdhw}, {ncdhw}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_avx2"}, "jit_avx2"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{ncdhw}, {ncdhw}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_sse42"}, "jit_sse42"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{ncdhw}, {ncdhw}, {"ref"}, "ref"});
    }
    return resCPUParams;
}

std::vector<CPUSpecificParams> filterCPUInfoForDevice4DBlock() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw16c}, {nChw16c}, {"jit_avx512"}, "jit_avx512"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_avx2"}, "jit_avx2"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_sse42"}, "jit_sse42"});
    }
    return resCPUParams;
}

std::vector<CPUSpecificParams> filterCPUInfoForDevice5DBlock() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {"jit_avx512"}, "jit_avx512"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_avx2"}, "jit_avx2"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_sse42"}, "jit_sse42"});
    }
    return resCPUParams;
}
/* ========== */

const std::vector<ElementType> inputElementType = {ElementType::f32, ElementType::bf16, ElementType::i8};

const auto shuffleChannelsParams4D = ::testing::Combine(::testing::ValuesIn(std::vector<int>{-4, -2, 0, 1, 3}),
                                                        ::testing::ValuesIn(std::vector<int>{1, 2, 4}));

const auto shuffleChannelsParams5D = ::testing::Combine(::testing::ValuesIn(std::vector<int>{-5, -3, -1, 0, 1, 3}),
                                                        ::testing::ValuesIn(std::vector<int>{1, 2, 3}));

const auto shuffleChannelsParams4DBlock = ::testing::Combine(::testing::ValuesIn(std::vector<int>{-4, -2, -1, 0, 2, 3}),
                                                             ::testing::ValuesIn(std::vector<int>{1, 2, 4}));

const auto shuffleChannelsParams5DBlock =
    ::testing::Combine(::testing::ValuesIn(std::vector<int>{-5, -2, -1, 0, 2, 3, 4}),
                       ::testing::ValuesIn(std::vector<int>{1, 2, 3}));

const std::vector<InputShape> inputShapesDynamic4D = {
    {{-1, -1, -1, -1}, {{8, 4, 4, 4}, {8, 16, 8, 4}, {8, 4, 4, 4}}},

    {{-1, 8, -1, -1}, {{8, 8, 8, 8}, {8, 8, 4, 16}, {8, 8, 8, 8}}},

    {{{4, 32}, {4, 32}, {4, 32}, {4, 32}}, {{4, 12, 8, 8}, {8, 32, 12, 4}, {4, 12, 8, 8}}},
};

const std::vector<InputShape> inputShapesDynamic5D = {
    {{-1, -1, -1, -1, -1}, {{6, 6, 6, 6, 6}, {12, 6, 12, 12, 12}, {6, 6, 6, 6, 6}}},

    {{-1, 18, -1, -1, -1}, {{6, 18, 12, 6, 12}, {6, 18, 6, 6, 6}, {6, 18, 12, 6, 12}}},

    {{{6, 24}, {6, 24}, {6, 24}, {6, 24}, {6, 24}}, {{24, 12, 6, 6, 6}, {12, 24, 6, 12, 12}, {24, 12, 6, 6, 6}}},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_ShuffleChannelsStatic4D,
    ShuffleChannelsLayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation({{16, 24, 32, 40}})),
                       ::testing::ValuesIn(inputElementType),
                       shuffleChannelsParams4D,
                       ::testing::ValuesIn(filterCPUInfoForDevice4D())),
    ShuffleChannelsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannelsDynamic4D,
                         ShuffleChannelsLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDynamic4D),
                                            ::testing::ValuesIn(inputElementType),
                                            shuffleChannelsParams4D,
                                            ::testing::ValuesIn(filterCPUInfoForDevice4D())),
                         ShuffleChannelsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_ShuffleChannelsStatic5D,
    ShuffleChannelsLayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation({{6, 24, 12, 12, 6}})),
                       ::testing::ValuesIn(inputElementType),
                       shuffleChannelsParams5D,
                       ::testing::ValuesIn(filterCPUInfoForDevice5D())),
    ShuffleChannelsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannelsDynamic5D,
                         ShuffleChannelsLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDynamic5D),
                                            ::testing::ValuesIn(inputElementType),
                                            shuffleChannelsParams5D,
                                            ::testing::ValuesIn(filterCPUInfoForDevice5D())),
                         ShuffleChannelsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_ShuffleChannelsStatic4DBlock,
    ShuffleChannelsLayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation({{40, 32, 24, 16}})),
                       ::testing::ValuesIn(inputElementType),
                       shuffleChannelsParams4DBlock,
                       ::testing::ValuesIn(filterCPUInfoForDevice4DBlock())),
    ShuffleChannelsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannelsDynamic4DBlock,
                         ShuffleChannelsLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDynamic4D),
                                            ::testing::ValuesIn(inputElementType),
                                            shuffleChannelsParams4DBlock,
                                            ::testing::ValuesIn(filterCPUInfoForDevice4DBlock())),
                         ShuffleChannelsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_ShuffleChannelsStatic5DBlock,
    ShuffleChannelsLayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation({{18, 12, 18, 12, 30}})),
                       ::testing::ValuesIn(inputElementType),
                       shuffleChannelsParams5DBlock,
                       ::testing::ValuesIn(filterCPUInfoForDevice5DBlock())),
    ShuffleChannelsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannelsDynamic5DBlock,
                         ShuffleChannelsLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDynamic5D),
                                            ::testing::ValuesIn(inputElementType),
                                            shuffleChannelsParams5DBlock,
                                            ::testing::ValuesIn(filterCPUInfoForDevice5DBlock())),
                         ShuffleChannelsLayerCPUTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
