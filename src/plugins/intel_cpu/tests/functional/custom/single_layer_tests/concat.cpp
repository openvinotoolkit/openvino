// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

typedef std::tuple<size_t,                   // Concat axis
                   std::vector<InputShape>,  // Input shapes
                   ElementType,              // Network precision
                   CPUSpecificParams>
    concatCPUTestParams;

class ConcatLayerCPUTest : public testing::WithParamInterface<concatCPUTestParams>,
                           virtual public SubgraphBaseTest,
                           public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<concatCPUTestParams> obj) {
        int axis;
        std::vector<InputShape> inputShapes;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(axis, inputShapes, netPrecision, cpuParams) = obj.param;

        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")_";
        }
        result << "axis=" << axis << "_";
        result << "netPRC=" << netPrecision << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        if (actual.front().get_size() == 0) {
            ASSERT_EQ(0, expected.front().get_size());
            for (const auto& shape : targetStaticShapes[inferNum]) {
                ASSERT_EQ(shape_size(shape), 0);
            }
        } else {
            SubgraphBaseTest::compare(expected, actual);
        }
        inferNum++;
    }

protected:
    size_t inferNum = 0;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        int axis;
        std::vector<InputShape> inputShape;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(axis, inputShape, netPrecision, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType += std::string("_") + ov::element::Type(netPrecision).get_type_name();

        init_input_shapes(inputShape);

        ov::ParameterVector params;
        ov::OutputVector paramsOuts;
        for (auto&& shape : inputDynamicShapes) {
            auto param = std::make_shared<ov::op::v0::Parameter>(netPrecision, shape);
            params.push_back(param);
            paramsOuts.push_back(param);
        }
        auto concat = std::make_shared<ov::op::v0::Concat>(paramsOuts, axis);

        function = makeNgraphFunction(netPrecision, params, concat, "ConcatCPU");
    }
};

TEST_P(ConcatLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Concatenation");
}

namespace {
const auto planar_4D_ref = CPUSpecificParams{{nchw}, {nchw}, {"ref"}, "ref"};
const auto planar_5D_ref = CPUSpecificParams{{ncdhw}, {ncdhw}, {"ref"}, "ref"};

const auto planar_4D = CPUSpecificParams{{nchw}, {nchw}, {}, "unknown"};
const auto planar_5D = CPUSpecificParams{{ncdhw}, {ncdhw}, {}, "unknown"};

const auto planarChannels_4D = CPUSpecificParams{{nhwc}, {nhwc}, {}, "ref"};
const auto planarChannels_5D = CPUSpecificParams{{ndhwc}, {ndhwc}, {}, "ref"};

const auto planarChannels_inplace_4D = CPUSpecificParams{{nhwc}, {nhwc}, {}, "unknown"};
const auto planarChannels_inplace_5D = CPUSpecificParams{{ndhwc}, {ndhwc}, {}, "unknown"};

const auto blocked8_4D = CPUSpecificParams{{nChw8c}, {nChw8c}, {}, "unknown"};
const auto blocked8_5D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {}, "unknown"};

const auto blocked8_4D_ref = CPUSpecificParams{{nChw8c}, {nChw8c}, {}, "ref"};
const auto blocked8_5D_ref = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {}, "ref"};

const auto blocked16_4D = CPUSpecificParams{{nChw16c}, {nChw16c}, {}, "unknown"};
const auto blocked16_5D = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {}, "unknown"};

const auto blocked16_4D_ref = CPUSpecificParams{{nChw16c}, {nChw16c}, {}, "ref"};
const auto blocked16_5D_ref = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {}, "ref"};

// List of precisions natively supported by onednn.
const std::vector<ElementType> netPrecisions = {ElementType::i8, ElementType::i32, ElementType::f32, ElementType::bf16};

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block8_static,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(1, -2, 3),
                                            ::testing::Values(static_shapes_to_test_representation({{2, 16, 3, 5},
                                                                                                    {2, 16, 3, 5}})),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planar_4D_ref, planarChannels_4D, blocked8_4D_ref)),
                         ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block16_static,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(1, 2, -1),
                                            ::testing::Values(static_shapes_to_test_representation({{3, 32, 3, 5},
                                                                                                    {3, 32, 3, 5}})),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(blocked16_4D_ref)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes4D_Block_axis1 = {
    {
        // {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
        {{-1, 32, -1, -1}, {{2, 32, 5, 7}, {1, 32, 10, 2}, {3, 32, 1, 8}}},  // input 0
        {{-1, 16, -1, -1}, {{2, 16, 5, 7}, {1, 16, 10, 2}, {3, 16, 1, 8}}},  // input 1
        {{-1, 64, -1, -1}, {{2, 64, 5, 7}, {1, 64, 10, 2}, {3, 64, 1, 8}}}   // input 2
    },
    {{{{1, 5}, 32, {1, 10}, {2, 8}}, {{2, 32, 5, 7}, {1, 32, 10, 2}, {3, 32, 1, 8}}},
     {{{1, 3}, 16, {1, 10}, {2, 8}}, {{2, 16, 5, 7}, {1, 16, 10, 2}, {3, 16, 1, 8}}},
     {{{1, 3}, 64, {1, 10}, {2, 8}}, {{2, 64, 5, 7}, {1, 64, 10, 2}, {3, 64, 1, 8}}}},
    {{{{1, 10}, 32, 2, 3}, {{2, 32, 2, 3}, {1, 32, 2, 3}}},
     {{{1, 10}, 16, 2, 3}, {{2, 16, 2, 3}, {1, 16, 2, 3}}},
     {{{1, 10}, 64, 2, 3}, {{2, 64, 2, 3}, {1, 64, 2, 3}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block_dynamic_axis_1,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(1, -3),
                                            ::testing::ValuesIn(inputShapes4D_Block_axis1),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(blocked8_4D_ref, blocked16_4D_ref)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes4D_axis1 = {
    {{{-1, -1, -1, -1},
      {{2, 32, 0, 7}, {2, 32, 5, 7}, {2, 32, 5, 7}, {1, 18, 10, 2}, {2, 32, 5, 7}, {3, 8, 1, 8}, {2, 0, 5, 7}}},
     {{-1, -1, -1, -1},
      {{2, 16, 0, 7}, {2, 16, 5, 7}, {2, 16, 5, 7}, {1, 5, 10, 2}, {2, 0, 5, 7}, {3, 3, 1, 8}, {2, 16, 5, 7}}},
     {{-1, -1, -1, -1},
      {{2, 64, 0, 7}, {2, 64, 5, 7}, {2, 0, 5, 7}, {1, 45, 10, 2}, {2, 64, 5, 7}, {3, 1, 1, 8}, {2, 64, 5, 7}}}},
    {{{{1, 3}, {8, 32}, {1, 10}, {2, 8}}, {{2, 32, 5, 7}, {1, 18, 10, 2}, {3, 8, 1, 8}}},
     {{{1, 3}, {3, 16}, {1, 10}, {2, 8}}, {{2, 16, 5, 7}, {1, 5, 10, 2}, {3, 3, 1, 8}}},
     {{{1, 3}, {1, 64}, {1, 10}, {2, 8}}, {{2, 64, 5, 7}, {1, 45, 10, 2}, {3, 1, 1, 8}}}},
    {{{{1, 18, 10, 2}}, {{1, 18, 10, 2}, {1, 18, 10, 2}}},
     {{-1, -1, -1, -1}, {{1, 3, 10, 2}, {1, 5, 10, 2}}},
     {{{1, 5, 10, 2}}, {{1, 5, 10, 2}, {1, 5, 10, 2}}}},
    {{{{-1, 8, -1, -1}}, {{2, 8, 5, 7}, {1, 8, 10, 2}}},
     {{{-1, 3, -1, -1}}, {{2, 3, 5, 7}, {1, 3, 10, 2}}},
     {{{-1, -1, -1, -1}}, {{2, 16, 5, 7}, {1, 7, 10, 2}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_dynamic_axis_1,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(1),
                                            ::testing::ValuesIn(inputShapes4D_axis1),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planar_4D_ref, planarChannels_4D)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes4D_Block_axis2 = {
    {
        {{-1, 16, -1, -1}, {{2, 16, 5, 7}, {1, 16, 16, 2}, {3, 16, 2, 8}}},
        {{-1, 16, -1, -1}, {{2, 16, 1, 7}, {1, 16, 3, 2}, {3, 16, 11, 8}}},
        {{-1, 16, -1, -1}, {{2, 16, 10, 7}, {1, 16, 5, 2}, {3, 16, 1, 8}}},
    },
    {
        {{{1, 3}, 16, {2, 16}, {2, 8}}, {{2, 16, 5, 7}, {1, 16, 16, 2}, {3, 16, 2, 8}}},
        {{{1, 3}, 16, {1, 11}, {2, 8}}, {{2, 16, 1, 7}, {1, 16, 3, 2}, {3, 16, 11, 8}}},
        {{{1, 3}, 16, {1, 10}, {2, 8}}, {{2, 16, 10, 7}, {1, 16, 5, 2}, {3, 16, 1, 8}}},
    },
    {
        {{{1, 5}, 16, 5, 7}, {{2, 16, 5, 7}, {1, 16, 5, 7}}},
        {{{1, 5}, 16, 1, 7}, {{2, 16, 1, 7}, {1, 16, 1, 7}}},
        {{{1, 5}, 16, 10, 7}, {{2, 16, 10, 7}, {1, 16, 10, 7}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block_dynamic_axis_2,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(2),
                                            ::testing::ValuesIn(inputShapes4D_Block_axis2),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(blocked8_4D_ref, blocked16_4D_ref)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes4D_axis2 = {
    {
        {{-1, -1, -1, -1}, {{2, 16, 5, 7}, {1, 16, 16, 2}, {3, 16, 2, 8}}},
        {{-1, -1, -1, -1}, {{2, 16, 1, 7}, {1, 16, 3, 2}, {3, 16, 11, 8}}},
        {{-1, -1, -1, -1}, {{2, 16, 10, 7}, {1, 16, 5, 2}, {3, 16, 1, 8}}},
    },
    {
        {{{1, 3}, {1, 16}, {2, 16}, {2, 8}}, {{2, 16, 5, 7}, {1, 16, 16, 2}, {3, 16, 2, 8}}},
        {{{1, 3}, {1, 16}, {1, 11}, {2, 8}}, {{2, 16, 1, 7}, {1, 16, 3, 2}, {3, 16, 11, 8}}},
        {{{1, 3}, {1, 16}, {1, 10}, {2, 8}}, {{2, 16, 10, 7}, {1, 16, 5, 2}, {3, 16, 1, 8}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_dynamic_axis_2,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(2, -2),
                                            ::testing::ValuesIn(inputShapes4D_axis2),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planar_4D_ref, planarChannels_4D)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes4D_Block_axis3 = {
    {
        {{-1, 32, -1, -1},
         {
             {2, 32, 4, 5},
             {1, 32, 1, 16},
             {3, 32, 7, 2},
         }},
        {{-1, 32, -1, -1}, {{2, 32, 4, 1}, {1, 32, 1, 3}, {3, 32, 7, 11}}},
        {{-1, 32, -1, -1}, {{2, 32, 4, 10}, {1, 32, 1, 5}, {3, 32, 7, 1}}},
    },
    {
        {{{1, 3}, 32, {1, 7}, {2, 16}}, {{2, 32, 4, 5}, {1, 32, 1, 16}, {3, 32, 7, 2}}},
        {{{1, 3}, 32, {1, 7}, {1, 11}}, {{2, 32, 4, 1}, {1, 32, 1, 3}, {3, 32, 7, 11}}},
        {{{1, 3}, 32, {1, 7}, {1, 10}}, {{2, 32, 4, 10}, {1, 32, 1, 5}, {3, 32, 7, 1}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block_dynamic_axis_3,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(3),
                                            ::testing::ValuesIn(inputShapes4D_Block_axis3),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(blocked8_4D_ref, blocked16_4D_ref)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes4D_axis3 = {
    {
        {{-1, -1, -1, -1}, {{2, 32, 4, 5}, {1, 32, 1, 16}, {3, 32, 7, 2}}},
        {{-1, -1, -1, -1}, {{2, 32, 4, 1}, {1, 32, 1, 3}, {3, 32, 7, 11}}},
        {{-1, -1, -1, -1}, {{2, 32, 4, 10}, {1, 32, 1, 5}, {3, 32, 7, 1}}},
    },
    {
        {{{1, 3}, {1, 32}, {1, 7}, {2, 16}}, {{2, 32, 4, 5}, {1, 32, 1, 16}, {3, 32, 7, 2}}},
        {{{1, 3}, {1, 32}, {1, 7}, {1, 11}}, {{2, 32, 4, 1}, {1, 32, 1, 3}, {3, 32, 7, 11}}},
        {{{1, 3}, {1, 32}, {1, 7}, {1, 10}}, {{2, 32, 4, 10}, {1, 32, 1, 5}, {3, 32, 7, 1}}},
    },
    {
        {{{1, 3}, 32, 4, 5}, {{1, 32, 4, 5}, {2, 32, 4, 5}}},
        {{{1, 3}, 32, 4, 1}, {{1, 32, 4, 1}, {2, 32, 4, 1}}},
        {{{1, 3}, 32, 4, 10}, {{1, 32, 4, 10}, {2, 32, 4, 10}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_dynamic_axis_3,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(3, -1),
                                            ::testing::ValuesIn(inputShapes4D_axis3),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planar_4D_ref, planarChannels_4D)),
                         ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block8_static,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(2, 3, -2),
                                            ::testing::Values(static_shapes_to_test_representation({{2, 16, 3, 5, 7},
                                                                                                    {2, 16, 3, 5, 7}})),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planar_5D_ref, planarChannels_5D, blocked8_5D_ref)),
                         ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block16_static,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(2, 3, 4),
                                            ::testing::Values(static_shapes_to_test_representation({{2, 32, 3, 5, 7},
                                                                                                    {2, 32, 3, 5, 7}})),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(blocked16_5D_ref)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes5D_Block_axis1 = {
    {
        {{-1, 32, -1, -1, -1}, {{2, 32, 5, 7, 6}, {1, 32, 10, 2, 8}, {3, 32, 1, 8, 10}}},
        {{-1, 16, -1, -1, -1}, {{2, 16, 5, 7, 6}, {1, 16, 10, 2, 8}, {3, 16, 1, 8, 10}}},
        {{-1, 64, -1, -1, -1}, {{2, 64, 5, 7, 6}, {1, 64, 10, 2, 8}, {3, 64, 1, 8, 10}}},
    },
    {
        {{{1, 3}, 32, {1, 10}, {2, 8}, {6, 10}}, {{2, 32, 5, 7, 6}, {1, 32, 10, 2, 8}, {3, 32, 1, 8, 10}}},
        {{{1, 3}, 16, {1, 10}, {2, 8}, {6, 10}}, {{2, 16, 5, 7, 6}, {1, 16, 10, 2, 8}, {3, 16, 1, 8, 10}}},
        {{{1, 3}, 64, {1, 10}, {2, 8}, {6, 10}}, {{2, 64, 5, 7, 6}, {1, 64, 10, 2, 8}, {3, 64, 1, 8, 10}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block_dynamic_axis_1,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(1),
                                            ::testing::ValuesIn(inputShapes5D_Block_axis1),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(blocked8_5D_ref, blocked16_5D_ref)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes5D_axis1 = {
    {
        {{-1, -1, -1, -1, -1}, {{2, 5, 5, 7, 6}, {1, 3, 10, 2, 8}, {3, 4, 1, 8, 10}}},
        {{-1, -1, -1, -1, -1},
         {
             {2, 16, 5, 7, 6},
             {1, 20, 10, 2, 8},
             {3, 5, 1, 8, 10},
         }},
        {{-1, -1, -1, -1, -1}, {{2, 1, 5, 7, 6}, {1, 17, 10, 2, 8}, {3, 5, 1, 8, 10}}},
    },
    {
        {{{1, 3}, {3, 5}, {1, 10}, {2, 8}, {6, 10}}, {{2, 5, 5, 7, 6}, {1, 3, 10, 2, 8}, {3, 4, 1, 8, 10}}},
        {{{1, 3}, {5, 20}, {1, 10}, {2, 8}, {4, 10}},
         {
             {2, 16, 5, 7, 6},
             {1, 20, 10, 2, 8},
             {3, 5, 1, 8, 10},
         }},
        {{{1, 3}, {1, 17}, {1, 10}, {2, 8}, {6, 10}}, {{2, 1, 5, 7, 6}, {1, 17, 10, 2, 8}, {3, 5, 1, 8, 10}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_dynamic_axis_1,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(1),
                                            ::testing::ValuesIn(inputShapes5D_axis1),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planar_5D_ref, planarChannels_5D)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes5D_Block_axis2 = {
    {
        {{-1, 16, -1, -1, -1},
         {
             {2, 16, 5, 8, 7},
             {1, 16, 16, 1, 2},
             {3, 16, 2, 5, 8},
         }},
        {{-1, 16, -1, -1, -1}, {{2, 16, 1, 8, 7}, {1, 16, 3, 1, 2}, {3, 16, 11, 5, 8}}},
        {{-1, 16, -1, -1, -1}, {{2, 16, 10, 8, 7}, {1, 16, 5, 1, 2}, {3, 16, 1, 5, 8}}},
    },
    {
        {{{1, 3}, 16, {2, 16}, {1, 8}, {2, 8}},
         {
             {2, 16, 5, 8, 7},
             {1, 16, 16, 1, 2},
             {3, 16, 2, 5, 8},
         }},
        {{{1, 5}, 16, {1, 11}, {1, 8}, {1, 8}}, {{2, 16, 1, 8, 7}, {1, 16, 3, 1, 2}, {3, 16, 11, 5, 8}}},
        {{{1, 6}, 16, {1, 10}, {1, 8}, {2, 10}}, {{2, 16, 10, 8, 7}, {1, 16, 5, 1, 2}, {3, 16, 1, 5, 8}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block_dynamic_axis_2,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(-3),
                                            ::testing::ValuesIn(inputShapes5D_Block_axis2),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(blocked8_5D_ref, blocked16_5D_ref)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes5D_axis2 = {
    {
        {{-1, -1, -1, -1, -1}, {{2, 4, 5, 8, 7}, {1, 20, 16, 1, 2}, {3, 8, 2, 5, 8}}},
        {{-1, -1, -1, -1, -1}, {{2, 4, 1, 8, 7}, {1, 20, 3, 1, 2}, {3, 8, 11, 5, 8}}},
        {{-1, -1, -1, -1, -1}, {{2, 4, 10, 8, 7}, {1, 20, 5, 1, 2}, {3, 8, 1, 5, 8}}},
    },
    {
        {{{1, 3}, {4, 20}, {1, 16}, {1, 8}, {2, 8}}, {{2, 4, 5, 8, 7}, {1, 20, 16, 1, 2}, {3, 8, 2, 5, 8}}},
        {{{1, 3}, {4, 20}, {1, 11}, {1, 10}, {1, 15}}, {{2, 4, 1, 8, 7}, {1, 20, 3, 1, 2}, {3, 8, 11, 5, 8}}},
        {{{1, 3}, {1, 20}, {1, 15}, {1, 10}, {2, 8}}, {{2, 4, 10, 8, 7}, {1, 20, 5, 1, 2}, {3, 8, 1, 5, 8}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_dynamic_axis_2,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(2),
                                            ::testing::ValuesIn(inputShapes5D_axis2),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planar_5D_ref, planarChannels_5D)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes5D_Block_axis3 = {
    {
        {{-1, 32, -1, -1, -1}, {{2, 32, 4, 5, 7}, {1, 32, 1, 16, 3}, {3, 32, 7, 2, 4}}},
        {{-1, 32, -1, -1, -1}, {{2, 32, 4, 1, 7}, {1, 32, 1, 3, 3}, {3, 32, 7, 11, 4}}},
        {{-1, 32, -1, -1, -1}, {{2, 32, 4, 10, 7}, {1, 32, 1, 5, 3}, {3, 32, 7, 1, 4}}},
    },
    {
        {{{1, 3}, 32, {1, 7}, {2, 16}, {3, 7}},
         {
             {2, 32, 4, 5, 7},
             {1, 32, 1, 16, 3},
             {3, 32, 7, 2, 4},
         }},
        {{{1, 5}, 32, {1, 7}, {1, 11}, {3, 7}}, {{2, 32, 4, 1, 7}, {1, 32, 1, 3, 3}, {3, 32, 7, 11, 4}}},
        {{{1, 6}, 32, {1, 15}, {1, 10}, {1, 20}}, {{2, 32, 4, 10, 7}, {1, 32, 1, 5, 3}, {3, 32, 7, 1, 4}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block_dynamic_axis_3,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(3),
                                            ::testing::ValuesIn(inputShapes5D_Block_axis3),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(blocked8_5D_ref, blocked16_5D_ref)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes5D_axis3 = {
    {
        {{-1, -1, -1, -1, -1}, {{2, 32, 4, 5, 7}, {1, 11, 1, 16, 3}, {3, 7, 7, 2, 4}}},
        {{-1, -1, -1, -1, -1}, {{2, 32, 4, 1, 7}, {1, 11, 1, 3, 3}, {3, 7, 7, 11, 4}}},
        {{-1, -1, -1, -1, -1}, {{2, 32, 4, 10, 7}, {1, 11, 1, 5, 3}, {3, 7, 7, 1, 4}}},
    },
    {
        {{{1, 7}, {7, 32}, {1, 7}, {1, 16}, {3, 14}},
         {
             {2, 32, 4, 5, 7},
             {1, 11, 1, 16, 3},
             {3, 7, 7, 2, 4},
         }},
        {{{1, 7}, {7, 32}, {1, 10}, {1, 11}, {3, 7}}, {{2, 32, 4, 1, 7}, {1, 11, 1, 3, 3}, {3, 7, 7, 11, 4}}},
        {{{1, 7}, {1, 32}, {1, 10}, {1, 10}, {1, 10}}, {{2, 32, 4, 10, 7}, {1, 11, 1, 5, 3}, {3, 7, 7, 1, 4}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_dynamic_axis_3,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(3),
                                            ::testing::ValuesIn(inputShapes5D_axis3),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planar_5D_ref, planarChannels_5D)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes5D_Block_axis4 = {
    {
        {{-1, 32, -1, -1, -1},
         {
             {2, 32, 4, 5, 5},
             {1, 32, 1, 1, 16},
             {3, 32, 7, 9, 2},
         }},
        {{-1, 32, -1, -1, -1}, {{2, 32, 4, 5, 1}, {1, 32, 1, 1, 3}, {3, 32, 7, 9, 11}}},
        {{-1, 32, -1, -1, -1}, {{2, 32, 4, 5, 10}, {1, 32, 1, 1, 5}, {3, 32, 7, 9, 1}}},
    },
    {
        {{{1, 15}, 32, {1, 10}, {1, 10}, {1, 16}},
         {
             {2, 32, 4, 5, 5},
             {1, 32, 1, 1, 16},
             {3, 32, 7, 9, 2},
         }},
        {{{1, 15}, 32, {1, 10}, {1, 10}, {1, 11}}, {{2, 32, 4, 5, 1}, {1, 32, 1, 1, 3}, {3, 32, 7, 9, 11}}},
        {{{1, 15}, 32, {1, 10}, {1, 10}, {1, 11}}, {{2, 32, 4, 5, 10}, {1, 32, 1, 1, 5}, {3, 32, 7, 9, 1}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_Block_dynamic_axis_4,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(4),
                                            ::testing::ValuesIn(inputShapes5D_Block_axis4),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(blocked8_5D_ref, blocked16_5D_ref)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes5D_axis4 = {
    {
        {{-1, -1, -1, -1, -1}, {{2, 1, 4, 5, 5}, {1, 4, 1, 1, 16}, {3, 14, 7, 9, 2}}},
        {{-1, -1, -1, -1, -1}, {{2, 1, 4, 5, 1}, {1, 4, 1, 1, 3}, {3, 14, 7, 9, 11}}},
        {{-1, -1, -1, -1, -1}, {{2, 1, 4, 5, 10}, {1, 4, 1, 1, 5}, {3, 14, 7, 9, 1}}},
    },
    {
        {{{1, 3}, {1, 14}, {1, 7}, {1, 10}, {2, 16}}, {{2, 1, 4, 5, 5}, {1, 4, 1, 1, 16}, {3, 14, 7, 9, 2}}},
        {{{1, 3}, {1, 14}, {1, 7}, {1, 9}, {1, 11}}, {{2, 1, 4, 5, 1}, {1, 4, 1, 1, 3}, {3, 14, 7, 9, 11}}},
        {{{1, 3}, {1, 14}, {1, 7}, {1, 9}, {1, 10}}, {{2, 1, 4, 5, 10}, {1, 4, 1, 1, 5}, {3, 14, 7, 9, 1}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat5D_CPU_dynamic_axis_4,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(4),
                                            ::testing::ValuesIn(inputShapes5D_axis4),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planar_5D_ref, planarChannels_5D)),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes_byBatch_static = {
    static_shapes_to_test_representation({{5, 2, 2, 2}, {2, 2, 2, 2}}),
    static_shapes_to_test_representation({{1, 3, 5}, {3, 3, 5}}),
    static_shapes_to_test_representation({{4, 3, 2}, {1, 3, 2}})};

const std::vector<std::vector<InputShape>> inputShapes_byBatch_dynamic = {
    // 5D
    {
        {{-1, -1, -1, -1, -1},
         {
             {10, 32, 4, 5, 5},
             {4, 7, 1, 1, 3},
             {3, 20, 7, 9, 1},
         }},
        {{-1, -1, -1, -1, -1}, {{5, 32, 4, 5, 5}, {7, 7, 1, 1, 3}, {3, 20, 7, 9, 1}}},
        {{-1, -1, -1, -1, -1}, {{1, 32, 4, 5, 5}, {1, 7, 1, 1, 3}, {6, 20, 7, 9, 1}}},
    },
    {
        {{{3, 10}, {7, 32}, {1, 9}, {1, 10}, {1, 5}},
         {
             {10, 32, 4, 5, 5},
             {4, 7, 1, 1, 3},
             {3, 20, 7, 9, 1},
         }},
        {{{3, 7}, {7, 32}, {1, 7}, {1, 9}, {1, 5}}, {{5, 32, 4, 5, 5}, {7, 7, 1, 1, 3}, {3, 20, 7, 9, 1}}},
        {{{1, 6}, {7, 32}, {1, 7}, {1, 9}, {1, 5}}, {{1, 32, 4, 5, 5}, {1, 7, 1, 1, 3}, {6, 20, 7, 9, 1}}},
    },
    // 4D
    {
        {{-1, -1, -1, -1},
         {
             {10, 32, 4, 5},
             {4, 7, 1, 1},
             {3, 20, 7, 9},
         }},
        {{-1, -1, -1, -1}, {{5, 32, 4, 5}, {7, 7, 1, 1}, {3, 20, 7, 9}}},
        {{-1, -1, -1, -1}, {{1, 32, 4, 5}, {1, 7, 1, 1}, {6, 20, 7, 9}}},
    },
    {
        {{{1, 10}, {1, 32}, {1, 7}, {1, 9}},
         {
             {10, 32, 4, 5},
             {4, 7, 1, 1},
             {3, 20, 7, 9},
         }},
        {{{3, 7}, {7, 32}, {1, 7}, {1, 9}}, {{5, 32, 4, 5}, {7, 7, 1, 1}, {3, 20, 7, 9}}},
        {{{1, 6}, {7, 32}, {1, 7}, {1, 9}}, {{1, 32, 4, 5}, {1, 7, 1, 1}, {6, 20, 7, 9}}},
    },
    {
        {{{1, 10}, 32, 4, 5}, {{10, 32, 4, 5}, {4, 32, 4, 5}}},
        {{{1, 10}, 32, 4, 5}, {{5, 32, 4, 5}, {7, 32, 4, 5}}},
        {{{1, 10}, 32, 4, 5}, {{1, 32, 4, 5}, {1, 32, 4, 5}}},
    }};

INSTANTIATE_TEST_SUITE_P(smoke_Concat_byBatch_static,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(0),
                                            ::testing::ValuesIn(inputShapes_byBatch_static),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"})),
                         ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat_byBatch_dynamic,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(0),
                                            ::testing::ValuesIn(inputShapes_byBatch_dynamic),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes3D_axis1 = {
    static_shapes_to_test_representation({{2, 4, 5}, {2, 4, 5}}),
    {
        {{-1, -1, -1},
         {
             {2, 5, 12},
             {1, 16, 1},
             {5, 2, 6},
         }},
        {{-1, -1, -1}, {{2, 1, 12}, {1, 3, 1}, {5, 11, 6}}},
        {{-1, -1, -1}, {{2, 10, 12}, {1, 5, 1}, {5, 1, 6}}},
    },
    {
        {{{1, 5}, {2, 16}, {1, 12}},
         {
             {2, 5, 12},
             {1, 16, 1},
             {5, 2, 6},
         }},
        {{{1, 5}, {1, 11}, {1, 21}}, {{2, 1, 12}, {1, 3, 1}, {5, 11, 6}}},
        {{{1, 5}, {1, 10}, {1, 12}}, {{2, 10, 12}, {1, 5, 1}, {5, 1, 6}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat_3D_axis1,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(1),
                                            ::testing::ValuesIn(inputShapes3D_axis1),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes3D_axis2 = {
    static_shapes_to_test_representation({{2, 4, 5}, {2, 4, 5}}),
    {
        {{-1, -1, -1}, {{4, 4, 5}, {3, 2, 16}, {1, 1, 2}}},
        {{-1, -1, -1}, {{4, 4, 1}, {3, 2, 3}, {1, 1, 11}}},
        {{-1, -1, -1}, {{4, 4, 10}, {3, 2, 5}, {1, 1, 1}}},
    },
    {
        {{{1, 4}, {1, 4}, {2, 16}},
         {
             {4, 4, 5},
             {3, 2, 16},
             {1, 1, 2},
         }},
        {{{1, 4}, {1, 4}, {1, 11}}, {{4, 4, 1}, {3, 2, 3}, {1, 1, 11}}},
        {{{1, 4}, {1, 4}, {1, 10}}, {{4, 4, 10}, {3, 2, 5}, {1, 1, 1}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat_3D_axis2,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(2),
                                            ::testing::ValuesIn(inputShapes3D_axis2),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes2D_axis1 = {
    static_shapes_to_test_representation({{3, 2}, {3, 10}}),
    {
        {{-1, -1},
         {
             {19, 5},
             {1, 16},
             {8, 2},
         }},
        {{-1, -1}, {{19, 1}, {1, 3}, {8, 11}}},
        {{-1, -1}, {{19, 10}, {1, 5}, {8, 1}}},
    },
    {
        {{{1, 19}, {2, 16}},
         {
             {19, 5},
             {1, 16},
             {8, 2},
         }},
        {{{1, 19}, {1, 11}}, {{19, 1}, {1, 3}, {8, 11}}},
        {{{1, 19}, {1, 10}}, {{19, 10}, {1, 5}, {8, 1}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat_2D_axis1,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(1),
                                            ::testing::ValuesIn(inputShapes2D_axis1),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                         ConcatLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes1D_static = {
    static_shapes_to_test_representation({ov::Shape{5}, ov::Shape{5}}),
    static_shapes_to_test_representation({ov::Shape{2}, ov::Shape{2}}),
    static_shapes_to_test_representation({ov::Shape{1}, ov::Shape{1}}),
    static_shapes_to_test_representation({ov::Shape{3}, ov::Shape{3}})};

const std::vector<std::vector<InputShape>> inputShapes1D_dynamic = {
    {
        {{-1}, {{19}, {8}, {5}}},
        {{-1}, {{19}, {8}, {5}}},
        {{-1}, {{19}, {8}, {5}}},
    },
    {
        {{{1, 20}}, {{19}, {8}, {5}}},
        {{{1, 20}}, {{19}, {8}, {5}}},
        {{{1, 20}}, {{19}, {8}, {5}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat_1D_static,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(0),
                                            ::testing::ValuesIn(inputShapes1D_static),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"})),
                         ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat_1D_dynamic,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(0),
                                            ::testing::ValuesIn(inputShapes1D_dynamic),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                         ConcatLayerCPUTest::getTestCaseName);

// ============================================== inPlace cases ============================================
INSTANTIATE_TEST_SUITE_P(concat_Concat4D_CPU_Block8inPlace,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(0, 1),
                                            ::testing::Values(
                                                std::vector<InputShape>{
                                                    {{}, {{1, 16, 5, 7}}},
                                                    {{}, {{1, 16, 5, 7}}},
                                                    {{}, {{1, 16, 5, 7}}},
                                                },
                                                std::vector<InputShape>{
                                                    {{1, 16, -1, -1}, {{1, 16, 5, 7}, {1, 16, 16, 2}, {1, 16, 2, 8}}},
                                                    {{1, 16, -1, -1}, {{1, 16, 5, 7}, {1, 16, 16, 2}, {1, 16, 2, 8}}},
                                                    {{1, 16, -1, -1}, {{1, 16, 5, 7}, {1, 16, 16, 2}, {1, 16, 2, 8}}},
                                                }),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(planar_4D, blocked8_4D)),
                         ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block16inPlace_0,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(0),
                                            ::testing::Values(
                                                std::vector<InputShape>{
                                                    {{}, {{1, 32, 5, 7}}},
                                                    {{}, {{1, 32, 5, 7}}},
                                                    {{}, {{1, 32, 5, 7}}},
                                                },
                                                std::vector<InputShape>{
                                                    {{1, 32, -1, -1}, {{1, 32, 5, 7}, {1, 32, 16, 2}, {1, 32, 2, 8}}},
                                                    {{1, 32, -1, -1}, {{1, 32, 5, 7}, {1, 32, 16, 2}, {1, 32, 2, 8}}},
                                                    {{1, 32, -1, -1}, {{1, 32, 5, 7}, {1, 32, 16, 2}, {1, 32, 2, 8}}},
                                                }),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(blocked16_4D)),
                         ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat4D_CPU_Block16inPlace_1,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(1),
                                            ::testing::Values(
                                                std::vector<InputShape>{
                                                    {{}, {{1, 32, 5, 7}}},
                                                    {{}, {{1, 16, 5, 7}}},
                                                    {{}, {{1, 32, 5, 7}}},
                                                },
                                                std::vector<InputShape>{
                                                    {{1, 32, -1, -1}, {{1, 32, 5, 7}, {1, 32, 16, 2}, {1, 32, 2, 8}}},
                                                    {{1, 16, -1, -1}, {{1, 16, 5, 7}, {1, 16, 16, 2}, {1, 16, 2, 8}}},
                                                    {{1, 32, -1, -1}, {{1, 32, 5, 7}, {1, 32, 16, 2}, {1, 32, 2, 8}}},
                                                }),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(blocked16_4D)),
                         ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    concat_Concat5D_CPU_Block8inPlace,
    ConcatLayerCPUTest,
    ::testing::Combine(::testing::Values(0, 1),
                       ::testing::Values(
                           std::vector<InputShape>{
                               {{}, {{1, 16, 3, 5, 7}}},
                               {{}, {{1, 16, 3, 5, 7}}},
                               {{}, {{1, 16, 3, 5, 7}}},
                           },
                           std::vector<InputShape>{
                               {{1, 32, -1, -1, -1}, {{1, 32, 5, 7, 3}, {1, 32, 16, 2, 3}, {1, 32, 2, 8, 3}}},
                               {{1, 32, -1, -1, -1}, {{1, 32, 5, 7, 3}, {1, 32, 16, 2, 3}, {1, 32, 2, 8, 3}}},
                               {{1, 32, -1, -1, -1}, {{1, 32, 5, 7, 3}, {1, 32, 16, 2, 3}, {1, 32, 2, 8, 3}}},
                           }),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(planar_5D, blocked8_5D)),
    ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Concat5D_CPU_Block16inPlace,
    ConcatLayerCPUTest,
    ::testing::Combine(::testing::Values(0, 1),
                       ::testing::Values(
                           std::vector<InputShape>{
                               {{}, {{1, 32, 3, 5, 7}}},
                               {{}, {{1, 32, 3, 5, 7}}},
                               {{}, {{1, 32, 3, 5, 7}}},
                           },
                           std::vector<InputShape>{
                               {{1, 32, -1, -1, -1}, {{1, 32, 5, 7, 3}, {1, 32, 16, 2, 3}, {1, 32, 2, 8, 3}}},
                               {{1, 32, -1, -1, -1}, {{1, 32, 5, 7, 3}, {1, 32, 16, 2, 3}, {1, 32, 2, 8, 3}}},
                               {{1, 32, -1, -1, -1}, {{1, 32, 5, 7, 3}, {1, 32, 16, 2, 3}, {1, 32, 2, 8, 3}}},
                           }),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(blocked16_5D)),
    ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat_inPlace,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(0, 1, 2, -1),
                                            ::testing::ValuesIn(std::vector<std::vector<InputShape>>{
                                                static_shapes_to_test_representation({{1, 1, 1, 10}, {1, 1, 1, 10}}),
                                                static_shapes_to_test_representation({{1, 1, 5}, {1, 1, 5}})}),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"})),
                         ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat_CPU_planarChannels_inplace_4D_static,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(1),
                                            ::testing::Values(static_shapes_to_test_representation({{1, 32, 1, 1},
                                                                                                    {1, 32, 1, 1}})),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planarChannels_inplace_4D)),
                         ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat_CPU_planarChannels_inplace_4D_sp_w_static,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(2),
                                            ::testing::Values(static_shapes_to_test_representation({{1, 1, 32, 32},
                                                                                                    {1, 1, 32, 32}})),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planarChannels_inplace_4D)),
                         ConcatLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat_CPU_planarChannels_inplace_5D_static,
                         ConcatLayerCPUTest,
                         ::testing::Combine(::testing::Values(1),
                                            ::testing::Values(static_shapes_to_test_representation({{1, 32, 1, 1, 1},
                                                                                                    {1, 32, 1, 1, 1}})),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planarChannels_inplace_5D)),
                         ConcatLayerCPUTest::getTestCaseName);
}  // namespace

}  // namespace test
}  // namespace ov
