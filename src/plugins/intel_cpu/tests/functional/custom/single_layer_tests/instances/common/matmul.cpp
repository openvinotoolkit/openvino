// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/matmul.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace MatMul {
/* ============= MatMul ============= */
namespace matmul {

const std::vector<ShapeRelatedParams> IS = {
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, false}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, false}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, true}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, true}},

    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {false, false}},
    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {true, false}},
    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {false, true}},
    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {true, true}},

    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {false, false}},
    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {true, false}},
    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {false, true}},
    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {true, true}},
};

const std::vector<ShapeRelatedParams> IS_Dynamic = {
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}, {33, 0}, {0, 33}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}, {0, 33}, {33, 0}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 16 }, {{ 4, 16 }, { 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 12, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {false, false}
    },
};

const std::vector<ShapeRelatedParams> IS_Dynamic_nightly = {
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{5, 15}, {1, 12}, {4, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 13}, {3, 15}, {1, 10}}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ {2, 10}, {3, 15}, -1, 16 }, {{ 2, 12, 4, 16 }, { 3, 12, 2, 16 }}}, // input 0
            {{ 1, 1, -1, 4 }, {{ 1, 1, 16, 4 }, { 1, 1, 16, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ 1, 1, -1, 16 }, {{ 1, 1, 4, 16 }, { 1, 1, 2, 16 }}}, // input 0
            {{ {2, 5}, {3, 15}, -1, 4 }, {{ 2, 12, 16, 4 }, { 2, 12, 16, 4 }}}  // input 1
        },
        {false, false}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 16 }, {{ 4, 16 }, { 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, {2, 15}, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ -1, 4 }, {{ 16, 4 }, { 16, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, {1, 15}, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ -1, 4 }, {{ 16, 4 }, { 16, 4 }}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ {1, 3}, {1, 9}, {1, 5}, {1, 10} }, {{ 1, 7, 4, 5 }, { 1, 7, 4, 4 }}}, // input 0
            {{ {1, 5}, {1, 7}, {1, 8}, {1, 5} }, {{ 1, 7, 5, 4 }, { 1, 7, 4, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ {1, 3}, {1, 9}, {1, 5}, {1, 10} }, {{ 1, 7, 4, 5 }, { 1, 7, 4, 4 }}}, // input 0
            {{ {1, 5}, {1, 7}, {1, 8}, {1, 5} }, {{ 1, 7, 5, 4 }, { 1, 7, 4, 4 }}}  // input 1
        },
        {false, false}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ 1, 7, 4, -1 }, {{ 1, 7, 4, 5 }, { 1, 7, 4, 4 }}}, // input 0
            {{ 1, 7, -1, 4 }, {{ 1, 7, 5, 4 }, { 1, 7, 4, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ 1, 7, 4, -1 }, {{ 1, 7, 4, 5 }, { 1, 7, 4, 4 }}}, // input 0
            {{ 1, 7, -1, 4 }, {{ 1, 7, 5, 4 }, { 1, 7, 4, 4 }}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 12, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 12, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {true, false}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 12, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {false, true}
    },
};

const auto matMulParams = ::testing::Combine(::testing::ValuesIn(IS),
                                             ::testing::ValuesIn(netPRCs()),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(utils::InputLayerType::PARAMETER),
                                             ::testing::Values(ov::test::utils::DEVICE_CPU),
                                             ::testing::ValuesIn(additionalConfig()));

const auto testParams = ::testing::Combine(matMulParams,
                                           ::testing::Values(MatMulNodeType::MatMul),
                                           ::testing::ValuesIn(matmulFusingParams()),
                                           ::testing::ValuesIn(filterCPUInfo(filterSpecificParams())));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Static, MatMulLayerCPUTest, testParams, MatMulLayerCPUTest::getTestCaseName);

const auto matMulParamsDynamic = ::testing::Combine(::testing::ValuesIn(IS_Dynamic),
                                             ::testing::ValuesIn(netPRCs()),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(utils::InputLayerType::PARAMETER),
                                             ::testing::Values(ov::test::utils::DEVICE_CPU),
                                             ::testing::ValuesIn(additionalConfig()));

const auto testParamsDynamic = ::testing::Combine(matMulParamsDynamic,
                                           ::testing::Values(MatMulNodeType::MatMul),
                                           ::testing::Values(emptyFusingSpec),
                                           ::testing::ValuesIn(filterCPUInfo(filterSpecificParams())));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Dynamic, MatMulLayerCPUTest, testParamsDynamic, MatMulLayerCPUTest::getTestCaseName);

const auto matMulParamsDynamic_nightly = ::testing::Combine(::testing::ValuesIn(IS_Dynamic_nightly),
                                             ::testing::ValuesIn(netPRCs()),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(utils::InputLayerType::PARAMETER),
                                             ::testing::Values(ov::test::utils::DEVICE_CPU),
                                             ::testing::ValuesIn(additionalConfig()));

const auto testParamsDynamic_nightly = ::testing::Combine(matMulParamsDynamic_nightly,
                                           ::testing::Values(MatMulNodeType::MatMul),
                                           ::testing::Values(emptyFusingSpec),
                                           ::testing::ValuesIn(filterCPUInfo(filterSpecificParams())));

INSTANTIATE_TEST_SUITE_P(nightly_MM_Dynamic, MatMulLayerCPUTest, testParamsDynamic_nightly, MatMulLayerCPUTest::getTestCaseName);

}  // namespace matmul
}  // namespace MatMul
}  // namespace test
}  // namespace ov