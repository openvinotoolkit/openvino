// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/matmul.hpp"
#include "shared_test_classes/single_layer/mat_mul.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace MatMul {
/* ============= Common params ============= */

std::vector<std::map<std::string, std::string>> additionalConfig {
#ifndef OV_CPU_WITH_MLAS
    // FP32 precision is covered by MLAS
    std::map<std::string, std::string>{/* empty config */},
#endif
    {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}}
};



/* ============= FullyConnected ============= */
namespace fullyConnected {

const std::vector<ShapeRelatedParams> IS2D_Brgconv1x1_smoke = {
    {static_shapes_to_test_representation({{49, 120}, {120, 120}}), {true, false}},
    {static_shapes_to_test_representation({{79, 120}, {120, 120}}), {true, false}},

    {static_shapes_to_test_representation({{256, 188}, {188, 120}}), {true, false}},
    {static_shapes_to_test_representation({{256, 188}, {188, 120}}), {true, true}},

    {static_shapes_to_test_representation({{71, 128}, {128, 200}}), {false, false}},
    {static_shapes_to_test_representation({{71, 128}, {128, 200}}), {false, true}},

    {
        {
            // ip->brg->ip->brg
            // {1, 120} are covered in 'IS2D_Brgemm_smoke' which is ip
            // {49, 120}, {79, 120} are covered above which is brg1x1
            {{-1, -1}, {{1, 120}, {49, 120}, {1, 120}, {79, 120}}},
            {{120, 120}, {{120, 120}, {120, 120}, {120, 120}, {120, 120}}}
        },
        {false, false}
    },
    {
        {
            // ip->brg->ip(cached)->brg(cached)
            {{{0, 200}, {0, 200}}, {{1, 128}, {199, 128}, {1, 128}, {199, 128}}},
            {{128, 166}, {{128, 166}, {128, 166}}}
        },
        {true, true}
    },
};




const std::vector<ShapeRelatedParams> IS3D_Brgconv1x1_smoke = {
    {static_shapes_to_test_representation({{2, 49, 120}, {120, 120}}), {true, false}},
    {static_shapes_to_test_representation({{4, 79, 120}, {120, 120}}), {true, false}},

    {static_shapes_to_test_representation({{1, 256, 188}, {188, 120}}), {true, false}},
    {static_shapes_to_test_representation({{2, 256, 188}, {188, 120}}), {true, true}},

    {static_shapes_to_test_representation({{2, 71, 128}, {128, 200}}), {false, false}},
    {static_shapes_to_test_representation({{3, 71, 128}, {128, 200}}), {false, true}},

    {
        {
            // ip->brg->ip->brg
            // {1, 1, 120}, {3, 1, 120} are covered in 'IS3D_smoke' which is ip
            // {2, 49, 120}, {4, 79, 120} are covered above which is brg1x1
            {{-1, -1, -1}, {{1, 1, 120}, {2, 49, 120}, {3, 1, 120}, {4, 79, 120}}},
            {{120, 120}, {{120, 120}, {120, 120}, {120, 120}, {120, 120}}}
        },
        {false, false}
    },
    {
        {
            // weights: Acb32a->Acb64a->Acb32a(cached)->Acb64a(cached)
            {{-1, -1, -1}, {{1, 54, 96}, {8, 54 * 2, 96}, {1, 54, 96}, {8, 54 * 2, 96}}},
            {{96, 128}, {{96, 128}, {96, 128}, {96, 128}, {96, 128}}}
        },
        {false, false}
    },
    {
        {
            // ip->brg->ip(cached)->brg(cached)
            {{{0, 200}, {0, 200}, {0, 200}}, {{1, 18, 128}, {2, 199, 128}, {3, 18, 128}, {4, 199, 128}}},
            {{128, 166}, {{128, 166}, {128, 166}}}
        },
        {true, true}
    },
};

} // namespace fullyConnected


/* ============= MatMul ============= */
namespace matmul {

const std::vector<ShapeRelatedParams> IS = {
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, false}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, false}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, true}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, true}},

    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {false, false}},
    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {true, false}},
    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {false, true}},
    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {true, true}},

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
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}}  // input 1
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

std::vector<fusingSpecificParams> matmulFusingParams {
        emptyFusingSpec,
        fusingElu,
        fusingSqrt,
        fusingPReluPerTensor,
        fusingMultiplyPerChannel,
        fusingAddPerTensor,
        fusingBias,
        fusingFakeQuantizePerChannel,
        /* @todo FQ unfolds into FQ + Convert + Substract + Multiply after LPT,
         * so Relu cannot be fused in this case. Should be analysed */
        // fusingFakeQuantizePerChannelRelu,
        fusingFakeQuantizePerTensorRelu,
        fusingScaleShiftAndFakeQuantizePerChannel,
};

const auto matMulParams = ::testing::Combine(::testing::ValuesIn(IS),
                                             ::testing::ValuesIn(netPRCs()),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(helpers::InputLayerType::PARAMETER),
                                             ::testing::Values(ov::test::utils::DEVICE_CPU),
                                             ::testing::ValuesIn(additionalConfig));

const auto testParams = ::testing::Combine(matMulParams,
                                           ::testing::Values(MatMulNodeType::MatMul),
                                           ::testing::ValuesIn(matmulFusingParams),
                                           ::testing::ValuesIn(filterCPUInfo(filterSpecificParams())));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Static, MatMulLayerCPUTest, testParams, MatMulLayerCPUTest::getTestCaseName);

const auto matMulParamsDynamic = ::testing::Combine(::testing::ValuesIn(IS_Dynamic),
                                             ::testing::ValuesIn(netPRCs()),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(helpers::InputLayerType::PARAMETER),
                                             ::testing::Values(ov::test::utils::DEVICE_CPU),
                                             ::testing::ValuesIn(additionalConfig));

const auto testParamsDynamic = ::testing::Combine(matMulParamsDynamic,
                                           ::testing::Values(MatMulNodeType::MatMul),
                                           ::testing::Values(emptyFusingSpec),
                                           ::testing::ValuesIn(filterCPUInfo(filterSpecificParams())));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Dynamic, MatMulLayerCPUTest, testParamsDynamic, MatMulLayerCPUTest::getTestCaseName);

const auto matMulParamsDynamic_nightly = ::testing::Combine(::testing::ValuesIn(IS_Dynamic_nightly),
                                             ::testing::ValuesIn(netPRCs()),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(helpers::InputLayerType::PARAMETER),
                                             ::testing::Values(ov::test::utils::DEVICE_CPU),
                                             ::testing::ValuesIn(additionalConfig));

const auto testParamsDynamic_nightly = ::testing::Combine(matMulParamsDynamic_nightly,
                                           ::testing::Values(MatMulNodeType::MatMul),
                                           ::testing::Values(emptyFusingSpec),
                                           ::testing::ValuesIn(filterCPUInfo(filterSpecificParams())));

INSTANTIATE_TEST_SUITE_P(nightly_MM_Dynamic, MatMulLayerCPUTest, testParamsDynamic_nightly, MatMulLayerCPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams> IS_Dynamic_Fusing = {
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{16, 12}, {33, 7}, {16, 12}}}, // input 0
            {{-1, 33}, {{12, 33}, {7, 33}, {12, 33}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, 5}, {{60, 5}, {30, 5}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, 25}, {{3, 7, 60, 25}, {3, 7, 30, 25}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}, {10, 10, 10}}}, // input 0
            {{-1, -1, 5}, {{10, 10, 5}, {5, 5, 5}, {10, 10, 5}}}  // input 1
        },
        {false, false}
    },
};

const auto matMulParamsDynamicFusing = ::testing::Combine(::testing::ValuesIn(IS_Dynamic_Fusing),
                                                        ::testing::ValuesIn(netPRCs()),
                                                        ::testing::Values(ElementType::undefined),
                                                        ::testing::Values(ElementType::undefined),
                                                        ::testing::Values(helpers::InputLayerType::PARAMETER),
                                                        ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                        ::testing::ValuesIn(additionalConfig));

const auto testParamsDynamicFusing = ::testing::Combine(matMulParamsDynamicFusing,
                                                  ::testing::Values(MatMulNodeType::MatMul),
                                                  ::testing::ValuesIn(matmulFusingParams),
                                                  ::testing::ValuesIn(filterCPUInfo(filterSpecificParams())));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Dynamic_Fusing, MatMulLayerCPUTest, testParamsDynamicFusing, MatMulLayerCPUTest::getTestCaseName);

} // namespace matmul
} // namespace MatMul
} // namespace CPULayerTestsDefinitions