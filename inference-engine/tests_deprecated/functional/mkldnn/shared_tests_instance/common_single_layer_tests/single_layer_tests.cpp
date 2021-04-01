// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests.hpp"


static CommonTestUtils::conv_common_params convParams =
        {
                PropertyVector<unsigned>{{2, 2}},  // stride
                PropertyVector<unsigned>{{3, 3}},  // kernel
                {},                                // pad_begin
                {},                                // pad_end
                PropertyVector<unsigned>{{1, 1}},  // dilation
                "same_upper",                      // auto_pad
                1,                                 // group
                2                                  // out_c
        };

static CommonTestUtils::pool_common_params poolParams =
        {
                PropertyVector<unsigned>{{2, 2}},  // stride
                PropertyVector<unsigned>{{3, 3}},  // kernel
                {},                                // pad_begin
                {},                                // pad_end
                "same_upper",                      // auto_pad
                true,                              // avg
                false                              // exclude_pad
        };

static CommonTestUtils::conv_common_params defConvParamsHeavy =
        {
                PropertyVector<unsigned>{{1, 1}},  // stride
                PropertyVector<unsigned>{{3, 3}},  // kernel
                {},                                // pad_begin
                {},                                // pad_end
                PropertyVector<unsigned>{{2, 2}},  // dilation
                "same_upper",                      // auto_pad
                1,                                 // group
                128                                // out_c
        };

static CommonTestUtils::conv_common_params defConvParamsLight0 =
        {
                PropertyVector<unsigned>{{1, 1}},  // stride
                PropertyVector<unsigned>{{3, 3}},  // kernel
                {},                                // pad_begin
                {},                                // pad_end
                PropertyVector<unsigned>{{2, 2}},  // dilation
                "same_upper",                      // auto_pad
                1,                                 // group
                4                                  // out_c
        };

static CommonTestUtils::conv_common_params defConvParamsLight1 =
        {
                PropertyVector<unsigned>{{2, 2}},  // stride
                PropertyVector<unsigned>{{3, 3}},  // kernel
                {},                                // pad_begin
                {},                                // pad_end
                PropertyVector<unsigned>{{1, 1}},  // dilation
                "same_upper",                      // auto_pad
                1,                                 // group
                16                                 // out_c
        };


static CommonTestUtils::conv_common_params defConvParamsLight2 =
        {
                PropertyVector<unsigned>{{2, 2}},  // stride
                PropertyVector<unsigned>{{3, 3}},  // kernel
                {},                                // pad_begin
                {},                                // pad_end
                PropertyVector<unsigned>{{2, 2}},  // dilation
                "same_upper",                      // auto_pad
                1,                                 // group
                15                                 // out_c
        };


static CommonTestUtils::conv_common_params defConvParamsLight3 =
        {
                PropertyVector<unsigned>{{1, 1}},  // stride
                PropertyVector<unsigned>{{3, 3}},  // kernel
                {},                                // pad_begin
                {},                                // pad_end
                PropertyVector<unsigned>{{2, 2}},  // dilation
                "same_upper",                      // auto_pad
                2,                                 // group
                4                                  // out_c
        };

static std::vector<PluginParams> pluginParams = {
        PluginDependentParam{"CPU", Layout::NCHW, Precision::FP32, 0.001f}
};

std::string
getTestCaseName(testing::TestParamInfo<std::tuple<InitialShapes, NewShapes, PluginParams, Helper>> obj) {
    auto params = obj.param;
    LayerTestHelper::Ptr helper = std::get<3>(params);
    return "MKLDNN" + helper->getType();
}

INSTANTIATE_TEST_CASE_P(
        // TODO: rewrite to ngraph to have reshape functionality
        DISABLED_Conv_smoke, CommonSingleLayerTest,
        ::testing::Combine(
        ::testing::Values(InitialShapes({
                                                {{1, 2, 16, 16}},           // input
                                                {{1, 2, 8,  8}}             // output
                                        })),
        ::testing::Values(NewShapes({
                                            {{1, 2, 15, 15}},               // input
                                            {{1, 2, 8,  8}}                 // output
                                    })),
        ::testing::ValuesIn(pluginParams),
        ::testing::Values(Helper(std::make_shared<ConvolutionTestHelper>(convParams)))
), getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        // TODO: rewrite to ngraph to have reshape functionality
        DISABLED_Deconv_smoke, CommonSingleLayerTest,
        ::testing::Combine(
        ::testing::Values(InitialShapes({
                                                {{1, 2, 8,  8}},             // input
                                                {{1, 2, 16, 16}}              // output
                                        })),
        ::testing::Values(NewShapes({
                                            {{1, 2, 7,  7}},                  // input
                                            {{1, 2, 14, 14}}                  // output
                                    })),
        ::testing::ValuesIn(pluginParams),
        ::testing::Values(Helper(std::make_shared<DeconvolutionTestHelper>(convParams)))
), getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        // TODO: rewrite to ngraph to have reshape functionality
        DISABLED_Pool_smoke, CommonSingleLayerTest,
        ::testing::Combine(
        ::testing::Values(InitialShapes({
                                                {{1, 2, 16, 16}},           // input
                                                {{1, 2, 8,  8}}             // output
                                        })),
        ::testing::Values(NewShapes({
                                            {{1, 2, 15, 15}},               // input
                                            {{1, 2, 8,  8}}                 // output
                                    })),
        ::testing::ValuesIn(pluginParams),
        ::testing::Values(Helper(std::make_shared<PoolingTestHelper>(poolParams)))
), getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        DISABLED_DefConvLight0_smoke, CommonSingleLayerTest,
        ::testing::Combine(
                ::testing::Values(InitialShapes({
                                                        {{1, 4, 4, 4}, {1, 36, 4, 4}}, // input, trans
                                                        {{1, 4, 4, 4}}                 // output
                                                })),
                ::testing::Values(NewShapes({
                                                    {{1, 4, 4, 4}, {1, 36, 4, 4}}, // input, trans
                                                    {{1, 4, 4, 4}}                 // output
                                            })),
                ::testing::ValuesIn(pluginParams),
                ::testing::Values(Helper(std::make_shared<DeformableConvolutionTestHelper>(defConvParamsLight0, 2)))
        ), getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        DISABLED_DefConvLight1_WithBatch_smoke, CommonSingleLayerTest,
        ::testing::Combine(
                ::testing::Values(InitialShapes({
                                                        {{2, 4, 8, 8}, {2, 36, 4, 4}}, // input, trans
                                                        {{2, 16, 4, 4}}                // output
                                                })),
                ::testing::Values(NewShapes({
                                                    {{2, 4, 8, 8}, {2, 36, 4, 4}}, // input, trans
                                                    {{2, 16, 4, 4}}                // output
                                            })),
                ::testing::ValuesIn(pluginParams),
                ::testing::Values(Helper(std::make_shared<DeformableConvolutionTestHelper>(defConvParamsLight1, 2)))
        ), getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        DISABLED_DefConvLight2_WithBatch_smoke, CommonSingleLayerTest,
        ::testing::Combine(
                ::testing::Values(InitialShapes({
                                                        {{2, 4, 8, 8}, {2, 18, 4, 4}}, // input, trans
                                                        {{2, 15, 4, 4}}                // output
                                                })),
                ::testing::Values(NewShapes({
                                                    {{2, 4, 8, 8}, {2, 18, 4, 4}}, // input, trans
                                                    {{2, 15, 4, 4}}                // output
                                            })),
                ::testing::ValuesIn(pluginParams),
                ::testing::Values(Helper(std::make_shared<DeformableConvolutionTestHelper>(defConvParamsLight2, 1)))
        ), getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        DISABLED_DefConvLight3_WithGroups_smoke, CommonSingleLayerTest,
        ::testing::Combine(
                ::testing::Values(InitialShapes({
                                                        {{1, 4, 4, 4}, {1, 18, 4, 4}}, // input, trans
                                                        {{1, 4, 4, 4}}                 // output
                                                })),
                ::testing::Values(NewShapes({
                                                    {{1, 4, 4, 4}, {1, 18, 4, 4}}, // input, trans
                                                    {{1, 4, 4, 4}}                 // output
                                            })),
                ::testing::ValuesIn(pluginParams),
                ::testing::Values(Helper(std::make_shared<DeformableConvolutionTestHelper>(defConvParamsLight3, 1)))
        ), getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        DISABLED_smoke_DefConvHeavy, CommonSingleLayerTest,
        ::testing::Combine(
                ::testing::Values(InitialShapes({
                                                        {{1, 512, 38, 38}, {1, 72, 38, 38}}, // input, trans
                                                        {{1, 128, 38, 38}}                   // output
                                                })),
                ::testing::Values(NewShapes({
                                                    {{1, 512, 38, 38}, {1, 72, 38, 38}}, // input, trans
                                                    {{1, 128, 38, 38}}                   // output
                                            })),
                ::testing::ValuesIn(pluginParams),
                ::testing::Values(Helper(std::make_shared<DeformableConvolutionTestHelper>(defConvParamsHeavy, 4)))
        ), getTestCaseName
);
