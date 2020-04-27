// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests.hpp"
#include "vpu_tests_config.hpp"

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

static std::vector<PluginParams> pluginParams = {
    PluginDependentParam{vpu::tests::deviceName(), Layout::NHWC, Precision::FP16, 0.01f},
};

std::string
getTestCaseName(testing::TestParamInfo<std::tuple<InitialShapes, NewShapes, PluginParams, Helper>> obj) {
    auto params = obj.param;
    PluginDependentParam pluginParams = std::get<2>(params);
    LayerTestHelper::Ptr helper = std::get<3>(params);
    // To correspond filter of functional tests
    std::map<std::string, std::string> device2FilterName{
        {vpu::tests::deviceName(), vpu::tests::pluginNameShort()},
    };
    return device2FilterName[pluginParams.deviceName] + helper->getType();
}

#if (defined INSTANTIATE_TESTS)

INSTANTIATE_TEST_CASE_P(
        Conv_nightly, CommonSingleLayerTest,
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
        Deconv_nightly, CommonSingleLayerTest,
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
        Pool_nightly, CommonSingleLayerTest,
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

#endif
