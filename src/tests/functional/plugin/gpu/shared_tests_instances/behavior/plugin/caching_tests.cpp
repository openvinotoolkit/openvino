// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/caching_tests.hpp"

using namespace LayerTestsDefinitions;

namespace {
    static const std::vector<ngraph::element::Type> precisionsGPU = {
            ngraph::element::f32,
            ngraph::element::f16,
            ngraph::element::i32,
            ngraph::element::i64,
            ngraph::element::i8,
            ngraph::element::u8,
            ngraph::element::i16,
            ngraph::element::u16,
    };

    static const std::vector<std::size_t> batchSizesGPU = {
            1, 2
    };

    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_GPU, LoadNetworkCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(LoadNetworkCacheTestBase::getStandardFunctions()),
                                    ::testing::ValuesIn(precisionsGPU),
                                    ::testing::ValuesIn(batchSizesGPU),
                                    ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                    ::testing::Values(std::map<std::string, std::string>())),
                            LoadNetworkCacheTestBase::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_KernelCachingSupportCase_GPU, LoadNetworkCompiledKernelsCacheTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                    ::testing::Values(std::make_pair(std::map<std::string, std::string>(), "cl_cache"))),
                            LoadNetworkCompiledKernelsCacheTest::getTestCaseName);

    typedef std::map<std::string, std::string> conftype;
    std::vector<std::pair<conftype, std::string>> autoConfigs = {
            std::make_pair(conftype{{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_GPU}}, "cl_cache"),
            std::make_pair(conftype{{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
            (std::string(CommonTestUtils::DEVICE_GPU) + "," + CommonTestUtils::DEVICE_CPU)}}, "blob,cl_cache"),
            std::make_pair(conftype{{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
            (std::string(CommonTestUtils::DEVICE_CPU) + "," + CommonTestUtils::DEVICE_GPU)}}, "blob")
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_KernelCachingSupportCase_GPU, LoadNetworkCompiledKernelsCacheTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(autoConfigs)),
                            LoadNetworkCompiledKernelsCacheTest::getTestCaseName);

} // namespace
