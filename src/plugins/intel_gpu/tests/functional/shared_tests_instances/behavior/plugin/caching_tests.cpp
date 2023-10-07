// Copyright (C) 2018-2023 Intel Corporation
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

    static const std::vector<ngraph::element::Type> floatPrecisionsGPU = {
            ngraph::element::f32,
            ngraph::element::f16
    };

    static const std::vector<std::size_t> batchSizesGPU = {
            1, 2
    };

    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_GPU, LoadNetworkCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(LoadNetworkCacheTestBase::getNumericAnyTypeFunctions()),
                                    ::testing::ValuesIn(precisionsGPU),
                                    ::testing::ValuesIn(batchSizesGPU),
                                    ::testing::Values(ov::test::utils::DEVICE_GPU)),
                            LoadNetworkCacheTestBase::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_GPU_Float, LoadNetworkCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(LoadNetworkCacheTestBase::getFloatingPointOnlyFunctions()),
                                    ::testing::ValuesIn(floatPrecisionsGPU),
                                    ::testing::ValuesIn(batchSizesGPU),
                                    ::testing::Values(ov::test::utils::DEVICE_GPU)),
                            LoadNetworkCacheTestBase::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_KernelCachingSupportCase_GPU, LoadNetworkCompiledKernelsCacheTest,
                            ::testing::Combine(
                                    ::testing::Values(ov::test::utils::DEVICE_GPU),
                                    ::testing::Values(std::make_pair(std::map<std::string, std::string>(), "blob"))),
                            LoadNetworkCompiledKernelsCacheTest::getTestCaseName);
} // namespace
