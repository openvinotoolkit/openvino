// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/caching_tests.hpp"

using namespace ov::test::behavior;

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

    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_GPU, CompileModelCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(CompileModelCacheTestBase::getStandardFunctions()),
                                    ::testing::ValuesIn(precisionsGPU),
                                    ::testing::ValuesIn(batchSizesGPU),
                                    ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                    ::testing::Values(ov::AnyMap{})),
                            CompileModelCacheTestBase::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_KernelCachingSupportCase_GPU, CompiledKernelsCacheTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                    ::testing::Values(std::make_pair(ov::AnyMap{}, "cl_cache"))),
                            CompiledKernelsCacheTest::getTestCaseName);

    auto autoConfigs = []() {
        return std::vector<std::pair<ov::AnyMap, std::string>>{
            std::make_pair(ov::AnyMap{{ov::device::priorities(CommonTestUtils::DEVICE_GPU)}}, "cl_cache"),
            std::make_pair(
                ov::AnyMap{{ov::device::priorities(CommonTestUtils::DEVICE_GPU, CommonTestUtils::DEVICE_CPU)}},
                "blob,cl_cache"),
            std::make_pair(
                ov::AnyMap{{ov::device::priorities(CommonTestUtils::DEVICE_CPU, CommonTestUtils::DEVICE_GPU)}},
                "blob")};
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_KernelCachingSupportCase_GPU, CompiledKernelsCacheTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(autoConfigs())),
                            CompiledKernelsCacheTest::getTestCaseName);

    const std::vector<ov::AnyMap> LoadFromFileConfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU), ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU), ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}
    };
    const std::vector<std::string> TestTargets =
    {CommonTestUtils::DEVICE_AUTO,
    CommonTestUtils::DEVICE_MULTI,
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_CachingSupportCase_GPU, CompileModelLoadFromFileTestBase,
                        ::testing::Combine(
                                ::testing::ValuesIn(TestTargets),
                                ::testing::ValuesIn(LoadFromFileConfigs)),
                        CompileModelLoadFromFileTestBase::getTestCaseName);

    const std::vector<ov::AnyMap> GPULoadFromFileConfigs = {
        {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
        {},
    };
    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_GPU, CompileModelLoadFromFileTestBase,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                ::testing::ValuesIn(GPULoadFromFileConfigs)),
                        CompileModelLoadFromFileTestBase::getTestCaseName);

} // namespace
