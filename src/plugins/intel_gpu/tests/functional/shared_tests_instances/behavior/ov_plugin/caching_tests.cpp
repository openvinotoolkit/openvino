// Copyright (C) 2018-2023 Intel Corporation
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

    static const std::vector<ov::element::Type> floatingPointPrecisionsGPU = {
            ngraph::element::f32,
            ngraph::element::f16,
    };

    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCaseAnyType_GPU, CompileModelCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(CompileModelCacheTestBase::getNumericAnyTypeFunctions()),
                                    ::testing::ValuesIn(precisionsGPU),
                                    ::testing::ValuesIn(batchSizesGPU),
                                    ::testing::Values(ov::test::utils::DEVICE_GPU),
                                    ::testing::Values(ov::AnyMap{})),
                            CompileModelCacheTestBase::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCaseFloat_GPU, CompileModelCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(CompileModelCacheTestBase::getFloatingPointOnlyFunctions()),
                                    ::testing::ValuesIn(floatingPointPrecisionsGPU),
                                    ::testing::ValuesIn(batchSizesGPU),
                                    ::testing::Values(ov::test::utils::DEVICE_GPU),
                                    ::testing::Values(ov::AnyMap{})),
                            CompileModelCacheTestBase::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_KernelCachingSupportCase_GPU, CompiledKernelsCacheTest,
                            ::testing::Combine(
                                    ::testing::Values(ov::test::utils::DEVICE_GPU),
                                    ::testing::Values(std::make_pair(ov::AnyMap{}, "blob"))),
                            CompiledKernelsCacheTest::getTestCaseName);

    auto autoConfigs = []() {
        return std::vector<std::pair<ov::AnyMap, std::string>>{
            std::make_pair(ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_GPU)}}, "blob"),
            std::make_pair(
                ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU)}},
                "blob"),
            std::make_pair(
                ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_CPU, ov::test::utils::DEVICE_GPU)}},
                "blob")};
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_KernelCachingSupportCase_GPU, CompiledKernelsCacheTest,
                            ::testing::Combine(
                                    ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                    ::testing::ValuesIn(autoConfigs())),
                            CompiledKernelsCacheTest::getTestCaseName);

    const std::vector<ov::AnyMap> LoadFromFileConfigs = {
        {ov::device::priorities(ov::test::utils::DEVICE_GPU), ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU), ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}
    };
    const std::vector<std::string> TestTargets =
    {ov::test::utils::DEVICE_AUTO,
    ov::test::utils::DEVICE_MULTI,
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_CachingSupportCase_GPU, CompileModelLoadFromFileTestBase,
                        ::testing::Combine(
                                ::testing::ValuesIn(TestTargets),
                                ::testing::ValuesIn(LoadFromFileConfigs)),
                        CompileModelLoadFromFileTestBase::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_CachingSupportCase_GPU,
                             CompileModelLoadFromMemoryTestBase,
                             ::testing::Combine(::testing::ValuesIn(TestTargets),
                                                ::testing::ValuesIn(LoadFromFileConfigs)),
                             CompileModelLoadFromMemoryTestBase::getTestCaseName);

    const std::vector<ov::AnyMap> GPULoadFromFileConfigs = {
        {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
        {},
    };
    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_GPU, CompileModelLoadFromFileTestBase,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(GPULoadFromFileConfigs)),
                        CompileModelLoadFromFileTestBase::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_GPU,
                             CompileModelLoadFromMemoryTestBase,
                             ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                                ::testing::ValuesIn(GPULoadFromFileConfigs)),
                             CompileModelLoadFromMemoryTestBase::getTestCaseName);
} // namespace
