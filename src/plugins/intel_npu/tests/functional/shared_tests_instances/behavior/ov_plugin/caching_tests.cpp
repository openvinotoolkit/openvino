// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/caching_tests.hpp"

#include <utility>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"

using namespace ov::test::behavior;

namespace {
static const std::vector<ov::element::Type> nightly_precisionsNPU = {
    ov::element::f32,
    // ov::element::f16,
    // ov::element::u8,
};

static const std::vector<ov::element::Type> smoke_precisionsNPU = {
    ov::element::f32,
};

static const std::vector<std::size_t> batchSizesNPU = {1};

static std::vector<ovModelWithName> smoke_functions() {
    auto funcs = CompileModelCacheTestBase::getStandardFunctions();
    if (funcs.size() > 2) {
        funcs.erase(funcs.begin() + 1, funcs.end());
    }
    return funcs;
}

static std::vector<ovModelWithName> NPU_functions() {
    auto funcs = CompileModelCacheTestBase::getStandardFunctions();

    std::vector<ovModelWithName>::iterator it = remove_if(funcs.begin(), funcs.end(), [](ovModelWithName func) {
        std::vector<std::string> bad_layers{"ReadConcatSplitAssign",
                                            "SimpleFunctionRelu",
                                            "2InputSubtract",
                                            "MatMulBias",
                                            "TIwithLSTMcell1",
                                            "KSOFunction"};
        return std::find(bad_layers.begin(), bad_layers.end(), std::get<1>(func)) != bad_layers.end();
    });

    funcs.erase(it, funcs.end());

    return funcs;
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_CachingSupportCase_NPU,
                         CompileModelCacheTestBase,
                         ::testing::Combine(::testing::ValuesIn(smoke_functions()),
                                            ::testing::ValuesIn(smoke_precisionsNPU),
                                            ::testing::ValuesIn(batchSizesNPU),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::Values(ov::AnyMap{})),
                         ov::test::utils::appendPlatformTypeTestName<CompileModelCacheTestBase>);

INSTANTIATE_TEST_SUITE_P(nightly_BehaviorTests_CachingSupportCase_NPU,
                         CompileModelCacheTestBase,
                         ::testing::Combine(::testing::ValuesIn(NPU_functions()),
                                            ::testing::ValuesIn(nightly_precisionsNPU),
                                            ::testing::ValuesIn(batchSizesNPU),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::Values(ov::AnyMap{})),
                         ov::test::utils::appendPlatformTypeTestName<CompileModelCacheTestBase>);

static std::string getTestCaseName(const testing::TestParamInfo<compileModelLoadFromFileParams>& obj) {
    std::string testCaseName = CompileModelLoadFromFileTestBase::getTestCaseName(obj);
    std::replace(testCaseName.begin(), testCaseName.end(), ':', '.');
    return testCaseName +
           "_targetPlatform=" + ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);
}

const std::vector<ov::AnyMap> LoadFromFileConfigs = {
    {ov::device::properties(ov::test::utils::DEVICE_NPU, {}),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::device::properties(ov::test::utils::DEVICE_NPU, {}),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}};

const std::vector<std::string> TestTargets = {
    ov::test::utils::DEVICE_AUTO,
    ov::test::utils::DEVICE_MULTI,
    ov::test::utils::DEVICE_BATCH,
};

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Auto_BehaviorTests_CachingSupportCase_NPU,
                         CompileModelLoadFromFileTestBase,
                         ::testing::Combine(::testing::ValuesIn(TestTargets), ::testing::ValuesIn(LoadFromFileConfigs)),
                         getTestCaseName);

const std::vector<ov::AnyMap> NPULoadFromFileConfigs = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}};

const std::vector<std::pair<ov::AnyMap, std::string>> NPUCompiledKernelsCacheTest = {
    std::make_pair<ov::AnyMap, std::string>({ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
                                            "blob"),
    std::make_pair<ov::AnyMap, std::string>({ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}, "blob")};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_CachingSupportCase_NPU,
                         CompileModelLoadFromFileTestBase,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(NPULoadFromFileConfigs)),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_CachingSupportCase_NPU,
                         CompileModelLoadFromMemoryTestBase,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(NPULoadFromFileConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<CompileModelLoadFromMemoryTestBase>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_CachingSupportCase_NPU,
                         CompiledKernelsCacheTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(NPUCompiledKernelsCacheTest)),
                         ov::test::utils::appendPlatformTypeTestName<CompiledKernelsCacheTest>);

const std::vector<ov::AnyMap> cachingProperties = {
    {ov::enable_profiling(true)},
    {ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
    {ov::hint::inference_precision(ov::element::i8)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::intel_npu::batch_compiler_mode_settings("batch-compile-method=unroll")},
    {ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::COMPILER)},
    {ov::intel_npu::compilation_mode("ReferenceSW")},
    {ov::intel_npu::compilation_mode_params("dummy-op-replacement=true")},
    {ov::intel_npu::compiler_dynamic_quantization()},
    {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)},
    {ov::intel_npu::dma_engines(1)},
    {ov::intel_npu::dynamic_shape_to_static("true")},
    {ov::intel_npu::max_tiles(64)},
    {ov::intel_npu::stepping(1)},
    {ov::intel_npu::tiles(2)},
    {ov::intel_npu::turbo(true)},
    {ov::intel_npu::qdq_optimization(true)},
    {ov::intel_npu::qdq_optimization_aggressive(true)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_CachingSupportCase_NPU_Check_Config,
                         CompileModelLoadFromCacheTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(cachingProperties)),
                         ov::test::utils::appendPlatformTypeTestName<CompileModelLoadFromCacheTest>);

}  // namespace
