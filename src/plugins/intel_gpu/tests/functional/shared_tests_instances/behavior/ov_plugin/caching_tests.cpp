// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/caching_tests.hpp"

#include <cstdlib>
#include <filesystem>

#include "openvino/pass/manager.hpp"
#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"

using namespace ov::test::behavior;

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    ::setenv(name, value, 1);
#endif
}

void unset_env(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    ::unsetenv(name);
#endif
}

static const std::vector<ov::element::Type> precisionsGPU = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i32,
    ov::element::i64,
    ov::element::i8,
    ov::element::u8,
    ov::element::i16,
    ov::element::u16,
};

static const std::vector<std::size_t> batchSizesGPU = {1, 2};

static const std::vector<ov::element::Type> floatingPointPrecisionsGPU = {
    ov::element::f32,
    ov::element::f16,
};

INSTANTIATE_TEST_SUITE_P(
    smoke_CachingSupportCaseAnyType_GPU,
    CompileModelCacheTestBase,
    ::testing::Combine(::testing::ValuesIn(CompileModelCacheTestBase::getNumericAnyTypeFunctions()),
                       ::testing::ValuesIn(precisionsGPU),
                       ::testing::ValuesIn(batchSizesGPU),
                       ::testing::Values(ov::test::utils::DEVICE_GPU),
                       ::testing::Values(ov::AnyMap{})),
    CompileModelCacheTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_CachingSupportCaseFloat_GPU,
    CompileModelCacheTestBase,
    ::testing::Combine(::testing::ValuesIn(CompileModelCacheTestBase::getFloatingPointOnlyFunctions()),
                       ::testing::ValuesIn(floatingPointPrecisionsGPU),
                       ::testing::ValuesIn(batchSizesGPU),
                       ::testing::Values(ov::test::utils::DEVICE_GPU),
                       ::testing::Values(ov::AnyMap{})),
    CompileModelCacheTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_KernelCachingSupportCase_GPU,
                         CompiledKernelsCacheTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(std::make_pair(ov::AnyMap{}, "blob,cl_cache"))),
                         CompiledKernelsCacheTest::getTestCaseName);

const std::vector<ov::AnyMap> GPULoadFromFileConfigs = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
    {},
};

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_GPU,
                         CompileModelLoadFromFileTestBase,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(GPULoadFromFileConfigs)),
                         CompileModelLoadFromFileTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_GPU,
                         CompileModelCacheRuntimePropertiesTestBase,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(GPULoadFromFileConfigs)),
                         CompileModelCacheRuntimePropertiesTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_GPU,
                         CompileModelLoadFromMemoryTestBase,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(GPULoadFromFileConfigs)),
                         CompileModelLoadFromMemoryTestBase::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_GPU,
                         CompileModelLoadFromCacheTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(GPULoadFromFileConfigs)),
                         CompileModelLoadFromCacheTest::getTestCaseName);

class CompileModelZeroCopySingleFileCacheLoadTest : public testing::WithParamInterface<CompileModelLoadFromCacheParams>,
                                                    virtual public ov::test::SubgraphBaseTest,
                                                    virtual public OVPluginTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CompileModelLoadFromCacheParams> obj) {
        return CompileModelLoadFromCacheTest::getTestCaseName(obj);
    }

    void SetUp() override {
        std::tie(targetDevice, configuration) = this->GetParam();
        target_device = targetDevice;
        APIBaseTest::SetUp();
        set_env("OV_GPU_ENABLE_ZERO_COPY_CACHE_LOAD", "YES");

        std::string filePrefix = ov::test::utils::generateTestFilePrefix();
        m_modelName = filePrefix + ".xml";
        m_weightsName = filePrefix + ".bin";
        // ".bin" extension makes ov::CoreConfig::CacheConfig::create() pick
        // ov::runtime::SingleFileStorage instead of the folder-based FileStorageCacheManager.
        m_cacheFilePath = filePrefix + ".cache.bin";
        std::filesystem::remove(m_cacheFilePath);

        core->set_property(ov::cache_dir());
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::Serialize>(m_modelName, m_weightsName);
        manager.run_passes(ov::test::utils::make_conv_pool_relu({1, 3, 227, 227}, ov::element::f32));
    }

    void TearDown() override {
        inferRequest = {};
        compiledModel = {};

        std::filesystem::remove(m_cacheFilePath);
        ov::test::utils::removeIRFiles(m_modelName, m_weightsName);
        core->set_property(ov::cache_dir());
        ov::test::utils::PluginCache::get().reset();

        unset_env("OV_GPU_ENABLE_ZERO_COPY_CACHE_LOAD");
        APIBaseTest::TearDown();
    }

    void run() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        core->set_property(ov::cache_dir(m_cacheFilePath));
        compiledModel = core->compile_model(m_modelName, targetDevice, configuration);
        EXPECT_EQ(false, compiledModel.get_property(ov::loaded_from_cache.name()).as<bool>());

        std::stringstream strm;
        compiledModel.export_model(strm);
        ov::CompiledModel importedCompiledModel = core->import_model(strm, target_device, configuration);
        EXPECT_EQ(false, importedCompiledModel.get_property(ov::loaded_from_cache.name()).as<bool>());

        compiledModel = core->compile_model(m_modelName, targetDevice, configuration);
        EXPECT_EQ(true, compiledModel.get_property(ov::loaded_from_cache.name()).as<bool>());
    }

private:
    std::string m_modelName;
    std::string m_weightsName;
    std::string m_cacheFilePath;
};

TEST_P(CompileModelZeroCopySingleFileCacheLoadTest, CanLoadFromSingleFileCache) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_ZeroCopySingleFileCacheLoad_GPU,
                         CompileModelZeroCopySingleFileCacheLoadTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(ov::AnyMap{})),
                         CompileModelZeroCopySingleFileCacheLoadTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_GPU,
                         CompileModelWithCacheEncryptionTest,
                         ::testing::Values(ov::test::utils::DEVICE_GPU),
                         CompileModelWithCacheEncryptionTest::getTestCaseName);
} // namespace
