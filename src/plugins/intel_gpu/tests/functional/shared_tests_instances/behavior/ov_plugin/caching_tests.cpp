// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/caching_tests.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "behavior/compiled_model/import_export.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "intel_gpu/plugin/hsm_common.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/codec_xor.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/parallel_io.hpp"

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

class GPUHsmRegressionTest : public OVCompiledModelHsmBlobTest {
public:
    void TearDown() override {
        if (!cache_path.empty()) {
            std::filesystem::remove(cache_path.string() + ".mapped");
        }
        OVCompiledModelHsmBlobTest::TearDown();
    }
};

TEST_P(GPUHsmRegressionTest, RejectsTensorPrefixOfLargerMappedFile) {
    namespace wire = ov::runtime::hsm;
    auto blob = export_blob(compile(false));
    const auto prefix_size = std::max(blob.size(), ov::util::default_parallel_io_threshold);
    blob.resize(prefix_size + ov::util::min_page_alignment, '\0');
    auto header = wire::Header::view(blob.data());
    header.container_size = blob.size();
    std::memcpy(blob.data(), &header, sizeof(header));
    const auto path = cache_path.string() + ".mapped";
    {
        std::ofstream file(path, std::ios::binary);
        ASSERT_LE(blob.size(), static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max()));
        file.write(blob.data(), static_cast<std::streamsize>(blob.size()));
        ASSERT_TRUE(file.good());
    }
    auto mapping = ov::load_mmap_object(path);
    // Keep the full mapping valid so a missing tensor-extent check fails without an unmapped read.
    ov::Tensor prefix(ov::element::u8, ov::Shape{prefix_size}, mapping->data());
    EXPECT_THROW(core.import_model(prefix, target_device, configuration), ov::Exception);

    ov::Tensor full(ov::element::u8, ov::Shape{blob.size()}, mapping->data());
    auto imported = core.import_model(full, target_device, configuration);
    expect_same_inference(imported);
}

TEST_P(GPUHsmRegressionTest, EncryptedPayloadRoundTripsThroughOffsetStreamAndTensor) {
    ov::EncryptionCallbacks callbacks;
    callbacks.encrypt = ov::util::codec_xor;
    callbacks.decrypt = ov::util::codec_xor;
    auto config = configuration;
    config.emplace(ov::cache_encryption_callbacks(callbacks));
    auto compiled = core.compile_model(model, target_device, config);
    const auto blob = export_blob(compiled);
    ASSERT_TRUE(is_hsm(blob));

    const std::string prefix = "embedded HSM blob";
    std::stringstream stream(prefix + blob);
    stream.seekg(prefix.size());
    auto from_stream = core.import_model(stream, target_device, config);
    expect_same_inference(from_stream);

    ov::CompiledModel from_tensor;
    {
        ov::Tensor tensor(ov::element::u8, ov::Shape{blob.size()});
        std::memcpy(tensor.data(), blob.data(), blob.size());
        from_tensor = core.import_model(tensor, target_device, config);
    }
    expect_same_inference(from_tensor);
}

TEST_P(GPUHsmRegressionTest, ForwardOnlyStreamRejectsHsmButReadsLegacy) {
    class ForwardOnlyBuffer final : public std::stringbuf {
    public:
        explicit ForwardOnlyBuffer(const std::string& data) : std::stringbuf(data, std::ios::in) {}

    protected:
        pos_type seekoff(off_type /*offset*/, std::ios_base::seekdir /*direction*/, std::ios_base::openmode /*mode*/) override {
            return pos_type(off_type(-1));
        }
        pos_type seekpos(pos_type /*position*/, std::ios_base::openmode /*mode*/) override {
            return pos_type(off_type(-1));
        }
    };

    ForwardOnlyBuffer hsm_buffer(export_blob(compile(false)));
    std::istream hsm_stream(&hsm_buffer);
    OV_EXPECT_THROW(core.import_model(hsm_stream, target_device, configuration),
                    ov::Exception,
                    ::testing::HasSubstr("HSM import requires a seekable input stream"));

    auto legacy_config = configuration;
    legacy_config[ov::intel_gpu::export_hsm_format.name()] = false;
    ForwardOnlyBuffer legacy_buffer(export_blob(core.compile_model(model, target_device, legacy_config)));
    std::istream legacy_stream(&legacy_buffer);
    auto imported = core.import_model(legacy_stream, target_device, configuration);
    expect_same_inference(imported);
}

TEST_P(GPUHsmRegressionTest, AlignedTensorImportRetainsStorageWithZeroCopyRequested) {
    struct RestoreZeroCopySetting {
        RestoreZeroCopySetting() {
            const auto* value = std::getenv("OV_GPU_ENABLE_ZERO_COPY_CACHE_LOAD");
            was_set = value != nullptr;
            previous = value ? value : "";
        }
        ~RestoreZeroCopySetting() {
            if (was_set)
                set_env("OV_GPU_ENABLE_ZERO_COPY_CACHE_LOAD", previous.c_str());
            else
                unset_env("OV_GPU_ENABLE_ZERO_COPY_CACHE_LOAD");
        }
        bool was_set;
        std::string previous;
    } restore;
    set_env("OV_GPU_ENABLE_ZERO_COPY_CACHE_LOAD", "YES");
    const auto blob = export_blob(compile(false));
    ov::CompiledModel imported;
    {
        const auto alignment = ov::util::min_page_alignment;
        ov::Tensor storage(ov::element::u8, ov::Shape{blob.size() + alignment});
        const auto offset = (alignment - reinterpret_cast<uintptr_t>(storage.data()) % alignment) % alignment;
        ov::Tensor tensor(storage, {offset}, {offset + blob.size()});
        std::memcpy(tensor.data(), blob.data(), blob.size());
        imported = core.import_model(tensor, target_device, configuration);
    }
    expect_same_inference(imported);
}

TEST_P(GPUHsmRegressionTest, RejectsMissingMalformedAndTruncatedSections) {
    namespace wire = ov::runtime::hsm;
    namespace gpu_hsm = ov::intel_gpu::hsm;
    const auto original = export_blob(compile(false));
    std::vector<std::pair<std::string, std::string>> invalid;
    for (const auto& id : {gpu_hsm::payload_info, gpu_hsm::runtime_requirements, gpu_hsm::compile_options, gpu_hsm::model}) {
        invalid.emplace_back("missing tag " + std::to_string(id.tag.id()),
                             hsm_repack(original, [&](std::vector<HsmEntry>& entries) {
                                 entries.erase(std::remove_if(entries.begin(), entries.end(), [&](const HsmEntry& entry) {
                                                   return entry.device == id.device && entry.tag.tag.id() == id.tag.id();
                                               }),
                                               entries.end());
                             }));
    }
    auto malformed = original;
    auto header = wire::Header::view(malformed.data());
    header.manifest_offset = header.container_size;
    std::memcpy(malformed.data(), &header, sizeof(header));
    invalid.emplace_back("manifest outside container", malformed);
    invalid.emplace_back("truncated container", original.substr(0, original.size() - 1));
    for (const auto& [name, blob] : invalid) {
        SCOPED_TRACE(name);
        std::stringstream stream(blob);
        EXPECT_THROW(core.import_model(stream, target_device, configuration), ov::Exception);
        ov::Tensor tensor(ov::element::u8, ov::Shape{blob.size()});
        std::memcpy(tensor.data(), blob.data(), blob.size());
        EXPECT_THROW(core.import_model(tensor, target_device, configuration), ov::Exception);
    }
    std::stringstream stream(original);
    auto imported = core.import_model(stream, target_device, configuration);
    expect_same_inference(imported);
}

INSTANTIATE_TEST_SUITE_P(smoke_Hsm_GPU,
                        OVCompiledModelHsmBlobTest,
                        ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                           ::testing::Values(ov::AnyMap{ov::intel_gpu::export_hsm_format(true)})),
                        OVCompiledModelHsmBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hsm_GPU,
                        GPUHsmRegressionTest,
                        ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                           ::testing::Values(ov::AnyMap{ov::intel_gpu::export_hsm_format(true)})),
                        GPUHsmRegressionTest::getTestCaseName);
} // namespace
