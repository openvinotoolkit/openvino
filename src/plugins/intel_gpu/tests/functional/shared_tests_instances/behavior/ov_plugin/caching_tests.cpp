// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/import_export.hpp"
#include "behavior/ov_plugin/caching_tests.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "intel_gpu/plugin/hsm_common.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/codec_xor.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/parallel_io.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
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

class GPUHsmRegressionTest : public OVCompiledModelHsmBlobTest {
public:
    void SetUp() override {
        std::tie(target_device, configuration) = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();

        core = ov::test::utils::create_core();
        core.set_property(target_device, configuration);
        model = ov::test::utils::make_conv_pool_relu({1, 3, 16, 16}, ov::element::f32);
        cache_path = "hsm_blob_cache_" + ov::test::utils::generateTestFilePrefix();
        input = ov::test::utils::create_and_fill_tensor(ov::element::f32, model->input().get_shape());
        auto compiled = compile(false);
        ASSERT_TRUE(is_hsm(export_blob(compiled)));
        auto request = compiled.create_infer_request();
        request.set_input_tensor(input);
        request.infer();
        const auto output = request.get_output_tensor();
        reference = ov::Tensor(output.get_element_type(), output.get_shape());
        output.copy_to(reference);
    }

    void TearDown() override {
        if (!cache_path.empty()) {
            std::filesystem::remove(cache_path.string() + ".xml");
            std::filesystem::remove(cache_path.string() + ".bin");
            std::filesystem::remove(cache_path.string() + ".mapped");
        }
        OVCompiledModelHsmBlobTest::TearDown();
    }
};

TEST_P(GPUHsmRegressionTest, CacheMissHitAndPlainImports) {
    auto compiled = compile(true);
    EXPECT_FALSE(compiled.get_property(ov::loaded_from_cache));
    const auto blob = export_blob(compiled);
    ASSERT_TRUE(is_hsm(blob));

    std::stringstream stream(blob);
    auto from_stream = core.import_model(stream, target_device, configuration);
    expect_same_inference(from_stream);

    ov::CompiledModel from_tensor;
    {
        ov::Tensor tensor(ov::element::u8, ov::Shape{blob.size()});
        std::memcpy(tensor.data(), blob.data(), blob.size());
        from_tensor = core.import_model(tensor, target_device, configuration);
    }
    expect_same_inference(from_tensor);

    auto cached = compile(true);
    EXPECT_TRUE(cached.get_property(ov::loaded_from_cache));
    expect_same_inference(cached);
}

TEST_P(GPUHsmRegressionTest, ReplacingSourceAtSamePathInvalidatesCache) {
    const auto xml = cache_path.string() + ".xml";
    const auto weights = cache_path.string() + ".bin";
    const auto make_model = [&](float value, const std::string& name) {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input.get_shape());
        auto constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {value});
        auto add = std::make_shared<ov::op::v1::Add>(parameter, constant);
        return std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{parameter}, name);
    };
    auto config = configuration;
    config.emplace(ov::cache_dir(ov::util::path_to_string(cache_path)));
    std::fill_n(input.data<float>(), input.get_size(), 1.f);
    const auto expect_output = [&](ov::CompiledModel& compiled, float expected) {
        auto request = compiled.create_infer_request();
        request.set_input_tensor(input);
        request.infer();
        const auto output = request.get_output_tensor();
        ASSERT_EQ(output.get_size(), input.get_size());
        for (size_t i = 0; i < output.get_size(); ++i)
            EXPECT_FLOAT_EQ(output.data<const float>()[i], expected);
    };
    ov::serialize(make_model(1.f, "before"), xml, weights);
    {
        auto first = core.compile_model(xml, target_device, config);
        EXPECT_FALSE(first.get_property(ov::loaded_from_cache));
        expect_output(first, 2.f);
    }
    {
        auto cached = core.compile_model(xml, target_device, config);
        EXPECT_TRUE(cached.get_property(ov::loaded_from_cache));
        expect_output(cached, 2.f);
    }
    // A different XML size makes the freshness change independent of timestamp granularity.
    ov::serialize(make_model(2.f, "after_source_replacement"), xml, weights);
    {
        auto replaced = core.compile_model(xml, target_device, config);
        EXPECT_FALSE(replaced.get_property(ov::loaded_from_cache));
        expect_output(replaced, 3.f);
    }
    auto cached = core.compile_model(xml, target_device, config);
    EXPECT_TRUE(cached.get_property(ov::loaded_from_cache));
    expect_output(cached, 3.f);
}

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
        file.write(blob.data(), blob.size());
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
    const auto original = export_blob(compile(false));
    std::vector<std::pair<std::string, std::string>> invalid;
    for (const auto tag : {ov::intel_gpu::hsm::payload_info_tag, wire::runtime_requirements_tag, wire::model_tag}) {
        invalid.emplace_back("missing tag " + std::to_string(tag.id()),
                             hsm_repack(original, [&](std::vector<HsmEntry>& entries) {
                                 entries.erase(std::remove_if(entries.begin(), entries.end(), [&](const HsmEntry& entry) {
                                                   return entry.device == wire::gpu_device_id && entry.tag.tag.id() == tag.id();
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
                        GPUHsmRegressionTest,
                        ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                           ::testing::Values(ov::AnyMap{ov::intel_gpu::export_hsm_format(true)})),
                        GPUHsmRegressionTest::getTestCaseName);
} // namespace
