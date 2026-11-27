// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/caching_tests.hpp"

#include <algorithm>
#include <cmath>
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
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "intel_gpu/plugin/hsm_common.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/codec_xor.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/native_streambuf.hpp"
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

// Minimal exported blob size; 0 selects the small conv_pool_relu model.
using ZeroCopySingleFileCacheLoadParams = std::tuple<std::string, ov::AnyMap, size_t>;

class CompileModelZeroCopySingleFileCacheLoadTest : public testing::WithParamInterface<ZeroCopySingleFileCacheLoadParams>,
                                                    virtual public ov::test::SubgraphBaseTest,
                                                    virtual public OVPluginTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ZeroCopySingleFileCacheLoadParams> obj) {
        const auto& [device, config, min_blob_size] = obj.param;
        return CompileModelLoadFromCacheTest::getTestCaseName({{device, config}, obj.index}) +
               "_MinBlobSize=" + std::to_string(min_blob_size);
    }

    void SetUp() override {
        std::tie(targetDevice, configuration, m_minBlobSize) = this->GetParam();
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
        std::shared_ptr<ov::Model> model;
        if (m_minBlobSize > 0) {
            // Smallest square matrix whose FP16 weights (2 B per element) exceed m_minBlobSize,
            // plus a small margin, so the blob lands slightly above the selected I/O threshold.
            const auto channels = static_cast<size_t>(std::ceil(std::sqrt(m_minBlobSize / 2.0))) + 16;
            auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, channels});
            std::vector<float> values(channels * channels);
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = static_cast<float>(i % 251) / 251.f;
            }
            auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{channels, channels}, values);
            auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights, false, true);
            model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input});
        } else {
            model = ov::test::utils::make_conv_pool_relu({1, 3, 227, 227}, ov::element::f32);
        }
        manager.run_passes(model);
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
        const auto expected_output = infer(compiledModel);

        std::stringstream strm;
        compiledModel.export_model(strm);
        if (m_minBlobSize > 0) {
            ASSERT_GT(strm.tellp(), std::streampos(m_minBlobSize));
        }
        ov::CompiledModel importedCompiledModel = core->import_model(strm, target_device, configuration);
        EXPECT_EQ(false, importedCompiledModel.get_property(ov::loaded_from_cache.name()).as<bool>());

        compiledModel = core->compile_model(m_modelName, targetDevice, configuration);
        EXPECT_EQ(true, compiledModel.get_property(ov::loaded_from_cache.name()).as<bool>());

        const auto actual_output = infer(compiledModel);
        ASSERT_EQ(expected_output.size(), actual_output.size());
        for (size_t i = 0; i < expected_output.size(); ++i) {
            EXPECT_NEAR(expected_output[i], actual_output[i], 1e-4f) << "Output mismatch at index " << i;
        }
    }

private:
    static std::vector<float> infer(ov::CompiledModel& model) {
        auto request = model.create_infer_request();
        auto input = request.get_input_tensor();
        for (size_t i = 0; i < input.get_size(); ++i) {
            input.data<float>()[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.01f;
        }
        request.infer();

        auto output = request.get_output_tensor();
        std::vector<float> values(output.get_size());
        for (size_t i = 0; i < output.get_size(); ++i) {
            values[i] = output.data<const float>()[i];
        }
        return values;
    }

    size_t m_minBlobSize = 0;
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
                                            ::testing::Values(ov::AnyMap{}),
                                            // Small model, then slightly above: NativeStreamBuf window bypass (2 MiB),
                                            // ParallelMemStreamBuf file-backed path (4 MiB), NativeStreamBuf parallel reads (8 MiB).
                                            ::testing::Values(size_t{0},
                                                              ov::util::default_native_threshold,
                                                              ov::util::default_parallel_io_threshold,
                                                              ov::util::default_native_window)),
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
            ov::test::utils::removeIRFiles(cache_path.string() + ".xml", cache_path.string() + ".bin");
        }
        OVCompiledModelHsmBlobTest::TearDown();
    }
};

TEST_P(GPUHsmRegressionTest, StandaloneExportCanReplacePathCacheEntry) {
    const auto xml = cache_path.string() + ".xml";
    const auto bin = cache_path.string() + ".bin";
    ov::serialize(model, xml, bin);
    const auto standalone = export_blob(core.compile_model(xml, target_device, configuration));
    ASSERT_TRUE(is_hsm(standalone));

    auto config = configuration;
    config.emplace(ov::cache_path(cache_path));
    auto first = core.compile_model(xml, target_device, config);
    ASSERT_FALSE(first.get_property(ov::loaded_from_cache));
    const auto cache_blob = read_cache_blob();
    EXPECT_EQ(cache_blob, export_blob(first));
    std::stringstream stream(cache_blob);
    auto imported = core.import_model(stream, target_device);
    expect_same_inference(imported);

    ASSERT_NO_FATAL_FAILURE(write_cache_blob(standalone));
    for (const bool use_mmap : {false, true}) {
        SCOPED_TRACE(use_mmap);
        core.set_property(ov::enable_mmap(use_mmap));
        auto cached = core.compile_model(xml, target_device, config);
        EXPECT_TRUE(cached.get_property(ov::loaded_from_cache));
        expect_same_inference(cached);
        EXPECT_EQ(read_cache_blob(), standalone);
    }
}

TEST_P(GPUHsmRegressionTest, ImportUsesStoredCacheMode) {
    const auto xml = cache_path.string() + ".xml";
    const auto bin = cache_path.string() + ".bin";
    ov::serialize(model, xml, bin);
    auto config = configuration;
    config.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    const auto blob = export_blob(core.compile_model(xml, target_device, config));
    ASSERT_TRUE(is_hsm(blob));

    std::stringstream stream(blob);
    auto imported = core.import_model(stream, target_device, {ov::weights_path(bin)});
    EXPECT_EQ(imported.get_property(ov::cache_mode), ov::CacheMode::OPTIMIZE_SIZE);
    expect_same_inference(imported);
}

TEST_P(GPUHsmRegressionTest, WeightlessExportRoundTripsSharedKernels) {
    const ov::Shape input_shape{4, 32};
    auto first = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto second = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    const auto weights = ov::op::v0::Constant::create(ov::element::f32, {32, 32}, {0.25f});
    auto first_matmul = std::make_shared<ov::op::v0::MatMul>(first, weights);
    auto second_matmul = std::make_shared<ov::op::v0::MatMul>(second, weights);
    auto shared_kernel_model = std::make_shared<ov::Model>(ov::OutputVector{first_matmul, second_matmul},
                                                           ov::ParameterVector{first, second});
    const auto xml = cache_path.string() + ".xml";
    const auto bin = cache_path.string() + ".bin";
    ov::serialize(shared_kernel_model, xml, bin);

    auto first_input = ov::test::utils::create_and_fill_tensor(ov::element::f32, input_shape);
    auto second_input = ov::test::utils::create_and_fill_tensor(ov::element::f32, input_shape);
    std::fill_n(second_input.data<float>(), second_input.get_size(), 2.f);
    const auto infer = [&](ov::CompiledModel& compiled) {
        auto request = compiled.create_infer_request();
        request.set_input_tensor(0, first_input);
        request.set_input_tensor(1, second_input);
        request.infer();
        return request;
    };

    for (const bool use_hsm : {false, true}) {
        SCOPED_TRACE(use_hsm);
        auto config = configuration;
        config.insert_or_assign(ov::intel_gpu::export_hsm_format.name(), use_hsm);
        config.insert_or_assign(ov::cache_mode.name(), ov::CacheMode::OPTIMIZE_SIZE);
        auto compiled = core.compile_model(xml, target_device, config);
        const auto blob = export_blob(compiled);
        ASSERT_EQ(is_hsm(blob), use_hsm);
        auto expected = infer(compiled);

        std::stringstream stream(blob);
        ov::AnyMap import_config{ov::weights_path(bin)};
        if (!use_hsm) {
            import_config.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        }
        auto imported = core.import_model(stream, target_device, import_config);
        EXPECT_EQ(imported.get_property(ov::cache_mode), ov::CacheMode::OPTIMIZE_SIZE);
        auto actual = infer(imported);
        for (size_t output = 0; output < shared_kernel_model->get_output_size(); ++output) {
            ov::test::utils::compare(expected.get_output_tensor(output), actual.get_output_tensor(output));
        }
    }
}

TEST_P(GPUHsmRegressionTest, CacheHitsUsePerCallExportFormat) {
    for (const bool use_mmap : {false, true}) {
        core.set_property(ov::enable_mmap(use_mmap));
        for (const bool use_hsm : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "mmap=" << use_mmap << ", HSM=" << use_hsm);
            ov::test::utils::removeFilesWithExt(ov::util::path_to_string(cache_path), "blob");
            core.set_property(target_device, ov::intel_gpu::export_hsm_format(!use_hsm));
            const ov::AnyMap config{ov::cache_path(cache_path), ov::intel_gpu::export_hsm_format(use_hsm)};
            auto first = core.compile_model(model, target_device, config);
            EXPECT_FALSE(first.get_property(ov::loaded_from_cache));
            const auto exported = export_blob(first);
            EXPECT_EQ(is_hsm(exported), use_hsm);
            const auto cached_blob = read_cache_blob();
            EXPECT_EQ(is_hsm(cached_blob), use_hsm);
            if (use_hsm) {
                EXPECT_EQ(cached_blob, exported);
            }
            auto cached = core.compile_model(model, target_device, config);
            EXPECT_TRUE(cached.get_property(ov::loaded_from_cache));
            EXPECT_EQ(is_hsm(export_blob(cached)), use_hsm);
            expect_same_inference(cached);
        }
    }
}

TEST_P(GPUHsmRegressionTest, CacheHitsUseSelectedDeviceExportFormat) {
    const auto devices = core.get_property(target_device, ov::available_devices);
    if (devices.size() < 2) {
        GTEST_SKIP() << "Requires two GPU device IDs";
    }

    const auto first_device = target_device + "." + devices[0];
    const auto second_device = target_device + "." + devices[1];
    core.set_property(first_device, ov::intel_gpu::export_hsm_format(false));
    core.set_property(second_device, ov::intel_gpu::export_hsm_format(true));
    const ov::AnyMap config{ov::cache_path(cache_path)};
    for (const auto& device : {first_device, second_device}) {
        SCOPED_TRACE(device);
        auto first = core.compile_model(model, device, config);
        EXPECT_EQ(is_hsm(export_blob(first)), device == second_device);
        auto cached = core.compile_model(model, device, config);
        EXPECT_TRUE(cached.get_property(ov::loaded_from_cache));
        expect_same_inference(cached);
    }
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

TEST_P(GPUHsmRegressionTest, EnablingEncryptionReplacesPlaintextCache) {
    ASSERT_FALSE(compile(true).get_property(ov::loaded_from_cache));
    ASSERT_TRUE(compile(true).get_property(ov::loaded_from_cache));

    size_t encrypt_calls = 0;
    size_t decrypt_calls = 0;
    ov::EncryptionCallbacks callbacks;
    callbacks.encrypt = [&](const std::string& data) {
        ++encrypt_calls;
        return ov::util::codec_xor(data);
    };
    callbacks.decrypt = [&](const std::string& data) {
        ++decrypt_calls;
        return ov::util::codec_xor(data);
    };
    auto config = configuration;
    config.emplace(ov::cache_encryption_callbacks(callbacks));
    config.emplace(ov::cache_path(cache_path));

    EXPECT_FALSE(core.compile_model(model, target_device, config).get_property(ov::loaded_from_cache));
    EXPECT_EQ(encrypt_calls, 1u);
    auto cached = core.compile_model(model, target_device, config);
    EXPECT_TRUE(cached.get_property(ov::loaded_from_cache));
    EXPECT_EQ(decrypt_calls, 1u);
    expect_same_inference(cached);
}

TEST_P(GPUHsmRegressionTest, EncryptedPayloadRoundTripsThroughOffsetStreamAndTensors) {
    ov::EncryptionCallbacks callbacks;
    callbacks.encrypt = ov::util::codec_xor;
    callbacks.decrypt = ov::util::codec_xor;
    auto config = configuration;
    config.insert_or_assign(ov::cache_mode.name(), ov::CacheMode::OPTIMIZE_SPEED);
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

    const auto path = cache_path.string() + ".mapped";
    {
        std::ofstream file(path, std::ios::binary);
        file.write(blob.data(), static_cast<std::streamsize>(blob.size()));
        ASSERT_TRUE(file.good());
    }
    const auto mapping = ov::load_mmap_object(path);
    ov::Tensor mapped_tensor(ov::element::u8, ov::Shape{blob.size()}, mapping->data());
    auto from_mapping = core.import_model(mapped_tensor, target_device, config);
    expect_same_inference(from_mapping);
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
    for (const bool pad_for_zero_copy : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "pad_for_zero_copy=" << pad_for_zero_copy);
        set_env("OV_GPU_ENABLE_ZERO_COPY_CACHE_LOAD", pad_for_zero_copy ? "YES" : "NO");
        const auto blob = export_blob(compile(false));
        set_env("OV_GPU_ENABLE_ZERO_COPY_CACHE_LOAD", "YES");
        ov::CompiledModel imported;
        {
            const auto alignment = std::max(ov::util::min_page_alignment, static_cast<size_t>(ov::util::get_system_page_size()));
            ov::Tensor storage(ov::element::u8, ov::Shape{blob.size() + alignment});
            const auto offset = (alignment - reinterpret_cast<uintptr_t>(storage.data()) % alignment) % alignment;
            ov::Tensor tensor(storage, {offset}, {offset + blob.size()});
            std::memcpy(tensor.data(), blob.data(), blob.size());
            imported = core.import_model(tensor, target_device, configuration);
        }
        expect_same_inference(imported);
    }
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
