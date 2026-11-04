// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <list>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "include/auto_unit_test.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/compilation_context.hpp"
#include "openvino/runtime/threading/immediate_executor.hpp"

using namespace ov::mock_auto_plugin;

namespace cache_warmup_test {
class ThrowOnCloneRelu : public ov::op::v0::Relu {
public:
    OPENVINO_OP("ThrowOnCloneRelu", "test", ov::op::v0::Relu);
    ThrowOnCloneRelu() = default;
    explicit ThrowOnCloneRelu(const ov::Output<ov::Node>& arg) : ov::op::v0::Relu(arg) {}
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector&) const override {
        OPENVINO_THROW("clone is not supported");
    }
};
}  // namespace cache_warmup_test

namespace {
bool wait_until(const std::function<bool()>& condition,
                std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!condition()) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return true;
}

enum class CacheKind { NONE, CACHE_DIR, CACHE_PATH_DIR, CACHE_PATH_FILE };

std::string to_string(CacheKind kind) {
    switch (kind) {
    case CacheKind::NONE:
        return "noCache";
    case CacheKind::CACHE_DIR:
        return "cacheDir";
    case CacheKind::CACHE_PATH_DIR:
        return "cachePathDir";
    default:
        return "cachePathFile";
    }
}
}  // namespace

class AutoCacheWarmupTestBase : public tests::AutoTest {
protected:
    const std::string cpu_device = ov::test::utils::DEVICE_CPU;
    const std::string actual_device = "GPU.1";

    // Like the Core, every device inherits the global setting unless it has its own.
    void set_global_cache(const std::string& cache_dir, const std::filesystem::path& cache_path = {}) {
        ON_CALL(*core, get_property(_, StrEq(ov::cache_dir.name()), _)).WillByDefault(Return(ov::Any(cache_dir)));
        ON_CALL(*core, get_property(_, StrEq(ov::cache_path.name()), _)).WillByDefault(Return(ov::Any(cache_path)));
    }

    void set_device_cache(const std::string& device, const std::string& cache_dir) {
        ON_CALL(*core, get_property(StrEq(device), StrEq(ov::cache_dir.name()), _))
            .WillByDefault(Return(ov::Any(cache_dir)));
        ON_CALL(*core, get_property(StrEq(device), StrEq(ov::cache_path.name()), _))
            .WillByDefault(Return(ov::Any(std::filesystem::path(cache_dir))));
    }

    // The last device is selected as the actual device.
    void set_devices(const std::vector<std::string>& devices) {
        metaDevices.clear();
        std::string priorities;
        for (const auto& device : devices) {
            metaDevices.push_back({device, {}, -1});
            priorities += (priorities.empty() ? "" : ",") + device;
        }
        ON_CALL(*plugin, parse_meta_devices(_, _)).WillByDefault(Return(metaDevices));
        ON_CALL(*plugin, select_device(_, _, _, _, _)).WillByDefault(Return(metaDevices.back()));
        config.insert(ov::device::priorities(priorities));
    }

    void on_compile(const std::string& device, std::function<ov::SoPtr<ov::ICompiledModel>()> action) {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(device)),
                              _))
            .WillByDefault(InvokeWithoutArgs(std::move(action)));
    }

    std::shared_ptr<NiceMock<ov::MockICompiledModel>> make_compiled_model() {
        auto compiled_model = std::make_shared<NiceMock<ov::MockICompiledModel>>(model, mock_plugin_gpu);
        ON_CALL(*compiled_model, inputs()).WillByDefault(ReturnRefOfCopy(model->inputs()));
        ON_CALL(*compiled_model, outputs()).WillByDefault(ReturnRefOfCopy(model->outputs()));
        return compiled_model;
    }

    // The actual device takes a while to compile.
    void use_default_startup_compilation() {
        on_compile(cpu_device, [this]() -> ov::SoPtr<ov::ICompiledModel> {
            return {mockIExeNet, {}};
        });
        on_compile(actual_device, [this]() -> ov::SoPtr<ov::ICompiledModel> {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return {mockIExeNetActual, {}};
        });
    }
};

using CacheWarmupParams = std::tuple<CacheKind, bool>;

class AutoCacheCompileForAllTest : public AutoCacheWarmupTestBase,
                                   public ::testing::TestWithParam<CacheWarmupParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CacheWarmupParams>& obj) {
        const auto& [cache_kind, compile_for_all] = obj.param;
        return to_string(cache_kind) + "_" + (compile_for_all ? "compileForAll" : "compileForActual");
    }
};

TEST_P(AutoCacheCompileForAllTest, compileForAllCompilesOtherDevicesWhenEnabled) {
    const auto& [cache_kind, compile_for_all] = this->GetParam();
    switch (cache_kind) {
    case CacheKind::NONE:
        set_global_cache("");
        break;
    case CacheKind::CACHE_DIR:
        set_global_cache("test_cache_dir");
        break;
    case CacheKind::CACHE_PATH_DIR:
        set_global_cache("", "test_cache_dir");
        break;
    case CacheKind::CACHE_PATH_FILE:
        set_global_cache("", "test_cache.bin");
        break;
    }
    set_devices({cpu_device, "GPU.0", actual_device});
    use_default_startup_compilation();
    auto warmup_compiled_model = make_compiled_model();
    std::atomic<int> warmup_count{0};
    on_compile("GPU.0", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        warmup_count++;
        return {warmup_compiled_model, {}};
    });
    config.insert(ov::intel_auto::compile_for_all(compile_for_all));
    const auto baseline = warmup_compiled_model.use_count();

    if (cache_kind == CacheKind::NONE && compile_for_all) {
        EXPECT_THROW(plugin->compile_model(model, config), ov::Exception);
        return;
    }
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    // A single-file cache is not warmed up because concurrent writes into one archive are unsafe.
    if (cache_kind != CacheKind::NONE && cache_kind != CacheKind::CACHE_PATH_FILE && compile_for_all) {
        EXPECT_TRUE(wait_until([&] {
            return warmup_count.load() == 1 && warmup_compiled_model.use_count() == baseline;
        }));
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        EXPECT_EQ(warmup_count.load(), 0);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         AutoCacheCompileForAllTest,
                         ::testing::Combine(::testing::Values(CacheKind::NONE,
                                                              CacheKind::CACHE_DIR,
                                                              CacheKind::CACHE_PATH_DIR,
                                                              CacheKind::CACHE_PATH_FILE),
                                            ::testing::Bool()),
                         AutoCacheCompileForAllTest::getTestCaseName);

class AutoCacheWarmupTest : public AutoCacheWarmupTestBase, public ::testing::Test {};

TEST_F(AutoCacheWarmupTest, ignoredWhenNoOtherCandidateDevice) {
    set_global_cache("test_cache_dir");
    set_devices({actual_device});
    use_default_startup_compilation();
    config.insert(ov::intel_auto::compile_for_all(true));
    OV_ASSERT_NO_THROW(plugin->compile_model(model, config));
}

TEST_F(AutoCacheWarmupTest, excludesCpuFromBackgroundPreCompilation) {
    set_global_cache("test_cache_dir");
    set_devices({cpu_device, "GPU.0", actual_device});
    std::atomic<int> cpu_count{0};
    std::atomic<int> warmup_count{0};
    on_compile(cpu_device, [&]() -> ov::SoPtr<ov::ICompiledModel> {
        cpu_count++;
        return {mockIExeNet, {}};
    });
    on_compile(actual_device, [this]() -> ov::SoPtr<ov::ICompiledModel> {
        return {mockIExeNetActual, {}};
    });
    auto warmup_compiled_model = make_compiled_model();
    on_compile("GPU.0", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        warmup_count++;
        return {warmup_compiled_model, {}};
    });
    config.insert(ov::intel_auto::compile_for_all(true));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_TRUE(wait_until([&] {
        return warmup_count.load() == 1;
    }));
    EXPECT_FALSE(wait_until([&] {
        return cpu_count.load() > 0;
    }, std::chrono::milliseconds(200)));
}

TEST_F(AutoCacheWarmupTest, doesNotDisableCpuHelperWhenOnlyCpuWouldBeWarmedUp) {
    set_global_cache("test_cache_dir");
    set_devices({cpu_device, actual_device});
    use_default_startup_compilation();
    std::atomic<int> cpu_count{0};
    on_compile(cpu_device, [&]() -> ov::SoPtr<ov::ICompiledModel> {
        cpu_count++;
        return {mockIExeNet, {}};
    });
    config.insert(ov::intel_auto::compile_for_all(true));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_TRUE(wait_until([&] {
        return cpu_count.load() == 1;
    }));
}

TEST_F(AutoCacheWarmupTest, disablesStartupAndRuntimeFallback) {
    set_global_cache("test_cache_dir");
    set_devices({cpu_device, "GPU.0", actual_device});
    std::atomic<int> cpu_count{0};
    std::atomic<int> warmup_count{0};
    on_compile(cpu_device, [&]() -> ov::SoPtr<ov::ICompiledModel> {
        cpu_count++;
        return {mockIExeNet, {}};
    });
    on_compile(actual_device, [this]() -> ov::SoPtr<ov::ICompiledModel> {
        return {mockIExeNetActual, {}};
    });
    auto warmup_compiled_model = make_compiled_model();
    on_compile("GPU.0", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        warmup_count++;
        return {warmup_compiled_model, {}};
    });
    auto executor = std::make_shared<ov::threading::ImmediateExecutor>();
    ON_CALL(*mockIExeNetActual, create_infer_request())
        .WillByDefault([this, executor]() -> std::shared_ptr<ov::IAsyncInferRequest> {
            auto sync_request = std::make_shared<ov::mock_auto_plugin::MockISyncInferRequest>(mockIExeNetActual);
            return std::make_shared<ov::mock_auto_plugin::MockAsyncInferRequest>(sync_request, executor, nullptr, true);
        });
    config.insert(ov::intel_auto::compile_for_all(true));
    config.insert(ov::intel_auto::enable_startup_fallback(true));
    config.insert(ov::intel_auto::enable_runtime_fallback(true));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    std::shared_ptr<ov::IAsyncInferRequest> request;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    request = compiled_model->create_infer_request();
    // No runtime fallback: the failure of the actual device reaches the caller.
    EXPECT_THROW(request->infer(), ov::Exception);
    EXPECT_TRUE(wait_until([&] {
        return warmup_count.load() == 1;
    }));
    // The CPU helper and CPU background pre-compilation are both disabled.
    EXPECT_EQ(cpu_count.load(), 0);
}

TEST_F(AutoCacheWarmupTest, delegatesCachedBlobValidationToCore) {
    const auto cache_dir = std::filesystem::temp_directory_path() /
                           ("ov_auto_cache_warmup_" +
                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(cache_dir);
    // The mocked core returns an empty compile config, so the blob id only depends on the model.
    std::ofstream(cache_dir / (ov::ModelCache::compute_hash(model, ov::AnyMap{}) + ".blob")) << "blob";
    set_global_cache(cache_dir.string());
    set_devices({cpu_device, "GPU.0", "NPU", actual_device});
    use_default_startup_compilation();
    std::atomic<int> gpu_count{0};
    std::atomic<int> npu_count{0};
    auto warmup_compiled_model = make_compiled_model();
    on_compile("GPU.0", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        gpu_count++;
        return {warmup_compiled_model, {}};
    });
    on_compile("NPU", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        npu_count++;
        return {warmup_compiled_model, {}};
    });
    // Core decides whether the existing GPU.0 blob is usable; NPU has a different cache key and also compiles.
    ON_CALL(*core, create_compile_config(StrEq("NPU"), _)).WillByDefault(Return(ov::AnyMap{{"NPU_KEY", "1"}}));
    config.insert(ov::intel_auto::compile_for_all(true));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_TRUE(wait_until([&] {
        return npu_count.load() == 1;
    }));
    EXPECT_EQ(gpu_count.load(), 1);
    compiled_model.reset();
    std::filesystem::remove_all(cache_dir);
}

TEST_F(AutoCacheWarmupTest, excludedDeviceIsNotProbedForCachedBlob) {
    const auto cache_dir = std::filesystem::temp_directory_path() /
                           ("ov_auto_cache_excluded_" +
                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(cache_dir);
    // The mocked core returns an empty compile config, so the blob id only depends on the model.
    std::ofstream(cache_dir / (ov::ModelCache::compute_hash(model, ov::AnyMap{}) + ".blob")) << "blob";
    ov::AnyMap properties = {ov::device::priorities("CPU,-GPU"), ov::cache_path(cache_dir)};
    std::string devices;
    OV_ASSERT_NO_THROW(devices = plugin->Plugin::get_device_list(properties, model, {}));
    EXPECT_EQ(devices, cpu_device);
    std::filesystem::remove_all(cache_dir);
}

TEST_F(AutoCacheWarmupTest, deviceWithCachingDisabledIsNotProbedForCachedBlob) {
    const auto cache_dir = std::filesystem::temp_directory_path() /
                           ("ov_auto_cache_device_disabled_" +
                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(cache_dir);
    // The mocked core returns an empty compile config, so the blob id only depends on the model.
    std::ofstream(cache_dir / (ov::ModelCache::compute_hash(model, ov::AnyMap{}) + ".blob")) << "blob";
    // The matching blob sits in the global cache, but caching is disabled for GPU itself.
    ov::AnyMap properties = {ov::device::priorities("CPU,GPU"),
                             ov::cache_path(cache_dir),
                             ov::device::properties(ov::AnyMap{{"GPU", ov::AnyMap{{ov::cache_path.name(), std::string{}}}}})};
    std::string devices;
    OV_ASSERT_NO_THROW(devices = plugin->Plugin::get_device_list(properties, model, {}));
    EXPECT_NE(devices.find(cpu_device), std::string::npos);
    EXPECT_EQ(properties.count(ov::intel_auto::enable_startup_fallback.name()), 0u);
    EXPECT_EQ(properties.count(ov::intel_auto::enable_runtime_fallback.name()), 0u);
    std::filesystem::remove_all(cache_dir);
}

TEST_F(AutoCacheWarmupTest, cacheProbeKeepsDeviceId) {
    const auto cache_dir = std::filesystem::temp_directory_path() /
                           ("ov_auto_cache_probe_device_id_" +
                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(cache_dir);
    std::string probed_id;
    ON_CALL(*core, create_compile_config(StrEq("GPU"), _))
        .WillByDefault(::testing::Invoke([&probed_id](const std::string&, const ov::AnyMap& config) -> ov::AnyMap {
            if (const auto it = config.find(ov::device::id.name()); it != config.end()) {
                probed_id = it->second.as<std::string>();
            }
            return {};
        }));
    ov::AnyMap properties = {ov::device::priorities("CPU,GPU.1"), ov::cache_path(cache_dir)};
    OV_ASSERT_NO_THROW(plugin->Plugin::get_device_list(properties, model, {}));
    EXPECT_EQ(probed_id, "1");
    std::filesystem::remove_all(cache_dir);
}

TEST_F(AutoCacheWarmupTest, failingCacheProbeDoesNotBreakDeviceList) {
    const auto cache_dir = std::filesystem::temp_directory_path() /
                           ("ov_auto_cache_probe_failure_" +
                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(cache_dir);
    ON_CALL(*core, create_compile_config(_, _)).WillByDefault(::testing::Invoke([](const std::string&, const ov::AnyMap&) -> ov::AnyMap {
        OPENVINO_THROW("no plugin");
    }));
    ov::AnyMap properties = {ov::device::priorities("CPU,GPU"), ov::cache_path(cache_dir)};
    std::string devices;
    OV_ASSERT_NO_THROW(devices = plugin->Plugin::get_device_list(properties, model, {}));
    EXPECT_NE(devices.find(cpu_device), std::string::npos);
    EXPECT_EQ(properties.count(ov::intel_auto::enable_startup_fallback.name()), 0u);
    std::filesystem::remove_all(cache_dir);
}

TEST_F(AutoCacheWarmupTest, failingConfigLookupDoesNotBreakDeviceList) {
    set_global_cache("");
    // e.g. GPU.1 configured with a conflicting ov::device::id; parse_meta_devices() discards it later.
    ON_CALL(*core, get_supported_property(StrEq("GPU.1"), _, _))
        .WillByDefault(::testing::Invoke([](const std::string&, const ov::AnyMap&, const bool) -> ov::AnyMap {
            OPENVINO_THROW("conflicting device id");
        }));
    ov::AnyMap properties = {ov::device::priorities("GPU.1,CPU")};
    std::string devices;
    OV_ASSERT_NO_THROW(devices = plugin->Plugin::get_device_list(properties, model, {}));
    EXPECT_NE(devices.find(cpu_device), std::string::npos);
}

TEST_F(AutoCacheWarmupTest, nonOvExceptionInCacheProbeDoesNotBreakDeviceList) {
    ON_CALL(*core, get_supported_property(StrEq("GPU.1"), _, _))
        .WillByDefault(::testing::Invoke([](const std::string&, const ov::AnyMap&, const bool) -> ov::AnyMap {
            throw std::bad_cast();
        }));
    ov::AnyMap properties = {ov::device::priorities("GPU.1,CPU"), ov::cache_path("test_cache_dir")};
    std::string devices;
    OV_ASSERT_NO_THROW(devices = plugin->Plugin::get_device_list(properties, model, {}));
    EXPECT_NE(devices.find(cpu_device), std::string::npos);
}

TEST_F(AutoCacheWarmupTest, customBlobIdDoesNotPruneCpuFallback) {
    const auto cache_dir = std::filesystem::temp_directory_path() /
                           ("ov_auto_cache_blob_id_prune_" +
                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(cache_dir);
    // A custom ID carries no device information, so this file must not count as a GPU cache hit.
    std::ofstream(cache_dir / "746352.blob") << "blob";
    ov::AnyMap properties = {ov::device::priorities("CPU,GPU"), ov::cache_path(cache_dir), ov::cache_blob_id(746352)};
    std::string devices;
    OV_ASSERT_NO_THROW(devices = plugin->Plugin::get_device_list(properties, model, {}));
    EXPECT_NE(devices.find(cpu_device), std::string::npos);
    EXPECT_EQ(properties.count(ov::intel_auto::enable_startup_fallback.name()), 0u);
    EXPECT_EQ(properties.count(ov::intel_auto::enable_runtime_fallback.name()), 0u);
    std::filesystem::remove_all(cache_dir);
}

TEST_F(AutoCacheWarmupTest, compileForAllKeepsCpuRetryCandidateWhenOtherDeviceHasCachedBlob) {
    const auto cache_dir = std::filesystem::temp_directory_path() /
                           ("ov_auto_cache_keep_cpu_blob_" +
                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(cache_dir);
    std::ofstream(cache_dir / (ov::ModelCache::compute_hash(model, ov::AnyMap{}) + ".blob")) << "blob";
    ov::AnyMap properties = {ov::device::priorities("CPU,GPU"),
                             ov::cache_path(cache_dir),
                             ov::intel_auto::compile_for_all(true)};
    std::string devices;
    OV_ASSERT_NO_THROW(devices = plugin->Plugin::get_device_list(properties, model, {}));
    EXPECT_NE(devices.find(cpu_device), std::string::npos);
    EXPECT_NE(devices.find("GPU"), std::string::npos);
    EXPECT_EQ(properties.count(ov::intel_auto::enable_startup_fallback.name()), 0u);
    EXPECT_EQ(properties.count(ov::intel_auto::enable_runtime_fallback.name()), 0u);
    std::filesystem::remove_all(cache_dir);
}

TEST_F(AutoCacheWarmupTest, compileForAllKeepsCpuRetryCandidateWhenBothFallbacksAreDisabled) {
    set_global_cache("test_cache_dir");
    ov::AnyMap properties = {ov::device::priorities("CPU,GPU"),
                             ov::intel_auto::compile_for_all(true),
                             ov::intel_auto::enable_startup_fallback(false),
                             ov::intel_auto::enable_runtime_fallback(false)};
    std::string devices;
    OV_ASSERT_NO_THROW(devices = plugin->Plugin::get_device_list(properties, model, {}));
    EXPECT_NE(devices.find(cpu_device), std::string::npos);

    ov::AnyMap legacy_properties = {ov::device::priorities("CPU,GPU"),
                                    ov::intel_auto::enable_startup_fallback(false),
                                    ov::intel_auto::enable_runtime_fallback(false)};
    OV_ASSERT_NO_THROW(devices = plugin->Plugin::get_device_list(legacy_properties, model, {}));
    EXPECT_EQ(devices.find(cpu_device), std::string::npos);
}

TEST_F(AutoCacheWarmupTest, deviceScopedBlobIdKeepsLegacyCpuPruning) {
    ON_CALL(*core, get_supported_property(StrEq("GPU"), _, _))
        .WillByDefault(Return(ov::AnyMap{{ov::cache_blob_id.name(), std::string{"746352"}}}));
    ov::AnyMap properties = {ov::device::priorities("CPU,GPU"),
                             ov::cache_path("test_cache_dir"),
                             ov::intel_auto::compile_for_all(true),
                             ov::intel_auto::enable_startup_fallback(false),
                             ov::intel_auto::enable_runtime_fallback(false),
                             ov::device::properties(
                                 ov::AnyMap{{"GPU", ov::AnyMap{{ov::cache_blob_id.name(), std::string{"746352"}}}}})};
    std::string devices;
    OV_ASSERT_NO_THROW(devices = plugin->Plugin::get_device_list(properties, model, {}));
    EXPECT_EQ(devices.find(cpu_device), std::string::npos);
    EXPECT_NE(devices.find("GPU"), std::string::npos);
}

class AutoCacheWarmupSharedManagerTest : public AutoCacheWarmupTestBase, public ::testing::Test {
protected:
    std::promise<void> gate;
    std::shared_future<void> gate_future = gate.get_future().share();
    std::once_flag gate_opened;
    std::atomic<int> actual_count{0};
    std::atomic<int> warmup_count{0};
    std::atomic<int> in_flight{0};
    std::atomic<int> max_in_flight{0};
    std::shared_ptr<NiceMock<ov::MockICompiledModel>> warmup_compiled_model;
    // Must be destroyed after the gate is opened, otherwise CacheWarmup::stop() blocks.
    std::shared_ptr<ov::ICompiledModel> first;
    std::shared_ptr<ov::ICompiledModel> second;

    // The first warm-up compilation blocks until the gate is opened.
    void SetUp() override {
        set_global_cache("test_cache_dir");
        set_devices({cpu_device, "GPU.0", actual_device});
        on_compile(cpu_device, [this]() -> ov::SoPtr<ov::ICompiledModel> {
            return {mockIExeNet, {}};
        });
        on_compile(actual_device, [this]() -> ov::SoPtr<ov::ICompiledModel> {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            actual_count++;
            return {mockIExeNetActual, {}};
        });
        warmup_compiled_model = make_compiled_model();
        on_compile("GPU.0", [this]() -> ov::SoPtr<ov::ICompiledModel> {
            const auto current = ++in_flight;
            max_in_flight = std::max(max_in_flight.load(), current);
            if (++warmup_count == 1) {
                gate_future.wait();
            }
            --in_flight;
            return {warmup_compiled_model, {}};
        });
        config.insert(ov::intel_auto::compile_for_all(true));
    }

    void open_gate() {
        std::call_once(gate_opened, [this] {
            gate.set_value();
        });
    }

    void TearDown() override {
        open_gate();
    }
};

TEST_F(AutoCacheWarmupSharedManagerTest, warmUpsOfDifferentModelsDoNotOverlap) {
    OV_ASSERT_NO_THROW(first = plugin->compile_model(model, config));
    ASSERT_TRUE(wait_until([&] {
        return warmup_count.load() == 1;
    }));
    OV_ASSERT_NO_THROW(second = plugin->compile_model(model->clone(), config));
    ASSERT_TRUE(wait_until([&] {
        return actual_count.load() == 2;
    }));
    // The second warm-up is queued behind the blocked first one.
    EXPECT_FALSE(wait_until(
        [&] {
            return warmup_count.load() > 1;
        },
        std::chrono::milliseconds(500)));
    open_gate();
    EXPECT_TRUE(wait_until([&] {
        return warmup_count.load() == 2;
    }));
    EXPECT_EQ(max_in_flight.load(), 1);
}

TEST_F(AutoCacheWarmupSharedManagerTest, destroyingQueuedModelCancelsItsWarmUp) {
    OV_ASSERT_NO_THROW(first = plugin->compile_model(model, config));
    ASSERT_TRUE(wait_until([&] {
        return warmup_count.load() == 1;
    }));
    OV_ASSERT_NO_THROW(second = plugin->compile_model(model->clone(), config));
    ASSERT_TRUE(wait_until([&] {
        return actual_count.load() == 2;
    }));
    // Give the second warm-up time to be queued behind the blocked first one.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    auto destroyed = std::async(std::launch::async, [this] {
        second.reset();
    });
    EXPECT_EQ(destroyed.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    open_gate();
    first.reset();
    // Neither the cancelled model nor the one just destroyed may compile GPU.0 again.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_EQ(warmup_count.load(), 1);
    EXPECT_EQ(max_in_flight.load(), 1);
}

TEST_F(AutoCacheWarmupTest, destructionSkipsRemainingDevices) {
    set_global_cache("test_cache_dir");
    set_devices({cpu_device, "GPU.0", "NPU", actual_device});
    use_default_startup_compilation();
    std::promise<void> warmup_started;
    std::atomic<int> npu_count{0};
    auto warmup_compiled_model = make_compiled_model();
    on_compile("GPU.0", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        warmup_started.set_value();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        return {warmup_compiled_model, {}};
    });
    on_compile("NPU", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        npu_count++;
        return {warmup_compiled_model, {}};
    });
    config.insert(ov::intel_auto::compile_for_all(true));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    ASSERT_EQ(warmup_started.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);
    compiled_model.reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(npu_count.load(), 0);
}

TEST_F(AutoCacheWarmupTest, failedActualDeviceIsNotWarmedUp) {
    set_global_cache("test_cache_dir");
    set_devices({cpu_device, "GPU.0", actual_device});
    std::atomic<int> select_count{0};
    ON_CALL(*plugin, select_device(_, _, _, _, _)).WillByDefault(InvokeWithoutArgs([&]() {
        return select_count++ == 0 ? metaDevices[2] : metaDevices[1];
    }));
    on_compile(cpu_device, [this]() -> ov::SoPtr<ov::ICompiledModel> {
        return {mockIExeNet, {}};
    });
    std::atomic<int> actual_count{0};
    on_compile(actual_device, [&]() -> ov::SoPtr<ov::ICompiledModel> {
        actual_count++;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        OPENVINO_THROW("compile failure on ", actual_device);
    });
    std::atomic<int> retry_count{0};
    on_compile("GPU.0", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        retry_count++;
        return {mockIExeNetActual, {}};
    });
    config.insert(ov::intel_auto::compile_for_all(true));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    ASSERT_TRUE(wait_until([&] {
        return retry_count.load() == 1;
    }));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(actual_count.load(), 1);
    EXPECT_EQ(retry_count.load(), 1);
}

TEST_F(AutoCacheWarmupTest, emptyConfigCpuIsRetriedWhenActualDeviceFails) {
    set_global_cache("test_cache_dir");
    set_devices({cpu_device, actual_device});
    config.insert(ov::intel_auto::enable_startup_fallback(false));
    std::atomic<int> select_count{0};
    ON_CALL(*plugin, select_device(_, _, _, _, _)).WillByDefault(InvokeWithoutArgs([&]() {
        return select_count++ == 0 ? metaDevices[1] : metaDevices[0];
    }));
    std::atomic<int> cpu_count{0};
    on_compile(cpu_device, [&]() -> ov::SoPtr<ov::ICompiledModel> {
        cpu_count++;
        return {mockIExeNet, {}};
    });
    on_compile(actual_device, [&]() -> ov::SoPtr<ov::ICompiledModel> {
        OPENVINO_THROW("compile failure on ", actual_device);
    });
    config.insert(ov::intel_auto::compile_for_all(true));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_TRUE(wait_until([&] {
        return cpu_count.load() == 1;
    }));
}

TEST_F(AutoCacheWarmupTest, warmsUpOtherDeviceWhenActualFailsAndCpuTakesOver) {
    set_global_cache("test_cache_dir");
    set_devices({actual_device, cpu_device, "GPU.0"});
    std::atomic<int> select_count{0};
    ON_CALL(*plugin, select_device(_, _, _, _, _)).WillByDefault(InvokeWithoutArgs([&]() {
        return select_count++ == 0 ? metaDevices[0] : metaDevices[1];
    }));
    std::atomic<int> cpu_count{0};
    on_compile(cpu_device, [&]() -> ov::SoPtr<ov::ICompiledModel> {
        cpu_count++;
        return {mockIExeNet, {}};
    });
    on_compile(actual_device, [&]() -> ov::SoPtr<ov::ICompiledModel> {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        OPENVINO_THROW("compile failure on ", actual_device);
    });
    std::atomic<int> warmup_count{0};
    auto warmup_compiled_model = make_compiled_model();
    on_compile("GPU.0", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        warmup_count++;
        return {warmup_compiled_model, {}};
    });
    config.insert(ov::intel_auto::compile_for_all(true));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_TRUE(wait_until([&] {
        return warmup_count.load() == 1;
    }));
    EXPECT_EQ(cpu_count.load(), 1);
}

TEST_F(AutoCacheWarmupTest, warmsUpOtherDeviceWhenCpuIsActualDevice) {
    set_global_cache("test_cache_dir");
    set_devices({"GPU.0", cpu_device});
    on_compile(cpu_device, [this]() -> ov::SoPtr<ov::ICompiledModel> {
        return {mockIExeNet, {}};
    });
    std::atomic<int> warmup_count{0};
    auto warmup_compiled_model = make_compiled_model();
    on_compile("GPU.0", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        warmup_count++;
        return {warmup_compiled_model, {}};
    });
    config.insert(ov::intel_auto::compile_for_all(true));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_TRUE(wait_until([&] {
        return warmup_count.load() == 1;
    }));
}

TEST_F(AutoCacheWarmupTest, singleNonCpuCandidateWithoutCacheThrows) {
    set_global_cache("");
    set_devices({"GPU.0", cpu_device});
    on_compile(cpu_device, [this]() -> ov::SoPtr<ov::ICompiledModel> {
        return {mockIExeNet, {}};
    });
    on_compile("GPU.0", [this]() -> ov::SoPtr<ov::ICompiledModel> {
        return {mockIExeNetActual, {}};
    });
    config.insert(ov::intel_auto::compile_for_all(true));
    EXPECT_THROW(plugin->compile_model(model, config), ov::Exception);
}

TEST_F(AutoCacheWarmupTest, noOtherCandidateWithoutCacheThrows) {
    set_global_cache("");
    set_devices({cpu_device, actual_device});
    use_default_startup_compilation();
    config.insert(ov::intel_auto::compile_for_all(true));
    EXPECT_THROW(plugin->compile_model(model, config), ov::Exception);
}

TEST_F(AutoCacheWarmupTest, ignoredWhenCacheBlobIdIsSet) {
    set_global_cache("test_cache_dir");
    set_devices({cpu_device, "GPU.0", actual_device});
    use_default_startup_compilation();
    std::atomic<int> warmup_count{0};
    auto warmup_compiled_model = make_compiled_model();
    on_compile("GPU.0", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        warmup_count++;
        return {warmup_compiled_model, {}};
    });
    config.insert(ov::intel_auto::compile_for_all(true));
    config.insert(ov::cache_blob_id(746352));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_FALSE(wait_until([&] {
        return warmup_count.load() > 0;
    }, std::chrono::milliseconds(500)));
}

TEST_F(AutoCacheWarmupTest, ignoredWhenCandidateHasCacheBlobId) {
    set_global_cache("test_cache_dir");
    set_devices({cpu_device, "GPU.0", actual_device});
    metaDevices[1].config = ov::AnyMap{{ov::cache_blob_id.name(), std::string{"746352"}}};
    ON_CALL(*plugin, parse_meta_devices(_, _)).WillByDefault(Return(metaDevices));
    use_default_startup_compilation();
    std::atomic<int> warmup_count{0};
    auto warmup_compiled_model = make_compiled_model();
    on_compile("GPU.0", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        warmup_count++;
        return {warmup_compiled_model, {}};
    });
    config.insert(ov::intel_auto::compile_for_all(true));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_FALSE(wait_until([&] {
        return warmup_count.load() > 0;
    }, std::chrono::milliseconds(500)));
}

TEST_F(AutoCacheWarmupTest, blobProbeHonorsCacheBlobId) {
    const auto cache_dir = std::filesystem::temp_directory_path() /
                           ("ov_auto_cache_blob_id_" +
                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(cache_dir);
    std::ofstream(cache_dir / "746352.blob") << "blob";
    const auto is_cached = [&](const ov::AnyMap& properties) {
        return ov::auto_plugin::CacheWarmupManager::is_blob_cached(*core, "GPU.0", properties, model, {}, cache_dir);
    };
    EXPECT_TRUE(is_cached({ov::cache_blob_id(746352)}));
    EXPECT_FALSE(is_cached({ov::cache_blob_id(1)}));
    // Without a user ID the hash is used, and no blob exists for it.
    EXPECT_FALSE(is_cached({}));
    std::filesystem::remove_all(cache_dir);
}

TEST_F(AutoCacheWarmupTest, skipsDeviceWithCachingDisabled) {
    set_global_cache("test_cache_dir");
    set_devices({cpu_device, "GPU.0", "NPU", actual_device});
    metaDevices[2].config = ov::AnyMap{{ov::cache_dir.name(), std::string{}}};
    ON_CALL(*plugin, parse_meta_devices(_, _)).WillByDefault(Return(metaDevices));
    use_default_startup_compilation();
    std::atomic<int> gpu_count{0};
    std::atomic<int> npu_count{0};
    auto warmup_compiled_model = make_compiled_model();
    on_compile("GPU.0", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        gpu_count++;
        return {warmup_compiled_model, {}};
    });
    on_compile("NPU", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        npu_count++;
        return {warmup_compiled_model, {}};
    });
    config.insert(ov::intel_auto::compile_for_all(true));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_TRUE(wait_until([&] {
        return gpu_count.load() == 1;
    }));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(npu_count.load(), 0);
}

TEST_F(AutoCacheWarmupTest, skipsDeviceWithCachingDisabledInCore) {
    set_global_cache("test_cache_dir");
    set_device_cache("NPU", "");
    set_devices({cpu_device, "GPU.0", "NPU", actual_device});
    use_default_startup_compilation();
    std::atomic<int> gpu_count{0};
    std::atomic<int> npu_count{0};
    auto warmup_compiled_model = make_compiled_model();
    on_compile("GPU.0", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        gpu_count++;
        return {warmup_compiled_model, {}};
    });
    on_compile("NPU", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        npu_count++;
        return {warmup_compiled_model, {}};
    });
    config.insert(ov::intel_auto::compile_for_all(true));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_TRUE(wait_until([&] {
        return gpu_count.load() == 1;
    }));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(npu_count.load(), 0);
}

TEST_F(AutoCacheWarmupTest, usesDeviceScopedCacheSetting) {
    set_global_cache("");
    set_device_cache("GPU.0", "device_cache_dir");
    set_device_cache(actual_device, "device_cache_dir");
    set_devices({cpu_device, "GPU.0", actual_device});
    use_default_startup_compilation();
    std::atomic<int> warmup_count{0};
    auto warmup_compiled_model = make_compiled_model();
    on_compile("GPU.0", [&]() -> ov::SoPtr<ov::ICompiledModel> {
        warmup_count++;
        return {warmup_compiled_model, {}};
    });
    config.insert(ov::intel_auto::compile_for_all(true));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    EXPECT_TRUE(wait_until([&] {
        return warmup_count.load() == 1;
    }));
}

TEST_F(AutoCacheWarmupTest, warmsUpPrivateModelSnapshot) {
    set_global_cache("test_cache_dir");
    set_devices({cpu_device, "GPU.0", actual_device});
    use_default_startup_compilation();
    std::mutex captured_mutex;
    std::shared_ptr<const ov::Model> captured_model;
    auto warmup_compiled_model = make_compiled_model();
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq("GPU.0")),
                          _))
        .WillByDefault(::testing::Invoke(
            [&](const std::shared_ptr<const ov::Model>& compiled, const std::string&, const ov::AnyMap&) {
                std::lock_guard<std::mutex> lock(captured_mutex);
                captured_model = compiled;
                return ov::SoPtr<ov::ICompiledModel>{warmup_compiled_model, {}};
            }));
    config.insert(ov::intel_auto::compile_for_all(true));

    std::shared_ptr<ov::ICompiledModel> compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, config));
    ASSERT_TRUE(wait_until([&] {
        std::lock_guard<std::mutex> lock(captured_mutex);
        return captured_model != nullptr;
    }));
    std::lock_guard<std::mutex> lock(captured_mutex);
    EXPECT_NE(captured_model, model);
}

TEST_F(AutoCacheWarmupTest, modelCloneFailureIsReportedToCaller) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 2, 2});
    auto relu = std::make_shared<cache_warmup_test::ThrowOnCloneRelu>(parameter);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    auto uncloneable_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
    set_global_cache("test_cache_dir");
    set_devices({cpu_device, "GPU.0", actual_device});
    std::atomic<int> compile_count{0};
    for (const auto& device : {cpu_device, std::string("GPU.0"), actual_device}) {
        on_compile(device, [&]() -> ov::SoPtr<ov::ICompiledModel> {
            compile_count++;
            return {mockIExeNetActual, {}};
        });
    }
    config.insert(ov::intel_auto::compile_for_all(true));

    // The snapshot is taken before any compilation starts, so the failure reaches the caller.
    EXPECT_THROW(plugin->compile_model(uncloneable_model, config), ov::Exception);
    EXPECT_EQ(compile_count.load(), 0);
}
