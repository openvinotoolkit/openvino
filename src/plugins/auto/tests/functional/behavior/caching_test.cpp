// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <thread>

#include "auto_func_test.hpp"
#include "common_test_utils/include/common_test_utils/file_utils.hpp"
#include "openvino/pass/serialize.hpp"

using namespace ov::auto_plugin::tests;

TEST_F(AutoFuncTests, compiled_with_cache_enabled) {
    core.set_property(ov::cache_dir(cache_path));
    core.set_property("MOCK_GPU", ov::device::id("test"));  // device id for cache property distinguish with MOCK_CPU
    auto compiled_model =
        core.compile_model(model_cannot_batch,
                           "AUTO",
                           {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)});
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 2);
    compiled_model = core.compile_model(model_cannot_batch,
                                        "AUTO",
                                        {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                         ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)});
    // can reuse the cache, no extra cache generated
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 2);
    core.set_property("MOCK_GPU", ov::device::id("test_regenerate"));
    compiled_model = core.compile_model(model_cannot_batch,
                                        "AUTO",
                                        {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                         ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)});
    // new cache file expected
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 3);
    core.set_property(ov::cache_dir(""));
}

TEST_F(AutoFuncTests, load_cached_model_to_actual_device_and_disable_CPU_accelerating_default_startup_fallback) {
    core.set_property(ov::cache_dir(cache_path));
    core.set_property("MOCK_GPU", ov::device::id("test"));  // device id for cache property distinguish with MOCK_CPU
    {
        auto compiled_model = core.compile_model(model_cannot_batch,
                                                 "AUTO",
                                                 {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                                  ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)});
    }
    // No cached model for actual device
    // will only cache model only for actual device. CVS-141026
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 1);
    ov::test::utils::removeFilesWithExt(cache_path, "blob");
    {
        auto compiled_model = core.compile_model(
            model_cannot_batch,
            "AUTO",
            {ov::device::priorities("MOCK_GPU"), ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)});
    }
    {
        auto compiled_model = core.compile_model(model_cannot_batch,
                                                 "AUTO",
                                                 {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                                  ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)});
    }
    // cached model exists for actual device
    // will reuse cached model for actual device without CPU accelerating(No cached model for CPU)
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 1);

    core.set_property("MOCK_GPU", ov::device::id("test_regenerate"));
    {
        auto compiled_model = core.compile_model(model_cannot_batch,
                                                 "AUTO",
                                                 {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                                  ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)});
    }
    // model hash id changed for actual device
    // will cache model for actual device with new device id
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 2);
    core.set_property(ov::cache_dir(""));
}

TEST_F(AutoFuncTests, load_model_path_to_actual_device_and_disable_CPU_accelerating_default_startup_fallback) {
    std::string filePrefix = ov::test::utils::generateTestFilePrefix();
    auto m_xml_path = filePrefix + ".xml";
    auto m_bin_path = filePrefix + ".bin";
    ov::pass::Serialize(m_xml_path, m_bin_path).run_on_model(model_cannot_batch);
    core.set_property(ov::cache_dir(cache_path));
    core.set_property("MOCK_GPU", ov::device::id("test"));  // device id for cache property distinguish with MOCK_CPU
    {
        auto compiled_model = core.compile_model(m_xml_path,
                                                 "AUTO",
                                                 {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                                  ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)});
    }
    // No cached model for actual device
    // will only cache model only for actual device. CVS-141026
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 1);
    ov::test::utils::removeFilesWithExt(cache_path, "blob");
    {
        auto compiled_model = core.compile_model(
            m_xml_path,
            "AUTO",
            {ov::device::priorities("MOCK_GPU"), ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)});
    }
    {
        auto compiled_model = core.compile_model(m_xml_path,
                                                 "AUTO",
                                                 {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                                  ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)});
    }
    // cached model exists for actual device
    // will reuse cached model for actual device without CPU accelerating(No cached model for CPU)
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 1);

    core.set_property("MOCK_GPU", ov::device::id("test_regenerate"));
    {
        auto compiled_model = core.compile_model(m_xml_path,
                                                 "AUTO",
                                                 {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                                  ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)});
    }
    // model hash id changed for actual device
    // will cache model for actual device with new device id
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 2);
    core.set_property(ov::cache_dir(""));
    ov::test::utils::removeIRFiles(m_xml_path, m_bin_path);
}

TEST_F(AutoFuncTests, load_cached_model_to_actual_device_and_disable_CPU_accelerating_set_startup_fallback) {
    core.set_property(ov::cache_dir(cache_path));
    core.set_property("MOCK_GPU", ov::device::id("test"));  // device id for cache property distinguish with MOCK_CPU
    {
        auto compiled_model = core.compile_model(model_cannot_batch,
                                                 "AUTO",
                                                 {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                                  ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)});
    }
    // No cached model for actual device
    // will only cache model only for actual device. CVS-141026
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 1);
    ov::test::utils::removeFilesWithExt(cache_path, "blob");
    {
        auto compiled_model = core.compile_model(
            model_cannot_batch,
            "AUTO",
            {ov::device::priorities("MOCK_GPU"), ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)});
    }
    {
        auto compiled_model = core.compile_model(model_cannot_batch,
                                                 "AUTO",
                                                 {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                                  ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                                  ov::intel_auto::enable_startup_fallback(true)});
    }
    // cached model exists for actual device
    // will reuse cached model for actual device without CPU accelerating(No cached model for CPU)
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 1);
    core.set_property("MOCK_GPU", ov::device::id("test_regenerate"));
    {
        auto compiled_model = core.compile_model(model_cannot_batch,
                                                 "AUTO",
                                                 {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                                  ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                                  ov::intel_auto::enable_startup_fallback(false)});
    }
    // model hash id changed for actual device
    // will cache model for actual device with new device id
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 2);
    core.set_property(ov::cache_dir(""));
}

TEST_F(AutoFuncTests, compile_for_all_populates_cache_reused_by_later_loads) {
    core.compile_model(model_cannot_batch, "MOCK_CPU");
    core.compile_model(model_cannot_batch, "MOCK_GPU");  // need to initialize the order of plugins in mock_engine
    register_plugin_mock_gpu(core, "MOCK_3", {});
    core.set_property(ov::cache_path(cache_path));
    // distinct device ids give each device its own blob
    core.set_property("MOCK_GPU", ov::device::id("gpu"));
    core.set_property("MOCK_3", ov::device::id("3"));
    {
        auto compiled_model =
            core.compile_model(model_cannot_batch,
                               "AUTO",
                               {ov::device::priorities("MOCK_GPU", "MOCK_3"), ov::intel_auto::compile_for_all(true)});
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (ov::test::utils::listFilesWithExt(cache_path, "blob").size() < 2 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    // Releasing AUTO waits for an in-flight pre-compilation, so both blobs are fully written here.
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 2);
    for (const std::string device : {"MOCK_GPU", "MOCK_3"}) {
        auto compiled_model = core.compile_model(model_cannot_batch, device);
        EXPECT_TRUE(compiled_model.get_property(ov::loaded_from_cache)) << device;
    }
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 2);
    core.set_property(ov::cache_path(""));
}

TEST_F(AutoFuncTests, compiled_with_cache_enabled_batch_enabled) {
#ifdef ENABLE_AUTO_BATCH
    core.set_property(ov::cache_dir(cache_path));
    core.set_property("MOCK_GPU", ov::device::id("test"));  // device id for cache property distinguish with MOCK_CPU
    auto compiled_model =
        core.compile_model(model_can_batch,
                           "AUTO",
                           {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)});
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 3);
    compiled_model = core.compile_model(model_can_batch,
                                        "AUTO",
                                        {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                         ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)});
    // can reuse the cache, no extra cache generated
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 3);
    core.set_property("MOCK_GPU", ov::device::id("test_regenerate"));
    compiled_model = core.compile_model(model_can_batch,
                                        "AUTO",
                                        {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                         ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)});
    // new cache file expected
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 5);
    core.set_property(ov::cache_dir(""));
#endif
}
