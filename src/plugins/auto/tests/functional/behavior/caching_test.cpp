// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    // will cache model for both actual device and CPU plugin
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 2);
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
    // will cache model for both actual device and CPU as accelerator
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 3);
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
    // will cache model for both actual device and CPU plugin
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 2);
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
    // will cache model for both actual device and CPU as accelerator
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 3);
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
    // will cache model for both actual device and CPU plugin
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 2);
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
    // will cache 2 models for actual device and no cached model for CPU
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 2);
    core.set_property(ov::cache_dir(""));
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