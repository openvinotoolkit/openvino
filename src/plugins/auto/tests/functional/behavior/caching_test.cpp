// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_func_test.hpp"
#include "common_test_utils/include/common_test_utils/file_utils.hpp"

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