// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_func_test.hpp"
#ifdef __GLIBC__
#    include <gnu/libc-version.h>
#    if __GLIBC_MINOR__ >= 34
#        define ENABLETESTTHREADING
#    endif
#endif

using namespace ov::auto_plugin::tests;

#ifdef ENABLETESTTHREADING
TEST_F(AutoFuncTests, can_compile_with_multiple_devices) {
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(
        compiled_model = core.compile_model(model_can_batch, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")}));
    compiled_model = core.compile_model(model_can_batch,
                                        "AUTO",
                                        {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                         ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)});
}

TEST_F(AutoFuncTests, threading_test) {
    ThreadingTest::runParallel(
        [&]() {
            (void)core.compile_model(model_can_batch, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")});
        },
        10,
        10);
    ThreadingTest::runParallel(
        [&]() {
            (void)core.compile_model(model_can_batch,
                                     "AUTO",
                                     {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                      ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)});
        },
        10,
        10);
}

TEST_F(AutoFuncTests, threading_test_cache_enabled) {
    core.set_property(ov::cache_dir(cache_path));
    ThreadingTest::runParallel(
        [&]() {
            (void)core.compile_model(model_can_batch,
                                     "AUTO",
                                     {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                      ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)});
        },
        10,
        10);
    core.set_property(ov::cache_dir(""));
}

TEST_F(AutoFuncTests, threading_test_get_version) {
    ThreadingTest::runParallel([&]() {
        auto versions = core.get_versions("AUTO");
        ASSERT_LE(1u, versions.size());
    });
}

TEST_F(AutoFuncTests, theading_compiled_with_cpu_help) {
    ThreadingTest::runParallel(
        [&]() {
            (void)core.compile_model(model_can_batch, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")});
        },
        10,
        10);
}

TEST_F(AutoFuncTests, threading_test_hardware_slower) {
    core.compile_model(model_cannot_batch, "MOCK_CPU");
    core.compile_model(model_cannot_batch, "MOCK_GPU");  // need to initialize the order of plugins in mock_engine
    register_plugin_mock_gpu_compile_slower(core, "MOCK_GPU_SLOWER", {});
    ThreadingTest::runParallel(
        [&]() {
            (void)core.compile_model(model_can_batch, "AUTO", {ov::device::priorities("MOCK_GPU_SLOWER", "MOCK_CPU")});
        },
        10,
        10);
}

TEST_F(AutoFuncTests, threading_test_cpu_help_slower) {
    core.compile_model(model_cannot_batch, "MOCK_CPU");
    core.compile_model(model_cannot_batch, "MOCK_GPU");  // need to initialize the order of plugins in mock_engine
    register_plugin_mock_cpu_compile_slower(core, "MOCK_CPU_SLOWER", {});
    ThreadingTest::runParallel(
        [&]() {
            (void)core.compile_model(model_can_batch, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU_SLOWER")});
        },
        10,
        10);
}
#endif