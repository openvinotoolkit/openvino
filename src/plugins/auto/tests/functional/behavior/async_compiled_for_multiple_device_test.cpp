// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_func_test.hpp"
#include <thread>

using namespace ov::auto_plugin::tests;
class CompileThreadingTest {
public:
    static void runParallel(std::function<void(void)> func,
                     const unsigned int iterations = 100,
                     const unsigned int threadsNum = 8) {
        std::vector<std::thread> threads(threadsNum);

        for (auto & thread : threads) {
            thread = std::thread([&](){
                for (unsigned int i = 0; i < iterations; ++i) {
                    func();
                }
            });
        }

        for (auto & thread : threads) {
            if (thread.joinable())
                thread.join();
        }
    }
};

TEST_F(AutoFuncTests, can_compile_with_multiple_devices) {
    ov::CompiledModel compiled_model;
    ASSERT_NO_THROW(compiled_model = core.compile_model(model_can_batch, "AUTO", {ov::device::priorities("MOCK_1", "MOCK_2")}));
    ASSERT_NO_THROW(compiled_model = core.compile_model(model_can_batch, "AUTO", {ov::device::priorities("MOCK_1", "MOCK_2"),
                                                        ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}));
}

TEST_F(AutoFuncTests, threading_test) {
    CompileThreadingTest::runParallel([&] () {
        (void)core.compile_model(model_can_batch, "AUTO", {ov::device::priorities("MOCK_1", "MOCK_2")});
    }, 10, 10);
    CompileThreadingTest::runParallel([&] () {
        (void)core.compile_model(model_can_batch, "AUTO", {ov::device::priorities("MOCK_1", "MOCK_2"),
                                 ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)});
    }, 10, 10);
}

TEST_F(AutoFuncTests, threading_test_cache_enabled) { 
    core.set_property(ov::cache_dir(cache_path));
    CompileThreadingTest::runParallel([&] () {
        (void)core.compile_model(model_can_batch, "AUTO", {ov::device::priorities("MOCK_1", "MOCK_2"),
                                 ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)});
    }, 10, 10);
    core.set_property(ov::cache_dir(""));
}