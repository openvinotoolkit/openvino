// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <thread>

#include "ov_test.hpp"

class ov_multithreading_test : public ov_capi_test_base {
    void SetUp() override {
        ov_capi_test_base::SetUp();
    }

    void TearDown() override {
        ov_capi_test_base::TearDown();
    }

public:
    void runParallel(std::function<void(void)> func,
                     const unsigned int iterations = 100,
                     const unsigned int threadsNum = 8) {
        std::vector<std::thread> threads(threadsNum);

        for (auto& thread : threads) {
            thread = std::thread([&]() {
                for (unsigned int i = 0; i < iterations; ++i) {
                    func();
                }
            });
        }

        for (auto& thread : threads) {
            if (thread.joinable())
                thread.join();
        }
    }
};

INSTANTIATE_TEST_SUITE_P(device_name, ov_multithreading_test, ::testing::Values("CPU"));

TEST_P(ov_multithreading_test, compile_model) {
    auto device_name = GetParam();
    std::atomic<unsigned int> counter{0u};
    runParallel([&]() {
        auto value = counter++;
        ov_core_t* core = nullptr;
        ov_core_create(&core);
        ov_model_t* model = nullptr;
        ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model);
        ov_compiled_model_t* compiled_model = nullptr;
        ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model);
    });
}