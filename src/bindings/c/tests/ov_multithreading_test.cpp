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

protected:
    unsigned int iterations;
    unsigned int threadsNum;

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
    std::vector<std::pair<std::string, std::string>> networks;
    void set_up_networks(const unsigned int iterations = 100) {
        for (unsigned i = 0; i < iterations; i++) {
            std::pair<std::string, std::string> network(xml_file_name, bin_file_name);
            networks.emplace_back(network);
        }
    }
};

INSTANTIATE_TEST_SUITE_P(device_name, ov_multithreading_test, ::testing::Values("CPU"));

TEST_P(ov_multithreading_test, get_property) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    const char* key = ov_property_key_supported_properties;
    char* result = nullptr;

    runParallel([&]() {
        OV_EXPECT_OK(ov_compiled_model_get_property(compiled_model, key, &result));
    });

    ov_free(result);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_multithreading_test, compile_model) {
    auto device_name = GetParam();

    std::atomic<unsigned int> counter{0u};
    set_up_networks();
    runParallel([&]() {
        auto value = counter++;
        ov_core_t* core = nullptr;
        OV_EXPECT_OK(ov_core_create(&core));
        EXPECT_NE(nullptr, core);
        ov_compiled_model_t* compiled_model = nullptr;
        ov_model_t* model = nullptr;
        std::string model_path = networks[value % networks.size()].first,
                    bin_path = networks[value % networks.size()].second;
        OV_EXPECT_OK(ov_core_read_model(core, model_path.c_str(), bin_path.c_str(), &model));
        EXPECT_NE(nullptr, model);
        OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
        EXPECT_NE(nullptr, compiled_model);

        ov_compiled_model_free(compiled_model);
        ov_model_free(model);
        ov_core_free(core);
    });
}

TEST_P(ov_multithreading_test, infer) {
    auto device_name = GetParam();

    std::atomic<unsigned int> counter{0u};
    set_up_networks();
    runParallel([&]() {
        auto value = counter++;
        ov_core_t* core = nullptr;
        OV_EXPECT_OK(ov_core_create(&core));
        EXPECT_NE(nullptr, core);
        ov_compiled_model_t* compiled_model = nullptr;
        ov_model_t* model = nullptr;
        std::string model_path = networks[value % networks.size()].first,
                    bin_path = networks[value % networks.size()].second;
        OV_EXPECT_OK(ov_core_read_model(core, model_path.c_str(), bin_path.c_str(), &model));
        EXPECT_NE(nullptr, model);
        OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
        EXPECT_NE(nullptr, compiled_model);
        ov_infer_request_t* infer_request = nullptr;
        OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
        OV_EXPECT_OK(ov_infer_request_infer(infer_request));

        ov_infer_request_free(infer_request);
        ov_compiled_model_free(compiled_model);
        ov_model_free(model);
        ov_core_free(core);
    });
}