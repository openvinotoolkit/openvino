// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../common/managers/thread_manager.h"
#include "../common/infer_api/infer_api.h"
#include "tests_pipelines/tests_pipelines.h"


#include <gtest/gtest.h>

using namespace InferenceEngine;

class MemLeaksTestSuiteNoModel : public ::testing::TestWithParam<MemLeaksTestCase> {
};

class MemLeaksTestSuiteNoDevice : public ::testing::TestWithParam<MemLeaksTestCase> {
};

class MemLeaksTestSuite : public ::testing::TestWithParam<MemLeaksTestCase> {
};

inline void test_runner(int numthreads, const std::function<TestResult()> &test_function) {
    ThreadManager<TestResult> thr_manager;
    for (int i = 0; i < numthreads; i++) thr_manager.add_task(test_function);
    thr_manager.run_parallel_n_wait();

    std::vector<ManagerStatus> statuses = thr_manager.get_all_statuses();
    std::vector<TestResult> results = thr_manager.get_all_results();

    for (int i = 0; i < numthreads; i++) {
        EXPECT_EQ(statuses[i], ManagerStatus::FINISHED_SUCCESSFULLY)
                            << "[Thread " << i << "] Thread not finished successfully";
        EXPECT_EQ(results[i].first, TestStatus::TEST_OK) << "[Thread " << i << "] " << results[i].second;
    }
}

// tests_pipelines/tests_pipelines.cpp
TEST_P(MemLeaksTestSuiteNoModel, load_unload_plugin) {
    auto test_params = GetParam();

    std::vector<std::function<void()>> pipeline = {load_unload_plugin(test_params.device, test_params.api_version)};
    auto test = [&] {
        log_info("Load/unload plugin for \"" << test_params.device << "\" device"
                                             << " for " << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuiteNoDevice, read_network) {
    auto test_params = GetParam();
    std::vector<std::function<void()>> pipeline;

    pipeline.reserve(test_params.models.size());
    for (int i = 0; i < test_params.models.size(); i++) {
        pipeline.push_back(read_cnnnetwork(test_params.models[i]["full_path"], test_params.api_version));
    }
    auto test = [&] {
        log_info("Read networks: " << test_params.model_name << " for " << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuiteNoDevice, cnnnetwork_reshape_batch_x2) {
    auto test_params = GetParam();
    std::vector<std::function<void()>> pipeline;

    pipeline.reserve(test_params.models.size());
    for (int i = 0; i < test_params.models.size(); i++) {
        pipeline.push_back(cnnnetwork_reshape_batch_x2(test_params.models[i]["full_path"], i, test_params.api_version));
    }
    auto test = [&] {
        log_info("Reshape to batch*=2 of CNNNetworks created from networks: " << test_params.model_name << " for "
                                                                              << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuiteNoDevice, set_input_params) {
    auto test_params = GetParam();
    std::vector<std::function<void()>> pipeline;

    pipeline.reserve(test_params.models.size());
    for (int i = 0; i < test_params.models.size(); i++) {
        pipeline.push_back(set_input_params(test_params.models[i]["full_path"], test_params.api_version));
    }
    auto test = [&] {
        log_info("Apply preprocessing for CNNNetworks from networks: " << test_params.model_name << " for "
                                                                       << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, recreate_exenetwork) {
    auto test_params = GetParam();
    std::vector<std::function<void()>> pipeline;
    auto ie_wrapper = create_infer_api_wrapper(test_params.api_version);

    pipeline.reserve(test_params.models.size());
    for (int i = 0; i < test_params.models.size(); i++) {
        pipeline.push_back(recreate_compiled_model(ie_wrapper, test_params.models[i]["full_path"], test_params.device,
                                                   test_params.api_version));
    }
    auto test = [&] {
        log_info("Recreate ExecutableNetworks within existing InferenceEngine::Core from networks: "
                         << test_params.model_name << " for \"" << test_params.device << "\" device for "
                         << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, recreate_infer_request) {
    auto test_params = GetParam();
    std::vector<std::function<void()>> pipeline;
    auto ie_wrapper = create_infer_api_wrapper(test_params.api_version);

    size_t n_models = test_params.models.size();

    for (int i = 0; i < n_models; i++) {
        ie_wrapper->read_network(test_params.models[i]["full_path"]);
        ie_wrapper->load_network(test_params.device);
        pipeline.push_back(recreate_infer_request(ie_wrapper));
    }

    auto test = [&] {
        log_info("Create InferRequests from networks: " << test_params.model_name << " for \"" << test_params.device
                                                        << "\" device for " << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, reinfer_request_inference) {
    auto test_params = GetParam();
    std::vector<std::function<void()>> pipeline;
    auto ie_wrapper = create_infer_api_wrapper(test_params.api_version);
    size_t n_models = test_params.models.size();

    for (int i = 0; i < n_models; i++) {
        ie_wrapper->read_network(test_params.models[i]["full_path"]);
        ie_wrapper->load_network(test_params.device);
        ie_wrapper->create_infer_request();
        ie_wrapper->prepare_input();
        pipeline.push_back(reinfer_request_inference(ie_wrapper));
    }

    auto test = [&] {
        log_info("Inference of InferRequests from networks: " << test_params.model_name << " for \""
                                                              << test_params.device << "\" device for "
                                                              << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, infer_request_inference) {
    auto test_params = GetParam();
    std::vector<std::function<void()>> pipeline;
    pipeline.reserve(test_params.models.size());
    for (int i = 0; i < test_params.models.size(); i++) {
        pipeline.push_back(infer_request_inference(test_params.models[i]["full_path"], test_params.device,
                                                   test_params.api_version));
    }
    auto test = [&] {
        log_info("Inference of InferRequests from networks: " << test_params.model_name << " for \""
                                                              << test_params.device << "\" device for "
                                                              << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, inference_with_streams) {
    auto test_params = GetParam();
    const auto nstreams = 2;
    std::vector<std::function<void()>> pipeline;
    pipeline.reserve(test_params.models.size());
    for (int i = 0; i < test_params.models.size(); i++) {
        pipeline.push_back(inference_with_streams(test_params.models[i]["full_path"], test_params.device, nstreams,
                                                  test_params.api_version));
    }
    auto test = [&] {
        log_info("Inference of InferRequests from networks: " << test_params.model_name << " for \""
                                                              << test_params.device << "\" device with " << nstreams
                                                              << " streams for " << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

using MultiThreadLeak = MemLeaksTestSuite;
TEST_P(MultiThreadLeak, inference_in_multi_thread) {
    auto test_params = GetParam();
    long vmsize = 0;
    long vmpeak = 0;
    long vmrss = 0;
    long vmhwm = 0;
    long last_vmsize = 0;
    long last_vmpeak = 0;
    long last_vmrss = 0;
    long last_vmhwm = 0;
    int loop_iters = test_params.numiters;
    if (loop_iters < 10) {
        GTEST_SKIP() << "Skipping the test case, loop niter < 10";
    }
    ov::Core core;
    for (int m = 0; m < test_params.models.size(); m++) {
        auto model = core.read_model(test_params.models[m]["full_path"]);
        auto compile_model = core.compile_model(model, test_params.device);
        int peak_count = 0;
        std::vector<long> memory_data;
        memory_data.reserve(loop_iters * 4);
        for (int z = 0; z < loop_iters; z++) {
            {
                std::vector<std::thread> threads;
                for (int i = 0; i < test_params.numthreads; i++) {
                    threads.push_back(std::thread([&compile_model](){
                                auto infer_request = compile_model.create_infer_request();
                                infer_request.infer();
                                }));
                }
                for (int i = 0; i < test_params.numthreads; i++) {
                    threads[i].join();
                }
            }
            getVmValues(vmsize, vmpeak, vmrss, vmhwm);
            memory_data.push_back(vmsize);
            memory_data.push_back(vmpeak);
            memory_data.push_back(vmrss);
            memory_data.push_back(vmhwm);
            if (vmpeak > last_vmpeak) {
                last_vmpeak = vmpeak;
                peak_count++;
            }
        }
        if (peak_count >  loop_iters * 2 / 3) {
            for(int i = 0, y = 0; i < loop_iters * 4; i = i + 4, y++) {
                std::cout << "loop:" << y << " vmsize:" << memory_data[i] <<
                    " vmpeak:" << memory_data[i+1] <<
                    " vmrss:" << memory_data[i+2] <<
                    " vmhwm:" << memory_data[i+3] << std::endl;
            }
        }
        EXPECT_LE(peak_count, loop_iters * 2 / 3);
    }
}


// tests_pipelines/tests_pipelines.cpp

INSTANTIATE_TEST_SUITE_P(MemLeaksTests, MemLeaksTestSuiteNoModel, ::testing::ValuesIn(generateTestsParamsMemLeaks()),
                         getTestCaseNameMemLeaks);

INSTANTIATE_TEST_SUITE_P(MemLeaksTests, MemLeaksTestSuiteNoDevice, ::testing::ValuesIn(generateTestsParamsMemLeaks()),
                         getTestCaseNameMemLeaks);

INSTANTIATE_TEST_SUITE_P(MemLeaksTests, MemLeaksTestSuite, ::testing::ValuesIn(generateTestsParamsMemLeaks()),
                         getTestCaseNameMemLeaks);

INSTANTIATE_TEST_SUITE_P(MemLeaksTests, MultiThreadLeak, ::testing::ValuesIn(generateTestsParamsMemLeaks()),
                         getTestCaseNameMemLeaks);
