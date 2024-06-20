// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../common/managers/thread_manager.h"
#include "../common/infer_api/infer_api.h"
#include "tests_pipelines/tests_pipelines.h"


#include <gtest/gtest.h>

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

    std::vector<std::function<void()>> pipeline = {load_unload_plugin(test_params.device)};
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
        pipeline.push_back(read_cnnnetwork(test_params.models[i]["full_path"]));
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
        pipeline.push_back(cnnnetwork_reshape_batch_x2(test_params.models[i]["full_path"], i));
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
        pipeline.push_back(set_input_params(test_params.models[i]["full_path"]));
    }
    auto test = [&] {
        log_info("Apply preprocessing for CNNNetworks from networks: " << test_params.model_name << " for "
                                                                       << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, recreate_compiled_model) {
    auto test_params = GetParam();
    std::vector<std::function<void()>> pipeline;

    pipeline.reserve(test_params.models.size());
    for (int i = 0; i < test_params.models.size(); i++) {
        auto ie_wrapper = create_infer_api_wrapper();
        ie_wrapper->read_network(test_params.models[i]["full_path"]);
        pipeline.push_back(recreate_compiled_model(ie_wrapper, test_params.device));
    }
    auto test = [&] {
        log_info("Recreate CompiledModels within existing ov::Core from networks: "
                         << test_params.model_name << " for \"" << test_params.device << "\" device for "
                         << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, recreate_infer_request) {
    auto test_params = GetParam();
    std::vector<std::function<void()>> pipeline;
    size_t n_models = test_params.models.size();

    for (int i = 0; i < n_models; i++) {
        auto ie_wrapper = create_infer_api_wrapper();
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
    size_t n_models = test_params.models.size();

    for (int i = 0; i < n_models; i++) {
        auto ie_wrapper = create_infer_api_wrapper();
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
        pipeline.push_back(infer_request_inference(test_params.models[i]["full_path"], test_params.device));
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
        pipeline.push_back(inference_with_streams(test_params.models[i]["full_path"], test_params.device, nstreams));
    }
    auto test = [&] {
        log_info("Inference of InferRequests from networks: " << test_params.model_name << " for \""
                                                              << test_params.device << "\" device with " << nstreams
                                                              << " streams for " << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, recreate_and_infer_in_thread) {
    auto test_params = GetParam();
    std::vector<std::function<void()>> pipeline;
    size_t n_models = test_params.models.size();

    std::vector<std::shared_ptr<InferApiBase>> ie_wrapper_vector;
    for (int i = 0; i < n_models; i++) {
        auto ie_wrapper = create_infer_api_wrapper();
        ie_wrapper_vector.push_back(ie_wrapper);
        ie_wrapper->read_network(test_params.models[i]["full_path"]);
        ie_wrapper->load_network(test_params.device);
        pipeline.push_back(recreate_and_infer_in_thread(ie_wrapper_vector[i], false));
    }

    auto test = [&] {
        log_info("Inference in separate thread of InferRequests from networks: " << test_params.model_name << " for \""
                                                              << test_params.device << "\" device for "
                                                              << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters, 0.02);
    };
    test_runner(test_params.numthreads, test);
}

// tests_pipelines/tests_pipelines.cpp

INSTANTIATE_TEST_SUITE_P(MemLeaksTests, MemLeaksTestSuiteNoModel, ::testing::ValuesIn(generateTestsParamsMemLeaks()),
                         getTestCaseNameMemLeaks);

INSTANTIATE_TEST_SUITE_P(MemLeaksTests, MemLeaksTestSuiteNoDevice, ::testing::ValuesIn(generateTestsParamsMemLeaks()),
                         getTestCaseNameMemLeaks);

INSTANTIATE_TEST_SUITE_P(MemLeaksTests, MemLeaksTestSuite, ::testing::ValuesIn(generateTestsParamsMemLeaks()),
                         getTestCaseNameMemLeaks);
