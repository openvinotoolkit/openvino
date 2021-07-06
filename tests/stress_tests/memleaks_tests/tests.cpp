// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../common/tests_utils.h"
#include "../common/ie_utils.h"
#include "../common/managers/thread_manager.h"
#include "tests_pipelines/tests_pipelines.h"

#include <inference_engine.hpp>

#include <gtest/gtest.h>

using namespace InferenceEngine;

class MemLeaksTestSuiteNoModel : public ::testing::TestWithParam<TestCase> {
};

class MemLeaksTestSuiteNoDevice : public ::testing::TestWithParam<TestCase> {
};

class MemLeaksTestSuite : public ::testing::TestWithParam<TestCase> {
};

inline void test_runner(int numthreads, const std::function<TestResult()> &test_function) {
    ThreadManager<TestResult> thr_manager;
    for (int i = 0; i < numthreads; i++)
        thr_manager.add_task(test_function);
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
    auto test = [&] {
        return test_load_unload_plugin(test_params.device, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuiteNoDevice, read_network) {
    auto test_params = GetParam();
    auto test = [&] {
        return test_read_network(test_params.model, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuiteNoDevice, cnnnetwork_reshape_batch_x2) {
    auto test_params = GetParam();
    auto test = [&] {
        return test_cnnnetwork_reshape_batch_x2(test_params.model, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuiteNoDevice, set_input_params) {
    auto test_params = GetParam();
    auto test = [&] {
        return test_set_input_params(test_params.model, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, recreate_exenetwork) {
    auto test_params = GetParam();
    Core ie;
    auto test = [&] {
        return test_recreate_exenetwork(ie, test_params.model, test_params.device, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, recreate_infer_request) {
    auto test_params = GetParam();
    Core ie;
    CNNNetwork cnnNetwork = ie.ReadNetwork(test_params.model);
    ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, test_params.device);
    auto test = [&] {
        return test_recreate_infer_request(exeNetwork, test_params.model, test_params.device, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, reinfer_request_inference) {
    auto test_params = GetParam();
    auto test = [&] {
        Core ie;
        CNNNetwork cnnNetwork = ie.ReadNetwork(test_params.model);
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, test_params.device);
        InferRequest infer_request = exeNetwork.CreateInferRequest();

        OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        auto batchSize = cnnNetwork.getBatchSize();
        batchSize = batchSize != 0 ? batchSize : 1;
        const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
        fillBlobs(infer_request, inputsInfo, batchSize);

        return test_reinfer_request_inference(infer_request, output_info, test_params.model, test_params.device, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, infer_request_inference) {
    auto test_params = GetParam();
    auto test = [&] {
        return test_infer_request_inference(test_params.model, test_params.device, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}
// tests_pipelines/tests_pipelines.cpp

INSTANTIATE_TEST_SUITE_P(MemLeaksTests, MemLeaksTestSuiteNoModel,
                        ::testing::ValuesIn(generateTestsParams({"processes", "threads", "iterations", "devices"})),
                        getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MemLeaksTests, MemLeaksTestSuiteNoDevice,
                        ::testing::ValuesIn(generateTestsParams({"processes", "threads", "iterations", "models"})),
                        getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MemLeaksTests, MemLeaksTestSuite,
                        ::testing::ValuesIn(
                                generateTestsParams({"processes", "threads", "iterations", "devices", "models"})),
                        getTestCaseName);

