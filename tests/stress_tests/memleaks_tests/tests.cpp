// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../common/tests_utils.h"
#include "common_utils.h"
#include "../common/managers/thread_manager.h"
#include "tests_pipelines/tests_pipelines.h"

#include <inference_engine.hpp>

#include <gtest/gtest.h>
#include <memory>

using namespace InferenceEngine;

class MemLeaksTestSuiteNoModel : public ::testing::TestWithParam<MemLeaksTestCase> {
};

class MemLeaksTestSuiteNoDevice : public ::testing::TestWithParam<MemLeaksTestCase> {
};

class MemLeaksTestSuite : public ::testing::TestWithParam<MemLeaksTestCase> {
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
    std::vector<std::function<void()>> pipeline = { load_unload_plugin(test_params.device) };
    auto test = [&] {
        log_info("Load/unload plugin for device: " << test_params.device
              << " for " << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuiteNoDevice, read_network) {
    auto test_params = GetParam();
    std::vector<std::function<void()>> pipeline;
    std::string path;
    for (int i = 0; i < test_params.models.size(); i++)
        pipeline.push_back(read_cnnnetwork(test_params.models[i]["path"]));
    auto test = [&] {
        log_info("Read networks: " << test_params.model_name
              << " for " << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuiteNoDevice, cnnnetwork_reshape_batch_x2) {
    auto test_params = GetParam();
    std::vector<std::function<void()>> pipeline;
    for (int i = 0; i < test_params.models.size(); i++)
        pipeline.push_back(cnnnetwork_reshape_batch_x2(test_params.models[i]["path"]));
    auto test = [&] {
        log_info("Reshape to batch*=2 of CNNNetwork created from networks: " << test_params.model_name
              << " for " << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuiteNoDevice, set_input_params) {
    auto test_params = GetParam();
    std::vector<std::function<void()>> pipeline;
    for (auto model: test_params.models) pipeline.push_back(set_input_params(model["path"]));
    auto test = [&] {
        log_info("Apply preprocessing for CNNNetwork from networks: " << test_params.model_name
              << " for " << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, recreate_exenetwork) {
    auto test_params = GetParam();
    Core ie;
    std::vector<std::function<void()>> pipeline;
    for (int i = 0; i < test_params.models.size(); i++)
        pipeline.push_back(recreate_exenetwork(ie, test_params.models[i]["path"], test_params.device));
    auto test = [&] {
        log_info("Recreate ExecutableNetwork from network within existing InferenceEngine::Core: " << test_params.model_name
              << " for device: \"" << test_params.device << "\" for " << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, recreate_infer_request) {
    auto test_params = GetParam();
    Core ie;
    std::vector<std::function<void()>> pipeline;
    for (int i = 0; i < test_params.models.size(); i++){
        CNNNetwork cnnNetwork = ie.ReadNetwork(test_params.models[i]["path"]);
        std::unique_ptr<ExecutableNetwork> exeNetwork(new ExecutableNetwork);
        *exeNetwork = ie.LoadNetwork(cnnNetwork, test_params.device);
        pipeline.push_back(recreate_infer_request(*exeNetwork));
    }
    auto test = [&] {
        log_info("Create InferRequest from networks: " << test_params.model_name
              << " for device: \"" << test_params.device << "\" for " << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, reinfer_request_inference) {
    auto test_params = GetParam();
    Core ie;
    std::vector<InferRequest> infer_requests;

    std::vector<OutputsDataMap> outputs_info;
    std::vector<std::function<void()>> pipeline;
    for (int i = 0; i < test_params.models.size(); i++){
        CNNNetwork cnnNetwork = ie.ReadNetwork(test_params.models[i]["path"]);
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, test_params.device);
        std::unique_ptr<InferRequest> infer_request(new InferRequest);
        *infer_request = exeNetwork.CreateInferRequest();
        std::unique_ptr<OutputsDataMap> output_info(new OutputsDataMap(cnnNetwork.getOutputsInfo()));
        auto batchSize = cnnNetwork.getBatchSize();
        batchSize = batchSize != 0 ? batchSize : 1;
        const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
        fillBlobs(* infer_request, inputsInfo, batchSize);
        pipeline.push_back(reinfer_request_inference(*infer_request, *output_info));
    }
    auto test = [&] {
        log_info("Inference of InferRequest from networks: " << test_params.model_name
              << " for device: \"" << test_params.device << "\" for " << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, infer_request_inference) {
    auto test_params = GetParam();
    std::vector<std::function<void()>> pipeline;
    for (int i = 0; i < test_params.models.size(); i++)
        pipeline.push_back(infer_request_inference(test_params.models[i]["path"], test_params.device));
    auto test = [&] {
        log_info("Inference of InferRequest from networks: " << test_params.model_name
              << " for device: \"" << test_params.device << "\" for " << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

TEST_P(MemLeaksTestSuite, inference_with_streams) {
    auto test_params = GetParam();
    const auto nstreams = 2;
    std::vector<std::function<void()>> pipeline;
    for (int i = 0; i < test_params.models.size(); i++){
        pipeline.push_back(inference_with_streams(test_params.models[i]["path"], test_params.device, nstreams));
    }
    auto test = [&] {
        log_info("Inference of InferRequest from networks: " << test_params.model_name
              << " for device: \"" << test_params.device << "\" with streams: " << nstreams << " for " << test_params.numiters << " times");
        return common_test_pipeline(pipeline, test_params.numiters);
    };
    test_runner(test_params.numthreads, test);
}

// tests_pipelines/tests_pipelines.cpp

INSTANTIATE_TEST_SUITE_P(MemLeaksTests, MemLeaksTestSuiteNoModel,
                         ::testing::ValuesIn(generateTestsParamsMemLeaks()),
                         getTestCaseNameMemLeaks);

INSTANTIATE_TEST_SUITE_P(MemLeaksTests, MemLeaksTestSuiteNoDevice,
                        ::testing::ValuesIn(generateTestsParamsMemLeaks()),
                        getTestCaseNameMemLeaks);

INSTANTIATE_TEST_SUITE_P(MemLeaksTests, MemLeaksTestSuite,
                        ::testing::ValuesIn(generateTestsParamsMemLeaks()),
                        getTestCaseNameMemLeaks);

