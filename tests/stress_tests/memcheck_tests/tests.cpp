// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_utils.h"
#include "../common/tests_utils.h"
#include "../common/ie_utils.h"
#include "../common/managers/thread_manager.h"
#include "tests_pipelines/tests_pipelines.h"

#include <gtest/gtest.h>

#include <inference_engine.hpp>

using namespace InferenceEngine;


class MemCheckTestSuite : public ::testing::TestWithParam<TestCase> {
public:
    std::string test_name, model, model_name, device, precision;
    TestReferences test_refs;

    void SetUp() override {
        const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        test_name = std::string(test_info->name()).substr(0, std::string(test_info->name()).find('/'));
        //const std::string full_test_name = std::string(test_info->test_case_name()) + "." + std::string(test_info->name());

        const auto& test_params = GetParam();
        model = test_params.model;
        model_name = test_params.model_name;
        device = test_params.device;
        precision = test_params.precision;

        test_refs.collect_vm_values_for_test(test_name, test_params);
        EXPECT_GT(test_refs.references[VMSIZE], 0) << "Reference value of VmSize is less than 0. Value: "
                                           << test_refs.references[VMSIZE];
        EXPECT_GT(test_refs.references[VMPEAK], 0) << "Reference value of VmPeak is less than 0. Value: "
                                           << test_refs.references[VMPEAK];
        EXPECT_GT(test_refs.references[VMRSS], 0) << "Reference value of VmRSS is less than 0. Value: "
                                          << test_refs.references[VMRSS];
        EXPECT_GT(test_refs.references[VMHWM], 0) << "Reference value of VmHWM is less than 0. Value: "
                                          << test_refs.references[VMHWM];
    }
};

// tests_pipelines/tests_pipelines.cpp
TEST_P(MemCheckTestSuite, create_exenetwork) {
    log_info("Create ExecutableNetwork from network: \"" << model
                                                         << "\" with precision: \"" << precision
                                                         << "\" for device: \"" << device << "\"");
    auto test_pipeline = [&]{
        MemCheckPipeline memCheckPipeline;

        Core ie;
        ie.GetVersions(device);
        CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, device);

        log_info("Memory consumption after LoadNetwork:");
        memCheckPipeline.record_measures(test_name);

        log_debug(memCheckPipeline.get_reference_record_for_test(test_name, model_name, precision, device));
        return memCheckPipeline.measure();
    };

    TestResult res = common_test_pipeline(test_pipeline, test_refs.references);
    EXPECT_EQ(res.first, TestStatus::TEST_OK) << res.second;
}

TEST_P(MemCheckTestSuite, infer_request_inference) {
    log_info("Inference of InferRequest from network: \"" << model
                                                          << "\" with precision: \"" << precision
                                                          << "\" for device: \"" << device << "\"");
    auto test_pipeline = [&]{
        MemCheckPipeline memCheckPipeline;

        Core ie;
        ie.GetVersions(device);
        CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, device);
        InferRequest inferRequest = exeNetwork.CreateInferRequest();

        auto batchSize = cnnNetwork.getBatchSize();
        batchSize = batchSize != 0 ? batchSize : 1;
        const ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
        fillBlobs(inferRequest, inputsInfo, batchSize);

        inferRequest.Infer();
        OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        for (auto &output : output_info)
            Blob::Ptr outputBlob = inferRequest.GetBlob(output.first);

        log_info("Memory consumption after Inference:");
        memCheckPipeline.record_measures(test_name);

        log_debug(memCheckPipeline.get_reference_record_for_test(test_name, model_name, precision, device));
        return memCheckPipeline.measure();
    };

    TestResult res = common_test_pipeline(test_pipeline, test_refs.references);
    EXPECT_EQ(res.first, TestStatus::TEST_OK) << res.second;
}
// tests_pipelines/tests_pipelines.cpp

INSTANTIATE_TEST_CASE_P(MemCheckTests, MemCheckTestSuite,
                        ::testing::ValuesIn(
                                generateTestsParams({"devices", "models"})),
                        getTestCaseName);

TEST_P(MemCheckTestSuite, inference_with_streams) {
    const auto nstreams = 2;
    log_info("Inference of InferRequest from network: \"" << model
                                                          << "\" with precision: \"" << precision
                                                          << "\" for device: \"" << device << "\""
                                                          << "\" with streams: \"" << nstreams);
    if ((device != "CPU") && (device != "GPU"))
        throw std::invalid_argument("This device is not supported");

    auto test_pipeline = [&] {
        MemCheckPipeline memCheckPipeline;

        std::map<std::string, std::string> config;
        const std::string key = device + "_THROUGHPUT_STREAMS";
        config[device + "_THROUGHPUT_STREAMS"] = std::to_string(nstreams);

        Core ie;
        ie.GetVersions(device);
        ie.SetConfig(config, device);

        InferRequest inferRequest;

        CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, device);
        auto batchSize = cnnNetwork.getBatchSize();
        batchSize = batchSize != 0 ? batchSize : 1;
        const ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());

        unsigned int nireq = nstreams;
        try {
            nireq = exeNetwork.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
        } catch (const std::exception &ex) {
            log_err("Failed to query OPTIMAL_NUMBER_OF_INFER_REQUESTS");
        }
        for (int counter = 0; counter < nireq; counter++) {
            inferRequest = exeNetwork.CreateInferRequest();
            fillBlobs(inferRequest, inputsInfo, batchSize);

            inferRequest.Infer();
            OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
            for (auto &output : output_info)
                Blob::Ptr outputBlob = inferRequest.GetBlob(output.first);
        }

        log_info("Memory consumption after Inference with streams: \"" << nstreams
                                                                       << "\" with number of infer request: " << nireq);
        memCheckPipeline.record_measures(test_name);

        log_debug(memCheckPipeline.get_reference_record_for_test(test_name, model_name, precision, device));
        return memCheckPipeline.measure();

    };

    TestResult res = common_test_pipeline(test_pipeline, test_refs.references);
    EXPECT_EQ(res.first, TestStatus::TEST_OK) << res.second;
}
