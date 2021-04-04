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
    const auto nstreams = "2";
    log_info("Inference of InferRequest from network: \"" << model
                                                          << "\" with precision: \"" << precision
                                                          << "\" for device: \"" << device << "\""
                                                          << "\" with strems: \"" << nstreams);
    auto test_pipeline = [&]{
        MemCheckPipeline memCheckPipeline;

        Core ie;
        ie.GetVersions(device);
        std::map<std::string, std::map<std::string, std::string>> config;

        if (!config.count(device)) 
            config[device] = {};

        std::map<std::string, std::string>& device_config = config.at(device);

        auto setThroughputStreams = [&] () {
            const std::string key = device + "_THROUGHPUT_STREAMS";
            // set to user defined value
            std::vector<std::string> supported_config_keys = ie.GetMetric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
            if (std::find(supported_config_keys.begin(), supported_config_keys.end(), key) == supported_config_keys.end()) {
                throw std::logic_error("Device " + device + " doesn't support config key '" + key + "'! " +
                                        "Please specify -nstreams for correct devices in format  <dev1>:<nstreams1>,<dev2>:<nstreams2>" +
                                        " or via configuration file.");
            }
            device_config[key] = nstreams;
        };

    if (device == "CPU") {  
        // CPU supports few special performance-oriented keys
        // for CPU execution, more throughput-oriented execution via streams
        setThroughputStreams();
    } else if (device == ("GPU")) {
        // for GPU execution, more throughput-oriented execution via streams
        setThroughputStreams();
    } else {
        throw std::invalid_argument("This device is not supported");
    }

    for (auto&& item : config) {
        ie.SetConfig(item.second, item.first);
    }

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
