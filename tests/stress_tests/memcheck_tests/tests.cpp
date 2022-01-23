// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_utils.h"
#include "../common/tests_utils.h"
#include "common_utils.h"
#include "../common/managers/thread_manager.h"
#include "tests_pipelines/tests_pipelines.h"

#include <gtest/gtest.h>

#include <inference_engine.hpp>
#include <openvino/runtime/core.hpp>


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
    auto test_params = GetParam();
    MemCheckPipeline memCheckPipeline;
        auto test_pipeline = [&] {
            if (test_params.api_version == 1) {
                InferenceEngine::Core ie;
                ie.GetVersions(device);
                InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
                InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, device);

                log_info("Memory consumption after LoadNetwork:");
                memCheckPipeline.record_measures(test_name);

                log_debug(memCheckPipeline.get_reference_record_for_test(test_name, model_name, precision, device));
                return memCheckPipeline.measure();
            }
            else {
                ov::Core ie;
                ie.get_versions(device);
                std::shared_ptr<ov::Model> network = ie.read_model(model);
                ov::CompiledModel compiled_model = ie.compile_model(network, device);

                log_info("Memory consumption after compile_model:");
                memCheckPipeline.record_measures(test_name);

                log_debug(memCheckPipeline.get_reference_record_for_test(test_name, model_name, precision, device));
                return memCheckPipeline.measure();
            }
        };


    TestResult res = common_test_pipeline(test_pipeline, test_refs.references);
    EXPECT_EQ(res.first, TestStatus::TEST_OK) << res.second;
}

TEST_P(MemCheckTestSuite, infer_request_inference) {
    log_info("Inference of InferRequest from network: \"" << model
                                                          << "\" with precision: \"" << precision
                                                          << "\" for device: \"" << device << "\"");
    auto test_params = GetParam();
    MemCheckPipeline memCheckPipeline;
    auto test_pipeline = [&]{
        if (test_params.api_version == 1) {
            InferenceEngine::Core ie;
            ie.GetVersions(device);
            InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
            InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, device);
            InferenceEngine::InferRequest inferRequest = exeNetwork.CreateInferRequest();

            auto batchSize = cnnNetwork.getBatchSize();
            batchSize = batchSize != 0 ? batchSize : 1;
            const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
            fillBlobs(inferRequest, inputsInfo, batchSize);

            inferRequest.Infer();
            InferenceEngine::OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
            for (auto &output: output_info)
                InferenceEngine::Blob::Ptr outputBlob = inferRequest.GetBlob(output.first);

            log_info("Memory consumption after Inference:");
            memCheckPipeline.record_measures(test_name);

            log_debug(memCheckPipeline.get_reference_record_for_test(test_name, model_name, precision, device));
            return memCheckPipeline.measure();
        }
        else {
            ov::Core ie;
            ie.get_versions(device);
            std::shared_ptr<ov::Model> network = ie.read_model(model);
            ov::CompiledModel compiled_model = ie.compile_model(network, device);
            ov::InferRequest infer_request = compiled_model.create_infer_request();
            const std::vector<ov::Output<ov::Node>> inputs = network->inputs();
            fillTensors(infer_request, inputs);

            infer_request.infer();
            auto outputs = network->outputs();
            for (size_t i = 0; i < outputs.size(); ++i) {
                const auto &output_tensor = infer_request.get_output_tensor(i);
            }
            log_info("Memory consumption after Inference:");
            memCheckPipeline.record_measures(test_name);

            log_debug(memCheckPipeline.get_reference_record_for_test(test_name, model_name, precision, device));
            return memCheckPipeline.measure();
        }
    };

    TestResult res = common_test_pipeline(test_pipeline, test_refs.references);
    EXPECT_EQ(res.first, TestStatus::TEST_OK) << res.second;
}
// tests_pipelines/tests_pipelines.cpp

INSTANTIATE_TEST_SUITE_P(MemCheckTests, MemCheckTestSuite,
                        ::testing::ValuesIn(
                                generateTestsParams({"devices", "models"})),
                        getTestCaseName);

TEST_P(MemCheckTestSuite, inference_with_streams) {
    auto test_params = GetParam();
    const auto nstreams = 2;
    log_info("Inference of InferRequest from network: \"" << model
                                                          << "\" with precision: \"" << precision
                                                          << "\" for device: \"" << device << "\""
                                                          << "\" with streams: \"" << nstreams);
    if ((device != "CPU") && (device != "GPU"))
        throw std::invalid_argument("This device is not supported");

    auto test_pipeline = [&] {
        MemCheckPipeline memCheckPipeline;
        unsigned int nireq = nstreams;

        if (test_params.api_version == 1) {
            std::map<std::string, std::string> config;
            config[device + "_THROUGHPUT_STREAMS"] = std::to_string(nstreams);
            InferenceEngine::Core ie;
            ie.GetVersions(device);
            ie.SetConfig(config, device);

            InferenceEngine::InferRequest inferRequest;

            InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
            InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, device);
            auto batchSize = cnnNetwork.getBatchSize();
            batchSize = batchSize != 0 ? batchSize : 1;
            const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());


            try {
                nireq = exeNetwork.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
            } catch (const std::exception &ex) {
                log_err("Failed to query OPTIMAL_NUMBER_OF_INFER_REQUESTS");
            }
            for (int counter = 0; counter < nireq; counter++) {
                inferRequest = exeNetwork.CreateInferRequest();
                fillBlobs(inferRequest, inputsInfo, batchSize);

                inferRequest.Infer();
                InferenceEngine::OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
                for (auto &output: output_info)
                    InferenceEngine::Blob::Ptr outputBlob = inferRequest.GetBlob(output.first);
            }
        }
        else {
            std::map<std::string, ov::Any> config;
            config[device + "_THROUGHPUT_STREAMS"] = std::to_string(nstreams);
            ov::Core ie;
            ie.get_versions(device);
            ie.set_property(device, config);
            auto network = ie.read_model(model);
            auto compiled_model = ie.compile_model(network, device);
            ov::InferRequest infer_request;
            std::vector<ov::Output<ov::Node>> inputs = network->inputs();

            try {
                nireq = compiled_model.get_property(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
            } catch (const std::exception &ex) {
                log_err("Failed to query OPTIMAL_NUMBER_OF_INFER_REQUESTS");
            }
            for (int counter = 0; counter < nireq; counter++) {
                infer_request = compiled_model.create_infer_request();
                fillTensors(infer_request, inputs);

                infer_request.infer();
                auto outputs = network->outputs();
                for (size_t i = 0; i < outputs.size(); ++i) {
                    const auto &output_tensor = infer_request.get_output_tensor(i);
                }
            }
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
