// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_utils.h"
#include "../common/tests_utils.h"
#include "../common/managers/thread_manager.h"
#include "tests_pipelines/tests_pipelines.h"

#include <gtest/gtest.h>

#include <inference_engine.hpp>

using namespace InferenceEngine;


#define checkRefVmValues()                                                                                  \
    if (!Environment::Instance().getCollectResultsOnly()) {                                                 \
        ASSERT_GT(test_refs.references[VMSIZE], 0) << "Reference value of VmSize is less than 0. Value: "   \
                                           << test_refs.references[VMSIZE];                                 \
        ASSERT_GT(test_refs.references[VMPEAK], 0) << "Reference value of VmPeak is less than 0. Value: "   \
                                           << test_refs.references[VMPEAK];                                 \
        ASSERT_GT(test_refs.references[VMRSS], 0) << "Reference value of VmRSS is less than 0. Value: "     \
                                          << test_refs.references[VMRSS];                                   \
        ASSERT_GT(test_refs.references[VMHWM], 0) << "Reference value of VmHWM is less than 0. Value: "     \
                                          << test_refs.references[VMHWM];                                   \
    }

class MemCheckTestSuite : public ::testing::TestWithParam<TestCase> {
};

// tests_pipelines/tests_pipelines.cpp
TEST_P(MemCheckTestSuite, create_exenetwork) {
    std::string test_name = "create_exenetwork";
    auto test_params = GetParam();

    TestReferences test_refs;
    test_refs.collect_vm_values_for_test(test_name, test_params);
    checkRefVmValues();

    log_info("Create ExecutableNetwork from network: \"" << test_params.model
                                                         << "\" for device: \"" << test_params.device << "\"");
    auto test_pipeline = [&]{
        MemCheckPipeline memCheckPipeline;

        log_info("Memory consumption before run:");
        memCheckPipeline.print_actual_measures();

        Core ie;
        log_info("Memory consumption after Core creation:");
        memCheckPipeline.print_actual_measures();

        ie.GetVersions(test_params.device);
        log_info("Memory consumption after GetCPPPluginByName (via GetVersions):");
        memCheckPipeline.print_actual_measures();

        CNNNetwork cnnNetwork = ie.ReadNetwork(test_params.model);
        log_info("Memory consumption after ReadNetwork:");
        memCheckPipeline.print_actual_measures();

        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, test_params.device);
        log_info("Memory consumption after LoadNetwork:");
        memCheckPipeline.upload_actual_measures("create_exenetwork");

        log_debug(memCheckPipeline.get_reference_record_for_test(test_name, test_params.model_name, test_params.device));
        return memCheckPipeline.get_measures();
    };

    TestResult res = common_test_pipeline(test_pipeline, test_refs.references);
    EXPECT_EQ(res.first, TestStatus::TEST_OK) << res.second;
}

TEST_P(MemCheckTestSuite, infer_request_inference) {
    std::string test_name = "infer_request_inference";
    auto test_params = GetParam();

    TestReferences test_refs;
    test_refs.collect_vm_values_for_test(test_name, test_params);
    checkRefVmValues();

    log_info("Inference of InferRequest from network: \"" << test_params.model
                                                          << "\" for device: \"" << test_params.device << "\"");
    auto test_pipeline = [&]{
        MemCheckPipeline memCheckPipeline;

        log_info("Memory consumption before run:");
        memCheckPipeline.print_actual_measures();

        Core ie;
        log_info("Memory consumption after Core creation:");
        memCheckPipeline.print_actual_measures();

        ie.GetVersions(test_params.device);
        log_info("Memory consumption after GetCPPPluginByName (via GetVersions):");
        memCheckPipeline.print_actual_measures();

        CNNNetwork cnnNetwork = ie.ReadNetwork(test_params.model);
        log_info("Memory consumption after ReadNetwork:");
        memCheckPipeline.print_actual_measures();

        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, test_params.device);
        log_info("Memory consumption after LoadNetwork:");
        memCheckPipeline.print_actual_measures();

        InferRequest inferRequest = exeNetwork.CreateInferRequest();
        log_info("Memory consumption after CreateInferRequest:");
        memCheckPipeline.print_actual_measures();

        inferRequest.Infer();
        OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        for (auto &output : output_info)
            Blob::Ptr outputBlob = inferRequest.GetBlob(output.first);
        log_info("Memory consumption after Inference:");
        memCheckPipeline.upload_actual_measures("infer_request_inference");

        log_debug(memCheckPipeline.get_reference_record_for_test(test_name, test_params.model_name, test_params.device));
        return memCheckPipeline.get_measures();
    };

    TestResult res = common_test_pipeline(test_pipeline, test_refs.references);
    EXPECT_EQ(res.first, TestStatus::TEST_OK) << res.second;
}
// tests_pipelines/tests_pipelines.cpp

INSTANTIATE_TEST_CASE_P(MemCheckTests, MemCheckTestSuite,
                        ::testing::ValuesIn(
                                generateTestsParams({"devices", "models"})),
                        getTestCaseName);
