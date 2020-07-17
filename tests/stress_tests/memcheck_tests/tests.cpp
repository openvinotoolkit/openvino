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
    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = std::string(test_info->name()).substr(0, std::string(test_info->name()).find('/'));
    //const std::string full_test_name = std::string(test_info->test_case_name()) + "." + std::string(test_info->name());
    auto test_params = GetParam();

    TestReferences test_refs;
    test_refs.collect_vm_values_for_test(test_name, test_params);
    checkRefVmValues();

    log_info("Create ExecutableNetwork from network: \"" << test_params.model
                                                         << "\" for device: \"" << test_params.device << "\"");
    auto test_pipeline = [&]{
        MemCheckPipeline memCheckPipeline;

        Core ie;
        ie.GetVersions(test_params.device);
        CNNNetwork cnnNetwork = ie.ReadNetwork(test_params.model);
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, test_params.device);

        log_info("Memory consumption after LoadNetwork:");
        memCheckPipeline.record_measures(test_name);

        log_debug(memCheckPipeline.get_reference_record_for_test(test_name, test_params.model_name, test_params.device));
        return memCheckPipeline.measure();
    };

    TestResult res = common_test_pipeline(test_pipeline, test_refs.references);
    EXPECT_EQ(res.first, TestStatus::TEST_OK) << res.second;
}

TEST_P(MemCheckTestSuite, infer_request_inference) {
    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = std::string(test_info->name()).substr(0, std::string(test_info->name()).find('/'));
    //const std::string full_test_name = std::string(test_info->test_case_name()) + "." + std::string(test_info->name());
    auto test_params = GetParam();

    TestReferences test_refs;
    test_refs.collect_vm_values_for_test(test_name, test_params);
    checkRefVmValues();

    log_info("Inference of InferRequest from network: \"" << test_params.model
                                                          << "\" for device: \"" << test_params.device << "\"");
    auto test_pipeline = [&]{
        MemCheckPipeline memCheckPipeline;

        Core ie;
        ie.GetVersions(test_params.device);
        CNNNetwork cnnNetwork = ie.ReadNetwork(test_params.model);
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, test_params.device);
        InferRequest inferRequest = exeNetwork.CreateInferRequest();
        inferRequest.Infer();
        OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        for (auto &output : output_info)
            Blob::Ptr outputBlob = inferRequest.GetBlob(output.first);

        log_info("Memory consumption after Inference:");
        memCheckPipeline.record_measures(test_name);

        log_debug(memCheckPipeline.get_reference_record_for_test(test_name, test_params.model_name, test_params.device));
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
