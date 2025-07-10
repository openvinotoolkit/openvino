// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_utils.h"
#include "../common/tests_utils.h"
#include "../common/infer_api/infer_api.h"
#include "common_utils.h"
#include "../common/managers/thread_manager.h"
#include "tests_pipelines/tests_pipelines.h"

#include <gtest/gtest.h>

#include <openvino/runtime/core.hpp>


class MemCheckTestSuite : public ::testing::TestWithParam<TestCase> {
public:
    std::string test_name, model, model_name, device, precision;
    TestReferences test_refs;

    void SetUp() override {
        const ::testing::TestInfo *const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        test_name = std::string(test_info->name()).substr(0, std::string(test_info->name()).find('/'));

        const auto &test_params = GetParam();
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
        auto ie_api_wrapper = create_infer_api_wrapper();
        ie_api_wrapper->load_plugin(device);
        ie_api_wrapper->read_network(model);
        ie_api_wrapper->load_network(device);
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
    auto test_params = GetParam();
    MemCheckPipeline memCheckPipeline;
    auto test_pipeline = [&] {
        auto ie_api_wrapper = create_infer_api_wrapper();
        ie_api_wrapper->load_plugin(device);
        ie_api_wrapper->read_network(model);
        ie_api_wrapper->load_network(device);
        ie_api_wrapper->create_infer_request();
        ie_api_wrapper->prepare_input();
        ie_api_wrapper->infer();
        log_info("Memory consumption after Inference:");
        memCheckPipeline.record_measures(test_name);

        log_debug(memCheckPipeline.get_reference_record_for_test(test_name, model_name, precision, device));
        return memCheckPipeline.measure();
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
        auto ie_api_wrapper = create_infer_api_wrapper();
        ie_api_wrapper->load_plugin(device);
        ie_api_wrapper->set_config(device, ov::AnyMap{ov::num_streams(nstreams)});
        ie_api_wrapper->read_network(model);
        ie_api_wrapper->load_network(device);
        try {
            nireq = ie_api_wrapper->get_property(ov::optimal_number_of_infer_requests.name());
        } catch (const std::exception &ex) {
            log_err("Failed to query OPTIMAL_NUMBER_OF_INFER_REQUESTS");
        }

        for (unsigned int counter = 0; counter < nireq; counter++) {
            ie_api_wrapper->create_infer_request();
            ie_api_wrapper->prepare_input();

            ie_api_wrapper->infer();
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
