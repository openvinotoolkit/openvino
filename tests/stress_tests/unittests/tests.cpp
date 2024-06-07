// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../common/tests_utils.h"
#include "tests_pipelines/tests_pipelines.h"

#include <gtest/gtest.h>

class UnitTestSuiteNoModel : public ::testing::TestWithParam<TestCase> {
};

class UnitTestSuiteNoDevice : public ::testing::TestWithParam<TestCase> {
};

class UnitTestSuite : public ::testing::TestWithParam<TestCase> {
};

// tests_pipelines/tests_pipelines.cpp
TEST_P(UnitTestSuiteNoModel, load_unload_plugin) {
    runTest(test_load_unload_plugin, GetParam());
}

TEST_P(UnitTestSuiteNoDevice, read_network) {
    runTest(test_read_network, GetParam());
}

TEST_P(UnitTestSuiteNoDevice, cnnnetwork_reshape_batch_x2) {
    runTest(test_cnnnetwork_reshape_batch_x2, GetParam());
}

TEST_P(UnitTestSuiteNoDevice, set_input_params) {
    runTest(test_set_input_params, GetParam());
}

TEST_P(UnitTestSuite, create_compiled_model) {
    runTest(test_create_compiled_model, GetParam());
}

TEST_P(UnitTestSuite, create_infer_request) {
    runTest(test_create_infer_request, GetParam());
}

TEST_P(UnitTestSuite, infer_request_inference) {
    runTest(test_infer_request_inference, GetParam());
}
// tests_pipelines/tests_pipelines.cpp


// tests_pipelines/tests_pipelines_full_pipeline.cpp
TEST_P(UnitTestSuite, load_unload_plugin_full_pipeline) {
    runTest(test_load_unload_plugin_full_pipeline, GetParam());
}

TEST_P(UnitTestSuite, read_network_full_pipeline) {
    runTest(test_read_network_full_pipeline, GetParam());
}

TEST_P(UnitTestSuite, set_input_params_full_pipeline) {
    runTest(test_set_input_params_full_pipeline, GetParam());
}

TEST_P(UnitTestSuite, cnnnetwork_reshape_batch_x2_full_pipeline) {
    runTest(test_cnnnetwork_reshape_batch_x2_full_pipeline, GetParam());
}

TEST_P(UnitTestSuite, create_exenetwork_full_pipeline) {
    runTest(test_create_exenetwork_full_pipeline, GetParam());
}

TEST_P(UnitTestSuite, create_infer_request_full_pipeline) {
    runTest(test_create_infer_request_full_pipeline, GetParam());
}

TEST_P(UnitTestSuite, infer_request_inference_full_pipeline) {
    runTest(test_infer_request_inference_full_pipeline, GetParam());
}

TEST_P(UnitTestSuite, recreate_and_infer_in_thread) {
    runTest(test_recreate_and_infer_in_thread, GetParam());
}


// tests_pipelines/tests_pipelines_full_pipeline.cpp

INSTANTIATE_TEST_SUITE_P(StressUnitTests, UnitTestSuiteNoModel,
                         ::testing::ValuesIn(generateTestsParams(
                                 {"processes", "threads", "iterations", "devices"})),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(StressUnitTests, UnitTestSuiteNoDevice,
                         ::testing::ValuesIn(generateTestsParams(
                                 {"processes", "threads", "iterations", "models"})),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(StressUnitTests, UnitTestSuite,
                         ::testing::ValuesIn(generateTestsParams(
                                 {"processes", "threads", "iterations", "devices", "models"})),
                         getTestCaseName);
