// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_utils.h"
#include "../common/tests_utils.h"
#include "../common/managers/thread_manager.h"
#include "tests_pipelines/tests_pipelines.h"

#include <gtest/gtest.h>


void checkReferences(long ref_vmsize, long ref_vmpeak, long ref_vmrss, long ref_vmhwm) {
    EXPECT_GT(ref_vmsize, 0) << "Reference value of VmSize is less than 0. Value: " << ref_vmsize;
    EXPECT_GT(ref_vmpeak, 0) << "Reference value of VmPeak is less than 0. Value: " << ref_vmpeak;
    EXPECT_GT(ref_vmrss, 0) << "Reference value of VmRSS is less than 0. Value: " << ref_vmrss;
    EXPECT_GT(ref_vmhwm, 0) << "Reference value of VmHWM is less than 0. Value: " << ref_vmhwm;
}

class MemCheckTestSuite : public ::testing::TestWithParam<TestCase> {
};

// tests_pipelines/tests_pipelines.cpp
TEST_P(MemCheckTestSuite, create_exenetwork) {
    std::string test_name = "create_exenetwork";
    auto test_params = GetParam();

    TestReferences test_refs;
    test_refs.collect_vm_values_for_test(test_name, test_params);

    checkReferences(test_refs.ref_vmsize, test_refs.ref_vmpeak, test_refs.ref_vmrss, test_refs.ref_vmhwm);

    TestResult res = test_create_exenetwork(test_params.model_name, test_params.model, test_params.device,
                                            test_refs.ref_vmsize, test_refs.ref_vmpeak, test_refs.ref_vmrss,
                                            test_refs.ref_vmhwm);
    EXPECT_EQ(res.first, TestStatus::TEST_OK) << res.second;
}

TEST_P(MemCheckTestSuite, infer_request_inference) {
    std::string test_name = "infer_request_inference";
    auto test_params = GetParam();

    TestReferences test_refs;
    test_refs.collect_vm_values_for_test(test_name, test_params);

    checkReferences(test_refs.ref_vmsize, test_refs.ref_vmpeak, test_refs.ref_vmrss, test_refs.ref_vmhwm);

    TestResult res = test_infer_request_inference(test_params.model_name, test_params.model, test_params.device,
                                                  test_refs.ref_vmsize, test_refs.ref_vmpeak, test_refs.ref_vmrss,
                                                  test_refs.ref_vmhwm);
    EXPECT_EQ(res.first, TestStatus::TEST_OK) << res.second;
}
// tests_pipelines/tests_pipelines.cpp

INSTANTIATE_TEST_CASE_P(MemCheckTests, MemCheckTestSuite,
                        ::testing::ValuesIn(
                                generateTestsParams({"devices", "models"})),
                        getTestCaseName);
