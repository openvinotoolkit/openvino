// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/perf_counters.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
TEST_P(InferRequestPerfCountersTest, CheckOperationInPerfMap) {
    InferenceEngine::CNNNetwork cnnNet(function);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_FATAL_FAILURE(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
    ASSERT_NO_THROW(perfMap = req.GetPerformanceCounts());
    for (const auto& op : function->get_ops()) {
        if (!strcmp(op->get_type_info().name, "Constant"))
            continue;
        auto it = perfMap.begin();
        while (true) {
            if (it->first.find(op->get_friendly_name() + "_") != std::string::npos || it->first == op->get_friendly_name()) {
                break;
            }
            it++;
            if (it == perfMap.end()) {
                GTEST_FAIL();
            }
        }
    }
}

const std::vector<std::map<std::string, std::string>> configs = {
        {}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                ::testing::ValuesIn(configs)),
                         InferRequestPerfCountersTest::getTestCaseName);
}  // namespace
