// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"
#include "details/ie_cnn_network_tools.h"
#include "exec_graph_info.hpp"

using namespace ::testing;
using namespace InferenceEngine;

namespace {
std::string getTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
    return obj.param.device + "_" + obj.param.input_blob_precision.name()
           + (obj.param.config.size() ? "_" + obj.param.config.begin()->second : "");
}
}

TEST_P(BehaviorPluginTestPerfCounters, EmptyWhenNotExecuted) {
    auto param = GetParam();

    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));

    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    ASSERT_EQ(StatusCode::GENERAL_ERROR, testEnv->inferRequest->GetPerformanceCounts(perfMap, &response)) << response.msg;
    ASSERT_EQ(perfMap.size(), 0);
}

TEST_P(BehaviorPluginTestPerfCounters, NotEmptyWhenExecuted) {
    auto param = GetParam();

    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv,
            {{ PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES }}));
    ASSERT_NO_THROW(sts = testEnv->inferRequest->Infer(&response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;

    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    ASSERT_EQ(StatusCode::OK, testEnv->inferRequest->GetPerformanceCounts(perfMap, &response)) << response.msg;
    ASSERT_NE(perfMap.size(), 0);
}
