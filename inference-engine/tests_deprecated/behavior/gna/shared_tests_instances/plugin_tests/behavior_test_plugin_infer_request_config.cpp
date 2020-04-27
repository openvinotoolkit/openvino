// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_infer_request_config.hpp"
#include "gna_test_data.hpp"

INSTANTIATE_TEST_CASE_P(BehaviorTest, BehaviorPluginTestInferRequestConfig,
                        ValuesIn(withCorrectConfValues),
                        getConfigTestCaseName);

INSTANTIATE_TEST_CASE_P(BehaviorTest, BehaviorPluginTestInferRequestConfigExclusiveAsync, ValuesIn(supportedValues),
                        getConfigTestCaseName);

bool CheckGnaHw() {
    if (auto envVar = std::getenv("IE_GNA_HW")) {
        return std::stoi(envVar) != 0;
    }
    return false;
}

class BehaviorPluginTestInferRequestWithGnaHw : public BehaviorPluginTestInferRequest {
};

TEST_P(BehaviorPluginTestInferRequestWithGnaHw, CanInferOrFailWithGnaHw) {
    TestEnv::Ptr testEnv;
    std::map<std::string, std::string> config = GetParam().config;

    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv, config));
    sts = testEnv->inferRequest->Infer(&response);

    if (CheckGnaHw()) {
        ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    } else {
        ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
        ASSERT_TRUE(strContains(response.msg, "Bad GNA status") ||         // GNA1 message
                    strContains(response.msg, "Unsuccessful Gna2Status")); // GNA2 message
    }
}

INSTANTIATE_TEST_CASE_P(BehaviorTest, BehaviorPluginTestInferRequestWithGnaHw,
                        ValuesIn(withGnaHwConfValue),
                        getConfigTestCaseName);
