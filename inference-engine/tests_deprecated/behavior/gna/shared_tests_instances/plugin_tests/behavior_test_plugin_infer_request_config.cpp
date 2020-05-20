// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_infer_request_config.hpp"
#include "gna_test_data.hpp"

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, BehaviorPluginTestInferRequestConfig,
                        ValuesIn(withCorrectConfValues),
                        getConfigTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, BehaviorPluginTestInferRequestConfigExclusiveAsync, ValuesIn(supportedValues),
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

    if (CheckGnaHw()) {
        ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv, config));
        sts = testEnv->inferRequest->Infer(&response);
        ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    } else {
        try {
            _createAndCheckInferRequest(GetParam(), testEnv, config);
        } catch (InferenceEngineException ex) {
            ASSERT_TRUE(strContains(ex.what(), "Unsuccessful Gna2Status"));
            return;
        } catch (...) {
            FAIL();
        }

        sts = testEnv->inferRequest->Infer(&response);
        ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
        ASSERT_TRUE(strContains(response.msg, "Bad GNA status"));
    }
}

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, BehaviorPluginTestInferRequestWithGnaHw,
                        ValuesIn(withGnaHwConfValue),
                        getConfigTestCaseName);
