// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helpers/myriad_protocol_case.hpp"
#include "XLinkLog.h"

std::shared_ptr<InferenceEngine::Core> MyriadProtocolTests::ie = nullptr;

TEST_P(MyriadProtocolTests, CanInferenceWithProtocol) {
    if (protocol != NC_ANY_PROTOCOL && !getAmountOfDevices(protocol)) {
        GTEST_SKIP();
    }

    auto network = ie->ReadNetwork(convReluNormPoolFcModelFP16.model_xml_str,
                                   convReluNormPoolFcModelFP16.weights_blob);

    std::map<std::string, std::string> config = getConfigForProtocol(protocol);

    InferenceEngine::ExecutableNetwork exe_network =
            ie->LoadNetwork(network, "MYRIAD", config);

    ASSERT_NO_THROW(request = exe_network.CreateInferRequest());
    ASSERT_NO_THROW(request.Infer());
}

TEST_P(MyriadProtocolTests, NoErrorsMessagesWhenLoadNetworkSuccessful) {
    if (protocol != NC_USB) {
        GTEST_SKIP();
    }

    char buff[8192] = {};
    setbuf(stdout, buff);

    auto network = ie->ReadNetwork(convReluNormPoolFcModelFP16.model_xml_str,
                                   convReluNormPoolFcModelFP16.weights_blob);

    std::map<std::string, std::string> config = {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_WARNING)}};

    InferenceEngine::ExecutableNetwork exe_network =
            ie->LoadNetwork(network, "MYRIAD", config);
    setbuf(stdout, NULL);

    std::string content(buff);
    for (int i = MVLOG_WARN; i < MVLOG_LAST; i++) {
        auto found = content.find(mvLogHeader[i]);
        ASSERT_TRUE(found == std::string::npos);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_VPUConfigProtocolTests,
                        MyriadProtocolTests,
                        ::testing::ValuesIn(myriadProtocols),
                        MyriadProtocolTests::getTestCaseName);