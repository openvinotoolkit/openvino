// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helpers/myriad_protocol_case.hpp"

std::shared_ptr<InferenceEngine::Core> MyriadProtocolTests::ie = nullptr;

TEST_P(MyriadProtocolTests, CanInferenceWithProtocol) {
    if (protocol != NC_ANY_PROTOCOL && !getAmountOfDevices(protocol)) {
        GTEST_SKIP();
    }

    auto network = ie->ReadNetwork(FuncTestUtils::TestModel::convReluNormPoolFcModelFP16.model_xml_str,
                                   FuncTestUtils::TestModel::convReluNormPoolFcModelFP16.weights_blob);

    std::map<std::string, std::string> config = getConfigForProtocol(protocol);

    InferenceEngine::IExecutableNetwork::Ptr exe_network =
            ie->LoadNetwork(network, "MYRIAD", config);

    ASSERT_NO_THROW(statusCode = exe_network->CreateInferRequest(request, &resp));
    ASSERT_EQ(statusCode, StatusCode::OK) << resp.msg;

    ASSERT_NO_THROW(statusCode = request->Infer(&resp));
    ASSERT_EQ(statusCode, StatusCode::OK) << resp.msg;
}



TEST_P(MyriadProtocolTests, NoWarningIfPatchFirmwareForUSBDevice) {
    if (protocol != NC_USB) {
        GTEST_SKIP();
    }

    char buff[8192] = {};
    setbuf(stdout, buff);

    auto network = ie->ReadNetwork(FuncTestUtils::TestModel::convReluNormPoolFcModelFP16.model_xml_str,
                                   FuncTestUtils::TestModel::convReluNormPoolFcModelFP16.weights_blob);

    std::map<std::string, std::string> config = {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_WARNING)}};

    InferenceEngine::IExecutableNetwork::Ptr exe_network =
            ie->LoadNetwork(network, "MYRIAD", config);
    setbuf(stdout, NULL);
    std::string warningMessege("Fail to patch");
    std::string content(buff);
    auto foundWarning = content.find(warningMessege);
    ASSERT_TRUE(foundWarning == std::string::npos);
}

INSTANTIATE_TEST_CASE_P(smoke_VPUConfigProtocolTests,
                        MyriadProtocolTests,
                        ::testing::ValuesIn(myriadProtocols),
                        MyriadProtocolTests::getTestCaseName);