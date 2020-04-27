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

INSTANTIATE_TEST_CASE_P(VPUConfigProtocolTests,
                        MyriadProtocolTests,
                        ::testing::ValuesIn(myriadProtocols),
                        MyriadProtocolTests::getTestCaseName);