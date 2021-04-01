// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_protocol_case.hpp"
#include "mvnc_ext.h"

void MyriadProtocolTests::SetUp() {
    protocol = GetParam();
}

void MyriadProtocolTests::SetUpTestCase() {
    try {
        ie = std::make_shared<InferenceEngine::Core>();
    }
    catch (...)
    {
        std::cerr << "Create core error";
    }
}

std::map<std::string, std::string> MyriadProtocolTests::getConfigForProtocol(const ncDeviceProtocol_t protocol) {
    switch (protocol) {
        case NC_ANY_PROTOCOL :
            return {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)},
                    {InferenceEngine::MYRIAD_PROTOCOL, ""}};
        case NC_USB:
            return {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)},
                    {InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_USB}};
        case NC_PCIE:
            return {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)},
                    {InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_PCIE}};
        default:
            return {};
    }
}

std::string MyriadProtocolTests::getTestCaseName(
    const ::testing::TestParamInfo<ncDeviceProtocol_t> param) {
    return std::string(ncProtocolToStr(param.param));
}

void MyriadProtocolTests::TearDownTestCase() {
    ie.reset();
}
