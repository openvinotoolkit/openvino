// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gtest/gtest.h>

#include <ie_core.hpp>
#include "myriad_devices.hpp"
#include <behavior_test_plugin.h>
#include <mvnc.h>

static const std::vector<ncDeviceProtocol_t> myriadProtocols = {
    NC_ANY_PROTOCOL,
    NC_USB,
    NC_PCIE
};

class MyriadProtocolTests : public testing::Test,
                            public testing::WithParamInterface<ncDeviceProtocol_t>,
                            public MyriadDevicesInfo {
public:
    // IE variables
    InferenceEngine::IInferRequest::Ptr request;
    InferenceEngine::ResponseDesc resp;
    StatusCode statusCode = StatusCode::GENERAL_ERROR;
    static std::shared_ptr<InferenceEngine::Core> ie;

    // MVNC variables
    ncDeviceProtocol_t protocol;

    void SetUp() override;
    static void SetUpTestCase();
    static void TearDownTestCase();

    static std::map<std::string, std::string> getConfigForProtocol(ncDeviceProtocol_t protocol);
    static std::string getTestCaseName(
        const ::testing::TestParamInfo<ncDeviceProtocol_t> param);
};
