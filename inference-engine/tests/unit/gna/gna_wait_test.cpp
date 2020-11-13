// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if GNA_LIB_VER == 2

#include <memory>

#include <gtest/gtest.h>

// to suppress deprecated definition errors
#define IMPLEMENT_INFERENCE_ENGINE_PLUGIN
#include "gna_infer_request.hpp"
#include "gna_mock_api.hpp"

using GNAPluginNS::GNAPlugin;
using GNAPluginNS::GNAInferRequest;
using ::testing::Return;
using ::testing::_;

class GNAWaitTest : public ::testing::Test {
};

class GNAPluginForGNAWaitTest : public GNAPlugin {
 public:
    // Prepare underlining object to enable GNAInferRequest::Wait() working
    GNAPluginForGNAWaitTest() {
        InferenceEngine::TensorDesc td{ InferenceEngine::Precision::FP32, {}, InferenceEngine::Layout::HW };
        auto fakeInfo = std::make_shared<InferenceEngine::InputInfo>();
        auto fakePtr = std::make_shared<InferenceEngine::Data>("fakeName", td);
        fakeInfo->setInputData(fakePtr);
        outputsDataMap["fakeOut"] = fakePtr;
        inputsDataMap["fakeIn"] = fakeInfo;
        gnaRequestConfigToRequestIdMap.push_back({ 0, 0, {} });
        InitGNADevice();
    }
};

class GNAInferRequestForGNAWaitTest : public GNAInferRequest {
 public:
    // Prepare underlining object to enable Wait() working
    GNAInferRequestForGNAWaitTest(std::shared_ptr<GNAPlugin> plugin) : GNAInferRequest{
                                            plugin,
                                            plugin->GetInputs(),
                                            plugin->GetOutputs() } {
        inferRequestIdx = 0;
    }
};

TEST_F(GNAWaitTest, ReturnsGna2StatusDriverQoSTimeoutExceeded) {
    GNACppApi enableMocks;
    EXPECT_CALL(enableMocks, Gna2RequestWait(_, _)).
        Times(1).
        WillOnce(Return(Gna2StatusDriverQoSTimeoutExceeded));
    auto plugin = std::make_shared<GNAPluginForGNAWaitTest>();
    GNAInferRequestForGNAWaitTest inferRequest{ plugin };
    ASSERT_EQ(InferenceEngine::INFER_NOT_STARTED, inferRequest.Wait(0));
}

TEST_F(GNAWaitTest, ReturnsGna2StatusWarningDeviceBusy) {
    GNACppApi enableMocks;
    EXPECT_CALL(enableMocks, Gna2RequestWait(_, _)).
        Times(1).
        WillOnce(Return(Gna2StatusWarningDeviceBusy));
    auto plugin = std::make_shared<GNAPluginForGNAWaitTest>();
    GNAInferRequestForGNAWaitTest inferRequest{ plugin };
    ASSERT_EQ(InferenceEngine::RESULT_NOT_READY, inferRequest.Wait(0));
}
#endif
