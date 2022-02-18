// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
        InferenceEngine::TensorDesc td{ InferenceEngine::Precision::FP32, {1, 1}, InferenceEngine::Layout::HW };
        auto fakeInfo = std::make_shared<InferenceEngine::InputInfo>();
        auto fakePtr = std::make_shared<InferenceEngine::Data>("fakeName", td);
        fakeInfo->setInputData(fakePtr);
        outputs_data_map_["fakeOut"] = fakePtr;
        inputs_data_map_["fakeIn"] = fakeInfo;
        gnaRequestConfigToRequestIdMap.push_back(std::tuple<uint32_t, int64_t, InferenceEngine::BlobMap>{ 0, 0, {} });
        InitGNADevice();
    }
};

class GNAInferRequestForGNAWaitTest : public GNAInferRequest {
 public:
    // Prepare underlining object to enable Wait() working
    GNAInferRequestForGNAWaitTest(std::shared_ptr<GNAPlugin> plugin) : GNAInferRequest {
                                            plugin,
                                            plugin->GetNetworkInputs(),
                                            plugin->GetNetworkOutputs() } {
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
