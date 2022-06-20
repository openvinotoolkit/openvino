// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

// to suppress deprecated definition errors
#define IMPLEMENT_INFERENCE_ENGINE_PLUGIN
#include "gna2_model_wrapper_factory.hpp"
#include "gna_infer_request.hpp"
#include "gna_mock_api.hpp"
#include "model_worker_factory.hpp"
#include "model_subrequest.hpp"
#include "model_worker_pool.hpp"
#include "model_worker_impl.hpp"

using GNAPluginNS::GNAInferRequest;
using GNAPluginNS::GNAPlugin;
using ::testing::_;
using ::testing::Return;

class GNAWaitTest : public ::testing::Test {};

class GNAPluginForGNAWaitTest : public GNAPlugin {
public:
    // Test should be rewritten.
    // Curretnly they expect to call to Gna2RequestWait but this may happen only
    // if network is loaded and rquestes is queued

    // Prepare underlining object to enable GNAInferRequest::Wait() working
    GNAPluginForGNAWaitTest() {
        InferenceEngine::TensorDesc td{InferenceEngine::Precision::FP32, {1, 1}, InferenceEngine::Layout::HW};
        auto fakeInfo = std::make_shared<InferenceEngine::InputInfo>();
        auto fakePtr = std::make_shared<InferenceEngine::Data>("fakeName", td);
        fakeInfo->setInputData(fakePtr);
        outputs_data_map_["fakeOut"] = fakePtr;
        inputs_data_map_["fakeIn"] = fakeInfo;

        std::vector<GNAPluginNS::ModelSubrequest> subrequests;

        // Code below could be replaced with LoadNetwork with at least one layer
        auto acceleration_mode = config.pluginGna2AccMode;
        std::weak_ptr<GNADevice> weak_device = gnadevice;
        auto enqueue = [](uint32_t request_config_id) -> uint32_t {
            return 1;
        };

        auto wait = [weak_device](uint32_t request_id, int64_t timeout_milliseconds) -> GNARequestWaitStatus {
            if (auto device = weak_device.lock()) {
                return device->wait_for_reuqest(request_id, timeout_milliseconds);
            }
            THROW_GNA_EXCEPTION << "device is nullptr";
        };

        auto model = GNAPluginNS::Gna2ModelWrapperFactory::create_with_number_of_empty_operations(1);
        subrequests.emplace_back(1, enqueue, wait);
        auto model_worker = std::make_shared<GNAPluginNS::ModelWorkerImpl>(model, std::move(subrequests));

        request_pool_->add_model_worker(model_worker);
        model_worker->enqueue_request();
    }
};

class GNAInferRequestForGNAWaitTest : public GNAInferRequest {
public:
    // Prepare underlining object to enable Wait() working
    GNAInferRequestForGNAWaitTest(std::shared_ptr<GNAPlugin> plugin)
        : GNAInferRequest{plugin, plugin->GetNetworkInputs(), plugin->GetNetworkOutputs()} {
        inferRequestIdx = 0;
    }
};

TEST_F(GNAWaitTest, ReturnsGna2StatusDriverQoSTimeoutExceeded) {
    GNACppApi enableMocks;
    EXPECT_CALL(enableMocks, Gna2RequestWait(_, _)).Times(1).WillOnce(Return(Gna2StatusDriverQoSTimeoutExceeded));
    auto plugin = std::make_shared<GNAPluginForGNAWaitTest>();
    GNAInferRequestForGNAWaitTest inferRequest{plugin};
    ASSERT_EQ(InferenceEngine::INFER_NOT_STARTED, inferRequest.Wait(0));
}

TEST_F(GNAWaitTest, ReturnsGna2StatusWarningDeviceBusy) {
    GNACppApi enableMocks;
    EXPECT_CALL(enableMocks, Gna2RequestWait(_, _)).Times(1).WillOnce(Return(Gna2StatusWarningDeviceBusy));
    auto plugin = std::make_shared<GNAPluginForGNAWaitTest>();
    GNAInferRequestForGNAWaitTest inferRequest{plugin};

    ASSERT_EQ(InferenceEngine::RESULT_NOT_READY, inferRequest.Wait(0));
}
