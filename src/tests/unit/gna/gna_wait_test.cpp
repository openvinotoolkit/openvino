// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

// to suppress deprecated definition errors
#define IMPLEMENT_INFERENCE_ENGINE_PLUGIN
#include "gna_infer_request.hpp"
#include "gna_mock_api.hpp"
#include "request/model_wrapper_factory.hpp"
#include "request/subrequest_impl.hpp"
#include "request/worker_factory.hpp"
#include "request/worker_impl.hpp"
#include "request/worker_pool.hpp"

using GNAPluginNS::GNAInferRequest;
using GNAPluginNS::GNAPlugin;
using ::testing::_;
using ::testing::Return;

class GNAWaitTest : public ::testing::Test {};

class GNAPluginForGNAWaitTest : public GNAPlugin {
public:
    // Prepare underlining object to enable GNAInferRequest::Wait() working
    GNAPluginForGNAWaitTest() {
        using namespace GNAPluginNS;
        using namespace request;

        InferenceEngine::TensorDesc td{InferenceEngine::Precision::FP32, {1, 1}, InferenceEngine::Layout::HW};
        auto fakeInfo = std::make_shared<InferenceEngine::InputInfo>();
        auto fakePtr = std::make_shared<InferenceEngine::Data>("fakeName", td);
        fakeInfo->setInputData(fakePtr);
        outputs_data_map_["fakeOut"] = fakePtr;
        inputs_data_map_["fakeIn"] = fakeInfo;

        std::vector<std::shared_ptr<Subrequest>> subrequests;

        // In the future code below could replaced with LoadNetwork with at least one layer and QueueInference
        std::weak_ptr<GNADevice> weakDevice = gnadevice;

        auto enqueue = []() -> uint32_t {
            return 1;
        };

        auto wait = [weakDevice](uint32_t requestID, int64_t timeoutMilliseconds) {
            if (auto device = weakDevice.lock()) {
                return device->waitForRequest(requestID, timeoutMilliseconds);
            }
            THROW_GNA_EXCEPTION << "device is nullptr";
        };

        auto model = ModelWrapperFactory::createWithNumberOfEmptyOperations(1);
        subrequests.push_back(std::make_shared<SubrequestImpl>(std::move(enqueue), std::move(wait)));
        auto worker = std::make_shared<WorkerImpl>(model, std::move(subrequests));

        requestWorkerPool_->addModelWorker(worker);
        worker->enqueueRequest();
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
