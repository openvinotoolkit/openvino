// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

// to suppress deprecated definition errors
#define IMPLEMENT_INFERENCE_ENGINE_PLUGIN
#include "gna_infer_request.hpp"
#include "gna_mock_api.hpp"
#include "gna_plugin.hpp"
#include "request/model_wrapper_factory.hpp"
#include "request/subrequest_impl.hpp"
#include "request/worker_factory.hpp"
#include "request/worker_impl.hpp"
#include "request/worker_pool.hpp"

using namespace ov::intel_gna::request;
using ::testing::_;
using ::testing::Return;

class GNAWaitTest : public ::testing::Test {};

class GNAPluginForGNAWaitTest : public GNAPlugin {
public:
    // Prepare underlining object to enable GNAInferRequest::Wait() working
    GNAPluginForGNAWaitTest() {
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
        _worker = std::make_shared<WorkerImpl>(model, std::move(subrequests));

        requestWorkerPool_->addModelWorker(_worker);
    }

    void EnqueTestRequest() {
        _worker->enqueueRequest();
    }

private:
    std::shared_ptr<Worker> _worker;
};

class GNAInferRequestForGNAWaitTest : public GNAInferRequest {
public:
    // Prepare underlining object to enable Wait() working
    GNAInferRequestForGNAWaitTest(std::shared_ptr<GNAPluginForGNAWaitTest> plugin)
        : GNAInferRequest{plugin, plugin->GetNetworkInputs(), plugin->GetNetworkOutputs()},
          _plugin(plugin) {}

    void EnqueTestRequest() {
        _plugin->EnqueTestRequest();
        SetRequestIndex(0);
    }

    std::shared_ptr<GNAPluginForGNAWaitTest> _plugin;
};

TEST_F(GNAWaitTest, ReturnsGna2StatusDriverQoSTimeoutExceeded) {
    GNACppApi enableMocks;
    EXPECT_CALL(enableMocks, Gna2DeviceGetVersion(_, _))
        .WillOnce(testing::Invoke([](uint32_t deviceIndex, enum Gna2DeviceVersion* deviceVersion) {
            *deviceVersion = Gna2DeviceVersion::Gna2DeviceVersionSoftwareEmulation;
            return Gna2StatusSuccess;
        }));
    EXPECT_CALL(enableMocks, Gna2RequestWait(_, _)).Times(1).WillOnce(Return(Gna2StatusDriverQoSTimeoutExceeded));
    auto plugin = std::make_shared<GNAPluginForGNAWaitTest>();
    GNAInferRequestForGNAWaitTest inferRequest{plugin};
    inferRequest.EnqueTestRequest();
    ASSERT_EQ(InferenceEngine::INFER_NOT_STARTED, inferRequest.Wait(0));
}

TEST_F(GNAWaitTest, ReturnsGna2StatusWarningDeviceBusy) {
    GNACppApi enableMocks;
    EXPECT_CALL(enableMocks, Gna2DeviceGetVersion(_, _))
        .WillOnce(testing::Invoke([](uint32_t deviceIndex, enum Gna2DeviceVersion* deviceVersion) {
            *deviceVersion = Gna2DeviceVersion::Gna2DeviceVersionSoftwareEmulation;
            return Gna2StatusSuccess;
        }));
    EXPECT_CALL(enableMocks, Gna2RequestWait(_, _)).Times(1).WillOnce(Return(Gna2StatusWarningDeviceBusy));
    auto plugin = std::make_shared<GNAPluginForGNAWaitTest>();
    GNAInferRequestForGNAWaitTest inferRequest{plugin};
    inferRequest.EnqueTestRequest();
    ASSERT_EQ(InferenceEngine::RESULT_NOT_READY, inferRequest.Wait(0));
}

TEST_F(GNAWaitTest, ReturnsGna2StatusDeviceParameterOutOfRange) {
    GNACppApi enableMocks;
    EXPECT_CALL(enableMocks, Gna2DeviceGetVersion(_, _))
        .WillOnce(testing::Invoke([](uint32_t deviceIndex, enum Gna2DeviceVersion* deviceVersion) {
            *deviceVersion = Gna2DeviceVersion::Gna2DeviceVersionSoftwareEmulation;
            return Gna2StatusSuccess;
        }));
    EXPECT_CALL(enableMocks, Gna2RequestWait(_, _)).Times(1).WillOnce(Return(Gna2StatusDeviceParameterOutOfRange));
    auto plugin = std::make_shared<GNAPluginForGNAWaitTest>();
    GNAInferRequestForGNAWaitTest inferRequest{plugin};
    inferRequest.EnqueTestRequest();
    ASSERT_THROW(inferRequest.Wait(0), std::exception);
}

TEST_F(GNAWaitTest, ReturnsGna2StatusDeviceParameterOutOfRange_Extra_Sync) {
    GNACppApi enableMocks;
    EXPECT_CALL(enableMocks, Gna2DeviceGetVersion(_, _))
        .WillOnce(testing::Invoke([](uint32_t deviceIndex, enum Gna2DeviceVersion* deviceVersion) {
            *deviceVersion = Gna2DeviceVersion::Gna2DeviceVersionSoftwareEmulation;
            return Gna2StatusSuccess;
        }));
    EXPECT_CALL(enableMocks, Gna2RequestWait(_, _)).Times(1).WillOnce(Return(Gna2StatusDeviceParameterOutOfRange));
    auto plugin = std::make_shared<GNAPluginForGNAWaitTest>();
    GNAInferRequestForGNAWaitTest inferRequest{plugin};
    inferRequest.EnqueTestRequest();
    ASSERT_THROW(inferRequest.Wait(0), std::exception);
    EXPECT_CALL(enableMocks, Gna2RequestWait(_, _)).Times(0);
    ASSERT_EQ(InferenceEngine::INFER_NOT_STARTED, inferRequest.Wait(0));
}

TEST_F(GNAWaitTest, ReturnsGna2StatusDeviceParameterOutOfRange_Another_Use) {
    GNACppApi enableMocks;
    EXPECT_CALL(enableMocks, Gna2DeviceGetVersion(_, _))
        .WillOnce(testing::Invoke([](uint32_t deviceIndex, enum Gna2DeviceVersion* deviceVersion) {
            *deviceVersion = Gna2DeviceVersion::Gna2DeviceVersionSoftwareEmulation;
            return Gna2StatusSuccess;
        }));
    EXPECT_CALL(enableMocks, Gna2RequestWait(_, _)).Times(1).WillOnce(Return(Gna2StatusDeviceParameterOutOfRange));
    auto plugin = std::make_shared<GNAPluginForGNAWaitTest>();
    GNAInferRequestForGNAWaitTest inferRequest{plugin};
    inferRequest.EnqueTestRequest();
    ASSERT_THROW(inferRequest.Wait(0), std::exception);
    inferRequest.EnqueTestRequest();
    EXPECT_CALL(enableMocks, Gna2RequestWait(_, _)).Times(1).WillOnce(Return(Gna2StatusSuccess));
    ASSERT_EQ(InferenceEngine::OK, inferRequest.Wait(0));
}
