// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <vector>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gna_mock_api.hpp"
#include "gna_plugin.hpp"
#include "gna_infer_request.hpp"
#include "ngraph_functions/builders.hpp"
#include "common_test_utils/data_utils.hpp"
#include "any_copy.hpp"

using namespace ::testing;
using GNAPluginNS::GNAPlugin;
using GNAPluginNS::GNAInferRequest;
using namespace InferenceEngine;

class GNAInferRequestTest : public ::testing::Test {};

std::shared_ptr<ngraph::Function> GetFunction() {
    auto ngPrc = ngraph::element::f32;
    std::vector<size_t> shape = {1, 10};
    auto params = ngraph::builder::makeParams(ngPrc, {shape});
    auto shape_size = ov::shape_size(shape);
    auto add_const =
        ngraph::builder::makeConstant<float>(ngPrc,
                                             shape,
                                             CommonTestUtils::generate_float_numbers(shape_size, -0.5f, 0.5f),
                                             false);

    auto add = std::make_shared<ngraph::opset9::Add>(params[0], add_const);
    auto res = std::make_shared<ngraph::op::Result>(add);
    auto function = std::make_shared<ngraph::Function>(res, params, "Add");
    return function;
}

void SetExpectsForLoadNetworhAndShutDown(GNACppApi& mock_api, std::vector<std::vector<uint8_t>>& data) {
    EXPECT_CALL(mock_api, Gna2MemoryAlloc(_, _, _))
        .Times(AtLeast(1))
        .WillRepeatedly(Invoke([&data](uint32_t size_requested, uint32_t* size_granted, void** memory_address) {
            data.push_back(std::vector<uint8_t>(size_requested));
            *size_granted = size_requested;
            *memory_address = data.back().data();
            return Gna2StatusSuccess;
        }));

    EXPECT_CALL(mock_api, Gna2DeviceGetVersion(_, _))
        .WillOnce(Invoke([](uint32_t deviceIndex, enum Gna2DeviceVersion* deviceVersion) {
            *deviceVersion = Gna2DeviceVersionSoftwareEmulation;
            return Gna2StatusSuccess;
        }));

    EXPECT_CALL(mock_api, Gna2DeviceOpen(_)).WillOnce(Return(Gna2StatusSuccess));

    EXPECT_CALL(mock_api, Gna2GetLibraryVersion(_, _)).Times(AtLeast(0)).WillRepeatedly(Return(Gna2StatusSuccess));

    EXPECT_CALL(mock_api, Gna2InstrumentationConfigCreate(_, _, _, _)).WillOnce(Return(Gna2StatusSuccess));

    EXPECT_CALL(mock_api, Gna2ModelCreate(_, _, _))
        .WillOnce(Invoke([](uint32_t deviceIndex, struct Gna2Model const* model, uint32_t* model_id) {
            *model_id = 0;
            return Gna2StatusSuccess;
        }));

    EXPECT_CALL(mock_api, Gna2RequestConfigCreate(_, _))
        .WillOnce(Invoke([](uint32_t model_Id, uint32_t* request_config_id) {
            *request_config_id = 0;
            return Gna2StatusSuccess;
        }));

    EXPECT_CALL(mock_api, Gna2InstrumentationConfigAssignToRequestConfig(_, _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(Gna2StatusSuccess));

    InSequence seq;
    EXPECT_CALL(mock_api, Gna2DeviceClose(_)).WillOnce(Return(Gna2StatusSuccess));
    EXPECT_CALL(mock_api, Gna2MemoryFree(_)).Times(AtLeast(1)).WillRepeatedly(Return(Gna2StatusSuccess));
}

IInferRequestInternal::Ptr CreateRequest(GNACppApi& mock_api, std::vector<std::vector<uint8_t>>& data) {
    auto function = GetFunction();
    CNNNetwork cnn_network = CNNNetwork{function};

    SetExpectsForLoadNetworhAndShutDown(mock_api, data);
    const ov::AnyMap gna_config = {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT)};
    auto plugin = std::make_shared<GNAPlugin>(any_copy(gna_config));
    plugin->LoadNetwork(cnn_network);

    return std::make_shared<GNAInferRequest>(plugin, cnn_network.getInputsInfo(), cnn_network.getOutputsInfo());
}

void SetExpectOnEnqueue(GNACppApi& mock_api, const Gna2Status& return_status = Gna2StatusSuccess) {
    EXPECT_CALL(mock_api, Gna2RequestEnqueue(_, _)).WillOnce(Return(return_status));
}

void SetExpectOnWait(GNACppApi& mock_api, const Gna2Status& return_status = Gna2StatusSuccess) {
    EXPECT_CALL(mock_api, Gna2RequestWait(_, _)).WillOnce(Return(return_status));
}

TEST_F(GNAInferRequestTest, start_async) {
    GNACppApi mock_api;
    std::vector<std::vector<uint8_t>> data;
    {
        auto request = CreateRequest(mock_api, data);
        SetExpectOnEnqueue(mock_api);
        // wait on shutdown neede if request was enqueued by not waited
        SetExpectOnWait(mock_api);
        EXPECT_NO_THROW(request->StartAsync());
    }
    ASSERT_TRUE(Mock::VerifyAndClearExpectations(&mock_api));
}

TEST_F(GNAInferRequestTest, start_async_with_enqueue_error) {
    GNACppApi mock_api;
    std::vector<std::vector<uint8_t>> data;
    {
        auto request = CreateRequest(mock_api, data);
        // trigger Gna2RequestEnqueue to fail
        SetExpectOnEnqueue(mock_api, Gna2StatusUnknownError);
        // no wait needed due the fact there are no enqueud requests
        EXPECT_THROW(request->StartAsync(), std::exception);
    }
    ASSERT_TRUE(Mock::VerifyAndClearExpectations(&mock_api));
}

TEST_F(GNAInferRequestTest, start_async_with_wait) {
    GNACppApi mock_api;
    std::vector<std::vector<uint8_t>> data;
    {
        auto request = CreateRequest(mock_api, data);
        SetExpectOnEnqueue(mock_api);
        // wait on shutdown needed if request was enqueued by not waited
        SetExpectOnWait(mock_api);
        EXPECT_NO_THROW(request->StartAsync());
        EXPECT_EQ(OK, request->Wait(0));
    }
    ASSERT_TRUE(Mock::VerifyAndClearExpectations(&mock_api));
}

TEST_F(GNAInferRequestTest, start_async_error_with_wait) {
    GNACppApi mock_api;
    std::vector<std::vector<uint8_t>> data;
    {
        auto request = CreateRequest(mock_api, data);
        SetExpectOnEnqueue(mock_api, Gna2StatusUnknownError);
        // wait on shutdown needed if request was enqueued by not waited
        // SetExpectOnWait();
        EXPECT_THROW(request->StartAsync(), std::exception);
        EXPECT_EQ(INFER_NOT_STARTED, request->Wait(0));
    }
    ASSERT_TRUE(Mock::VerifyAndClearExpectations(&mock_api));
}

TEST_F(GNAInferRequestTest, start_async_with_wait_error) {
    GNACppApi mock_api;
    std::vector<std::vector<uint8_t>> data;
    {
        auto request = CreateRequest(mock_api, data);
        SetExpectOnEnqueue(mock_api);
        // wait on shutdown needed if request was enqueued by not waited
        SetExpectOnWait(mock_api, Gna2StatusUnknownError);
        EXPECT_NO_THROW(request->StartAsync());
        EXPECT_THROW(request->Wait(0), std::exception);
    }
    ASSERT_TRUE(Mock::VerifyAndClearExpectations(&mock_api));
}

TEST_F(GNAInferRequestTest, start_async_wait_check_recovery_after_wait_error) {
    GNACppApi mock_api;
    std::vector<std::vector<uint8_t>> data;
    {
        auto request = CreateRequest(mock_api, data);
        SetExpectOnEnqueue(mock_api);
        SetExpectOnWait(mock_api, Gna2StatusUnknownError);
        EXPECT_NO_THROW(request->StartAsync());
        EXPECT_THROW(request->Wait(0), std::exception);
        // check that no there is exception on second wiat after first failing
        EXPECT_EQ(INFER_NOT_STARTED, request->Wait(0));

        // start new request
        SetExpectOnEnqueue(mock_api);
        SetExpectOnWait(mock_api);
        EXPECT_NO_THROW(request->StartAsync());
        EXPECT_EQ(OK, request->Wait(0));
    }
    ASSERT_TRUE(Mock::VerifyAndClearExpectations(&mock_api));
}

TEST_F(GNAInferRequestTest, infer) {
    std::vector<std::vector<uint8_t>> data;
    GNACppApi mock_api;
    {
        auto request = CreateRequest(mock_api, data);
        SetExpectOnEnqueue(mock_api);
        SetExpectOnWait(mock_api);
        EXPECT_NO_THROW(request->Infer());
    }
    ASSERT_TRUE(Mock::VerifyAndClearExpectations(&mock_api));
}

TEST_F(GNAInferRequestTest, infer_enque_error) {
    GNACppApi mock_api;
    std::vector<std::vector<uint8_t>> data;
    {
        auto request = CreateRequest(mock_api, data);
        SetExpectOnEnqueue(mock_api, Gna2StatusUnknownError);
        EXPECT_THROW(request->Infer(), std::exception);
    }
    ASSERT_TRUE(Mock::VerifyAndClearExpectations(&mock_api));
}

TEST_F(GNAInferRequestTest, infer_wait_error_check_recovery) {
    GNACppApi mock_api;
    std::vector<std::vector<uint8_t>> data;
    {
        auto request = CreateRequest(mock_api, data);
        SetExpectOnEnqueue(mock_api);
        EXPECT_THROW(request->Infer(), std::exception);
        // check if next infer will execute properly after wait throwing
        SetExpectOnEnqueue(mock_api);
        SetExpectOnWait(mock_api);
        EXPECT_NO_THROW(request->Infer());
    }
    ASSERT_TRUE(Mock::VerifyAndClearExpectations(&mock_api));
}
