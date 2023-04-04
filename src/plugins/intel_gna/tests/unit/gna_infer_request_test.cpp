// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_infer_request.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "any_copy.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "gna_mock_api.hpp"
#include "gna_plugin.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ::testing;
using namespace InferenceEngine;

using ov::intel_gna::GNAInferRequest;
using ov::intel_gna::GNAPlugin;
using ::testing::InSequence;

class GNAInferRequestTest : public ::testing::Test {
public:
    IInferRequestInternal::Ptr CreateRequest() {
        auto function = GetFunction();
        CNNNetwork cnn_network = CNNNetwork{function};

        SetExpectsForLoadNetworkAndShutDown(_data);
        const ov::AnyMap gna_config = {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT)};
        auto plugin = std::make_shared<GNAPlugin>(any_copy(gna_config));
        plugin->LoadNetwork(cnn_network);

        return std::make_shared<GNAInferRequest>(plugin, cnn_network.getInputsInfo(), cnn_network.getOutputsInfo());
    }

protected:
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

    void SetExpectsForLoadNetworkAndShutDown(std::vector<std::vector<uint8_t>>& data) {
        EXPECT_CALL(*_mock_api, Gna2MemoryAlloc(_, _, _))
            .Times(AtLeast(1))
            .WillRepeatedly(Invoke([&data](uint32_t size_requested, uint32_t* size_granted, void** memory_address) {
                data.push_back(std::vector<uint8_t>(size_requested));
                *size_granted = size_requested;
                *memory_address = data.back().data();
                return Gna2StatusSuccess;
            }));

        EXPECT_CALL(*_mock_api, Gna2DeviceGetVersion(_, _))
            .WillOnce(Invoke([](uint32_t deviceIndex, enum Gna2DeviceVersion* deviceVersion) {
                *deviceVersion = Gna2DeviceVersion::Gna2DeviceVersionSoftwareEmulation;
                return Gna2StatusSuccess;
            }));

        EXPECT_CALL(*_mock_api, Gna2DeviceOpen(_)).WillOnce(Return(Gna2StatusSuccess));

        EXPECT_CALL(*_mock_api, Gna2GetLibraryVersion(_, _))
            .Times(AtLeast(0))
            .WillRepeatedly(Return(Gna2StatusSuccess));

        EXPECT_CALL(*_mock_api, Gna2InstrumentationConfigCreate(_, _, _, _)).WillOnce(Return(Gna2StatusSuccess));

        EXPECT_CALL(*_mock_api, Gna2ModelCreate(_, _, _))
            .WillOnce(Invoke([](uint32_t deviceIndex, struct Gna2Model const* model, uint32_t* model_id) {
                *model_id = 0;
                return Gna2StatusSuccess;
            }));

        EXPECT_CALL(*_mock_api, Gna2RequestConfigCreate(_, _))
            .WillOnce(Invoke([](uint32_t model_Id, uint32_t* request_config_id) {
                *request_config_id = 0;
                return Gna2StatusSuccess;
            }));

        EXPECT_CALL(*_mock_api, Gna2InstrumentationConfigAssignToRequestConfig(_, _))
            .Times(AtLeast(1))
            .WillRepeatedly(Return(Gna2StatusSuccess));

        InSequence seq;
        EXPECT_CALL(*_mock_api, Gna2DeviceClose(_)).WillOnce(Return(Gna2StatusSuccess));
        EXPECT_CALL(*_mock_api, Gna2MemoryFree(_)).Times(AtLeast(1)).WillRepeatedly(Return(Gna2StatusSuccess));
    }

    void SetExpectOnEnqueue(const Gna2Status& return_status = Gna2StatusSuccess) {
        EXPECT_CALL(*_mock_api, Gna2RequestEnqueue(_, _)).WillOnce(Return(return_status));
    }

    void SetExpectOnWait(const Gna2Status& return_status = Gna2StatusSuccess) {
        EXPECT_CALL(*_mock_api, Gna2RequestWait(_, _)).WillOnce(Return(return_status));
    }

    void SetUp() override {
        _mock_api = std::make_shared<StrictMock<GNACppApi>>();
    }

    void TearDown() override {
        ASSERT_TRUE(Mock::VerifyAndClearExpectations(_mock_api.get()));
    }

    std::shared_ptr<StrictMock<GNACppApi>> _mock_api;
    std::vector<std::vector<uint8_t>> _data;
};

TEST_F(GNAInferRequestTest, start_async) {
    auto request = CreateRequest();
    SetExpectOnEnqueue();
    // wait on shutdown neede if request was enqueued by not waited
    SetExpectOnWait();
    EXPECT_NO_THROW(request->StartAsync());
}

TEST_F(GNAInferRequestTest, start_async_with_enqueue_error) {
    auto request = CreateRequest();
    // trigger Gna2RequestEnqueue to fail
    SetExpectOnEnqueue(Gna2StatusUnknownError);
    // no wait needed due the fact there are no enqueud requests
    EXPECT_THROW(request->StartAsync(), std::exception);
}

TEST_F(GNAInferRequestTest, start_async_with_wait) {
    auto request = CreateRequest();
    SetExpectOnEnqueue();
    // wait on shutdown needed if request was enqueued by not waited
    SetExpectOnWait();
    EXPECT_NO_THROW(request->StartAsync());
    EXPECT_EQ(OK, request->Wait(0));
}

TEST_F(GNAInferRequestTest, start_async_error_with_wait) {
    auto request = CreateRequest();
    SetExpectOnEnqueue(Gna2StatusUnknownError);
    // wait on shutdown needed if request was enqueued by not waited
    // SetExpectOnWait();
    EXPECT_THROW(request->StartAsync(), std::exception);
    EXPECT_EQ(INFER_NOT_STARTED, request->Wait(0));
}

TEST_F(GNAInferRequestTest, start_async_with_wait_error) {
    auto request = CreateRequest();
    SetExpectOnEnqueue();
    // wait on shutdown needed if request was enqueued by not waited
    SetExpectOnWait(Gna2StatusUnknownError);
    EXPECT_NO_THROW(request->StartAsync());
    EXPECT_THROW(request->Wait(0), std::exception);
}

TEST_F(GNAInferRequestTest, start_async_wait_check_recovery_after_wait_error) {
    auto request = CreateRequest();
    SetExpectOnEnqueue();
    SetExpectOnWait(Gna2StatusUnknownError);
    EXPECT_NO_THROW(request->StartAsync());
    EXPECT_THROW(request->Wait(0), std::exception);
    // check that no there is exception on second wiat after first failing
    EXPECT_EQ(INFER_NOT_STARTED, request->Wait(0));

    // start new request
    SetExpectOnEnqueue();
    SetExpectOnWait();
    EXPECT_NO_THROW(request->StartAsync());
    EXPECT_EQ(OK, request->Wait(0));
}

TEST_F(GNAInferRequestTest, infer) {
    auto request = CreateRequest();
    SetExpectOnEnqueue();
    SetExpectOnWait();
    EXPECT_NO_THROW(request->Infer());
}

TEST_F(GNAInferRequestTest, infer_enque_error) {
    auto request = CreateRequest();
    SetExpectOnEnqueue(Gna2StatusUnknownError);
    EXPECT_THROW(request->Infer(), std::exception);
}

TEST_F(GNAInferRequestTest, infer_wait_error_check_recovery) {
    auto request = CreateRequest();
    SetExpectOnEnqueue();
    SetExpectOnWait(Gna2StatusUnknownError);
    EXPECT_THROW(request->Infer(), std::exception);
    // check if next infer will execute properly after wait throwing
    SetExpectOnEnqueue();
    SetExpectOnWait();
    EXPECT_NO_THROW(request->Infer());
}
