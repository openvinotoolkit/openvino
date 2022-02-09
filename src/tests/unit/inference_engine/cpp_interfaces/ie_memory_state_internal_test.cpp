// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <cpp/ie_executable_network.hpp>

#include <cpp/ie_executable_network_base.hpp>
#include <cpp/ie_infer_async_request_base.hpp>

#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "cpp/ie_plugin.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;


class InferRequestVariableStateTests : public ::testing::Test {
 protected:
    shared_ptr<MockIExecutableNetworkInternal> mockExeNetworkInternal;
    shared_ptr<MockIInferRequestInternal> mockInferRequestInternal;
    shared_ptr<MockIVariableStateInternal> mockVariableStateInternal;
    MockIInferencePlugin*                           mockIPlugin;
    InferencePlugin                                 plugin;
    ov::SoPtr<IExecutableNetworkInternal>  net;
    IInferRequestInternal::Ptr                      req;

    void SetUp() override {
        mockExeNetworkInternal = make_shared<MockIExecutableNetworkInternal>();
        mockInferRequestInternal = make_shared<MockIInferRequestInternal>();
        mockVariableStateInternal = make_shared<MockIVariableStateInternal>();
        ON_CALL(*mockExeNetworkInternal, CreateInferRequest()).WillByDefault(Return(mockInferRequestInternal));
        auto mockIPluginPtr = std::make_shared<MockIInferencePlugin>();
        ON_CALL(*mockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).WillByDefault(Return(mockExeNetworkInternal));
        plugin = InferenceEngine::InferencePlugin{mockIPluginPtr, {}};
        net = plugin.LoadNetwork(CNNNetwork{}, {});
        req = net->CreateInferRequest();
    }
};

class VariableStateInternalMockImpl : public IVariableStateInternal {
 public:
    VariableStateInternalMockImpl(const char* name) : IVariableStateInternal(name) {}
    MOCK_METHOD0(Reset, void());
};

TEST_F(InferRequestVariableStateTests, VariableStateInternalCanSaveName) {
    IVariableStateInternal::Ptr pState(new VariableStateInternalMockImpl("VariableStateInternalMockImpl"));
    ASSERT_STREQ(pState->GetName().c_str(), "VariableStateInternalMockImpl");
}

TEST_F(InferRequestVariableStateTests, VariableStateInternalCanSaveState) {
    IVariableStateInternal::Ptr pState(new VariableStateInternalMockImpl("VariableStateInternalMockImpl"));
    float data[] = {123, 124, 125};
    auto stateBlob = make_shared_blob<float>({ Precision::FP32, {3}, C }, data, sizeof(data) / sizeof(*data));

    pState->SetState(stateBlob);
    auto saver = pState->GetState();

    ASSERT_NE(saver, nullptr);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float *>()[0], 123);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float *>()[1], 124);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float *>()[2], 125);
}


TEST_F(InferRequestVariableStateTests, VariableStateInternalCanSaveStateByReference) {
    IVariableStateInternal::Ptr pState(new VariableStateInternalMockImpl("VariableStateInternalMockImpl"));
    float data[] = {123, 124, 125};
    auto stateBlob = make_shared_blob<float>({ Precision::FP32, {3}, C }, data, sizeof(data) / sizeof(*data));

    pState->SetState(stateBlob);

    data[0] = 121;
    data[1] = 122;
    data[2] = 123;
    auto saver = pState->GetState();

    ASSERT_NE(saver, nullptr);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float *>()[0], 121);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float *>()[1], 122);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float *>()[2], 123);
}

// Tests for InferRequest::QueryState
TEST_F(InferRequestVariableStateTests, InferRequestCanConvertOneVariableStateFromCppToAPI) {
    std::vector<IVariableStateInternal::Ptr> toReturn(1);
    toReturn[0] = mockVariableStateInternal;

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).Times(1).WillRepeatedly(Return(toReturn));

    auto state = req->QueryState();
    ASSERT_EQ(state.size(), 1);
}

TEST_F(InferRequestVariableStateTests, InferRequestCanConvertZeroVariableStateFromCppToAPI) {
    std::vector<IVariableStateInternal::Ptr> toReturn;

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).WillOnce(Return(toReturn));

    auto state = req->QueryState();
    ASSERT_EQ(state.size(), 0);
}

TEST_F(InferRequestVariableStateTests, InferRequestCanConvert2VariableStatesFromCPPtoAPI) {
    std::vector<IVariableStateInternal::Ptr> toReturn;
    toReturn.push_back(mockVariableStateInternal);
    toReturn.push_back(mockVariableStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).Times(1).WillRepeatedly(Return(toReturn));

    auto state = req->QueryState();
    ASSERT_EQ(state.size(), 2);
}

TEST_F(InferRequestVariableStateTests, InfReqVariableStatePropagatesReset) {
    std::vector<IVariableStateInternal::Ptr> toReturn;
    toReturn.push_back(mockVariableStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).Times(1).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockVariableStateInternal.get(), Reset()).Times(1);

    auto state = req->QueryState();
    state.front()->Reset();
}

TEST_F(InferRequestVariableStateTests, InfReqVariableStatePropagatesExceptionsFromReset) {
    std::vector<IVariableStateInternal::Ptr> toReturn;
    toReturn.push_back(mockVariableStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).Times(1).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockVariableStateInternal.get(), Reset()).WillOnce(Throw(std::logic_error("some error")));

    auto state = req->QueryState();
    EXPECT_ANY_THROW(state.front()->Reset());
}

TEST_F(InferRequestVariableStateTests, InfReqVariableStatePropagatesGetName) {
    std::vector<IVariableStateInternal::Ptr> toReturn;
    toReturn.push_back(mockVariableStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).Times(1).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockVariableStateInternal.get(), GetName()).WillOnce(Return("someName"));

    auto state = req->QueryState();
    EXPECT_STREQ(state.front()->GetName().c_str(), "someName");
}

TEST_F(InferRequestVariableStateTests, InfReqVariableStateCanPropagateSetState) {
    std::vector<IVariableStateInternal::Ptr> toReturn;
    Blob::Ptr saver;
    toReturn.push_back(mockVariableStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockVariableStateInternal.get(), SetState(_)).WillOnce(SaveArg<0>(&saver));

    float data[] = {123, 124, 125};
    auto stateBlob = make_shared_blob<float>({ Precision::FP32, {3}, C }, data, sizeof(data) / sizeof(*data));

    EXPECT_NO_THROW(req->QueryState().front()->SetState(stateBlob));
    ASSERT_FLOAT_EQ(saver->buffer().as<float*>()[0], 123);
    ASSERT_FLOAT_EQ(saver->buffer().as<float*>()[1], 124);
    ASSERT_FLOAT_EQ(saver->buffer().as<float*>()[2], 125);
}

TEST_F(InferRequestVariableStateTests, InfReqVariableStateCanPropagateGetLastState) {
    std::vector<IVariableStateInternal::Ptr> toReturn;

    float data[] = {123, 124, 125};
    auto stateBlob = make_shared_blob<float>({ Precision::FP32, {3}, C }, data, sizeof(data) / sizeof(*data));

    toReturn.push_back(mockVariableStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockVariableStateInternal.get(), GetState()).WillOnce(Return(stateBlob));

    auto saver = req->QueryState().front()->GetState();
    ASSERT_NE(saver, nullptr);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float*>()[0], 123);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float*>()[1], 124);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float*>()[2], 125);
}
