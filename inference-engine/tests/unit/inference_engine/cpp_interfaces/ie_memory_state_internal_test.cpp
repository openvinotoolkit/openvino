// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <cpp/ie_executable_network.hpp>

#include <cpp_interfaces/base/ie_executable_network_base.hpp>
#include <cpp_interfaces/base/ie_infer_async_request_base.hpp>

#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_imemory_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iasync_infer_request_internal.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

template <class T>
inline typename InferenceEngine::InferRequest make_infer_request(std::shared_ptr<T> impl) {
    typename InferRequestBase<T>::Ptr req(new InferRequestBase<T>(impl), [](IInferRequest* p) {
        p->Release();
    });
    return InferenceEngine::InferRequest(req);
}


class MemoryStateTests : public ::testing::Test {
 protected:
    shared_ptr<MockIExecutableNetworkInternal> mockExeNetworkInternal;
    shared_ptr<MockIAsyncInferRequestInternal> mockInferRequestInternal;
    shared_ptr<MockIMemoryStateInternal> mockMemoryStateInternal;

    virtual void SetUp() {
        mockExeNetworkInternal = make_shared<MockIExecutableNetworkInternal>();
        mockInferRequestInternal = make_shared<MockIAsyncInferRequestInternal>();
        mockMemoryStateInternal = make_shared<MockIMemoryStateInternal>();
    }
};

TEST_F(MemoryStateTests, ExecutableNetworkCanConvertOneMemoryStateFromCppToAPI) {
    IE_SUPPRESS_DEPRECATED_START
    auto net = make_executable_network(mockExeNetworkInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn(1);
    toReturn[0] = mockMemoryStateInternal;

    EXPECT_CALL(*mockExeNetworkInternal.get(), QueryState()).Times(2).WillRepeatedly(Return(toReturn));

    auto state = net.QueryState();
    ASSERT_EQ(state.size(), 1);
    IE_SUPPRESS_DEPRECATED_END
}

TEST_F(MemoryStateTests, ExecutableNetworkCanConvertZeroMemoryStateFromCppToAPI) {
    IE_SUPPRESS_DEPRECATED_START
    auto net = make_executable_network(mockExeNetworkInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;

    EXPECT_CALL(*mockExeNetworkInternal.get(), QueryState()).WillOnce(Return(toReturn));

    auto state = net.QueryState();
    ASSERT_EQ(state.size(), 0);
    IE_SUPPRESS_DEPRECATED_END
}

TEST_F(MemoryStateTests, ExecutableNetworkCanConvert2MemoryStatesFromCPPtoAPI) {
    IE_SUPPRESS_DEPRECATED_START
    auto net = make_executable_network(mockExeNetworkInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    toReturn.push_back(mockMemoryStateInternal);
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockExeNetworkInternal.get(), QueryState()).Times(3).WillRepeatedly(Return(toReturn));

    auto state = net.QueryState();
    ASSERT_EQ(state.size(), 2);
    IE_SUPPRESS_DEPRECATED_END
}

TEST_F(MemoryStateTests, MemoryStatePropagatesReset) {
    IE_SUPPRESS_DEPRECATED_START
    auto net = make_executable_network(mockExeNetworkInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockExeNetworkInternal.get(), QueryState()).Times(2).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), Reset()).Times(1);

    auto state = net.QueryState();
    state.front().Reset();
    IE_SUPPRESS_DEPRECATED_END
}

TEST_F(MemoryStateTests, MemoryStatePropagatesExceptionsFromReset) {
    IE_SUPPRESS_DEPRECATED_START
    auto net = make_executable_network(mockExeNetworkInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockExeNetworkInternal.get(), QueryState()).Times(2).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), Reset()).WillOnce(Throw(std::logic_error("some error")));

    auto state = net.QueryState();
    EXPECT_ANY_THROW(state.front().Reset());
    IE_SUPPRESS_DEPRECATED_END
}

TEST_F(MemoryStateTests, MemoryStatePropagatesGetName) {
    IE_SUPPRESS_DEPRECATED_START
    auto net = make_executable_network(mockExeNetworkInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockExeNetworkInternal.get(), QueryState()).Times(2).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), GetName()).WillOnce(Return("someName"));

    auto state = net.QueryState();
    EXPECT_STREQ(state.front().GetName().c_str(), "someName");
    IE_SUPPRESS_DEPRECATED_END
}

TEST_F(MemoryStateTests, MemoryStatePropagatesGetNameWithZeroLen) {
    IE_SUPPRESS_DEPRECATED_START
    auto net = make_executable_network(mockExeNetworkInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockExeNetworkInternal.get(), QueryState()).Times(1).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), GetName()).WillOnce(Return("someName"));

    IMemoryState::Ptr pState;

    static_cast<IExecutableNetwork::Ptr>(net)->QueryState(pState, 0, nullptr);
    char *name = reinterpret_cast<char *>(1);
    EXPECT_NO_THROW(pState->GetName(name, 0, nullptr));
    IE_SUPPRESS_DEPRECATED_END
}


TEST_F(MemoryStateTests, MemoryStatePropagatesGetNameWithLenOfOne) {
    IE_SUPPRESS_DEPRECATED_START
    auto net = make_executable_network(mockExeNetworkInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockExeNetworkInternal.get(), QueryState()).Times(1).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), GetName()).WillOnce(Return("someName"));

    IMemoryState::Ptr pState;

    static_cast<IExecutableNetwork::Ptr>(net)->QueryState(pState, 0, nullptr);
    char name[1];
    EXPECT_NO_THROW(pState->GetName(name, 1, nullptr));
    EXPECT_STREQ(name, "");
    IE_SUPPRESS_DEPRECATED_END
}

TEST_F(MemoryStateTests, MemoryStatePropagatesGetNameWithLenOfTwo) {
    IE_SUPPRESS_DEPRECATED_START
    auto net = make_executable_network(mockExeNetworkInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockExeNetworkInternal.get(), QueryState()).Times(1).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), GetName()).WillOnce(Return("someName"));

    IMemoryState::Ptr pState;

    static_cast<IExecutableNetwork::Ptr>(net)->QueryState(pState, 0, nullptr);
    char name[2];
    EXPECT_NO_THROW(pState->GetName(name, 2, nullptr));
    EXPECT_STREQ(name, "s");
    IE_SUPPRESS_DEPRECATED_END
}

TEST_F(MemoryStateTests, MemoryStateCanPropagateSetState) {
    IE_SUPPRESS_DEPRECATED_START
    auto net = make_executable_network(mockExeNetworkInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    Blob::Ptr saver;
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockExeNetworkInternal.get(), QueryState()).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), SetState(_)).WillOnce(SaveArg<0>(&saver));

    float data[] = {123, 124, 125};
    auto stateBlob = make_shared_blob<float>({ Precision::FP32, {3}, C }, data, sizeof(data) / sizeof(*data));

    EXPECT_NO_THROW(net.QueryState().front().SetState(stateBlob));
    ASSERT_FLOAT_EQ(saver->buffer().as<float*>()[0], 123);
    ASSERT_FLOAT_EQ(saver->buffer().as<float*>()[1], 124);
    ASSERT_FLOAT_EQ(saver->buffer().as<float*>()[2], 125);
    IE_SUPPRESS_DEPRECATED_END
}

TEST_F(MemoryStateTests, MemoryStateCanPropagateGetLastState) {
    IE_SUPPRESS_DEPRECATED_START
    auto net = make_executable_network(mockExeNetworkInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;

    float data[] = {123, 124, 125};
    auto stateBlob = make_shared_blob<float>({ Precision::FP32, {3}, C }, data, sizeof(data) / sizeof(*data));


    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockExeNetworkInternal.get(), QueryState()).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), GetState()).WillOnce(Return(stateBlob));


    auto saver = net.QueryState().front().GetState();
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float*>()[0], 123);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float*>()[1], 124);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float*>()[2], 125);
    IE_SUPPRESS_DEPRECATED_END
}

class MemoryStateInternalMockImpl : public MemoryStateInternal {
 public:
    using MemoryStateInternal::MemoryStateInternal;
    MOCK_METHOD0(Reset, void());
};

TEST_F(MemoryStateTests, MemoryStateInternalCanSaveName) {
    IMemoryStateInternal::Ptr pState(new MemoryStateInternalMockImpl("name"));
    ASSERT_STREQ(pState->GetName().c_str(), "name");
}


TEST_F(MemoryStateTests, MemoryStateInternalCanSaveState) {
    IMemoryStateInternal::Ptr pState(new MemoryStateInternalMockImpl("name"));
    float data[] = {123, 124, 125};
    auto stateBlob = make_shared_blob<float>({ Precision::FP32, {3}, C }, data, sizeof(data) / sizeof(*data));

    pState->SetState(stateBlob);
    auto saver = pState->GetState();

    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float *>()[0], 123);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float *>()[1], 124);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float *>()[2], 125);
}


TEST_F(MemoryStateTests, MemoryStateInternalCanSaveStateByReference) {
    IMemoryStateInternal::Ptr pState(new MemoryStateInternalMockImpl("name"));
    float data[] = {123, 124, 125};
    auto stateBlob = make_shared_blob<float>({ Precision::FP32, {3}, C }, data, sizeof(data) / sizeof(*data));

    pState->SetState(stateBlob);

    data[0] = 121;
    data[1] = 122;
    data[2] = 123;
    auto saver = pState->GetState();

    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float *>()[0], 121);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float *>()[1], 122);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float *>()[2], 123);
}

// Tests for InferRequest::QueryState
TEST_F(MemoryStateTests, InferRequestCanConvertOneMemoryStateFromCppToAPI) {
    auto req = make_infer_request(mockInferRequestInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn(1);
    toReturn[0] = mockMemoryStateInternal;

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).Times(2).WillRepeatedly(Return(toReturn));

    auto state = req.QueryState();
    ASSERT_EQ(state.size(), 1);
}

TEST_F(MemoryStateTests, InferRequestCanConvertZeroMemoryStateFromCppToAPI) {
    auto req = make_infer_request(mockInferRequestInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).WillOnce(Return(toReturn));

    auto state = req.QueryState();
    ASSERT_EQ(state.size(), 0);
}

TEST_F(MemoryStateTests, InferRequestCanConvert2MemoryStatesFromCPPtoAPI) {
    auto req = make_infer_request(mockInferRequestInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    toReturn.push_back(mockMemoryStateInternal);
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).Times(3).WillRepeatedly(Return(toReturn));

    auto state = req.QueryState();
    ASSERT_EQ(state.size(), 2);
}

TEST_F(MemoryStateTests, InfReqMemoryStatePropagatesReset) {
    auto req = make_infer_request(mockInferRequestInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).Times(2).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), Reset()).Times(1);

    auto state = req.QueryState();
    state.front().Reset();
}

TEST_F(MemoryStateTests, InfReqMemoryStatePropagatesExceptionsFromReset) {
    auto req = make_infer_request(mockInferRequestInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).Times(2).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), Reset()).WillOnce(Throw(std::logic_error("some error")));

    auto state = req.QueryState();
    EXPECT_ANY_THROW(state.front().Reset());
}

TEST_F(MemoryStateTests, InfReqMemoryStatePropagatesGetName) {
auto req = make_infer_request(mockInferRequestInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).Times(2).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), GetName()).WillOnce(Return("someName"));

    auto state = req.QueryState();
    EXPECT_STREQ(state.front().GetName().c_str(), "someName");
}

TEST_F(MemoryStateTests, InfReqMemoryStatePropagatesGetNameWithZeroLen) {
    auto req = make_infer_request(mockInferRequestInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).Times(1).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), GetName()).WillOnce(Return("someName"));

    IMemoryState::Ptr pState;

    static_cast<IInferRequest::Ptr>(req)->QueryState(pState, 0, nullptr);
    char *name = reinterpret_cast<char *>(1);
    EXPECT_NO_THROW(pState->GetName(name, 0, nullptr));
}

TEST_F(MemoryStateTests, InfReqMemoryStatePropagatesGetNameWithLenOfOne) {
    auto req = make_infer_request(mockInferRequestInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).Times(1).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), GetName()).WillOnce(Return("someName"));

    IMemoryState::Ptr pState;

    static_cast<IInferRequest::Ptr>(req)->QueryState(pState, 0, nullptr);
    char name[1];
    EXPECT_NO_THROW(pState->GetName(name, 1, nullptr));
    EXPECT_STREQ(name, "");
}

TEST_F(MemoryStateTests, InfReqMemoryStatePropagatesGetNameWithLenOfTwo) {
    auto req = make_infer_request(mockInferRequestInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).Times(1).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), GetName()).WillOnce(Return("someName"));

    IMemoryState::Ptr pState;

    static_cast<IInferRequest::Ptr>(req)->QueryState(pState, 0, nullptr);
    char name[2];
    EXPECT_NO_THROW(pState->GetName(name, 2, nullptr));
    EXPECT_STREQ(name, "s");
}

TEST_F(MemoryStateTests, InfReqMemoryStateCanPropagateSetState) {
    auto req = make_infer_request(mockInferRequestInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;
    Blob::Ptr saver;
    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), SetState(_)).WillOnce(SaveArg<0>(&saver));

    float data[] = {123, 124, 125};
    auto stateBlob = make_shared_blob<float>({ Precision::FP32, {3}, C }, data, sizeof(data) / sizeof(*data));

    EXPECT_NO_THROW(req.QueryState().front().SetState(stateBlob));
    ASSERT_FLOAT_EQ(saver->buffer().as<float*>()[0], 123);
    ASSERT_FLOAT_EQ(saver->buffer().as<float*>()[1], 124);
    ASSERT_FLOAT_EQ(saver->buffer().as<float*>()[2], 125);
}

TEST_F(MemoryStateTests, InfReqMemoryStateCanPropagateGetLastState) {
    auto req = make_infer_request(mockInferRequestInternal);
    std::vector<IMemoryStateInternal::Ptr> toReturn;

    float data[] = {123, 124, 125};
    auto stateBlob = make_shared_blob<float>({ Precision::FP32, {3}, C }, data, sizeof(data) / sizeof(*data));

    toReturn.push_back(mockMemoryStateInternal);

    EXPECT_CALL(*mockInferRequestInternal.get(), QueryState()).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mockMemoryStateInternal.get(), GetState()).WillOnce(Return(stateBlob));

    auto saver = req.QueryState().front().GetState();
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float*>()[0], 123);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float*>()[1], 124);
    ASSERT_FLOAT_EQ(saver->cbuffer().as<const float*>()[2], 125);
}
