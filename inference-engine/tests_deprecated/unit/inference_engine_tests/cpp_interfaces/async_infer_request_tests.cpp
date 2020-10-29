// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <gmock/gmock-generated-actions.h>
#include <ie_version.hpp>
#include <cpp/ie_infer_request.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include <cpp_interfaces/base/ie_infer_async_request_base.hpp>

#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/mock_not_empty_icnn_network.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_async_infer_request_internal.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

constexpr const char* MockNotEmptyICNNNetwork::INPUT_BLOB_NAME;
constexpr const char* MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME;

class InferRequestTests : public ::testing::Test {
protected:
    std::shared_ptr<MockIInferRequest> mock_request;
    InferRequest::Ptr requestWrapper;
    ResponseDesc dsc;

    shared_ptr<MockAsyncInferRequestInternal> mockInferRequestInternal;
    MockNotEmptyICNNNetwork mockNotEmptyNet;
    std::string _incorrectName;
    std::string _inputName;
    std::string _failedToFindInOutError;
    std::string _inputDataNotAllocatedError;
    std::string _inputDataIsEmptyError;

    virtual void TearDown() {
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockInferRequestInternal.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mock_request.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(requestWrapper.get()));
    }

    virtual void SetUp() {
        mock_request = make_shared<MockIInferRequest>();
        requestWrapper = make_shared<InferRequest>(mock_request);
        _incorrectName = "incorrect_name";
        _inputName = MockNotEmptyICNNNetwork::INPUT_BLOB_NAME;
        _failedToFindInOutError =
                NOT_FOUND_str + "Failed to find input or output with name: \'" + _incorrectName + "\'";
        _inputDataNotAllocatedError = std::string("Input data was not allocated. Input name: \'")
                                      + _inputName + "\'";
        _inputDataIsEmptyError = std::string("Input data is empty. Input name: \'")
                                 + _inputName + "\'";
    }

    InferRequest::Ptr getInferRequestWithMockImplInside() {
        IInferRequest::Ptr inferRequest;
        InputsDataMap inputsInfo;
        mockNotEmptyNet.getInputsInfo(inputsInfo);
        OutputsDataMap outputsInfo;
        mockNotEmptyNet.getOutputsInfo(outputsInfo);
        mockInferRequestInternal = make_shared<MockAsyncInferRequestInternal>(inputsInfo, outputsInfo);
        inferRequest = shared_from_irelease(
                new InferRequestBase<MockAsyncInferRequestInternal>(mockInferRequestInternal));
        return make_shared<InferRequest>(inferRequest);
    }

    std::string getExceptionMessage(std::function<void()> function) {
        std::string actualError;
        try {
            function();
        } catch (const InferenceEngineException &iie) {
            actualError = iie.what();
        }
        return actualError;
    }

    BlobMap getBlobMapWithIncorrectName() const {
        Blob::Ptr Blob = make_shared_blob<float>({ Precision::FP32, {1, 1, 1, 1}, NCHW });
        Blob->allocate();
        return BlobMap{{_incorrectName, Blob}};
    }

    BlobMap getBlobMapWithNotAllocatedInput() const {
        Blob::Ptr Blob = make_shared_blob<float>({ Precision::FP32, {1, 1, 1, 1}, NCHW });
        return BlobMap{{_inputName, Blob}};
    }

    BlobMap getBlobMapWithEmptyDimensions() const {
        Blob::Ptr Blob = make_shared_blob<float>({ Precision::FP32, {}, NCHW });
        Blob->allocate();
        return BlobMap{{_inputName, Blob}};
    }
};

// constructor tests
TEST_F(InferRequestTests, constructorsTests) {
    // construction from the non-null should not throw
    ASSERT_NO_THROW(InferRequest req(mock_request));
    IInferRequest::Ptr tmp;
    // InferRequest's "actual" is nullptr, let's check it throws on construction
    ASSERT_THROW(InferRequest req(tmp), InferenceEngineException);
}

// StartAsync
TEST_F(InferRequestTests, canForwardStartAsync) {
    EXPECT_CALL(*mock_request.get(), StartAsync(_)).WillOnce(Return(OK));
    ASSERT_NO_THROW(requestWrapper->StartAsync());
}

TEST_F(InferRequestTests, throwsIfStartAsyncReturnNotOK) {
    EXPECT_CALL(*mock_request.get(), StartAsync(_)).WillOnce(Return(GENERAL_ERROR));
    ASSERT_THROW(requestWrapper->StartAsync(), InferenceEngineException);
}

// Wait
TEST_F(InferRequestTests, canForwardWait) {
    int64_t ms = 0;
    EXPECT_CALL(*mock_request.get(), Wait(ms, _)).WillOnce(Return(OK));
    ASSERT_TRUE(OK == requestWrapper->Wait(ms));
}

TEST_F(InferRequestTests, canForwardStatusFromWait) {
    EXPECT_CALL(*mock_request.get(), Wait(_, _)).WillOnce(Return(RESULT_NOT_READY));
    ASSERT_EQ(requestWrapper->Wait(0), RESULT_NOT_READY);
}

// Infer
TEST_F(InferRequestTests, canForwardInfer) {
    EXPECT_CALL(*mock_request.get(), Infer(_)).WillOnce(Return(OK));
    ASSERT_NO_THROW(requestWrapper->Infer());
}

TEST_F(InferRequestTests, throwsIfInferReturnNotOK) {
    EXPECT_CALL(*mock_request.get(), Infer(_)).WillOnce(Return(GENERAL_ERROR));
    ASSERT_THROW(requestWrapper->Infer(), InferenceEngineException);
}

// GetPerformanceCounts
TEST_F(InferRequestTests, canForwardGetPerformanceCounts) {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> info;
    EXPECT_CALL(*mock_request.get(), GetPerformanceCounts(_, _)).WillOnce(Return(OK));
    ASSERT_NO_THROW(info = requestWrapper->GetPerformanceCounts());
}

TEST_F(InferRequestTests, throwsIfGetPerformanceCountsReturnNotOK) {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> info;
    EXPECT_CALL(*mock_request.get(), GetPerformanceCounts(_, _)).WillOnce(Return(GENERAL_ERROR));
    ASSERT_THROW(info = requestWrapper->GetPerformanceCounts(), InferenceEngineException);
}

MATCHER_P(blob_in_map_pointer_is_same, ref_blob, "") {
    auto a = arg.begin()->second.get();
    return (float *) (arg.begin()->second->buffer()) == (float *) (ref_blob->buffer());
}

// SetInput
TEST_F(InferRequestTests, getInputCallsSetBlob) {
    Blob::Ptr inblob;
    std::string blobName1 = "blob1";
    std::string blobName2 = "blob2";
    BlobMap blobMap{{blobName1, inblob},
                    {blobName2, inblob}};

    EXPECT_CALL(*mock_request.get(), SetBlob(StrEq(blobName1.c_str()), inblob, _)).WillOnce(Return(OK));
    EXPECT_CALL(*mock_request.get(), SetBlob(StrEq(blobName2.c_str()), inblob, _)).WillOnce(Return(OK));
    ASSERT_NO_THROW(requestWrapper->SetInput(blobMap));
}

TEST_F(InferRequestTests, throwsIfSetInputReturnNotOK) {
    EXPECT_CALL(*mock_request.get(), SetBlob(_, _, _)).WillOnce(Return(GENERAL_ERROR));
    BlobMap blobMap{{{}, {}}};
    ASSERT_THROW(requestWrapper->SetInput(blobMap), InferenceEngineException);
}

// SetOutput
TEST_F(InferRequestTests, getOutputCallsSetBlob) {
    Blob::Ptr inblob;
    std::string blobName1 = "blob1";
    std::string blobName2 = "blob2";
    BlobMap blobMap{{blobName1, inblob},
                    {blobName2, inblob}};

    EXPECT_CALL(*mock_request.get(), SetBlob(StrEq(blobName1.c_str()), inblob, _)).WillOnce(Return(OK));
    EXPECT_CALL(*mock_request.get(), SetBlob(StrEq(blobName2.c_str()), inblob, _)).WillOnce(Return(OK));
    ASSERT_NO_THROW(requestWrapper->SetOutput(blobMap));
}

// GetBlob
TEST_F(InferRequestTests, canForwardGetBlob) {
    Blob::Ptr blob = make_shared_blob<float>({ Precision::FP32, {}, NCHW });
    blob->allocate();
    std::string name = "blob1";

    EXPECT_CALL(*mock_request.get(), GetBlob(StrEq(name.c_str()), _, _)).WillOnce(DoAll(SetArgReferee<1>(blob), Return(OK)));
    ASSERT_NO_THROW(requestWrapper->GetBlob(name));
}

TEST_F(InferRequestTests, throwsIfGetBlobReturnNotOK) {
    Blob::Ptr blob;
    std::string name = "blob1";

    EXPECT_CALL(*mock_request.get(), GetBlob(_, _, _)).WillOnce(Return(GENERAL_ERROR));
    ASSERT_THROW(blob = requestWrapper->GetBlob(name), InferenceEngineException);
}

// SetBlob
TEST_F(InferRequestTests, canForwardSetBlob) {
    Blob::Ptr blob;
    std::string name = "blob1";

    EXPECT_CALL(*mock_request.get(), SetBlob(StrEq(name.c_str()), blob, _)).WillOnce(Return(OK));
    ASSERT_NO_THROW(requestWrapper->SetBlob(name, blob));
}

TEST_F(InferRequestTests, throwsIfSetBlobReturnNotOK) {
    Blob::Ptr blob;
    std::string name = "blob1";

    EXPECT_CALL(*mock_request.get(), SetBlob(_, _, _)).WillOnce(Return(GENERAL_ERROR));
    ASSERT_THROW(requestWrapper->SetBlob(name, blob), InferenceEngineException);
}

TEST_F(InferRequestTests, throwsIfSetOutputReturnNotOK) {
    EXPECT_CALL(*mock_request.get(), SetBlob(_, _, _)).WillOnce(Return(GENERAL_ERROR));
    BlobMap blobMap{{{}, {}}};
    ASSERT_THROW(requestWrapper->SetOutput(blobMap), InferenceEngineException);
}

// SetCompletionCallback API
void callme(InferenceEngine::IInferRequest::Ptr p, InferenceEngine::StatusCode) {
    void *data = nullptr;
    p->GetUserData(&data, nullptr);
    ASSERT_NE(nullptr, data);
}

TEST_F(InferRequestTests, canForwardCompletionCallback) {
    void *data = nullptr;
    EXPECT_CALL(*mock_request.get(), SetCompletionCallback(_)).WillOnce(
            DoAll(InvokeArgument<0>(static_pointer_cast<IInferRequest>(mock_request), OK), Return(OK)));
    EXPECT_CALL(*mock_request.get(), GetUserData(_, _)).WillRepeatedly(
            DoAll(Invoke([&](void **pData, ResponseDesc *resp) {
                *pData = data;
            }), Return(OK)));
    EXPECT_CALL(*mock_request.get(), SetUserData(_, _)).WillOnce(DoAll(SaveArg<0>(&data), Return(OK)));
    ASSERT_NO_THROW(requestWrapper->SetCompletionCallback(&callme));
}

TEST_F(InferRequestTests, canForwardAnyCallback) {
    void *data = nullptr;
    EXPECT_CALL(*mock_request.get(), SetCompletionCallback(_)).WillOnce(
            DoAll(InvokeArgument<0>(static_pointer_cast<IInferRequest>(mock_request), OK), Return(OK)));
    EXPECT_CALL(*mock_request.get(), GetUserData(_, _)).WillRepeatedly(
            DoAll(Invoke([&](void **pData, ResponseDesc *resp) {
                *pData = data;
            }), Return(OK)));
    EXPECT_CALL(*mock_request.get(), SetUserData(_, _)).WillOnce(DoAll(SaveArg<0>(&data), Return(OK)));

    ASSERT_NO_THROW(requestWrapper->SetCompletionCallback([&]() {
        // data used to store callback pointer
        ASSERT_NE(data, nullptr);
    }));
}

TEST_F(InferRequestTests, failToSetInputWithInCorrectName) {
    auto InferRequest = getInferRequestWithMockImplInside();
    auto blobMap = getBlobMapWithIncorrectName();
    auto exceptionMessage = getExceptionMessage([&]() { InferRequest->SetInput(blobMap); });
    ASSERT_EQ(_failedToFindInOutError, exceptionMessage.substr(0, _failedToFindInOutError.size()));
}

TEST_F(InferRequestTests, failToSetOutputWithInCorrectName) {
    auto InferRequest = getInferRequestWithMockImplInside();
    auto blobMap = getBlobMapWithIncorrectName();
    auto exceptionMessage = getExceptionMessage([&]() { InferRequest->SetOutput(blobMap); });
    ASSERT_EQ(_failedToFindInOutError, exceptionMessage.substr(0, _failedToFindInOutError.size()));
}

TEST_F(InferRequestTests, failToSetInputWithNotAllocatedInput) {
    auto InferRequest = getInferRequestWithMockImplInside();
    auto blobMap = getBlobMapWithNotAllocatedInput();
    auto exceptionMessage = getExceptionMessage([&]() { InferRequest->SetInput(blobMap); });
    ASSERT_EQ(_inputDataNotAllocatedError, exceptionMessage.substr(0, _inputDataNotAllocatedError.size()));
}

TEST_F(InferRequestTests, failToSetInputWithEmptyDimensions) {
    auto InferRequest = getInferRequestWithMockImplInside();
    auto blobMap = getBlobMapWithEmptyDimensions();
    auto exceptionMessage = getExceptionMessage([&]() { InferRequest->SetInput(blobMap); });
    ASSERT_EQ(_inputDataIsEmptyError, exceptionMessage.substr(0, _inputDataIsEmptyError.size()));
}
