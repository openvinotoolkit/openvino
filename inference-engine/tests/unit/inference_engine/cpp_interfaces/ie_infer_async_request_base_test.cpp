// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cpp/ie_infer_request.hpp>
#include <cpp/ie_executable_network.hpp>
#include <cpp/ie_plugin.hpp>
#include <cpp/ie_infer_async_request_base.hpp>

#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/mock_not_empty_icnn_network.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinfer_request_internal.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

constexpr const char* MockNotEmptyICNNNetwork::INPUT_BLOB_NAME;
constexpr const char* MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME;

IE_SUPPRESS_DEPRECATED_START

class InferRequestBaseTests : public ::testing::Test {
protected:
    std::shared_ptr<MockIInferRequestInternal> mock_impl;
    shared_ptr<IInferRequest> request;
    ResponseDesc dsc;

    void SetUp() override {
        mock_impl.reset(new MockIInferRequestInternal());
        request = std::make_shared<InferRequestBase>(mock_impl);
    }
};

// StartAsync
TEST_F(InferRequestBaseTests, canForwardStartAsync) {
    EXPECT_CALL(*mock_impl.get(), StartAsync()).Times(1);
    ASSERT_EQ(OK, request->StartAsync(&dsc));
}

TEST_F(InferRequestBaseTests, canReportErrorInStartAsync) {
    EXPECT_CALL(*mock_impl.get(), StartAsync()).WillOnce(Throw(std::runtime_error("compare")));
    ASSERT_NE(request->StartAsync(&dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(InferRequestBaseTests, canCatchUnknownErrorInStartAsync) {
    EXPECT_CALL(*mock_impl.get(), StartAsync()).WillOnce(Throw(5));
    ASSERT_EQ(UNEXPECTED, request->StartAsync(nullptr));
}

// Wait
TEST_F(InferRequestBaseTests, canForwardWait) {
    int64_t ms = 0;
    EXPECT_CALL(*mock_impl.get(), Wait(ms)).WillOnce(Return(StatusCode::OK));
    ASSERT_EQ(OK, request->Wait(ms, &dsc)) << dsc.msg;
}

TEST_F(InferRequestBaseTests, canReportErrorInWait) {
    EXPECT_CALL(*mock_impl.get(), Wait(_)).WillOnce(Throw(std::runtime_error("compare")));
    int64_t ms = 0;
    ASSERT_NE(request->Wait(ms, &dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(InferRequestBaseTests, canCatchUnknownErrorInWait) {
    EXPECT_CALL(*mock_impl.get(), Wait(_)).WillOnce(Throw(5));
    int64_t ms = 0;
    ASSERT_EQ(UNEXPECTED, request->Wait(ms, nullptr));
}

// Infer
TEST_F(InferRequestBaseTests, canForwardInfer) {
    EXPECT_CALL(*mock_impl.get(), Infer()).Times(1);
    ASSERT_EQ(OK, request->Infer(&dsc));
}

TEST_F(InferRequestBaseTests, canReportErrorInInfer) {
    EXPECT_CALL(*mock_impl.get(), Infer()).WillOnce(Throw(std::runtime_error("compare")));
    ASSERT_NE(request->Infer(&dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(InferRequestBaseTests, canCatchUnknownErrorInInfer) {
    EXPECT_CALL(*mock_impl.get(), Infer()).WillOnce(Throw(5));
    ASSERT_EQ(UNEXPECTED, request->Infer(nullptr));
}

// GetPerformanceCounts
TEST_F(InferRequestBaseTests, canForwardGetPerformanceCounts) {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> info;
    EXPECT_CALL(*mock_impl.get(), GetPerformanceCounts()).WillOnce(Return(std::map<std::string, InferenceEngineProfileInfo>{}));
    ASSERT_EQ(OK, request->GetPerformanceCounts(info, &dsc)) << dsc.msg;
}

TEST_F(InferRequestBaseTests, canReportErrorInGetPerformanceCounts) {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> info;
    EXPECT_CALL(*mock_impl.get(), GetPerformanceCounts()).WillOnce(Throw(std::runtime_error("compare")));
    ASSERT_NE(request->GetPerformanceCounts(info, &dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(InferRequestBaseTests, canCatchUnknownErrorInGetPerformanceCounts) {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> info;
    EXPECT_CALL(*mock_impl.get(), GetPerformanceCounts()).WillOnce(Throw(5));
    ASSERT_EQ(UNEXPECTED, request->GetPerformanceCounts(info, nullptr));
}

// GetBlob
TEST_F(InferRequestBaseTests, canForwardGetBlob) {
    Blob::Ptr data;
    EXPECT_CALL(*mock_impl.get(), GetBlob(_)).WillOnce(Return(Blob::Ptr{}));
    ASSERT_EQ(OK, request->GetBlob("", data, &dsc)) << dsc.msg;
}

TEST_F(InferRequestBaseTests, canReportErrorInGetBlob) {
    EXPECT_CALL(*mock_impl.get(), GetBlob(_)).WillOnce(Throw(std::runtime_error("compare")));
    Blob::Ptr data;
    ASSERT_NE(request->GetBlob("", data, &dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(InferRequestBaseTests, canCatchUnknownErrorInGetBlob) {
    Blob::Ptr data;
    EXPECT_CALL(*mock_impl.get(), GetBlob(_)).WillOnce(Throw(5));
    ASSERT_EQ(UNEXPECTED, request->GetBlob("notEmpty", data, nullptr));
}

// SetBlob
TEST_F(InferRequestBaseTests, canForwardSetBlob) {
    Blob::Ptr data;
    const char *name = "";
    EXPECT_CALL(*mock_impl.get(), SetBlob(name, Ref(data))).Times(1);
    ASSERT_EQ(OK, request->SetBlob(name, data, &dsc));
}

TEST_F(InferRequestBaseTests, canReportErrorInSetBlob) {
    EXPECT_CALL(*mock_impl.get(), SetBlob(_, _)).WillOnce(Throw(std::runtime_error("compare")));
    Blob::Ptr data;
    ASSERT_NE(request->SetBlob("", data, &dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(InferRequestBaseTests, canCatchUnknownErrorInSetBlob) {
    Blob::Ptr data;
    EXPECT_CALL(*mock_impl.get(), SetBlob(_, _)).WillOnce(Throw(5));
    ASSERT_EQ(UNEXPECTED, request->SetBlob("notEmpty", data, nullptr));
}

// SetCompletionCallback
TEST_F(InferRequestBaseTests, canForwardSetCompletionCallback) {
    InferenceEngine::IInferRequest::CompletionCallback callback = nullptr;
    EXPECT_CALL(*mock_impl.get(), SetCallback(_)).Times(1);
    ASSERT_NO_THROW(request->SetCompletionCallback(callback));
}

TEST_F(InferRequestBaseTests, canReportErrorInSetCompletionCallback) {
    EXPECT_CALL(*mock_impl.get(), SetCallback(_)).WillOnce(Throw(std::runtime_error("compare")));
    ASSERT_NE(request->SetCompletionCallback(nullptr), OK);
}


class InferRequestTests : public ::testing::Test {
protected:
    std::shared_ptr<MockIInferRequestInternal> mock_request;
    IInferRequestInternal::Ptr request;
    ResponseDesc dsc;

    std::shared_ptr<MockIExecutableNetworkInternal> mockIExeNet;
    InferenceEngine::SoExecutableNetworkInternal    exeNetwork;
    MockIInferencePlugin*                           mockIPlugin;
    InferencePlugin                                 plugin;
    shared_ptr<MockIInferRequestInternal> mockInferRequestInternal;
    MockNotEmptyICNNNetwork mockNotEmptyNet;
    std::string _incorrectName;
    std::string _inputName;
    std::string _failedToFindInOutError;
    std::string _inputDataNotAllocatedError;
    std::string _inputDataIsEmptyError;

    void TearDown() override {
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockInferRequestInternal.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mock_request.get()));
        request = {};
    }

    void SetUp() override {
        mock_request = make_shared<MockIInferRequestInternal>();
        mockIExeNet = std::make_shared<MockIExecutableNetworkInternal>();
        ON_CALL(*mockIExeNet, CreateInferRequest()).WillByDefault(Return(mock_request));
        auto mockIPluginPtr = std::make_shared<MockIInferencePlugin>();
        ON_CALL(*mockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).WillByDefault(Return(mockIExeNet));
        plugin = InferenceEngine::InferencePlugin{{}, mockIPluginPtr};
        exeNetwork = plugin.LoadNetwork(CNNNetwork{}, {});
        request = exeNetwork->CreateInferRequest();
        _incorrectName = "incorrect_name";
        _inputName = MockNotEmptyICNNNetwork::INPUT_BLOB_NAME;
        _failedToFindInOutError =
                "[ NOT_FOUND ] Failed to find input or output with name: \'" + _incorrectName + "\'";
        _inputDataNotAllocatedError = std::string("[ NOT_ALLOCATED ] Input data was not allocated. Input name: \'")
                                      + _inputName + "\'";
        _inputDataIsEmptyError = std::string("Input data is empty. Input name: \'")
                                 + _inputName + "\'";
    }

    IInferRequestInternal::Ptr getInferRequestWithMockImplInside() {
        IInferRequest::Ptr inferRequest;
        InputsDataMap inputsInfo;
        mockNotEmptyNet.getInputsInfo(inputsInfo);
        OutputsDataMap outputsInfo;
        mockNotEmptyNet.getOutputsInfo(outputsInfo);
        mockInferRequestInternal = make_shared<MockIInferRequestInternal>(inputsInfo, outputsInfo);
        auto mockIExeNet = std::make_shared<MockIExecutableNetworkInternal>();
        ON_CALL(*mockIExeNet, CreateInferRequest()).WillByDefault(Return(mockInferRequestInternal));
        auto mockIPluginPtr = std::make_shared<MockIInferencePlugin>();
        ON_CALL(*mockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).WillByDefault(Return(mockIExeNet));
        auto plugin = InferenceEngine::InferencePlugin{{}, mockIPluginPtr};
        auto exeNetwork = plugin.LoadNetwork(CNNNetwork{}, {});
        return exeNetwork->CreateInferRequest();
    }

    std::string getExceptionMessage(std::function<void()> function) {
        std::string actualError;
        try {
            function();
        } catch (const Exception &iie) {
            actualError = iie.what();
        }
        return actualError;
    }
};


// StartAsync
TEST_F(InferRequestTests, canForwardStartAsync) {
    EXPECT_CALL(*mock_request.get(), StartAsync());
    ASSERT_NO_THROW(request->StartAsync());
}

TEST_F(InferRequestTests, throwsIfStartAsyncReturnNotOK) {
    EXPECT_CALL(*mock_request.get(), StartAsync()).WillOnce(Throw(GeneralError{""}));
    ASSERT_THROW(request->StartAsync(), Exception);
}

// Wait
TEST_F(InferRequestTests, canForwardWait) {
    int64_t ms = 0;
    EXPECT_CALL(*mock_request.get(), Wait(_)).WillOnce(Return(OK));
    ASSERT_TRUE(OK == request->Wait(ms));
}

TEST_F(InferRequestTests, canForwardStatusFromWait) {
    EXPECT_CALL(*mock_request.get(), Wait(_)).WillOnce(Return(RESULT_NOT_READY));
    ASSERT_EQ(request->Wait(0), RESULT_NOT_READY);
}

// Infer
TEST_F(InferRequestTests, canForwardInfer) {
    EXPECT_CALL(*mock_request.get(), Infer());
    ASSERT_NO_THROW(request->Infer());
}

TEST_F(InferRequestTests, throwsIfInferReturnNotOK) {
    EXPECT_CALL(*mock_request.get(), Infer()).WillOnce(Throw(GeneralError{""}));
    ASSERT_THROW(request->Infer(), Exception);
}

// GetPerformanceCounts
TEST_F(InferRequestTests, canForwardGetPerformanceCounts) {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> info;
    EXPECT_CALL(*mock_request.get(), GetPerformanceCounts()).WillOnce(Return(info));
    ASSERT_NO_THROW(info = request->GetPerformanceCounts());
}

TEST_F(InferRequestTests, throwsIfGetPerformanceCountsReturnNotOK) {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> info;
    EXPECT_CALL(*mock_request.get(), GetPerformanceCounts()).WillOnce(Throw(GeneralError{""}));
    ASSERT_THROW(info = request->GetPerformanceCounts(), Exception);
}

MATCHER_P(blob_in_map_pointer_is_same, ref_blob, "") {
    auto a = arg.begin()->second.get();
    return reinterpret_cast<float*>(arg.begin()->second->buffer()) == reinterpret_cast<float*>(ref_blob->buffer());
}

// GetBlob
TEST_F(InferRequestTests, canForwardGetBlob) {
    Blob::Ptr blob = make_shared_blob<float>({ Precision::FP32, {}, NCHW });
    blob->allocate();
    std::string name = "blob1";

    EXPECT_CALL(*mock_request.get(), GetBlob(_)).WillOnce(Return(blob));
    ASSERT_NO_THROW(request->GetBlob(name));
}

TEST_F(InferRequestTests, throwsIfGetBlobReturnNotOK) {
    Blob::Ptr blob;
    std::string name = "blob1";

    EXPECT_CALL(*mock_request.get(), GetBlob(_)).WillOnce(Throw(GeneralError{""}));
    ASSERT_THROW(blob = request->GetBlob(name), Exception);
}

// SetBlob
TEST_F(InferRequestTests, canForwardSetBlob) {
    Blob::Ptr blob;
    std::string name = "blob1";

    EXPECT_CALL(*mock_request.get(), SetBlob(name, blob));
    ASSERT_NO_THROW(request->SetBlob(name, blob));
}

TEST_F(InferRequestTests, throwsIfSetBlobReturnNotOK) {
    Blob::Ptr blob;
    std::string name = "blob1";

    EXPECT_CALL(*mock_request.get(), SetBlob(_, _)).WillOnce(Throw(GeneralError{""}));
    ASSERT_THROW(request->SetBlob(name, blob), Exception);
}

TEST_F(InferRequestTests, canForwardAnyCallback) {
    EXPECT_CALL(*mock_request.get(), SetCallback(_));
    ASSERT_NO_THROW(request->SetCallback([] (std::exception_ptr) {}));
}
