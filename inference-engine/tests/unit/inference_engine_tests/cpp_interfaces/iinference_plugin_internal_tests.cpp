// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <ie_version.hpp>
#include <inference_engine/cnn_network_impl.hpp>
#include <cpp_interfaces/base/ie_plugin_base.hpp>

#include <mock_icnn_network.hpp>
#include <mock_iexecutable_network.hpp>
#include <mock_not_empty_icnn_network.hpp>
#include <cpp_interfaces/mock_plugin_impl.hpp>
#include <cpp_interfaces/impl/mock_inference_plugin_internal.hpp>
#include <cpp_interfaces/impl/mock_executable_thread_safe_default.hpp>
#include <cpp_interfaces/interface/mock_iinfer_request_internal.hpp>
#include <mock_iasync_infer_request.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class InferenceEnginePluginInternalTest : public ::testing::Test {
protected:
    shared_ptr<IInferencePlugin> plugin;
    shared_ptr<MockInferencePluginInternal> mock_plugin_impl;
    shared_ptr<MockExecutableNetworkInternal> mockExeNetworkInternal;
    shared_ptr<MockExecutableNetworkThreadSafe> mockExeNetworkTS;
    shared_ptr<MockInferRequestInternal> mockInferRequestInternal;
    MockNotEmptyICNNNetwork mockNotEmptyNet;

    ResponseDesc dsc;
    StatusCode sts;

    virtual void TearDown() {
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mock_plugin_impl.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockExeNetworkInternal.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockExeNetworkTS.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockInferRequestInternal.get()));
    }

    virtual void SetUp() {
        mock_plugin_impl.reset(new MockInferencePluginInternal());
        plugin = details::shared_from_irelease(make_ie_compatible_plugin({1, 2, "test", "version"}, mock_plugin_impl));
        mockExeNetworkInternal = make_shared<MockExecutableNetworkInternal>();
    }

    void getInferRequestWithMockImplInside(IInferRequest::Ptr &request) {
        IExecutableNetwork::Ptr exeNetwork;
        InputsDataMap inputsInfo;
        mockNotEmptyNet.getInputsInfo(inputsInfo);
        OutputsDataMap outputsInfo;
        mockNotEmptyNet.getOutputsInfo(outputsInfo);
        mockInferRequestInternal = make_shared<MockInferRequestInternal>(inputsInfo, outputsInfo);
        mockExeNetworkTS = make_shared<MockExecutableNetworkThreadSafe>();
        EXPECT_CALL(*mock_plugin_impl.get(), LoadExeNetworkImpl(_, _)).WillOnce(Return(mockExeNetworkTS));
        EXPECT_CALL(*mockExeNetworkTS.get(), CreateInferRequestImpl(_, _)).WillOnce(Return(mockInferRequestInternal));
        sts = plugin->LoadNetwork(exeNetwork, mockNotEmptyNet, {}, &dsc);
        ASSERT_EQ((int) StatusCode::OK, sts) << dsc.msg;
        ASSERT_NE(exeNetwork, nullptr) << dsc.msg;
        sts = exeNetwork->CreateInferRequest(request, &dsc);
        ASSERT_EQ((int) StatusCode::OK, sts) << dsc.msg;
    }
};

MATCHER_P(blob_in_map_pointer_is_same, ref_blob, "") {
    auto a = arg.begin()->second.get();
    return (float *) (arg.begin()->second->buffer()) == (float *) (ref_blob->buffer());
}

TEST_F(InferenceEnginePluginInternalTest, canUseNewInferViaOldAPI) {
    shared_ptr<Blob> inblob(new TBlob<float>(Precision::FP32, NCHW));
    shared_ptr<Blob> resblob(new TBlob<float>(Precision::FP32, NCHW));

    inblob->Resize({1}, Layout::C);
    resblob->Resize({1}, Layout::C);

    inblob->allocate();
    resblob->allocate();

    BlobMap blbi;
    blbi[""] = inblob;
    BlobMap blbo;
    blbo[""] = resblob;
    EXPECT_CALL(*mock_plugin_impl.get(), Infer(Matcher<const BlobMap &>(blob_in_map_pointer_is_same(inblob)),
                                               Matcher<BlobMap &>(blob_in_map_pointer_is_same(resblob)))).Times(1);

    EXPECT_NO_THROW(plugin->Infer((Blob &) *inblob.get(), (Blob &) *resblob.get(), nullptr));
}

TEST_F(InferenceEnginePluginInternalTest, loadExeNetworkCallsSetNetworkIO) {
    IExecutableNetwork::Ptr exeNetwork;
    map<string, string> config;
    EXPECT_CALL(*mockExeNetworkInternal.get(), setNetworkInputs(_)).Times(1);
    EXPECT_CALL(*mockExeNetworkInternal.get(), setNetworkOutputs(_)).Times(1);
    EXPECT_CALL(*mock_plugin_impl.get(), LoadExeNetworkImpl(Ref(mockNotEmptyNet), Ref(config))).WillOnce(
            Return(mockExeNetworkInternal));
    EXPECT_NO_THROW(plugin->LoadNetwork(exeNetwork, mockNotEmptyNet, config, nullptr));
}

TEST_F(InferenceEnginePluginInternalTest, failToSetBlobWithInCorrectName) {
    Blob::Ptr inBlob = make_shared_blob<float>(Precision::FP32, NCHW, {});
    inBlob->allocate();
    string inputName = "not_input";
    std::string refError = NOT_FOUND_str + "Failed to find input or output with name: \'" + inputName + "\'";
    IInferRequest::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);

    ASSERT_NO_THROW(sts = inferRequest->SetBlob(inputName.c_str(), inBlob, &dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    dsc.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, dsc.msg);
}

TEST_F(InferenceEnginePluginInternalTest, failToSetBlobWithNullPtr) {
    Blob::Ptr inBlob = make_shared_blob<float>(Precision::FP32, NCHW, {});
    inBlob->allocate();
    string inputName = "not_input";
    std::string refError = NOT_FOUND_str + "Failed to set blob with empty name";
    IInferRequest::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);

    ASSERT_NO_THROW(sts = inferRequest->SetBlob(nullptr, inBlob, &dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    dsc.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, dsc.msg);
}

TEST_F(InferenceEnginePluginInternalTest, failToSetNullPtr) {
    string inputName = MockNotEmptyICNNNetwork::INPUT_BLOB_NAME;
    std::string refError = NOT_ALLOCATED_str + "Failed to set empty blob with name: \'" + inputName + "\'";
    IInferRequest::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);
    Blob::Ptr inBlob = nullptr;

    ASSERT_NO_THROW(sts = inferRequest->SetBlob(inputName.c_str(), inBlob, &dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    dsc.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, dsc.msg);
}

TEST_F(InferenceEnginePluginInternalTest, failToSetEmptyBlob) {
    Blob::Ptr inBlob;
    string inputName = MockNotEmptyICNNNetwork::INPUT_BLOB_NAME;
    std::string refError = NOT_ALLOCATED_str + "Failed to set empty blob with name: \'" + inputName + "\'";
    IInferRequest::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);

    ASSERT_NO_THROW(sts = inferRequest->SetBlob(inputName.c_str(), inBlob, &dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    dsc.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, dsc.msg);
}

TEST_F(InferenceEnginePluginInternalTest, failToSetNotAllocatedBlob) {
    string inputName = MockNotEmptyICNNNetwork::INPUT_BLOB_NAME;
    std::string refError = "Input data was not allocated. Input name: \'" + inputName + "\'";
    IInferRequest::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);
    Blob::Ptr blob = make_shared_blob<float>(Precision::FP32, NCHW, {});

    ASSERT_NO_THROW(sts = inferRequest->SetBlob(inputName.c_str(), blob, &dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    dsc.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, dsc.msg);
}

class InferenceEnginePluginInternal2Test : public ::testing::Test {
protected:
    shared_ptr<IInferencePlugin> plugin;
    shared_ptr<MockInferencePluginInternal2> mockPluginImpl;
    shared_ptr<MockIExecutableNetwork> mockExeNetwork;
    shared_ptr<MockIInferRequest> mockIInferRequest;
    MockICNNNetwork mockEmptyNet;
    MockNotEmptyICNNNetwork mockNotEmptyNet;

    ResponseDesc dsc;
    StatusCode sts;

    virtual void TearDown() {}

    virtual void SetUp() {
        mockPluginImpl = make_shared<MockInferencePluginInternal2>();
        plugin = details::shared_from_irelease(make_ie_compatible_plugin({1, 2, "test", "version"}, mockPluginImpl));
        mockExeNetwork = make_shared<MockIExecutableNetwork>();
    }

    shared_ptr<MockIInferRequest> getMockIInferRequestPtr() {
        auto mockRequest = make_shared<MockIInferRequest>();
        EXPECT_CALL(*mockPluginImpl.get(), LoadNetwork(_, _, _)).WillOnce(SetArgReferee<0>(mockExeNetwork));
        EXPECT_CALL(*mockExeNetwork.get(), CreateInferRequest(_, _)).WillOnce(DoAll(SetArgReferee<0>(mockRequest),
                                                                                    Return(StatusCode::OK)));
        plugin->LoadNetwork(mockNotEmptyNet, nullptr);
        return mockRequest;
    }
};

TEST_F(InferenceEnginePluginInternal2Test, loadExeNetworkWithEmptyNetworkReturnsError) {
    string refError = "The network doesn't have inputs/outputs";
    EXPECT_CALL(mockEmptyNet, getInputsInfo(_)).Times(1);
    EXPECT_CALL(mockEmptyNet, getOutputsInfo(_)).Times(1);
    EXPECT_NO_THROW(sts = plugin->LoadNetwork(mockEmptyNet, &dsc));
    ASSERT_EQ(GENERAL_ERROR, sts);
    dsc.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, dsc.msg);
}

TEST_F(InferenceEnginePluginInternal2Test, canForwardGetPerfCount) {
    mockIInferRequest = getMockIInferRequestPtr();
    map<string, InferenceEngineProfileInfo> profileInfo;
    EXPECT_CALL(*mockIInferRequest.get(), GetPerformanceCounts(Ref(profileInfo), _)).WillOnce(Return(StatusCode::OK));
    ASSERT_EQ(OK, plugin->GetPerformanceCounts(profileInfo, &dsc)) << dsc.msg;
}

TEST_F(InferenceEnginePluginInternal2Test, deprecatedInferCallsSetterAndInfer) {
    mockIInferRequest = getMockIInferRequestPtr();

    Blob::Ptr inBlob, resBlob;
    BlobMap inBlobMap, resBlobMap;
    inBlobMap[MockNotEmptyICNNNetwork::INPUT_BLOB_NAME] = inBlob;
    resBlobMap[MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME] = resBlob;

    EXPECT_CALL(*mockIInferRequest.get(), SetBlob(StrEq(MockNotEmptyICNNNetwork::INPUT_BLOB_NAME), inBlob, _)).WillOnce(
            Return(StatusCode::OK));
    EXPECT_CALL(*mockIInferRequest.get(),
                SetBlob(StrEq(MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME), resBlob, _)).WillOnce(Return(StatusCode::OK));
    EXPECT_CALL(*mockIInferRequest.get(), Infer(_)).WillOnce(Return(StatusCode::OK));

    ASSERT_EQ(OK, plugin->Infer(inBlobMap, resBlobMap, &dsc)) << dsc.msg;
}

TEST_F(InferenceEnginePluginInternal2Test, deprecatedLoadNetworkCallsCreateInferRequest) {
    EXPECT_CALL(*mockPluginImpl.get(), LoadNetwork(_, _, _)).WillOnce(SetArgReferee<0>(mockExeNetwork));
    EXPECT_CALL(*mockExeNetwork.get(), CreateInferRequest(_, _)).WillOnce(Return(StatusCode::OK));
    ASSERT_EQ(OK, plugin->LoadNetwork(mockNotEmptyNet, &dsc)) << dsc.msg;
}
