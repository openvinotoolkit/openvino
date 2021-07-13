// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>

#include <ie_version.hpp>
#include <cpp/ie_plugin.hpp>

#include <cpp/ie_infer_async_request_base.hpp>
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>

#include "unit_test_utils/mocks/mock_not_empty_icnn_network.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_executable_thread_safe_default.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinfer_request_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class InferenceEnginePluginInternalTest : public ::testing::Test {
protected:
    shared_ptr<IInferencePlugin> plugin;
    shared_ptr<MockInferencePluginInternal> mock_plugin_impl;
    shared_ptr<MockIExecutableNetworkInternal> mockIExeNetworkInternal;
    shared_ptr<MockExecutableNetworkThreadSafe> mockExeNetworkTS;
    shared_ptr<MockIInferRequestInternal> mockInferRequestInternal;
    std::shared_ptr<MockNotEmptyICNNNetwork> mockNotEmptyNet = std::make_shared<MockNotEmptyICNNNetwork>();
    std::string pluginId;

    ResponseDesc dsc;
    StatusCode sts;

    void TearDown() override {
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mock_plugin_impl.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockIExeNetworkInternal.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockExeNetworkTS.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockInferRequestInternal.get()));
    }

    void SetUp() override {
        pluginId = "TEST";
        mock_plugin_impl.reset(new MockInferencePluginInternal());
        mock_plugin_impl->SetName(pluginId);
        plugin = std::static_pointer_cast<IInferencePlugin>(mock_plugin_impl);
        mockIExeNetworkInternal = make_shared<MockIExecutableNetworkInternal>();
        mockIExeNetworkInternal->SetPointerToPlugin(mock_plugin_impl);
    }

    void getInferRequestWithMockImplInside(IInferRequestInternal::Ptr &request) {
        IExecutableNetworkInternal::Ptr exeNetwork;
        InputsDataMap inputsInfo;
        mockNotEmptyNet->getInputsInfo(inputsInfo);
        OutputsDataMap outputsInfo;
        mockNotEmptyNet->getOutputsInfo(outputsInfo);
        mockInferRequestInternal = make_shared<MockIInferRequestInternal>(inputsInfo, outputsInfo);
        mockExeNetworkTS = make_shared<MockExecutableNetworkThreadSafe>();
        EXPECT_CALL(*mock_plugin_impl.get(), LoadExeNetworkImpl(_, _)).WillOnce(Return(mockExeNetworkTS));
        EXPECT_CALL(*mockExeNetworkTS.get(), CreateInferRequestImpl(_, _)).WillOnce(Return(mockInferRequestInternal));
        IE_SUPPRESS_DEPRECATED_START
        ASSERT_NO_THROW(exeNetwork = plugin->LoadNetwork(InferenceEngine::CNNNetwork(mockNotEmptyNet), {}));
        IE_SUPPRESS_DEPRECATED_END
        ASSERT_NO_THROW(request = exeNetwork->CreateInferRequest());
    }
};

MATCHER_P(blob_in_map_pointer_is_same, ref_blob, "") {
    return reinterpret_cast<float*>(arg.begin()->second->buffer()) == reinterpret_cast<float*>(ref_blob->buffer());
}

TEST_F(InferenceEnginePluginInternalTest, failToSetBlobWithInCorrectName) {
    Blob::Ptr inBlob = make_shared_blob<float>({ Precision::FP32, {1, 1, 1, 1}, NCHW });
    inBlob->allocate();
    string inputName = "not_input";
    std::string refError = "[ NOT_FOUND ] Failed to find input or output with name: \'" + inputName + "\'";
    IInferRequestInternal::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);
    try {
        inferRequest->SetBlob(inputName, inBlob);
    } catch(InferenceEngine::NotFound& ex) {
        ASSERT_TRUE(std::string{ex.what()}.find(refError) != std::string::npos)
            << "\tExpected: " << refError
            << "\n\tActual: " << ex.what();
    }
}

TEST_F(InferenceEnginePluginInternalTest, failToSetBlobWithEmptyName) {
    Blob::Ptr inBlob = make_shared_blob<float>({ Precision::FP32, {}, NCHW });
    inBlob->allocate();
    string inputName = "not_input";
    std::string refError = "[ NOT_FOUND ] Failed to set blob with empty name";
    IInferRequestInternal::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);
    try {
        inferRequest->SetBlob(inputName, inBlob);
    } catch(InferenceEngine::NotFound& ex) {
        ASSERT_TRUE(std::string{ex.what()}.find(refError) != std::string::npos)
            << "\tExpected: " << refError
            << "\n\tActual: " << ex.what();
    }
}

TEST_F(InferenceEnginePluginInternalTest, failToSetNullPtr) {
    string inputName = MockNotEmptyICNNNetwork::INPUT_BLOB_NAME;
    std::string refError = "[ NOT_ALLOCATED ] Failed to set empty blob with name: \'" + inputName + "\'";
    IInferRequestInternal::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);
    Blob::Ptr inBlob = nullptr;
    try {
        inferRequest->SetBlob(inputName, inBlob);
    } catch(InferenceEngine::NotAllocated& ex) {
        ASSERT_TRUE(std::string{ex.what()}.find(refError) != std::string::npos)
            << "\tExpected: " << refError
            << "\n\tActual: " << ex.what();
    }
}

TEST_F(InferenceEnginePluginInternalTest, failToSetEmptyBlob) {
    Blob::Ptr inBlob;
    string inputName = MockNotEmptyICNNNetwork::INPUT_BLOB_NAME;
    std::string refError = "[ NOT_ALLOCATED ] Failed to set empty blob with name: \'" + inputName + "\'";
    IInferRequestInternal::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);
    try {
        inferRequest->SetBlob(inputName, inBlob);
    } catch(InferenceEngine::NotAllocated& ex) {
        ASSERT_TRUE(std::string{ex.what()}.find(refError) != std::string::npos)
            << "\tExpected: " << refError
            << "\n\tActual: " << ex.what();
    }
}

TEST_F(InferenceEnginePluginInternalTest, failToSetNotAllocatedBlob) {
    string inputName = MockNotEmptyICNNNetwork::INPUT_BLOB_NAME;
    std::string refError = "[ NOT_ALLOCATED ] Input data was not allocated. Input name: \'" + inputName + "\'";
    IInferRequestInternal::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);
    Blob::Ptr blob = make_shared_blob<float>({ Precision::FP32, {}, NCHW });
    try {
        inferRequest->SetBlob(inputName, blob);
    } catch(InferenceEngine::NotAllocated& ex) {
        ASSERT_TRUE(std::string{ex.what()}.find(refError) != std::string::npos)
            << "\tExpected: " << refError
            << "\n\tActual: " << ex.what();
    }
}

TEST(InferencePluginTests, throwsOnUninitializedGetVersion) {
    InferencePlugin plg;
    ASSERT_THROW(plg.GetVersion(), Exception);
}

TEST(InferencePluginTests, throwsOnUninitializedLoadNetwork) {
    InferencePlugin plg;
    ASSERT_THROW(plg.LoadNetwork(CNNNetwork(), {}), Exception);
}

TEST(InferencePluginTests, throwsOnUninitializedImportNetwork) {
    InferencePlugin plg;
    ASSERT_THROW(plg.ImportNetwork({}, {}), Exception);
}

TEST(InferencePluginTests, throwsOnUninitializedAddExtension) {
    InferencePlugin plg;
    ASSERT_THROW(plg.AddExtension(IExtensionPtr()), Exception);
}

TEST(InferencePluginTests, throwsOnUninitializedSetConfig) {
    InferencePlugin plg;
    ASSERT_THROW(plg.SetConfig({{}}), Exception);
}
