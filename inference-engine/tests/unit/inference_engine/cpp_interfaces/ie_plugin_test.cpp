// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>

#include <ie_version.hpp>
#include <ie_plugin_cpp.hpp>

#include "unit_test_utils/mocks/mock_not_empty_icnn_network.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_executable_thread_safe_default.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinfer_request_internal.hpp"

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
    std::shared_ptr<MockNotEmptyICNNNetwork> mockNotEmptyNet = std::make_shared<MockNotEmptyICNNNetwork>();
    std::string pluginId;

    ResponseDesc dsc;
    StatusCode sts;

    virtual void TearDown() {
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mock_plugin_impl.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockExeNetworkInternal.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockExeNetworkTS.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockInferRequestInternal.get()));
    }

    virtual void SetUp() {
        pluginId = "TEST";
        mock_plugin_impl.reset(new MockInferencePluginInternal());
        mock_plugin_impl->SetName(pluginId);
        plugin = std::static_pointer_cast<IInferencePlugin>(mock_plugin_impl);
        mockExeNetworkInternal = make_shared<MockExecutableNetworkInternal>();
        mockExeNetworkInternal->SetPointerToPlugin(mock_plugin_impl);
    }

    void getInferRequestWithMockImplInside(IInferRequest::Ptr &request) {
        ExecutableNetwork exeNetwork;
        InputsDataMap inputsInfo;
        mockNotEmptyNet->getInputsInfo(inputsInfo);
        OutputsDataMap outputsInfo;
        mockNotEmptyNet->getOutputsInfo(outputsInfo);
        mockInferRequestInternal = make_shared<MockInferRequestInternal>(inputsInfo, outputsInfo);
        mockExeNetworkTS = make_shared<MockExecutableNetworkThreadSafe>();
        EXPECT_CALL(*mock_plugin_impl.get(), LoadExeNetworkImpl(_, _)).WillOnce(Return(mockExeNetworkTS));
        EXPECT_CALL(*mockExeNetworkTS.get(), CreateInferRequestImpl(_, _)).WillOnce(Return(mockInferRequestInternal));
        ASSERT_NO_THROW(exeNetwork = plugin->LoadNetwork(InferenceEngine::CNNNetwork(mockNotEmptyNet), {}));
        ASSERT_NO_THROW(request = exeNetwork.CreateInferRequest());
    }
};

MATCHER_P(blob_in_map_pointer_is_same, ref_blob, "") {
    return reinterpret_cast<float*>(arg.begin()->second->buffer()) == reinterpret_cast<float*>(ref_blob->buffer());
}

TEST_F(InferenceEnginePluginInternalTest, failToSetBlobWithInCorrectName) {
    Blob::Ptr inBlob = make_shared_blob<float>({ Precision::FP32, {1, 1, 1, 1}, NCHW });
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

TEST_F(InferenceEnginePluginInternalTest, failToSetBlobWithEmptyName) {
    Blob::Ptr inBlob = make_shared_blob<float>({ Precision::FP32, {}, NCHW });
    inBlob->allocate();
    string inputName = "not_input";
    std::string refError = NOT_FOUND_str + "Failed to set blob with empty name";
    IInferRequest::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);

    ASSERT_NO_THROW(sts = inferRequest->SetBlob("", inBlob, &dsc));
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
    Blob::Ptr blob = make_shared_blob<float>({ Precision::FP32, {}, NCHW });

    ASSERT_NO_THROW(sts = inferRequest->SetBlob(inputName.c_str(), blob, &dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    dsc.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, dsc.msg);
}

TEST_F(InferenceEnginePluginInternalTest, executableNetworkInternalExportsMagicAndName) {
    std::stringstream strm;
    ASSERT_NO_THROW(mockExeNetworkInternal->WrapOstreamExport(strm));
    ExportMagic actualMagic = {};
    strm.read(actualMagic.data(), actualMagic.size());
    ASSERT_EQ(exportMagic, actualMagic);
    std::string pluginName;
    std::getline(strm, pluginName);
    ASSERT_EQ(pluginId, pluginName);
    std::string exportedString;
    std::getline(strm, exportedString);
    ASSERT_EQ(mockExeNetworkInternal->exportString, exportedString);
}

TEST_F(InferenceEnginePluginInternalTest, pluginInternalEraseMagicAndNameWhenImports) {
    std::stringstream strm;
    ASSERT_NO_THROW(mockExeNetworkInternal->WrapOstreamExport(strm));
    ASSERT_NO_THROW(mock_plugin_impl->ImportNetwork(strm, {}));
    ASSERT_EQ(mockExeNetworkInternal->exportString, mock_plugin_impl->importedString);
    mock_plugin_impl->importedString = {};
}


TEST(InferencePluginTests, throwsOnNullptrCreation) {
    InferenceEnginePluginPtr nulptr;
    InferencePlugin plugin;
    ASSERT_THROW(plugin = InferencePlugin(nulptr), details::InferenceEngineException);
}

TEST(InferencePluginTests, throwsOnUninitializedGetVersion) {
    InferencePlugin plg;
    ASSERT_THROW(plg.GetVersion(), details::InferenceEngineException);
}

TEST(InferencePluginTests, throwsOnUninitializedLoadNetwork) {
    InferencePlugin plg;
    ASSERT_THROW(plg.LoadNetwork(CNNNetwork(), {}), details::InferenceEngineException);
}

TEST(InferencePluginTests, throwsOnUninitializedImportNetwork) {
    InferencePlugin plg;
    ASSERT_THROW(plg.ImportNetwork({}, {}), details::InferenceEngineException);
}

TEST(InferencePluginTests, throwsOnUninitializedAddExtension) {
    InferencePlugin plg;
    ASSERT_THROW(plg.AddExtension(IExtensionPtr()), details::InferenceEngineException);
}

TEST(InferencePluginTests, throwsOnUninitializedSetConfig) {
    InferencePlugin plg;
    ASSERT_THROW(plg.SetConfig({{}}), details::InferenceEngineException);
}

TEST(InferencePluginTests, nothrowsUninitializedCast) {
    InferencePlugin plg;
    ASSERT_NO_THROW(auto plgPtr = static_cast<InferenceEnginePluginPtr>(plg));
}
