// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <ie_version.hpp>
#include "cpp_interfaces/mock_plugin_impl.hpp"
#include "cpp_interfaces/base/ie_plugin_base.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class PluginBaseTests: public ::testing::Test {
 protected:
    std::shared_ptr<MockPluginImpl> mock_impl;
    shared_ptr<IInferencePlugin> plugin;
    ResponseDesc dsc;
    virtual void TearDown() {
    }
    virtual void SetUp() {
        mock_impl.reset(new MockPluginImpl());
        plugin = details::shared_from_irelease(make_ie_compatible_plugin({1,2,"test", "version"}, mock_impl));
    }
};

TEST_F(PluginBaseTests, canReportVersion) {
    const Version *V;
    plugin->GetVersion(V);

    EXPECT_STREQ(V->buildNumber, "test");
    EXPECT_STREQ(V->description, "version");
    EXPECT_EQ(V->apiVersion.major, 1);
    EXPECT_EQ(V->apiVersion.minor, 2);

}

TEST_F(PluginBaseTests, canForwardLoadNetwork) {

    EXPECT_CALL(*mock_impl.get(), LoadNetwork(_)).Times(1);

    ICNNNetwork * network = nullptr;
    ASSERT_EQ(OK, plugin->LoadNetwork(*network, &dsc));
}


TEST_F(PluginBaseTests, canReportErrorInLoadNetwork) {

    EXPECT_CALL(*mock_impl.get(), LoadNetwork(_)).WillOnce(Throw(std::runtime_error("compare")));

    ICNNNetwork * network = nullptr;
    ASSERT_NE(plugin->LoadNetwork(*network, &dsc), OK);

    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(PluginBaseTests, canCatchUnknownErrorInLoadNetwork) {

    EXPECT_CALL(*mock_impl.get(), LoadNetwork(_)).WillOnce(Throw(5));
    ICNNNetwork * network = nullptr;
    ASSERT_EQ(UNEXPECTED, plugin->LoadNetwork(*network, nullptr));
}

TEST_F(PluginBaseTests, canForwardLoadExeNetwork) {

    EXPECT_CALL(*mock_impl.get(), LoadExeNetwork(_,_,_)).Times(1);

    ICNNNetwork * network = nullptr;
    IExecutableNetwork::Ptr exeNetwork = nullptr;
    ASSERT_EQ(OK, plugin->LoadNetwork(exeNetwork, *network, {}, &dsc));
}


TEST_F(PluginBaseTests, canReportErrorInLoadExeNetwork) {

    EXPECT_CALL(*mock_impl.get(), LoadExeNetwork(_,_,_)).WillOnce(Throw(std::runtime_error("compare")));

    ICNNNetwork * network = nullptr;
    IExecutableNetwork::Ptr exeNetwork = nullptr;
    ASSERT_NE(plugin->LoadNetwork(exeNetwork, *network, {}, &dsc), OK);

    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(PluginBaseTests, canCatchUnknownErrorInLoadExeNetwork) {

    EXPECT_CALL(*mock_impl.get(), LoadExeNetwork(_,_,_)).WillOnce(Throw(5));
    ICNNNetwork * network = nullptr;
    IExecutableNetwork::Ptr exeNetwork = nullptr;
    ASSERT_EQ(UNEXPECTED, plugin->LoadNetwork(exeNetwork, *network, {}, nullptr));
}

TEST_F(PluginBaseTests, canForwarInfer) {

    TBlob<float>  input(Precision::FP32, NCHW);
    TBlob<float>  result(Precision::FP32, NCHW);


    EXPECT_CALL(*mock_impl.get(), Infer(Ref(input), Ref(result))).Times(1);

    ASSERT_EQ(OK, plugin->Infer(input, result, &dsc));
}

TEST_F(PluginBaseTests, canReportErrorInInfer) {

    EXPECT_CALL(*mock_impl.get(), Infer(_,_)).WillOnce(Throw(std::runtime_error("error")));

    Blob * input = nullptr;
    ASSERT_NE(plugin->Infer(*input, *input, &dsc), OK);

    ASSERT_STREQ(dsc.msg, "error");
}

TEST_F(PluginBaseTests, canCatchUnknownErrorInInfer) {
    EXPECT_CALL(*mock_impl.get(), Infer(_,_)).WillOnce(Throw(5));
    Blob * input = nullptr;
    ASSERT_EQ(UNEXPECTED, plugin->Infer(*input, *input, nullptr));
}

TEST_F(PluginBaseTests, canForwarBlobMapInfer) {
    BlobMap  input;
    BlobMap  result;

    EXPECT_CALL(*mock_impl.get(), InferBlobMap(Ref(input), Ref(result))).Times(1);

    ASSERT_EQ(OK, plugin->Infer(input, result, &dsc));
}

TEST_F(PluginBaseTests, canReportErrorInBlobMapInfer) {

    EXPECT_CALL(*mock_impl.get(), InferBlobMap(_,_)).WillOnce(Throw(std::runtime_error("error")));

    BlobMap * input = nullptr;
    ASSERT_NE(plugin->Infer(*input, *input, &dsc), OK);

    ASSERT_STREQ(dsc.msg, "error");
}

TEST_F(PluginBaseTests, canCatchUnknownErrorInBlobMapInfer) {
    EXPECT_CALL(*mock_impl.get(), InferBlobMap(_,_)).WillOnce(Throw(5));
    BlobMap * input = nullptr;
    ASSERT_EQ(UNEXPECTED, plugin->Infer(*input, *input, nullptr));
}

TEST_F(PluginBaseTests, canForwarGetPerformanceCounts) {

    std::map <std::string, InferenceEngineProfileInfo> profileInfo;

    EXPECT_CALL(*mock_impl.get(), GetPerformanceCounts(Ref(profileInfo))).Times(1);

    ASSERT_EQ(OK, plugin->GetPerformanceCounts(profileInfo, &dsc));
}


TEST_F(PluginBaseTests, canReportErrorInGetPerformanceCounts) {

    std::map <std::string, InferenceEngineProfileInfo> profileInfo;

    EXPECT_CALL(*mock_impl.get(), GetPerformanceCounts(_)).WillOnce(Throw(std::runtime_error("error")));

    ASSERT_NE(OK, plugin->GetPerformanceCounts(profileInfo, &dsc));

    ASSERT_STREQ(dsc.msg, "error");
}

TEST_F(PluginBaseTests, canCatchUnknownErrorInGetPerformanceCounts) {
    EXPECT_CALL(*mock_impl.get(), GetPerformanceCounts(_)).WillOnce(Throw(5));
    std::map <std::string, InferenceEngineProfileInfo> profileInfo;
    ASSERT_EQ(UNEXPECTED, plugin->GetPerformanceCounts(profileInfo, nullptr));
}

TEST_F(PluginBaseTests, canForwarSetConfig) {

    const std::map <std::string, std::string> config;
    EXPECT_CALL(*mock_impl.get(), SetConfig(Ref(config))).Times(1);
    ASSERT_EQ(OK, plugin->SetConfig(config, &dsc));
}

TEST_F(PluginBaseTests, canReportErrorInSetConfig) {
    const std::map <std::string, std::string> config;
    EXPECT_CALL(*mock_impl.get(), SetConfig(_)).WillOnce(Throw(std::runtime_error("error")));

    ASSERT_NE(OK, plugin->SetConfig(config, &dsc));
    ASSERT_STREQ(dsc.msg, "error");
}

TEST_F(PluginBaseTests, canCatchUnknownErrorInSetConfig) {
    EXPECT_CALL(*mock_impl.get(), SetConfig(_)).WillOnce(Throw(5));
    const std::map <std::string, std::string> config;
    ASSERT_EQ(UNEXPECTED, plugin->SetConfig(config, nullptr));
}
