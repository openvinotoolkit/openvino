// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>

#include <ie_version.hpp>
#include <ie_plugin_cpp.hpp>

#include "unit_test_utils/mocks/cpp_interfaces/mock_plugin_impl.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class PluginBaseTests: public ::testing::Test {
protected:
    std::shared_ptr<MockPluginImpl> mock_impl;
    std::shared_ptr<IInferencePlugin> plugin;
    virtual void TearDown() {
    }
    virtual void SetUp() {
        mock_impl.reset(new MockPluginImpl());
        mock_impl->SetVersion({{2, 1}, "test", "version"});
        plugin = std::static_pointer_cast<IInferencePlugin>(mock_impl);
    }
};

TEST_F(PluginBaseTests, canReportVersion) {
    const Version V = plugin->GetVersion();

    EXPECT_STREQ(V.buildNumber, "test");
    EXPECT_STREQ(V.description, "version");
    EXPECT_EQ(V.apiVersion.major, 2);
    EXPECT_EQ(V.apiVersion.minor, 1);
}

TEST_F(PluginBaseTests, canForwardLoadExeNetwork) {
    EXPECT_CALL(*mock_impl.get(), LoadExeNetwork(_, _, _)).Times(1);
    ICNNNetwork * network = nullptr;
    IExecutableNetwork::Ptr exeNetwork = nullptr;
    ASSERT_NO_THROW(plugin->LoadNetwork(exeNetwork, *network, {}));
}

TEST_F(PluginBaseTests, canReportErrorInLoadExeNetwork) {
    EXPECT_CALL(*mock_impl.get(), LoadExeNetwork(_, _, _)).WillOnce(Throw(std::runtime_error("compare")));

    ICNNNetwork * network = nullptr;
    IExecutableNetwork::Ptr exeNetwork = nullptr;
    ASSERT_THROW(plugin->LoadNetwork(exeNetwork, *network, {}), details::InferenceEngineException);
}

TEST_F(PluginBaseTests, canCatchUnknownErrorInLoadExeNetwork) {
    EXPECT_CALL(*mock_impl.get(), LoadExeNetwork(_, _, _)).WillOnce(Throw(5));
    ICNNNetwork * network = nullptr;
    IExecutableNetwork::Ptr exeNetwork = nullptr;
    ASSERT_THROW(plugin->LoadNetwork(exeNetwork, *network, {}), details::InferenceEngineException);
}

TEST_F(PluginBaseTests, canForwardSetConfig) {
    const std::map <std::string, std::string> config;
    EXPECT_CALL(*mock_impl.get(), SetConfig(Ref(config))).Times(1);
    ASSERT_NO_THROW(plugin->SetConfig(config));
}

TEST_F(PluginBaseTests, canReportErrorInSetConfig) {
    const std::map <std::string, std::string> config;
    EXPECT_CALL(*mock_impl.get(), SetConfig(_)).WillOnce(Throw(std::runtime_error("error")));

    ASSERT_THROW(plugin->SetConfig(config), details::InferenceEngineException);
}

TEST_F(PluginBaseTests, canCatchUnknownErrorInSetConfig) {
    EXPECT_CALL(*mock_impl.get(), SetConfig(_)).WillOnce(Throw(5));
    const std::map <std::string, std::string> config;
    ASSERT_THROW(plugin->SetConfig(config), details::InferenceEngineException);
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
