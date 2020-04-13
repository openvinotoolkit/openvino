// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <ie_version.hpp>
#include "cpp_interfaces/base/ie_plugin_base.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/mock_plugin_impl.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class PluginBaseTests: public ::testing::Test {
 protected:
    std::shared_ptr<MockPluginImpl> mock_impl;
    IE_SUPPRESS_DEPRECATED_START
    shared_ptr<IInferencePlugin> plugin;
    IE_SUPPRESS_DEPRECATED_END
    ResponseDesc dsc;
    virtual void TearDown() {
    }
    virtual void SetUp() {
        mock_impl.reset(new MockPluginImpl());
        plugin = details::shared_from_irelease(make_ie_compatible_plugin({{2, 1}, "test", "version"}, mock_impl));
    }
};

TEST_F(PluginBaseTests, canReportVersion) {
    const Version *V;
    plugin->GetVersion(V);

    EXPECT_STREQ(V->buildNumber, "test");
    EXPECT_STREQ(V->description, "version");
    EXPECT_EQ(V->apiVersion.major, 2);
    EXPECT_EQ(V->apiVersion.minor, 1);

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
