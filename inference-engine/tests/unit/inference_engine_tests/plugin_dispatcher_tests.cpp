// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_common.hpp"
#include "mock_plugin_dispatcher.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "ie_plugin_dispatcher.hpp"
#include "ie_plugin_ptr.hpp"
#include "ie_device.hpp"

using namespace InferenceEngine;
using namespace ::testing;

class PluginDispatcherTests : public ::testing::Test {
public:
    const std::string nameExt(const std::string& name) { return name + IE_BUILD_POSTFIX;}
};

TEST_F(PluginDispatcherTests, canLoadMockPlugin) {
    PluginDispatcher dispatcher({ "", "./", "./lib" });
    ASSERT_NO_THROW(dispatcher.getPluginByName(nameExt("mock_engine")));
}

TEST_F(PluginDispatcherTests, throwsOnUnknownPlugin) {
    PluginDispatcher dispatcher({ "./", "./lib" });
    ASSERT_THROW(dispatcher.getPluginByName(nameExt("unknown_plugin")), InferenceEngine::details::InferenceEngineException);
}

TEST_F(PluginDispatcherTests, throwsOnDeviceWithoutPlugins) {
    PluginDispatcher dispatcher({ "./", "./lib" });
    ASSERT_THROW(dispatcher.getSuitablePlugin(TargetDevice::eBalanced),
                                                    InferenceEngine::details::InferenceEngineException);
}

ACTION(ThrowException)
{
    THROW_IE_EXCEPTION << "Exception!";
}

TEST_F(PluginDispatcherTests, triesToLoadEveryPluginSuitableForDevice) {
    MockDispatcher disp({ "./", "./lib" });

    ON_CALL(disp, getPluginByName(_)).WillByDefault(ThrowException());
#ifdef ENABLE_MKL_DNN
    EXPECT_CALL(disp, getPluginByName(nameExt("MKLDNNPlugin"))).Times(1);
#endif
#ifdef ENABLE_OPENVX_CVE
    EXPECT_CALL(disp, getPluginByName(nameExt("OpenVXPluginCVE"))).Times(1);
#elif defined ENABLE_OPENVX
    EXPECT_CALL(disp, getPluginByName(nameExt("OpenVXPlugin"))).Times(1);
#endif
    ASSERT_THROW(disp.getSuitablePlugin(TargetDevice::eCPU), InferenceEngine::details::InferenceEngineException);
}

#if defined(ENABLE_OPENVX) || defined(ENABLE_MKL_DNN) || defined(ENABLE_OPENVX_CVE)
TEST_F(PluginDispatcherTests, returnsIfLoadSuccessfull) {
    MockDispatcher disp({ "./", "./lib" });
    PluginDispatcher dispatcher({ "", "./", "./lib" });
    auto ptr = dispatcher.getPluginByName(nameExt("mock_engine"));

    EXPECT_CALL(disp, getPluginByName(_)).WillOnce(Return(ptr));
    ASSERT_NO_THROW(disp.getSuitablePlugin(TargetDevice::eCPU));
}
#endif
