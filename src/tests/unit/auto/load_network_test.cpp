// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <common_test_utils/test_constants.hpp>
#include <ie_core.hpp>
#include <ie_metric_helpers.hpp>
#include <multi-device/multi_device_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>

#include "cpp/ie_plugin.hpp"
#include "mock_common.hpp"
#include "plugin/mock_auto_device_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"

using ::testing::_;
using ::testing::AllOf;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::Eq;
using ::testing::InvokeWithoutArgs;
using ::testing::MatcherCast;
using ::testing::Matches;
using ::testing::Property;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::Throw;
using namespace MockMultiDevice;

TEST(LoadNetworkToDefaultDeviceTest, LoadNetwork) {
    std::string mockEngineName("mock_engine");
    std::string libraryName = CommonTestUtils::pre + mockEngineName + IE_BUILD_POSTFIX + CommonTestUtils::ext;
    std::shared_ptr<void> sharedObjectLoader = ov::util::load_shared_object(libraryName.c_str());
    std::function<void(IInferencePlugin*)> injectProxyEngine(
        reinterpret_cast<void (*)(IInferencePlugin*)>(ov::util::get_symbol(sharedObjectLoader, "InjectProxyEngine")));

    // prepare mockicore and cnnNetwork for loading
    Core ie;
    auto* origin_plugin = new MockMultiDeviceInferencePlugin();
    // replace core with mock Icore
    injectProxyEngine(origin_plugin);
    //ie.RegisterPlugin(std::string("mock_engine") + IE_BUILD_POSTFIX, "AUTO");

    EXPECT_CALL(*origin_plugin, LoadNetworkImpl(_, _, _, _)).Times(AnyNumber());

    InferenceEngine::CNNNetwork actualCnnNetwork;
    std::shared_ptr<ngraph::Function> actualNetwork = ngraph::builder::subgraph::makeSplitConvConcat();
    ASSERT_NO_THROW(actualCnnNetwork = InferenceEngine::CNNNetwork(actualNetwork));
    ie.LoadNetwork(actualCnnNetwork);
    delete origin_plugin;
}