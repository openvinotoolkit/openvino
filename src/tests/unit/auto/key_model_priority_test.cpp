// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <common_test_utils/test_constants.hpp>
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/core.hpp"
#include <ie_core.hpp>
#include <multi-device/multi_device_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "plugin/mock_auto_device_plugin.hpp"
#include "cpp/ie_plugin.hpp"
#include "mock_common.hpp"

using namespace MockMultiDevice;
using ConfigParams = std::tuple<
        std::map<std::string, std::string>,                     // key-value
        std::string                                             // exptected model priority in AUTO
        >;
class KeyModelPriorityTest: public ::testing::TestWithParam<ConfigParams> {
public:
    ov::Core                       core;
    std::shared_ptr<MultiDeviceInferencePlugin>     plugin;
    std::shared_ptr<ngraph::Function>               actualNetwork;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::map<std::string, std::string> inputPriority;
        std::string expectedPriority;
        std::tie(inputPriority, expectedPriority) = obj.param;
        auto it = inputPriority.begin();
        std::ostringstream result;
        result <<  "_input_model_priority_" << it->second;
        result <<  "_expect_return_" << expectedPriority;
        return result.str();
    }

    void TearDown() override {
        plugin.reset();
    }

    void SetUp() override {
       // prepare mockicore and cnnNetwork for loading
       auto* origin_plugin = new MultiDeviceInferencePlugin();
       plugin  = std::shared_ptr<MultiDeviceInferencePlugin>(origin_plugin);
       // Generic network
       actualNetwork = ngraph::builder::subgraph::makeSplitConvConcat();
    }
};

TEST_P(KeyModelPriorityTest, ModelPriorityTest) {
    // get Parameter
    std::map<std::string, std::string> inputPriority;
    std::string expectedPriority;
    std::tie(inputPriority, expectedPriority) = this->GetParam();
    ASSERT_NO_THROW(plugin->SetConfig(inputPriority));
    std::map<std::string, InferenceEngine::Parameter> opt;
    std::string priority;
    ASSERT_NO_THROW(priority = plugin->GetConfig(MultiDeviceConfigParams::KEY_AUTO_NETWORK_PRIORITY, opt).as<std::string>());
    EXPECT_EQ(priority, expectedPriority);
}

// ConfigParams details
const std::vector<ConfigParams> testConfigs = {
                                               ConfigParams {{{"MODEL_PRIORITY", "LOW"}}, "2"},
                                               ConfigParams {{{"MODEL_PRIORITY", "MEDIUM"}}, "1"},
                                               ConfigParams {{{"MODEL_PRIORITY", "HIGH"}}, "0"}
                                              };


INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, KeyModelPriorityTest,
                ::testing::ValuesIn(testConfigs),
            KeyModelPriorityTest::getTestCaseName);

