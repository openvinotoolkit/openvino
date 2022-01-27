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

using ::testing::MatcherCast;
using ::testing::AllOf;
using ::testing::Throw;
using ::testing::Matches;
using ::testing::_;
using ::testing::StrEq;
using ::testing::Return;
using ::testing::Property;
using ::testing::Eq;
using ::testing::ReturnRef;
using ::testing::AtLeast;
using ::testing::InvokeWithoutArgs;
using Config = std::map<std::string, std::string>;
using namespace MockMultiDevice;

using PriorityParams = std::tuple<unsigned int, std::string>; //{priority, deviceUniquName}

using ConfigParams = std::tuple<
        ov::hint::ModelPriority,                        // input model priority from IE
        std::string                                             // exptected model priority in AUTO
        >;
class KeyModelPriorityTest: public ::testing::TestWithParam<ConfigParams> {
public:
    ov::Core                       core;
    std::shared_ptr<MultiDeviceInferencePlugin>     plugin;
    std::shared_ptr<ngraph::Function>               actualNetwork;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        ov::hint::ModelPriority inputPriority;
        std::string expectedPriority;
        std::tie(inputPriority, expectedPriority) = obj.param;
        std::ostringstream result;
        result <<  "_input_priority_" << inputPriority;
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
    ov::hint::ModelPriority inputPriority;
    std::string expectedPriority;
    std::tie(inputPriority, expectedPriority) = this->GetParam();
    ASSERT_NO_THROW(core.compile_model(actualNetwork, CommonTestUtils::DEVICE_AUTO, ov::hint::model_priority(inputPriority)));
    const std::map<std::string, InferenceEngine::Parameter> opt;
    auto priority = plugin->GetConfig(MultiDeviceConfigParams::KEY_AUTO_NETWORK_PRIORITY, opt);
    EXPECT_EQ(priority, expectedPriority);
}

// ConfigParams details

const std::vector<ConfigParams> testConfigs = {
                                               ConfigParams {ov::hint::ModelPriority::LOW, "2"},
                                               ConfigParams {ov::hint::ModelPriority::MEDIUM, "1"},
                                               ConfigParams {ov::hint::ModelPriority::HIGH, "0"}
                                              };


INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, KeyModelPriorityTest,
                ::testing::ValuesIn(testConfigs),
            KeyModelPriorityTest::getTestCaseName);

