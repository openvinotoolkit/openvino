// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <common_test_utils/test_constants.hpp>
#include <ie_core.hpp>
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
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
using ::testing::AnyNumber;
using ::testing::AtLeast;
using namespace MockMultiDevice;

TEST(LoadNetworkToDefaultDeviceTest, LoadNetwork) {
    std::string pluginXML{"mock_engine_valid.xml"};
    std::string content{"<ie><plugins><plugin name=\"AUTO\" location=\"libmock_engine.so\"></plugin></plugins></ie>"};
    std::ofstream outfile(pluginXML);
    ASSERT_NO_THROW(outfile << content);
    ASSERT_NO_THROW(outfile.close());
    std::cout<< "File: " << __FILE__ << ", Line: " << __LINE__ <<", Call Function: " << __FUNCTION__ << std::endl;
    Core ie(pluginXML);
    //ov::Core ie_core(pluginXML);
    std::remove(pluginXML.c_str());
    std::string mockEngineName("mock_engine");
    std::string libraryName = CommonTestUtils::pre + mockEngineName + IE_BUILD_POSTFIX + CommonTestUtils::ext;
    std::shared_ptr<void> sharedObjectLoader = ov::util::load_shared_object(libraryName.c_str());
    std::function<void(IInferencePlugin*)> injectProxyEngine(
        reinterpret_cast<void (*)(IInferencePlugin*)>(ov::util::get_symbol(sharedObjectLoader, "InjectProxyEngine")));

    auto* origin_plugin = new MockMultiDeviceLoadNetworkTestPlugin();
    auto plugin  = std::shared_ptr<MockMultiDeviceLoadNetworkTestPlugin>(origin_plugin);
    injectProxyEngine(origin_plugin);

    InferenceEngine::CNNNetwork actualCnnNetwork;
    std::shared_ptr<ngraph::Function> actualNetwork = ngraph::builder::subgraph::makeSplitConvConcat();
    ASSERT_NO_THROW(actualCnnNetwork = InferenceEngine::CNNNetwork(actualNetwork));

    auto mockIExeNet = std::make_shared<MockIExecutableNetworkInternal>();
    EXPECT_CALL(*mockIExeNet, GetOutputsInfo())
        .WillRepeatedly([&]() -> ConstOutputsDataMap {
            ConstOutputsDataMap outputMap;
            for (const auto& output : actualCnnNetwork.getOutputsInfo()) {
                outputMap.emplace(output.first, output.second);
            }
            return outputMap;
        });

    EXPECT_CALL(*mockIExeNet, GetInputsInfo())
        .WillRepeatedly([&]() -> ConstInputsDataMap {
            ConstInputsDataMap inputMap;
            for (const auto& input : actualCnnNetwork.getInputsInfo()) {
                inputMap.emplace(input.first, input.second);
            }
            return inputMap;
        });

    EXPECT_CALL(*plugin, LoadExeNetworkImpl(_, _)).Times(AtLeast(1))
        .WillRepeatedly([&](const InferenceEngine::CNNNetwork&,
                      const std::map<std::string, std::string>&) -> InferenceEngine::IExecutableNetworkInternal::Ptr {
            std::cout<< "File: " << __FILE__ << ", Line: " << __LINE__ <<", Call Function: " <<  __FUNCTION__ << std::endl;
            return mockIExeNet;
        });

    ASSERT_NO_THROW(ie.LoadNetwork(actualCnnNetwork));
    //ASSERT_NO_THROW(ie.LoadNetwork(actualCnnNetwork, CommonTestUtils::DEVICE_AUTO));
}
//
//TEST(CompileModelToDefaultDeviceTest, compileModel) {
//    std::string pluginXML{"mock_engine_valid.xml"};
//    std::string content{"<ie><plugins><plugin name=\"AUTO\" location=\"libmock_engine.so\"></plugin></plugins></ie>"};
//    std::ofstream outfile(pluginXML);
//    outfile << content;
//    outfile.close();
//    ov::Core ie(pluginXML);
//    std::remove(pluginXML.c_str());
//    std::string mockEngineName("mock_engine");
//    std::string libraryName = CommonTestUtils::pre + mockEngineName + IE_BUILD_POSTFIX + CommonTestUtils::ext;
//    std::shared_ptr<void> sharedObjectLoader = ov::util::load_shared_object(libraryName.c_str());
//    std::function<void(IInferencePlugin*)> injectProxyEngine(
//        reinterpret_cast<void (*)(IInferencePlugin*)>(ov::util::get_symbol(sharedObjectLoader, "InjectProxyEngine")));
//
//    auto* origin_plugin = new MockMultiDeviceLoadNetworkTestPlugin();
//    auto plugin  = std::shared_ptr<MockMultiDeviceLoadNetworkTestPlugin>(origin_plugin);
//    injectProxyEngine(origin_plugin);
//
//    InferenceEngine::CNNNetwork actualCnnNetwork;
//    std::shared_ptr<ngraph::Function> actualNetwork = ngraph::builder::subgraph::makeSplitConvConcat();
//    ASSERT_NO_THROW(actualCnnNetwork = InferenceEngine::CNNNetwork(actualNetwork));
//
//    auto mockIExeNet = std::make_shared<MockIExecutableNetworkInternal>();
//    EXPECT_CALL(*mockIExeNet, GetOutputsInfo())
//        .WillRepeatedly([&]() -> ConstOutputsDataMap {
//            ConstOutputsDataMap outputMap;
//            for (const auto& output : actualCnnNetwork.getOutputsInfo()) {
//                outputMap.emplace(output.first, output.second);
//            }
//            return outputMap;
//        });
//
//    EXPECT_CALL(*mockIExeNet, GetInputsInfo())
//        .WillRepeatedly([&]() -> ConstInputsDataMap {
//            ConstInputsDataMap inputMap;
//            for (const auto& input : actualCnnNetwork.getInputsInfo()) {
//                inputMap.emplace(input.first, input.second);
//            }
//            return inputMap;
//        });
//
//    EXPECT_CALL(*plugin, LoadExeNetworkImpl(_, _)).Times(2)
//        .WillRepeatedly([&](const InferenceEngine::CNNNetwork&,
//                      const std::map<std::string, std::string>&) -> InferenceEngine::IExecutableNetworkInternal::Ptr {
//            return mockIExeNet;
//        });
//
//    ASSERT_NO_THROW(ie.compile_model(actualNetwork));
//    ASSERT_NO_THROW(ie.compile_model(actualNetwork, CommonTestUtils::DEVICE_AUTO));
//}