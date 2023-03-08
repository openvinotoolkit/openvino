// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <common_test_utils/common_utils.hpp>
#include <common_test_utils/test_constants.hpp>
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/mock_task_executor.hpp"
#include <ie_core.hpp>
#include <multi-device/multi_device_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "mock_common.hpp"
#include "plugin/mock_auto_device_plugin.hpp"

using ::testing::HasSubstr;
using ::testing::_;
using ::testing::StrNe;
using ::testing::Return;
using ::testing::InvokeWithoutArgs;
using ::testing::MatcherCast;
using ::testing::StrEq;
using ::testing::NiceMock;

using Config = std::map<std::string, std::string>;

const std::vector<std::string>  availableDevs = {"CPU", "GPU", "VPUX"};
using DynamicOutputConfigParams = std::tuple<
        bool,                     // is newAPI or not
        ov::Any,                  // priority device list
        ov::AnyMap,               // hint setting
        ov::Any                   // expected device to run inference on
        >;
class DynamicShapeTest {
public:
    static unsigned int                                                 target_request_num;
    std::shared_ptr<ngraph::Function>                                   function;
    InferenceEngine::CNNNetwork                                         cnnNet;
    std::shared_ptr<NiceMock<MockICore>>                                core;
    std::shared_ptr<IInferencePlugin>                                   plugin; // real auto plugin used

    //mock hardware exeNetwork
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>>           mockIExeNet;
    ov::SoPtr<IExecutableNetworkInternal>                               mockExeNetwork;

    // config for Auto device
    std::vector<DeviceInformation>                                      metaDevices;
    std::vector<std::shared_ptr<MockIInferRequestInternal>>             inferReqInternal;
    std::vector<std::shared_ptr<AsyncInferRequestThreadSafeDefault>>    asyncInferRequest;
    ImmediateExecutor::Ptr taskExecutor;

public:
    ~DynamicShapeTest() {
        core.reset();
        plugin.reset();
        mockIExeNet.reset();
        mockExeNetwork = {};

        metaDevices.clear();
        inferReqInternal.clear();
        asyncInferRequest.clear();
    }

    DynamicShapeTest() {
        // prepare cpuMockExeNetwork
        mockIExeNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        mockExeNetwork =  {mockIExeNet, {}};

        // prepare mockicore and cnnNetwork for loading
        core = std::make_shared<NiceMock<MockICore>>();
        auto* origin_plugin = new MultiDeviceInferencePlugin();
        plugin  = std::shared_ptr<MultiDeviceInferencePlugin>(origin_plugin);
        function = getFunctionDynamicOutput();
        cnnNet = InferenceEngine::CNNNetwork(function);
        // mock execNetwork can work
        IE_SET_METRIC(SUPPORTED_METRICS, metrics, {METRIC_KEY(SUPPORTED_CONFIG_KEYS), METRIC_KEY(FULL_DEVICE_NAME)});
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_METRICS)), _))
            .WillByDefault(RETURN_MOCK_VALUE(metrics));
        ON_CALL(*core, GetMetric(_,
                    StrEq(METRIC_KEY(FULL_DEVICE_NAME)), _)).WillByDefault(Return(""));
        ON_CALL(*core, GetConfig(_,
                    StrEq(CONFIG_KEY(DEVICE_ID)))).WillByDefault(Return(0));
        IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, supportConfigs, {});
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _))
            .WillByDefault(RETURN_MOCK_VALUE(supportConfigs));
        ON_CALL(*core, GetAvailableDevices()).WillByDefault(Return(availableDevs));
        IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, optimalNum, 1);
        ON_CALL(*mockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(Return(optimalNum));
        std::vector<std::string> cability{"FP32", "FP16", "INT8", "BIN"};
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _))
            .WillByDefault(Return(cability));
        ON_CALL(*core, GetConfig(_, StrEq(ov::compilation_num_threads.name())))
           .WillByDefault(Return(12));
        ON_CALL(*mockIExeNet.get(), GetConfig(StrEq(ov::inference_num_threads.name())))
           .WillByDefault(Return(12));
        ON_CALL(*mockIExeNet.get(), GetConfig(StrEq(ov::num_streams.name())))
            .WillByDefault(Return(4));
        ON_CALL(*mockIExeNet.get(), GetConfig(StrEq(ov::compilation_num_threads.name())))
           .WillByDefault(Return(12));
        // test auto plugin
        plugin->SetName("MULTI"); // repace with auto when multi integrated into auto
        taskExecutor = std::make_shared<ImmediateExecutor>();
        // set up mock infer request
        for (size_t i = 0; i < target_request_num; i++) {
            auto inferReq = std::make_shared<NiceMock<MockIInferRequestInternal>>();
            auto asyncRequest = std::make_shared<AsyncInferRequestThreadSafeDefault>(inferReq, taskExecutor, taskExecutor);
            inferReqInternal.push_back(inferReq);
            asyncInferRequest.push_back(asyncRequest);
        }
        EXPECT_CALL(*mockIExeNet.get(), CreateInferRequest()).WillOnce(Return(asyncInferRequest[0]))
                                                              .WillOnce(Return(asyncInferRequest[1]))
                                                              .WillRepeatedly(Return(asyncInferRequest[0]));
    }

protected:
    // constructing dynamic output model
    std::shared_ptr<ngraph::Function> getFunctionDynamicOutput() {
        auto boxes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});
        boxes->set_friendly_name("param_1");
        boxes->get_output_tensor(0).set_names({"input_tensor_1"});
        auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 2});
        scores->set_friendly_name("param_2");
        scores->get_output_tensor(0).set_names({"input_tensor_2"});
        auto max_output_boxes_per_class = ov::op::v0::Constant::create(ov::element::i64,  ov::Shape{}, {10});
        auto iou_threshold = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.75});
        auto score_threshold = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.7});
        auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                                                    iou_threshold, score_threshold);
        auto res = std::make_shared<ov::op::v0::Result>(nms);
        res->set_friendly_name("output_dynamic");
        auto func = std::make_shared<ngraph::Function>(ov::NodeVector{nms}, ngraph::ParameterVector{boxes, scores});
        return func;
    }
};
unsigned int DynamicShapeTest::target_request_num = 2;
class DynamicOutputTest :   public DynamicShapeTest,
                            public ::testing::TestWithParam<DynamicOutputConfigParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DynamicOutputConfigParams> obj) {
            bool isNewAPI;
            ov::Any priorityList;
            ov::AnyMap property;
            ov::Any targetList;
            std::tie(isNewAPI, priorityList, property, targetList) = obj.param;
            std::ostringstream result;
            result << "_isNewAPI_" << isNewAPI;
            result << "_withList_" << priorityList.as<std::string>();
            for (auto& iter : property)
                result << "_hint_" << iter.first << "_as_" << iter.second.as<std::string>();
            result << "_expect_";
            auto targets = targetList.as<std::vector<std::string>>();
            for (auto& iter : targets)
                result << "_" << iter;
            auto string = result.str();
            CommonTestUtils::replaceSubstringInString(string, ",", "_");
            return string;
        }
    void SetUp() override {
        std::tie(isNewAPI, priorityList, property, targetList) = GetParam();
        if (isNewAPI) {
            ON_CALL(*core.get(), isNewAPI()).WillByDefault(Return(true));
        } else {
            ON_CALL(*core.get(), isNewAPI()).WillByDefault(Return(false));
        }
        // replace core with mock Icore
        plugin->SetCore(core);
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(HasSubstr(CommonTestUtils::DEVICE_GPU)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        return mockExeNetwork; }));
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
            ::testing::Matcher<const std::string&>(HasSubstr(CommonTestUtils::DEVICE_CPU)),
            ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));
    }
    void TearDown() override {
    }

protected:
        bool isNewAPI;
        ov::Any priorityList;
        ov::AnyMap property;
        ov::Any targetList;
};

TEST_P(DynamicOutputTest, CanSelectCorrectTargetDeviceandInitizeBlobWithCorrectSize) {
    auto targets = targetList.as<std::vector<std::string>>();
    std::map<std::string, std::string> config;
    for (auto& iter : property)
        config.insert({iter.first, iter.second.as<std::string>()});
    config.insert({ov::device::priorities.name(), priorityList.as<std::string>()});
    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> exeNetwork;
    for (auto& iter : targets) {
        EXPECT_CALL(
                *core,
                LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                            ::testing::Matcher<const std::string&>(HasSubstr(iter)),
                            ::testing::Matcher<const Config&>(_)))
                .Times(1);
    }
    ASSERT_NO_THROW(exeNetwork = plugin->LoadNetwork(cnnNet, config));
    std::shared_ptr<InferenceEngine::IInferRequestInternal> auto_request;
    ASSERT_NO_THROW(auto_request = exeNetwork->CreateInferRequest());
    for (auto & iter : exeNetwork->GetOutputsInfo()) {
        auto outBlob = auto_request->GetBlob(iter.first);
        ASSERT_NE(outBlob->size(), 0);
    }
    ASSERT_NO_THROW(auto_request->StartAsync());
}

const std::vector<DynamicOutputConfigParams> testConfigs = {
    DynamicOutputConfigParams {false, "CPU,GPU", {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)},
                    std::vector<std::string>{"CPU", "GPU"}},
    DynamicOutputConfigParams {true, "CPU,GPU", {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)},
                    std::vector<std::string>{"CPU", "GPU"}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, DynamicOutputTest,
                ::testing::ValuesIn(testConfigs),
            DynamicOutputTest::getTestCaseName);