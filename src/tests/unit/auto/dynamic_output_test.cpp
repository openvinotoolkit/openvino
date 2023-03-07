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
#include <ie_core.hpp>
#include <multi-device/multi_device_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "plugin/mock_auto_device_plugin.hpp"
#include "mock_common.hpp"

using ::testing::MatcherCast;
using ::testing::HasSubstr;
using ::testing::AllOf;
using ::testing::Throw;
using ::testing::Matches;
using ::testing::_;
using ::testing::StrEq;
using ::testing::StrNe;
using ::testing::Return;
using ::testing::Property;
using ::testing::Eq;
using ::testing::AnyNumber;
using ::testing::ReturnRef;
using ::testing::AtLeast;
using ::testing::InvokeWithoutArgs;
using ::testing::NiceMock;

using Config = std::map<std::string, std::string>;
// const char cpuFullDeviceName[] = "Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz";
const char igpuFullDeviceName[] = "Intel(R) Gen9 HD Graphics (iGPU)";
const char dgpuFullDeviceName[] = "Intel(R) Iris(R) Xe MAX Graphics (dGPU)";
// const char myriadFullDeviceName[] = "Intel Movidius Myriad X VPU";
// const char vpuxFullDeviceName[] = "";
const std::vector<std::string>  availableDevs = {"CPU", "GPU.0", "GPU.1", "VPUX"};
const std::vector<std::string>  availableDevsNoID = {"CPU", "GPU", "VPUX"};
using DynamicOutputConfigParams = std::tuple<
        bool,                     // is newAPI or not
        ov::Any,                  // priority device list
        ov::AnyMap,               // hint setting
        ov::Any                  // expected device to run inference on
        >;
class DynamicShapeTest {
public:
    std::shared_ptr<ngraph::Function>               function;
    InferenceEngine::CNNNetwork                     cnnNet;
    std::shared_ptr<NiceMock<MockICore>>            core;
    std::shared_ptr<MultiDeviceInferencePlugin>     plugin; // real auto plugin used

    //mock cpu exeNetwork
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>> cpuMockIExeNet;
    ov::SoPtr<IExecutableNetworkInternal>  cpuMockExeNetwork;

    //mock accelerator exeNetwork
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>> accMockIExeNet;
    ov::SoPtr<IExecutableNetworkInternal>  accMockExeNetwork;

    // config for Auto device
    std::map<std::string, std::string>              config;
    std::vector<DeviceInformation>                  metaDevices;
    std::shared_ptr<MockIInferRequestInternal>      inferReqInternal;

public:
    ~DynamicShapeTest() {
        core.reset();
        plugin.reset();
        cpuMockIExeNet.reset();
        cpuMockExeNetwork = {};
        accMockIExeNet.reset();
        accMockExeNetwork = {};
        config.clear();
        metaDevices.clear();
        inferReqInternal.reset();
    }

    DynamicShapeTest() {
        // prepare cpuMockExeNetwork
        cpuMockIExeNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        cpuMockExeNetwork =  {cpuMockIExeNet, {}};

        // prepare acceleratorMockExeNetwork
        accMockIExeNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        accMockExeNetwork = {accMockIExeNet, {}};

        // prepare mockicore and cnnNetwork for loading
        core = std::make_shared<NiceMock<MockICore>>();
        auto* origin_plugin = new MultiDeviceInferencePlugin();
        plugin  = std::shared_ptr<MultiDeviceInferencePlugin>(origin_plugin);
        function = getFunctionDynamicOutput();
        cnnNet = InferenceEngine::CNNNetwork(function);
        // mock execNetwork can work
        inferReqInternal = std::make_shared<NiceMock<MockIInferRequestInternal>>();
        ON_CALL(*cpuMockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));
        ON_CALL(*accMockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));

        //EXPECT_CALL(*inferReqInternal, SetCallback).Times(AtLeast(1));
        IE_SET_METRIC(SUPPORTED_METRICS, metrics, {METRIC_KEY(SUPPORTED_CONFIG_KEYS), METRIC_KEY(FULL_DEVICE_NAME)});
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_METRICS)), _))
            .WillByDefault(RETURN_MOCK_VALUE(metrics));
        ON_CALL(*core, GetMetric(StrEq("GPU"),
                    StrEq(METRIC_KEY(FULL_DEVICE_NAME)), _)).WillByDefault(Return(igpuFullDeviceName));
        ON_CALL(*core, GetConfig(StrEq("GPU"),
                    StrEq(CONFIG_KEY(DEVICE_ID)))).WillByDefault(Return(0));
        ON_CALL(*core, GetMetric(StrEq("GPU.0"),
                    StrEq(METRIC_KEY(FULL_DEVICE_NAME)), _)).WillByDefault(Return(igpuFullDeviceName));
        ON_CALL(*core, GetMetric(StrEq("GPU.1"),
                    StrEq(METRIC_KEY(FULL_DEVICE_NAME)), _)).WillByDefault(Return(dgpuFullDeviceName));
        IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, supportConfigs, {});
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _))
            .WillByDefault(RETURN_MOCK_VALUE(supportConfigs));
        ON_CALL(*core, GetAvailableDevices()).WillByDefault(Return(availableDevs));
        IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, optimalNum, 2);
        ON_CALL(*cpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(Return(optimalNum));
        ON_CALL(*accMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(Return(optimalNum));
        std::vector<std::string> cpuCability{"FP32", "FP16", "INT8", "BIN"};
        std::vector<std::string> gpuCability{"FP32", "FP16", "BATCHED_BLOB", "BIN", "INT8"};
        ON_CALL(*core, GetMetric(StrEq(CommonTestUtils::DEVICE_CPU), StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _))
            .WillByDefault(Return(cpuCability));
        ON_CALL(*core, GetMetric(HasSubstr(CommonTestUtils::DEVICE_GPU), StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _))
            .WillByDefault(Return(gpuCability));
        // test auto plugin
        plugin->SetName("AUTO");
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
        res->get_output_tensor(0).set_names({"output_dynamic"});
        auto func = std::make_shared<ngraph::Function>(ov::NodeVector{nms}, ngraph::ParameterVector{boxes, scores});
        return func;
    }
};

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
            result << "_expect_" << targetList.as<std::string>();
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
                    ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        return accMockExeNetwork; }));
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
            ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
            ::testing::Matcher<const Config&>(_))).WillByDefault(Return(cpuMockExeNetwork));
    }
    void TearDown() override {
    }

protected:
        bool isNewAPI;
        ov::Any priorityList;
        ov::AnyMap property;
        ov::Any targetList;
};

TEST_P(DynamicOutputTest, CanSelectCorrectTargetDevice) {
    std::map<std::string, std::string> config;
    for (auto& iter : property)
        config.insert({iter.first, iter.second.as<std::string>()});
    config.insert({ov::device::priorities.name(), priorityList.as<std::string>()});
    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> exeNetwork;
    EXPECT_CALL(
            *core,
            LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                        ::testing::Matcher<const std::string&>(HasSubstr(targetList.as<std::string>())),
                        ::testing::Matcher<const Config&>(_)))
            .Times(1);
    ASSERT_NO_THROW(exeNetwork = plugin->LoadExeNetworkImpl(cnnNet, config));
}

const std::vector<DynamicOutputConfigParams> testConfigs = {
    DynamicOutputConfigParams {true, "CPU,GPU", {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)},
                    "CPU,GPU"},
    DynamicOutputConfigParams {true, "GPU.0,GPU.1,CPU", {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)},
                    "GPU.0,GPU.1,CPU"},
    DynamicOutputConfigParams {false, "CPU,GPU", {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)},
                    "CPU,GPU"}
};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, DynamicOutputTest,
                ::testing::ValuesIn(testConfigs),
            DynamicOutputTest::getTestCaseName);