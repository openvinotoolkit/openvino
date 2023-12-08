// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/common_utils.hpp>
#include <thread>

#include "include/auto_unit_test.hpp"
#include "openvino/opsets/opset11.hpp"

using StatefulModelConfigParams =
    std::tuple<std::string,                              // device candidate list
               bool,                                     // is dynamic model
               bool,                                     // is stateful model
               std::map<std::string, bool>,              // device and the flag if device supports stateful
               bool,                                     // is cumulative mode
               std::vector<std::pair<std::string, int>>  // expected compiling model times on each device
               >;

class StatefulModelSupportedTest : public tests::AutoTest, public ::testing::TestWithParam<StatefulModelConfigParams> {
public:
    std::shared_ptr<ov::Model> create_dynamic_output_model();
    std::shared_ptr<ov::Model> create_stateful_model();
    static std::string getTestCaseName(testing::TestParamInfo<StatefulModelConfigParams> obj);
    void SetUp() override;

protected:
    bool isDynamicModel;
    bool isStatefulModel;
    std::map<std::string, bool> isDevSupportStatefulMap;
    std::vector<std::pair<std::string, int>> expectedCalledTimes;
    bool isCumulative;
    std::string devicesList;
};

std::string StatefulModelSupportedTest::getTestCaseName(testing::TestParamInfo<StatefulModelConfigParams> obj) {
    bool isDynamicModel;
    bool isStatefulModel;
    std::map<std::string, bool> isDevSupportStatefulMap;
    std::vector<std::pair<std::string, int>> expectedCalledTimes;
    bool isCumulative;
    std::string devicesList;

    std::tie(devicesList, isDynamicModel, isStatefulModel, isDevSupportStatefulMap, isCumulative, expectedCalledTimes) =
        obj.param;
    std::ostringstream result;
    result << "_devicesList_" << devicesList;
    result << "_isDynamic_" << isDynamicModel;
    result << "_isStatefulModel_" << isStatefulModel;
    for (auto& item : isDevSupportStatefulMap) {
        result << "_" << item.first << "_" << item.second;
    }
    result << "_isCumulative_" << isCumulative;
    for (auto& item : expectedCalledTimes) {
        result << "_calling_on_" << item.first << "_expected_times_" << item.second;
    }
    auto string = result.str();
    return string;
}

std::shared_ptr<ov::Model> StatefulModelSupportedTest::create_dynamic_output_model() {
    auto boxes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});
    boxes->set_friendly_name("param_1");
    boxes->get_output_tensor(0).set_names({"input_tensor_1"});
    auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 2});
    scores->set_friendly_name("param_2");
    scores->get_output_tensor(0).set_names({"input_tensor_2"});
    auto max_output_boxes_per_class = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {10});
    auto iou_threshold = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.75});
    auto score_threshold = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.7});
    auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold);
    auto res = std::make_shared<ov::op::v0::Result>(nms);
    res->set_friendly_name("output_dynamic");
    return std::make_shared<ov::Model>(ov::NodeVector{nms}, ov::ParameterVector{boxes, scores});
}

std::shared_ptr<ov::Model> StatefulModelSupportedTest::create_stateful_model() {
    auto arg = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{1, 1});
    auto init_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1}, {0});
    // The ReadValue/Assign operations must be used in pairs in the model.
    // For each such a pair, its own variable object must be created.
    const std::string variable_name("variable0");
    // auto variable = std::make_shared<ov::op::util::Variable>(
    //     ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, variable_name});
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{init_const->get_shape(), ov::element::f32, variable_name});
    // Creating ov::Model
    auto read = std::make_shared<ov::opset11::ReadValue>(init_const, variable);
    std::vector<std::shared_ptr<ov::Node>> args = {arg, read};
    auto add = std::make_shared<ov::opset11::Add>(arg, read);
    add->set_friendly_name("add_sum");
    auto assign = std::make_shared<ov::opset11::Assign>(add, variable);
    assign->set_friendly_name("save");
    auto res = std::make_shared<ov::opset11::Result>(add);
    res->set_friendly_name("res");

    auto model =
        std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), ov::ParameterVector({arg}));

    return model;
}

void StatefulModelSupportedTest::SetUp() {
    std::tie(devicesList, isDynamicModel, isStatefulModel, isDevSupportStatefulMap, isCumulative, expectedCalledTimes) =
        GetParam();
    if (isDynamicModel) {
        model = create_dynamic_output_model();
    } else if (isStatefulModel) {
        model = create_stateful_model();
    }

    std::map<std::string, ov::SupportedOpsMap> devicesSupportedLayers;
    for (auto& item : isDevSupportStatefulMap) {
        ov::SupportedOpsMap res;
        auto deviceName = item.first;
        auto isSupportStateful = item.second;
        std::unordered_set<std::string> device_supported_layers;
        for (auto& op : model->get_ops()) {
            if (!std::dynamic_pointer_cast<ngraph::op::AssignBase>(op) &&
                !std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(op)) {
                res[op->get_friendly_name()] = deviceName;
                continue;
            }
            if (isSupportStateful) {
                res[op->get_friendly_name()] = deviceName;
            }
        }
        devicesSupportedLayers[deviceName] = res;
    }

    for (auto& item : devicesSupportedLayers) {
        ON_CALL(*core,
                query_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                            ::testing::Matcher<const std::string&>(StrEq(item.first)),
                            _))
            .WillByDefault(Return(item.second));
    }

    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                          (_)))
        .WillByDefault(Return(mockExeNetwork));

    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_GPU)),
                          (_)))
        .WillByDefault(Return(mockExeNetworkActual));
    if (isCumulative)
        plugin->set_device_name("MULTI");
    else
        plugin->set_device_name("AUTO");
}

TEST_P(StatefulModelSupportedTest, CanFilterOutCorrectTargetDeviceWithStatefulModel) {
    metaDevices.clear();
    int priority = 0;
    for (auto& item : expectedCalledTimes) {
        auto deviceName = item.first;
        auto times = item.second;
        DeviceInformation devInfo(deviceName, {}, -1, {}, deviceName, priority++);
        metaDevices.push_back(devInfo);
        if (times >= 0) {
            EXPECT_CALL(*core,
                        compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                                      ::testing::Matcher<const std::string&>(StrEq(deviceName)),
                                      ::testing::Matcher<const ov::AnyMap&>(_)))
                .Times(times);
        }
    }
    int expectedTimes = expectedCalledTimes.begin()->second;
    ov::AnyMap config = {};

    if (!devicesList.empty())
        config.insert(ov::device::priorities(devicesList));

    ON_CALL(*plugin, parse_meta_devices(_, _)).WillByDefault(Return(metaDevices));
    ON_CALL(*plugin, get_valid_device)
        .WillByDefault([](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
            std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
            return devices;
        });

    config.insert(ov::intel_auto::enable_runtime_fallback(false));
    if (isCumulative) {
        config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
    }
    if (expectedTimes < 0) {
        ASSERT_THROW(plugin->compile_model(model, config), ov::Exception);
    } else {
        ASSERT_NO_THROW(plugin->compile_model(model, config));
    }
}

const std::vector<StatefulModelConfigParams> testConfigs = {
    // test cases for dynamic model
    StatefulModelConfigParams{
        "CPU",                                                  // device candidate list is CPU
        true,                                                   // model is dynamic model
        true,                                                   // model is stateful model
        std::map<std::string, bool>{{"CPU", true}},             // device CPU supports stateful model
        true,                                                   // performance mode is cumulative mode
        std::vector<std::pair<std::string, int>>{{"CPU", 1}}},  // expected compiling model count is 1 on device CPU
    StatefulModelConfigParams{"CPU",
                              true,
                              false,
                              std::map<std::string, bool>{{"CPU", true}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}}},
    StatefulModelConfigParams{"CPU",
                              true,
                              true,
                              std::map<std::string, bool>{{"CPU", false}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}}},
    StatefulModelConfigParams{"CPU",
                              true,
                              true,
                              std::map<std::string, bool>{{"CPU", true}},
                              false,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}}},
    StatefulModelConfigParams{"CPU",
                              true,
                              true,
                              std::map<std::string, bool>{{"CPU", false}},
                              false,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}}},

    StatefulModelConfigParams{"GPU",
                              true,
                              false,
                              std::map<std::string, bool>{{"GPU", true}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}}},
    StatefulModelConfigParams{"GPU",
                              true,
                              false,
                              std::map<std::string, bool>{{"GPU", false}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}}},
    StatefulModelConfigParams{"GPU",
                              true,
                              false,
                              std::map<std::string, bool>{{"GPU", true}},
                              false,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}}},

    StatefulModelConfigParams{"CPU,GPU",
                              true,
                              true,
                              std::map<std::string, bool>{{"CPU", true}, {"GPU", true}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}, {"GPU", 0}}},
    StatefulModelConfigParams{"GPU,CPU",
                              true,
                              true,
                              std::map<std::string, bool>{{"CPU", true}, {"GPU", true}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 0}, {"CPU", 1}}},
    StatefulModelConfigParams{"CPU,GPU",
                              true,
                              false,
                              std::map<std::string, bool>{{"CPU", true}, {"GPU", true}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}, {"GPU", 0}}},
    StatefulModelConfigParams{"GPU,CPU",
                              true,
                              false,
                              std::map<std::string, bool>{{"CPU", true}, {"GPU", true}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 0}, {"CPU", 1}}},
    StatefulModelConfigParams{"CPU",
                              false,
                              false,
                              std::map<std::string, bool>{{"CPU", true}, {"GPU", true}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}}},
    StatefulModelConfigParams{"CPU",
                              false,
                              false,
                              std::map<std::string, bool>{{"CPU", false}, {"GPU", true}},
                              false,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}}},
    StatefulModelConfigParams{"GPU",
                              false,
                              false,
                              std::map<std::string, bool>{{"CPU", true}, {"GPU", true}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}}},
    StatefulModelConfigParams{"GPU",
                              false,
                              false,
                              std::map<std::string, bool>{{"CPU", false}, {"GPU", true}},
                              false,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}}},
    StatefulModelConfigParams{"CPU,GPU",
                              false,
                              false,
                              std::map<std::string, bool>{{"CPU", false}, {"GPU", true}},
                              false,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}, {"GPU", 0}}},
    StatefulModelConfigParams{"CPU,GPU",
                              false,
                              false,
                              std::map<std::string, bool>{{"CPU", false}, {"GPU", true}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}, {"GPU", 1}}},
    StatefulModelConfigParams{"GPU,CPU",
                              false,
                              false,
                              std::map<std::string, bool>{{"CPU", false}, {"GPU", true}},
                              false,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", 1}}},
    StatefulModelConfigParams{"GPU,CPU",
                              false,
                              false,
                              std::map<std::string, bool>{{"CPU", false}, {"GPU", true}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", 1}}},
    StatefulModelConfigParams{"CPU",
                              false,
                              true,
                              std::map<std::string, bool>{{"CPU", false}, {"GPU", true}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}}},
    StatefulModelConfigParams{"CPU",
                              false,
                              true,
                              std::map<std::string, bool>{{"CPU", false}, {"GPU", true}},
                              false,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}}},
    StatefulModelConfigParams{"CPU",
                              false,
                              true,
                              std::map<std::string, bool>{{"CPU", true}, {"GPU", false}},
                              false,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}}},
    StatefulModelConfigParams{"GPU",
                              false,
                              true,
                              std::map<std::string, bool>{{"CPU", true}, {"GPU", false}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}}},
    StatefulModelConfigParams{"GPU",
                              false,
                              true,
                              std::map<std::string, bool>{{"CPU", false}, {"GPU", true}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}}},
    StatefulModelConfigParams{"GPU",
                              false,
                              true,
                              std::map<std::string, bool>{{"CPU", false}, {"GPU", true}},
                              false,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}}},
    StatefulModelConfigParams{"CPU,GPU",
                              false,
                              true,
                              std::map<std::string, bool>{{"CPU", true}, {"GPU", false}},
                              false,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}, {"GPU", 0}}},
    StatefulModelConfigParams{"CPU,GPU",
                              false,
                              true,
                              std::map<std::string, bool>{{"CPU", false}, {"GPU", true}},
                              false,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", 0}}},
    StatefulModelConfigParams{"CPU,GPU",
                              false,
                              true,
                              std::map<std::string, bool>{{"CPU", true}, {"GPU", true}},
                              false,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", 0}}},
    StatefulModelConfigParams{"CPU,GPU",
                              false,
                              true,
                              std::map<std::string, bool>{{"CPU", false}, {"GPU", false}},
                              false,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", 0}}},

    StatefulModelConfigParams{"CPU,GPU",
                              false,
                              true,
                              std::map<std::string, bool>{{"CPU", true}, {"GPU", false}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 0}, {"CPU", 1}}},
    StatefulModelConfigParams{"CPU,GPU",
                              false,
                              true,
                              std::map<std::string, bool>{{"CPU", false}, {"GPU", true}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", 0}}},
    StatefulModelConfigParams{"CPU,GPU",
                              false,
                              true,
                              std::map<std::string, bool>{{"CPU", true}, {"GPU", true}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", -1}, {"CPU", -1}}},
    StatefulModelConfigParams{"CPU,GPU",
                              false,
                              true,
                              std::map<std::string, bool>{{"CPU", false}, {"GPU", false}},
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", -1}, {"CPU", -1}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         StatefulModelSupportedTest,
                         ::testing::ValuesIn(testConfigs),
                         StatefulModelSupportedTest::getTestCaseName);
