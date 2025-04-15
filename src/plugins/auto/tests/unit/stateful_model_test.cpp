// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/common_utils.hpp>
#include <thread>

#include "include/auto_unit_test.hpp"
#include "openvino/opsets/opset11.hpp"

using StatefulModelConfigParams =
    std::tuple<std::string,                               // device candidate list
               bool,                                      // is dynamic model
               bool,                                      // is stateful model
               bool,                                      // is cumulative mode
               bool,                                      // is actual device compiled model successfully
               std::vector<std::pair<std::string, int>>,  // expected compiling model times on each device
               std::string                                // expected execution devices list
               >;

class StatefulModelSupportedTest : public tests::AutoTest, public ::testing::TestWithParam<StatefulModelConfigParams> {
public:
    std::shared_ptr<ov::Model> create_dynamic_output_model();
    std::shared_ptr<ov::Model> create_stateful_model();
    std::shared_ptr<ov::Model> create_stateful_dynamic_model();
    static std::string getTestCaseName(testing::TestParamInfo<StatefulModelConfigParams> obj);
    void SetUp() override;

protected:
    std::string devicesList;
    bool isDynamicModel;
    bool isStatefulModel;
    bool isActualSuccessful;
    bool isCumulative;
    std::vector<std::pair<std::string, int>> expectedCalledTimes;
    std::string expectedExecuteDev;
};

std::string StatefulModelSupportedTest::getTestCaseName(testing::TestParamInfo<StatefulModelConfigParams> obj) {
    bool isDynamicModel;
    bool isStatefulModel;
    bool isActualSuccessful;
    bool isCumulative;
    std::vector<std::pair<std::string, int>> expectedCalledTimes;
    std::string devicesList;
    std::string expectedExecuteDev;
    std::tie(devicesList,
             isDynamicModel,
             isStatefulModel,
             isCumulative,
             isActualSuccessful,
             expectedCalledTimes,
             expectedExecuteDev) = obj.param;
    std::ostringstream result;
    result << "_devicesList_" << devicesList;
    result << "_isDynamic_" << isDynamicModel;
    result << "_isStatefulModel_" << isStatefulModel;
    result << "_isCumulative_" << isCumulative;
    result << "_isActualCompileSuccessful" << isActualSuccessful;
    for (auto& item : expectedCalledTimes) {
        result << "_calling_on_" << item.first << "_expected_times_" << item.second;
    }
    result << "_expectedExecuteDevice_" << expectedExecuteDev;
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

std::shared_ptr<ov::Model> StatefulModelSupportedTest::create_stateful_dynamic_model() {
    ov::PartialShape shape({ov::Dimension::dynamic(), 1});
    ov::element::Type type(ov::element::Type_t::f32);
    auto arg = std::make_shared<ov::op::v0::Parameter>(type, shape);
    auto init_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1}, {0});
    // The ReadValue/Assign operations must be used in pairs in the model.
    // For each such a pair, its own variable object must be created.
    const std::string variable_name("variable0");
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
    std::tie(devicesList,
             isDynamicModel,
             isStatefulModel,
             isCumulative,
             isActualSuccessful,
             expectedCalledTimes,
             expectedExecuteDev) = GetParam();
    if (isDynamicModel && isStatefulModel) {
        model = create_stateful_dynamic_model();
    } else if (isDynamicModel) {
        model = create_dynamic_output_model();
    } else if (isStatefulModel) {
        model = create_stateful_model();
    }

    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                          (_)))
        .WillByDefault(InvokeWithoutArgs([this]() {
            return mockExeNetwork;
        }));

    if (isActualSuccessful) {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_GPU)),
                              (_)))
            .WillByDefault(InvokeWithoutArgs([this]() {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(100));
                return mockExeNetworkActual;
            }));
    } else {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_GPU)),
                              _))
            .WillByDefault(InvokeWithoutArgs([this]() {
                OPENVINO_THROW("");
                return mockExeNetworkActual;
            }));
    }
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
        } else if (times == -2) {
            EXPECT_CALL(*core,
                        compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                                      ::testing::Matcher<const std::string&>(StrEq(deviceName)),
                                      ::testing::Matcher<const ov::AnyMap&>(_)))
                .Times(AtLeast(1));
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

    if (isCumulative) {
        config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
    }
    std::shared_ptr<ov::ICompiledModel> exeNetwork;
    if (expectedTimes < 0) {
        ASSERT_THROW(plugin->compile_model(model, config), ov::Exception);
    } else {
        OV_ASSERT_NO_THROW(exeNetwork = plugin->compile_model(model, config));
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        EXPECT_EQ(exeNetwork->get_property(ov::execution_devices.name()).as<std::string>(), expectedExecuteDev);
    }
}

const std::vector<StatefulModelConfigParams> testConfigs = {
    // test cases for dynamic model
    StatefulModelConfigParams{
        "CPU",                                                 // device candidate list
        false,                                                 // is dynamic model
        false,                                                 // is stateful model
        false,                                                 // is cumulative mode
        false,                                                 // is actual device compiled model successfully
        std::vector<std::pair<std::string, int>>{{"CPU", 1}},  // expected compiling model times on each device
        "CPU"},                                                // expected compiling model times on each device
    StatefulModelConfigParams{"CPU",
                              true,
                              false,
                              false,
                              false,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}},
                              "CPU"},
    StatefulModelConfigParams{"CPU",
                              false,
                              true,
                              false,
                              true,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}},
                              "CPU"},
    StatefulModelConfigParams{"CPU",
                              true,
                              true,
                              true,
                              false,
                              std::vector<std::pair<std::string, int>>{{"CPU", 1}},
                              "CPU"},
    StatefulModelConfigParams{"GPU",
                              false,
                              false,
                              false,
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}},
                              "GPU"},
    StatefulModelConfigParams{"GPU",
                              true,
                              false,
                              true,
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}},
                              "GPU"},
    StatefulModelConfigParams{"GPU",
                              false,
                              true,
                              false,
                              false,
                              std::vector<std::pair<std::string, int>>{{"GPU", -1}},
                              ""},
    StatefulModelConfigParams{"GPU",
                              true,
                              true,
                              true,
                              false,
                              std::vector<std::pair<std::string, int>>{{"GPU", -1}},
                              ""},

    StatefulModelConfigParams{"GPU,CPU",
                              false,
                              false,
                              false,
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", 1}},
                              "GPU"},
    StatefulModelConfigParams{"GPU,CPU",
                              true,
                              false,
                              false,
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", 1}},
                              "GPU"},
    StatefulModelConfigParams{
        "GPU,CPU",
        false,
        false,
        false,
        false,
        std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", -2}},  // -2 means at least 1 times for CPU
        "CPU"},
    StatefulModelConfigParams{
        "GPU,CPU",
        true,
        false,
        false,
        false,
        std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", -2}},  // -2 means at least 1 times for CPU
        "CPU"},

    StatefulModelConfigParams{"GPU,CPU",
                              true,
                              false,
                              true,
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", 1}},
                              "GPU CPU"},
    StatefulModelConfigParams{"GPU,CPU",
                              false,
                              false,
                              true,
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", 1}},
                              "GPU CPU"},

    StatefulModelConfigParams{"GPU,CPU",
                              true,
                              false,
                              true,
                              false,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", 1}},
                              "CPU"},
    StatefulModelConfigParams{"GPU,CPU",
                              false,
                              false,
                              true,
                              false,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", 1}},
                              "CPU"},
    // AUTO with stateful model
    StatefulModelConfigParams{"GPU,CPU",
                              false,
                              true,
                              false,
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", 0}},
                              "GPU"},
    StatefulModelConfigParams{"GPU,CPU",
                              true,
                              true,
                              false,
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", 1}, {"CPU", 0}},
                              "GPU"},

    // MULTI with stateful model
    StatefulModelConfigParams{"GPU,CPU",
                              false,
                              true,
                              true,
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", -1}, {"CPU", -1}},
                              ""},
    StatefulModelConfigParams{"GPU,CPU",
                              true,
                              true,
                              true,
                              true,
                              std::vector<std::pair<std::string, int>>{{"GPU", -1}, {"CPU", -1}},
                              ""}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         StatefulModelSupportedTest,
                         ::testing::ValuesIn(testConfigs),
                         StatefulModelSupportedTest::getTestCaseName);
