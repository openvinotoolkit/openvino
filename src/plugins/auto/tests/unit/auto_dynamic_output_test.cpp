// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "include/mock_auto_device_plugin.hpp"
#include "include/auto_infer_request_test_base.hpp"

std::string DynamicOutputInferenceTest::getTestCaseName(testing::TestParamInfo<DynamicOutputConfigParams> obj) {
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

std::shared_ptr<ngraph::Function> DynamicOutputInferenceTest::getFunction() {
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

void DynamicOutputInferenceTest::SetUp() {
        std::tie(isNewAPI, priorityList, property, targetList) = GetParam();
        if (isNewAPI) {
            ON_CALL(*core.get(), isNewAPI()).WillByDefault(Return(true));
        } else {
            ON_CALL(*core.get(), isNewAPI()).WillByDefault(Return(false));
        }
        auto function = getFunction();
        cnnNet = InferenceEngine::CNNNetwork(function);
        // replace core with mock Icore
        plugin->SetCore(core);
        plugin->SetName("MULTI"); // change to AUTO when multi logic been unified to auto
        makeAsyncRequest();
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(HasSubstr(CommonTestUtils::DEVICE_GPU)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        return mockExeNetwork; }));
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
            ::testing::Matcher<const std::string&>(HasSubstr(CommonTestUtils::DEVICE_CPU)),
            ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));
}

void DynamicOutputInferenceTest::TearDown() {
}

TEST_P(DynamicOutputInferenceTest, CanSelectCorrectTargetDeviceandInitizeBlobWithCorrectSize) {
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

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, DynamicOutputInferenceTest,
                ::testing::ValuesIn(testConfigs),
            DynamicOutputInferenceTest::getTestCaseName);