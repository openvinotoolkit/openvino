// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/common_utils.hpp>
#include <thread>

#include "include/auto_unit_test.hpp"

using DynamicOutputConfigParams = std::tuple<
        ov::Any,                  // priority device list
        ov::Any                   // expected device to run inference on
        >;

class DynamicOutputInferenceTest : public tests::AutoTest, public ::testing::TestWithParam<DynamicOutputConfigParams> {
public:
    std::shared_ptr<ov::Model> create_dynamic_output_model();
    static std::string getTestCaseName(testing::TestParamInfo<DynamicOutputConfigParams> obj);
    void SetUp() override;

protected:
    ov::Any priorityList;
    ov::Any targetList;
};

std::string DynamicOutputInferenceTest::getTestCaseName(testing::TestParamInfo<DynamicOutputConfigParams> obj) {
    ov::Any priorityList;
    ov::Any targetList;
    std::tie(priorityList, targetList) = obj.param;
    std::ostringstream result;
    result << "_withList_" << priorityList.as<std::string>();
    result << "_expect_";
    auto targets = targetList.as<std::vector<std::string>>();
    for (auto& iter : targets)
        result << "_" << iter;
    auto string = result.str();
    ov::test::utils::replaceSubstringInString(string, ",", "_");
    return string;
}

std::shared_ptr<ov::Model> DynamicOutputInferenceTest::create_dynamic_output_model() {
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

void DynamicOutputInferenceTest::SetUp() {
    model = create_dynamic_output_model();
    std::tie(priorityList, targetList) = GetParam();
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_GPU)),
                          _))
        .WillByDefault(InvokeWithoutArgs([this]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            return mockExeNetworkActual;
        }));
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq(ov::test::utils::DEVICE_CPU)),
                          (_)))
        .WillByDefault(Return(mockExeNetwork));
}

TEST_P(DynamicOutputInferenceTest, CanSelectCorrectTargetDeviceandInitizeBlobWithCorrectSize) {
    auto targets = targetList.as<std::vector<std::string>>();
    config.insert(ov::device::priorities(priorityList.as<std::string>()));
    config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
    std::shared_ptr<ov::ICompiledModel> exeNetwork;
    for (auto& iter : targets) {
        EXPECT_CALL(*core,
                    compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                                  ::testing::Matcher<const std::string&>(HasSubstr(iter)),
                                  ::testing::Matcher<const ov::AnyMap&>(_)))
            .Times(1);
    }
    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(HasSubstr("GPU")),
                              ::testing::Matcher<const ov::AnyMap&>(_)))
        .Times(0);
    ASSERT_NO_THROW(exeNetwork = plugin->compile_model(model, config));
}

const std::vector<DynamicOutputConfigParams> testConfigs = {
    DynamicOutputConfigParams{"CPU,GPU", std::vector<std::string>{"CPU"}},
    DynamicOutputConfigParams{"GPU,CPU", std::vector<std::string>{"CPU"}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         DynamicOutputInferenceTest,
                         ::testing::ValuesIn(testConfigs),
                         DynamicOutputInferenceTest::getTestCaseName);
