// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/common_utils.hpp>
#include <thread>

#include "include/auto_unit_test.hpp"
#include "openvino/runtime/threading/immediate_executor.hpp"
using DynamicOutputConfigParams = std::tuple<ov::Any,  // priority device list
                                             ov::Any   // expected device to run inference on
                                             >;

class DynamicOutputInferenceTest : public tests::AutoTest, public ::testing::TestWithParam<DynamicOutputConfigParams> {
public:
    DynamicOutputInferenceTest(const tests::MODELTYPE modelType = tests::MODELTYPE::DYNAMIC) : AutoTest(modelType) {}
    static std::string getTestCaseName(testing::TestParamInfo<DynamicOutputConfigParams> obj);
    void SetUp() override;
    void TearDown() override {
        mockExecutor.reset();
        mockExecutorActual.reset();
        mockInferrequest.reset();
        mockInferrequest_2.reset();
        mockInferrequestActual.reset();
        mockInferrequestActual_2.reset();
    }

protected:
    ov::Any priorityList;
    ov::Any targetList;
    std::shared_ptr<ov::mock_auto_plugin::MockAsyncInferRequest> mockInferrequest;
    std::shared_ptr<ov::mock_auto_plugin::MockAsyncInferRequest> mockInferrequest_2;
    std::shared_ptr<ov::mock_auto_plugin::MockAsyncInferRequest> mockInferrequestActual;
    std::shared_ptr<ov::mock_auto_plugin::MockAsyncInferRequest> mockInferrequestActual_2;
    std::shared_ptr<ov::threading::ImmediateExecutor> mockExecutor;
    std::shared_ptr<ov::threading::ImmediateExecutor> mockExecutorActual;
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

void DynamicOutputInferenceTest::SetUp() {
    mockExecutor = std::make_shared<ov::threading::ImmediateExecutor>();
    mockExecutorActual = std::make_shared<ov::threading::ImmediateExecutor>();
    mockInferrequest =
        std::make_shared<ov::mock_auto_plugin::MockAsyncInferRequest>(inferReqInternal, mockExecutor, nullptr, false);
    // will be at least 2 infer requests for mocked CPU/GPU
    auto inferReqInternal_2 = std::make_shared<ov::mock_auto_plugin::MockISyncInferRequest>(mockIExeNet);
    mockInferrequest_2 =
        std::make_shared<ov::mock_auto_plugin::MockAsyncInferRequest>(inferReqInternal_2, mockExecutor, nullptr, false);

    auto inferReqInternalActual_2 = std::make_shared<ov::mock_auto_plugin::MockISyncInferRequest>(mockIExeNetActual);

    mockInferrequestActual = std::make_shared<ov::mock_auto_plugin::MockAsyncInferRequest>(inferReqInternalActual,
                                                                                           mockExecutorActual,
                                                                                           nullptr,
                                                                                           false);
    mockInferrequestActual_2 = std::make_shared<ov::mock_auto_plugin::MockAsyncInferRequest>(inferReqInternalActual_2,
                                                                                             mockExecutorActual,
                                                                                             nullptr,
                                                                                             false);

    std::tie(priorityList, targetList) = GetParam();
    auto targets = targetList.as<std::vector<std::string>>();
    ON_CALL(*core, get_available_devices()).WillByDefault(Return(targets));
        std::vector<std::string> deviceIDs = {};
        ON_CALL(*core, get_property(StrEq("GPU"), StrEq(ov::available_devices.name()), _))
            .WillByDefault(RETURN_MOCK_VALUE(deviceIDs));
    for (auto device : targets) {
        ON_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                              ::testing::Matcher<const std::string&>(StrEq(device)),
                              _))
            .WillByDefault(InvokeWithoutArgs([this, device]() {
                if (device.find("GPU") != std::string::npos) {
                    if (device == "GPU.1") {
                        return mockExeNetwork;
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        return mockExeNetworkActual;
                    }
                }
                return mockExeNetwork;
            }));
    }
}

TEST_P(DynamicOutputInferenceTest, CanSelectCorrectTargetDeviceandInitizeBlobWithCorrectSize) {
    auto targets = targetList.as<std::vector<std::string>>();
    config.insert(ov::device::priorities(priorityList.as<std::string>()));
    config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
    std::shared_ptr<ov::ICompiledModel> exeNetwork;
    for (auto& iter : targets) {
        EXPECT_CALL(*core,
                    compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                                  ::testing::Matcher<const std::string&>(StrEq(iter)),
                                  ::testing::Matcher<const ov::AnyMap&>(_)))
            .Times(1);
    }
    OV_ASSERT_NO_THROW(exeNetwork = plugin->compile_model(model, config));
}

TEST_P(DynamicOutputInferenceTest, CanInferWithOutputChangedFromDynamicOnAutoToStaticOnActualDevice) {
    plugin->set_device_name("AUTO");
    // change the tensor shape from dynamic to static for CPU/GPU.0 infer request
    for (auto& it : inferReqInternal->get_outputs()) {
        if (!it.get_partial_shape().is_dynamic())
            continue;
        auto tensor = inferReqInternal->get_tensor(it);
        tensor->set_shape(ov::Shape{2, 3});
    }
    EXPECT_CALL(*mockIExeNet.get(), create_infer_request())
        .WillOnce(Return(mockInferrequest))
        .WillOnce(Return(mockInferrequest_2));
    EXPECT_CALL(*mockIExeNetActual.get(), create_infer_request())
        .WillOnce(Return(mockInferrequestActual))
        .WillOnce(Return(mockInferrequestActual_2));
    config.insert(ov::device::priorities(priorityList.as<std::string>()));
    config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
    std::shared_ptr<ov::ICompiledModel> exeNetwork;
    OV_ASSERT_NO_THROW(exeNetwork = plugin->compile_model(model, config));
    std::shared_ptr<ov::IAsyncInferRequest> infer_request;
    OV_ASSERT_NO_THROW(infer_request = exeNetwork->create_infer_request());
    OV_ASSERT_NO_THROW(infer_request->infer());
}

const std::vector<DynamicOutputConfigParams> testConfigs = {
    DynamicOutputConfigParams{"CPU,GPU", std::vector<std::string>{"CPU", "GPU"}},
    DynamicOutputConfigParams{"GPU,CPU", std::vector<std::string>{"CPU", "GPU"}},
    DynamicOutputConfigParams{"GPU.0,GPU.1", std::vector<std::string>{"GPU.0", "GPU.1"}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         DynamicOutputInferenceTest,
                         ::testing::ValuesIn(testConfigs),
                         DynamicOutputInferenceTest::getTestCaseName);
