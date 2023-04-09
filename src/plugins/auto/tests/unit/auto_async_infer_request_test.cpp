// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "include/mock_auto_device_plugin.hpp"
#include "include/auto_infer_request_test_base.hpp"

std::string AsyncInferenceTest::getTestCaseName(testing::TestParamInfo<AsyncInferRequestTestParams> obj) {
    bool isCpuFast;
    bool isSingleDevice;
    std::tie(isCpuFast, isSingleDevice) = obj.param;
    std::ostringstream result;
    result << "_isCPUFaster_" << isCpuFast << "_isSingleDevice" << isSingleDevice;
    return result.str();
}

void AsyncInferenceTest::SetUp() {
        std::tie(isCpuFaster, isSingleDevice) = GetParam();
        ON_CALL(*core.get(), isNewAPI()).WillByDefault(Return(true));
        if (isSingleDevice) {
            std::vector<std::string>  testDevs = {CommonTestUtils::DEVICE_GPU};
            ON_CALL(*core, GetAvailableDevices()).WillByDefault(Return(testDevs));
        } else if (isCpuFaster) {
             ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(Not(HasSubstr(CommonTestUtils::DEVICE_CPU))),
                ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    return mockExeNetwork; }));
            ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(HasSubstr(CommonTestUtils::DEVICE_CPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));
        } else {
            ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(HasSubstr(CommonTestUtils::DEVICE_CPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    return mockExeNetwork; }));
            ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(Not(HasSubstr(CommonTestUtils::DEVICE_CPU))),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));
        }
        taskExecutor = std::make_shared<DeferedExecutor>();
        // replace core with mock Icore
        plugin->SetCore(core);
        makeAsyncRequest();
        exeNetwork = plugin->LoadNetwork(cnnNet, {});
        auto_request = exeNetwork->CreateInferRequest();
}

void AsyncInferenceTest::TearDown() {
}

TEST_P(AsyncInferenceTest, returnRequestBusyOnStartAsync) {
    for (size_t i = 0; i < request_num_pool ; i++)
        ON_CALL(*inferReqInternal[i], InferImpl()).WillByDefault(Return());
    ASSERT_NO_THROW(auto_request->StartAsync());
    ASSERT_THROW(auto_request->StartAsync(), RequestBusy);
    std::dynamic_pointer_cast<DeferedExecutor>(taskExecutor)->executeAll();
}

TEST_P(AsyncInferenceTest, canInferIfStartAsyncSuccess) {
    auto_request->SetCallback([&](std::exception_ptr exceptionPtr_) {
        auto exceptionPtr = exceptionPtr_;
        ASSERT_EQ(exceptionPtr, nullptr);
    });
    auto_request->StartAsync();
    std::dynamic_pointer_cast<DeferedExecutor>(taskExecutor)->executeAll();
    ASSERT_NO_THROW(auto_request->Wait(InferRequest::WaitMode::RESULT_READY));
}

TEST_P(AsyncInferenceTest, canRethrowIfStartAsyncFails) {
    // create additional resource for infer fail restart
    for (size_t i = 0; i < request_num_pool; i++)
        ON_CALL(*inferReqInternal[i], InferImpl()).WillByDefault(Throw(InferenceEngine::GeneralError("runtime error")));
    auto_request = exeNetwork->CreateInferRequest();
    auto_request->SetCallback([&](std::exception_ptr exceptionPtr_) {
        auto exceptionPtr = exceptionPtr_;
        ASSERT_NE(exceptionPtr, nullptr);
    });
    auto_request->StartAsync();
    std::dynamic_pointer_cast<DeferedExecutor>(taskExecutor)->executeAll();
    EXPECT_THROW(auto_request->Wait(InferRequest::WaitMode::RESULT_READY), std::exception);
}

TEST_P(AsyncInferenceTest, canStart2AsyncInferRequests) {
    auto another_auto_request = exeNetwork->CreateInferRequest();
    auto_request->SetCallback([&](std::exception_ptr exceptionPtr_) {
        auto exceptionPtr = exceptionPtr_;
        ASSERT_EQ(exceptionPtr, nullptr);
    });
    another_auto_request->SetCallback([&](std::exception_ptr exceptionPtr_) {
        auto exceptionPtr = exceptionPtr_;
        ASSERT_EQ(exceptionPtr, nullptr);
    });
    auto_request->StartAsync();
    another_auto_request->StartAsync();
    std::dynamic_pointer_cast<DeferedExecutor>(taskExecutor)->executeAll();
    ASSERT_NO_THROW(auto_request->Wait(InferRequest::WaitMode::RESULT_READY));
    ASSERT_NO_THROW(another_auto_request->Wait(InferRequest::WaitMode::RESULT_READY));
}

TEST_P(AsyncInferenceTest, returnRequestBusyOnGetPerformanceCounts) {
    for (size_t i = 0; i < request_num_pool ; i++)
        ON_CALL(*inferReqInternal[i], InferImpl()).WillByDefault(Return());
    ASSERT_NO_THROW(auto_request->StartAsync());
    ASSERT_THROW(auto_request->GetPerformanceCounts(), RequestBusy);
    std::dynamic_pointer_cast<DeferedExecutor>(taskExecutor)->executeAll();
}

const std::vector<AsyncInferRequestTestParams> testConfigs = {
    AsyncInferRequestTestParams {true, true},
    AsyncInferRequestTestParams {true, false},
    AsyncInferRequestTestParams {false, true},
    AsyncInferRequestTestParams {false, false}
};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, AsyncInferenceTest,
                ::testing::ValuesIn(testConfigs),
            AsyncInferenceTest::getTestCaseName);