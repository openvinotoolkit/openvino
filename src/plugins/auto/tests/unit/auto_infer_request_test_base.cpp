// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_infer_request_test_base.hpp"
AutoInferRequestTestBase::AutoInferRequestTestBase() {
    // prepare cpuMockExeNetwork
    mockIExeNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
    mockExeNetwork =  {mockIExeNet, {}};

    // prepare mockicore and cnnNetwork for loading
    core = std::make_shared<NiceMock<MockICore>>();
    auto* origin_plugin = new MultiDeviceInferencePlugin();
    plugin  = std::shared_ptr<MultiDeviceInferencePlugin>(origin_plugin);
    function = getFunction();
    cnnNet = InferenceEngine::CNNNetwork(function);
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(HasSubstr(CommonTestUtils::DEVICE_GPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    return mockExeNetwork; }));
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(HasSubstr(CommonTestUtils::DEVICE_CPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(HasSubstr(CommonTestUtils::DEVICE_KEEMBAY)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    return mockExeNetwork; }));
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
    IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, optimalNum, 2);
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
    plugin->SetName("AUTO");
}

void AutoInferRequestTestBase::makeAsyncRequest() {
    auto taskExecutor = std::make_shared<ImmediateExecutor>();
    // set up mock infer request
    for (size_t i = 0; i < target_request_num; i++) {
        auto inferReq = std::make_shared<NiceMock<MockIInferRequestInternal>>();
        auto asyncRequest = std::make_shared<AsyncInferRequestThreadSafeDefault>(inferReq, taskExecutor, taskExecutor);
        inferReqInternal.push_back(inferReq);
        asyncInferRequest.push_back(asyncRequest);
    }
    EXPECT_CALL(*mockIExeNet.get(), CreateInferRequest()).WillOnce(Return(asyncInferRequest[0]))
                                                            .WillOnce(Return(asyncInferRequest[1]))
                                                            .WillOnce(Return(asyncInferRequest[2]))
                                                            .WillOnce(Return(asyncInferRequest[3]))
                                                            .WillRepeatedly(Return(asyncInferRequest[0]));
}
AutoInferRequestTestBase::~AutoInferRequestTestBase() {
    core.reset();
    plugin.reset();
    mockIExeNet.reset();
    mockExeNetwork = {};

    metaDevices.clear();
    inferReqInternal.clear();
    asyncInferRequest.clear();
}

    // constructing default test model
std::shared_ptr<ngraph::Function> AutoInferRequestTestBase::getFunction() {
    return ngraph::builder::subgraph::makeSingleConv();
}