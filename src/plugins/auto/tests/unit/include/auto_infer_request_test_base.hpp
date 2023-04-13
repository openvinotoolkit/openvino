// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock-spec-builders.h>
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
#include "mock_auto_device_plugin.hpp"
#include <atomic>

using namespace ::testing;

using Config = std::map<std::string, std::string>;
const std::vector<std::string>  availableDevs = {CommonTestUtils::DEVICE_CPU, CommonTestUtils::DEVICE_GPU, CommonTestUtils::DEVICE_KEEMBAY};
struct DeferedExecutor : public ITaskExecutor {
    using Ptr = std::shared_ptr<DeferedExecutor>;
    DeferedExecutor() = default;

    void executeOne() {
        tasks.front()();
        tasks.pop_front();
    }

    void executeAll() {
        while (!tasks.empty()) {
            executeOne();
        }
    }

    ~DeferedExecutor() override {
        executeAll();
    };

    void run(Task task) override {
        tasks.push_back(task);
    }

    std::deque<Task> tasks;
};

class AutoInferRequestTestBase {
protected:
    // by default will create 4 infer request for tests and 4 reserverd for fallback
    const unsigned int                                                  request_num_pool{8};
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

    std::atomic_int                                                     index{0};
    ITaskExecutor::Ptr                                                  taskExecutor;

public:
    ~AutoInferRequestTestBase();

    AutoInferRequestTestBase();

protected:
    // constructing default test model
    std::shared_ptr<ngraph::Function> getFunction();
    void makeAsyncRequest();
};

using DynamicOutputConfigParams = std::tuple<
        bool,                     // is newAPI or not
        ov::Any,                  // priority device list
        ov::AnyMap,               // hint setting
        ov::Any                   // expected device to run inference on
        >;
class DynamicOutputInferenceTest : public AutoInferRequestTestBase,
                            public ::testing::TestWithParam<DynamicOutputConfigParams> {
public:
    std::shared_ptr<ngraph::Function> getFunction();
    static std::string getTestCaseName(testing::TestParamInfo<DynamicOutputConfigParams> obj);
    void SetUp() override;
    void TearDown() override;

protected:
        bool isNewAPI;
        ov::Any priorityList;
        ov::AnyMap property;
        ov::Any targetList;
};

using AsyncInferRequestTestParams = std::tuple<
        bool,                      // is cpu loaded faster
        bool                       // is single device
>;
class AsyncInferenceTest : public AutoInferRequestTestBase,
                            public ::testing::TestWithParam<AsyncInferRequestTestParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AsyncInferRequestTestParams> obj);
    void SetUp() override;
    void TearDown() override;

protected:
    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> exeNetwork;
    std::shared_ptr<InferenceEngine::IInferRequestInternal> auto_request;
    bool isCpuFaster;
    bool isSingleDevice;
};

class mockAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Parent = InferenceEngine::AsyncInferRequestThreadSafeDefault;
    mockAsyncInferRequest(const InferenceEngine::IInferRequestInternal::Ptr &inferRequest,
                      const ImmediateExecutor::Ptr& taskExecutor,
                      const ImmediateExecutor::Ptr& callbackExecutor,
                      bool ifThrow);

    ~mockAsyncInferRequest() override = default;

private:
    bool _throw;
};