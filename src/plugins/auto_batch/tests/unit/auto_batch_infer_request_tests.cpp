// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thread>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_ngraph_utils.hpp"
#include "mock_auto_batch_plugin.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/utils/utils.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/mock_task_executor.hpp"

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::Eq;
using ::testing::MatcherCast;
using ::testing::Matches;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::StrNe;
using ::testing::Throw;
using namespace MockAutoBatchPlugin;
using namespace MockAutoBatchDevice;
using namespace InferenceEngine;

using AutoBatchRequestTestParams = std::tuple<int,                      // batch_size
                                              ngraph::element::Type_t,  // data type
                                              int>;                     // inference interval
class AutoBatchRequestTest : public ::testing::TestWithParam<AutoBatchRequestTestParams> {
public:
    // Mock inferRequest
    std::shared_ptr<NiceMock<MockIInferRequestInternal>> mockInferRequestBatched;

    std::vector<std::shared_ptr<AutoBatchInferRequest>> autoBatchInferRequests;
    std::map<std::string, InferenceEngine::Blob::Ptr> blobMap;

    std::vector<std::shared_ptr<const ov::Node>> inputs, outputs;
    std::set<std::string> batchedInputs, batchedOutputs;
    std::shared_ptr<AutoBatchExecutableNetwork::WorkerInferRequest> workerRequestPtr;

public:
    static std::string getTestCaseName(testing::TestParamInfo<AutoBatchRequestTestParams> obj) {
        int batch_size, infer_interval;
        ngraph::element::Type_t element_type;
        std::tie(batch_size, element_type, infer_interval) = obj.param;

        std::string res;
        res = "batch_size_" + std::to_string(batch_size);
        res += "_element_type_" + std::to_string(static_cast<int>(element_type));
        if (infer_interval > 0)
            res += "_infer_interval_" + std::to_string(infer_interval);
        return res;
    }

    void TearDown() override {
        mockInferRequestBatched = {};
        autoBatchInferRequests.clear();
        blobMap.clear();

        inputs.clear();
        outputs.clear();
        batchedInputs.clear();
        batchedOutputs.clear();
        clear_worker();
    }

    void SetUp() override {
        mockInferRequestBatched = std::make_shared<NiceMock<MockIInferRequestInternal>>();
    }

    void create_worker(int batch_size) {
        workerRequestPtr = std::make_shared<AutoBatchExecutableNetwork::WorkerInferRequest>();

        workerRequestPtr->_inferRequestBatched = {mockInferRequestBatched, {}};
        workerRequestPtr->_batchSize = batch_size;
        workerRequestPtr->_completionTasks.resize(workerRequestPtr->_batchSize);
        workerRequestPtr->_inferRequestBatched->SetCallback([this](std::exception_ptr exceptionPtr) mutable {
            if (exceptionPtr)
                workerRequestPtr->_exceptionPtr = exceptionPtr;
        });
        workerRequestPtr->_thread = std::thread([this] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
        return;
    }

    void clear_worker() {
        workerRequestPtr->_inferRequestBatched = {};
        workerRequestPtr->_completionTasks.clear();
        workerRequestPtr->_thread.join();
    }

    void prepare_input(std::shared_ptr<ov::Model>& function, int batch_size) {
        for (auto& input : function->inputs()) {
            std::shared_ptr<const ov::Node> n = input.get_node_shared_ptr();
            inputs.emplace_back(n);
        }

        for (auto& output : function->outputs()) {
            std::shared_ptr<const ov::Node> n = output.get_node_shared_ptr();
            outputs.emplace_back(n);
        }

        const auto& params = function->get_parameters();
        for (size_t i = 0; i < params.size(); i++) {
            batchedInputs.insert(ov::op::util::get_ie_output_name(params[i]->output(0)));
        }
        const auto& results = function->get_results();
        for (size_t i = 0; i < results.size(); i++) {
            const auto& output = results[i];
            const auto& node = output->input_value(0);
            batchedOutputs.insert(
                ov::op::util::get_ie_output_name(ov::Output<const ov::Node>(node.get_node(), node.get_index())));
        }

        ON_CALL(*mockInferRequestBatched, GetBlob(StrEq(*batchedInputs.begin())))
            .WillByDefault([this, batch_size](const std::string& name) {
                auto item = blobMap.find(name);
                if (item != blobMap.end()) {
                    return item->second;
                }
                auto shape = inputs[0]->get_shape();
                shape[0] = batch_size;
                auto element_type = inputs[0]->get_element_type();
                InferenceEngine::TensorDesc tensorDesc = {InferenceEngine::details::convertPrecision(element_type),
                                                          shape,
                                                          InferenceEngine::TensorDesc::getLayoutByRank(shape.size())};
                auto blob = make_blob_with_precision(tensorDesc);
                blob->allocate();
                blobMap[name] = blob;
                return blob;
            });

        ON_CALL(*mockInferRequestBatched, GetBlob(StrEq(*batchedOutputs.begin())))
            .WillByDefault([this, batch_size](const std::string& name) {
                auto item = blobMap.find(name);
                if (item != blobMap.end()) {
                    return item->second;
                }
                auto shape = outputs[0]->get_shape();
                shape[0] = batch_size;
                auto element_type = outputs[0]->get_element_type();
                InferenceEngine::TensorDesc tensorDesc = {InferenceEngine::details::convertPrecision(element_type),
                                                          shape,
                                                          InferenceEngine::TensorDesc::getLayoutByRank(shape.size())};
                auto blob = make_blob_with_precision(tensorDesc);
                blob->allocate();
                blobMap[name] = blob;
                return blob;
            });
    }
};

TEST_P(AutoBatchRequestTest, AutoBatchRequestCreateTestCase) {
    int batch_size, infer_interval;
    ngraph::element::Type_t element_type;
    std::tie(batch_size, element_type, infer_interval) = this->GetParam();

    std::vector<size_t> inputShape = {1, 3, 24, 24};
    auto function = ngraph::builder::subgraph::makeMultiSingleConv(inputShape, element_type);
    prepare_input(function, batch_size);
    create_worker(batch_size);

    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        auto req = std::make_shared<AutoBatchInferRequest>(inputs,
                                                           outputs,
                                                           *workerRequestPtr,
                                                           batch_id,
                                                           batch_size,
                                                           batchedInputs,
                                                           batchedOutputs);
        EXPECT_NE(req, nullptr);
        autoBatchInferRequests.emplace_back(req);

        std::vector<std::string> names = {*batchedInputs.begin(), *batchedOutputs.begin()};
        for (auto& name : names) {
            auto blob = req->GetBlob(name);
            auto ptr = blob->buffer().as<char*>();
            auto size = blob->byteSize();
            auto batch_blob = mockInferRequestBatched->GetBlob(name);
            auto batch_ptr = batch_blob->buffer().as<char*>();
            EXPECT_EQ(ptr, batch_ptr + size * batch_id);
        }
    }
}

TEST_P(AutoBatchRequestTest, AutoBatchRequestCopyBlobTestCase) {
    int batch_size, infer_interval;
    ngraph::element::Type_t element_type;
    std::tie(batch_size, element_type, infer_interval) = this->GetParam();

    std::vector<size_t> inputShape = {1, 3, 24, 24};
    auto function = ngraph::builder::subgraph::makeMultiSingleConv(inputShape, element_type);
    prepare_input(function, batch_size);
    create_worker(batch_size);

    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        auto req = std::make_shared<AutoBatchInferRequest>(inputs,
                                                           outputs,
                                                           *workerRequestPtr,
                                                           batch_id,
                                                           batch_size,
                                                           batchedInputs,
                                                           batchedOutputs);
        EXPECT_NE(req, nullptr);
        autoBatchInferRequests.emplace_back(req);

        EXPECT_NO_THROW(req->CopyInputsIfNeeded());
        EXPECT_NO_THROW(req->CopyOutputsIfNeeded());
    }
}

class AutoBatchAsyncInferRequestTest : public AutoBatchRequestTest {
public:
    std::shared_ptr<NiceMock<MockIInferRequestInternal>> mockInferRequestWithoutBatched;
    MockTaskExecutor::Ptr mockTaskExecutor;
    std::vector<AutoBatchAsyncInferRequest::Ptr> autoBatchAsyncInferRequestVec;
    bool terminate;

public:
    void TearDown() override {
        terminate = true;
        autoBatchAsyncInferRequestVec.clear();
        AutoBatchRequestTest::TearDown();
        mockInferRequestWithoutBatched = {};
    }

    void SetUp() override {
        AutoBatchRequestTest::SetUp();
        mockInferRequestWithoutBatched = std::make_shared<NiceMock<MockIInferRequestInternal>>();
        terminate = false;

        mockTaskExecutor = std::make_shared<MockTaskExecutor>();
    }

    void create_worker(int batch_size) {
        workerRequestPtr = std::make_shared<AutoBatchExecutableNetwork::WorkerInferRequest>();

        workerRequestPtr->_inferRequestBatched = {mockInferRequestBatched, {}};
        workerRequestPtr->_batchSize = batch_size;
        workerRequestPtr->_completionTasks.resize(workerRequestPtr->_batchSize);
        workerRequestPtr->_inferRequestBatched->SetCallback([this](std::exception_ptr exceptionPtr) mutable {
            if (exceptionPtr)
                workerRequestPtr->_exceptionPtr = exceptionPtr;
        });

        ON_CALL(*mockInferRequestBatched, StartAsync()).WillByDefault([this]() {
            IE_ASSERT(workerRequestPtr->_completionTasks.size() == (size_t)workerRequestPtr->_batchSize);
            for (int c = 0; c < workerRequestPtr->_batchSize; c++) {
                workerRequestPtr->_completionTasks[c]();
            }
            workerRequestPtr->_cond.notify_one();
        });

        workerRequestPtr->_thread = std::thread([this] {
            while (1) {
                std::cv_status status;
                {
                    std::unique_lock<std::mutex> lock(workerRequestPtr->_mutex);
                    status = workerRequestPtr->_cond.wait_for(lock, std::chrono::milliseconds(10));
                }
                if (terminate) {
                    break;
                } else {
                    const int sz = static_cast<int>(workerRequestPtr->_tasks.size());
                    if (sz == workerRequestPtr->_batchSize) {
                        std::pair<AutoBatchAsyncInferRequest*, InferenceEngine::Task> t;
                        for (int n = 0; n < sz; n++) {
                            IE_ASSERT(workerRequestPtr->_tasks.try_pop(t));
                            workerRequestPtr->_completionTasks[n] = std::move(t.second);
                            t.first->_inferRequest->_wasBatchedRequestUsed =
                                AutoBatchInferRequest::eExecutionFlavor::BATCH_EXECUTED;
                        }
                        workerRequestPtr->_inferRequestBatched->StartAsync();
                    } else if ((status == std::cv_status::timeout) && sz) {
                        std::pair<AutoBatchAsyncInferRequest*, InferenceEngine::Task> t;
                        for (int n = 0; n < sz; n++) {
                            IE_ASSERT(workerRequestPtr->_tasks.try_pop(t));
                            t.first->_inferRequest->_wasBatchedRequestUsed =
                                AutoBatchInferRequest::eExecutionFlavor::TIMEOUT_EXECUTED;
                            t.first->_inferRequestWithoutBatch->StartAsync();
                            t.second();
                        }
                    }
                }
            }
        });
        return;
    }
};

TEST_P(AutoBatchAsyncInferRequestTest, AutoBatchAsyncInferRequestCreateTest) {
    int batch_size, infer_interval;
    ngraph::element::Type_t element_type;
    std::tie(batch_size, element_type, infer_interval) = this->GetParam();

    std::vector<size_t> inputShape = {1, 3, 24, 24};
    auto function = ngraph::builder::subgraph::makeMultiSingleConv(inputShape, element_type);
    prepare_input(function, batch_size);
    create_worker(batch_size);

    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        auto autoRequestImpl = std::make_shared<AutoBatchInferRequest>(inputs,
                                                                       outputs,
                                                                       *workerRequestPtr,
                                                                       batch_id,
                                                                       batch_size,
                                                                       batchedInputs,
                                                                       batchedOutputs);
        EXPECT_NE(autoRequestImpl, nullptr);
        autoBatchInferRequests.emplace_back(autoRequestImpl);

        InferenceEngine::SoIInferRequestInternal inferRequestWithoutBatched = {mockInferRequestWithoutBatched, {}};
        auto asyncInferRequest =
            std::make_shared<AutoBatchAsyncInferRequest>(autoRequestImpl, inferRequestWithoutBatched, nullptr);
        EXPECT_NE(asyncInferRequest, nullptr);
        autoBatchAsyncInferRequestVec.emplace_back(asyncInferRequest);
    }
}

TEST_P(AutoBatchAsyncInferRequestTest, AutoBatchAsyncInferRequestStartAsyncTest) {
    int batch_size, infer_interval;
    ngraph::element::Type_t element_type;
    std::tie(batch_size, element_type, infer_interval) = this->GetParam();

    std::vector<size_t> inputShape = {1, 3, 24, 24};
    auto function = ngraph::builder::subgraph::makeMultiSingleConv(inputShape, element_type);
    prepare_input(function, batch_size);
    create_worker(batch_size);

    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        auto autoRequestImpl = std::make_shared<AutoBatchInferRequest>(inputs,
                                                                       outputs,
                                                                       *workerRequestPtr,
                                                                       batch_id,
                                                                       batch_size,
                                                                       batchedInputs,
                                                                       batchedOutputs);
        EXPECT_NE(autoRequestImpl, nullptr);
        autoBatchInferRequests.emplace_back(autoRequestImpl);

        InferenceEngine::SoIInferRequestInternal inferRequestWithoutBatched = {mockInferRequestWithoutBatched, {}};
        auto asyncInferRequest =
            std::make_shared<AutoBatchAsyncInferRequest>(autoRequestImpl, inferRequestWithoutBatched, nullptr);
        EXPECT_NE(asyncInferRequest, nullptr);
        autoBatchAsyncInferRequestVec.emplace_back(asyncInferRequest);
    }

    for (auto& req : autoBatchAsyncInferRequestVec) {
        if (infer_interval > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(infer_interval));
        EXPECT_NO_THROW(req->StartAsync());
    }

    for (auto& req : autoBatchAsyncInferRequestVec)
        EXPECT_NO_THROW(req->Wait(InferRequest::WaitMode::RESULT_READY));
}

const std::vector<ngraph::element::Type_t> element_type{ngraph::element::Type_t::f16,
                                                        ngraph::element::Type_t::f32,
                                                        ngraph::element::Type_t::f64,
                                                        ngraph::element::Type_t::i8,
                                                        ngraph::element::Type_t::i16,
                                                        ngraph::element::Type_t::i32,
                                                        ngraph::element::Type_t::i64,
                                                        ngraph::element::Type_t::u8,
                                                        ngraph::element::Type_t::u16,
                                                        ngraph::element::Type_t::u32,
                                                        ngraph::element::Type_t::u64};
const std::vector<int> batch_size{1, 8, 16, 32, 64, 128};
const std::vector<int> infer_interval{0};
const std::vector<int> infer_interval_timeout{0, 10};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         AutoBatchRequestTest,
                         ::testing::Combine(::testing::ValuesIn(batch_size),
                                            ::testing::ValuesIn(element_type),
                                            ::testing::ValuesIn(infer_interval)),
                         AutoBatchRequestTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         AutoBatchAsyncInferRequestTest,
                         ::testing::Combine(::testing::ValuesIn(batch_size),
                                            ::testing::ValuesIn(element_type),
                                            ::testing::ValuesIn(infer_interval_timeout)),
                         AutoBatchAsyncInferRequestTest::getTestCaseName);