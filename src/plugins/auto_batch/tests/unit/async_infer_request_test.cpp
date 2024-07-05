// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.hpp"

#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "mock_common.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/threading/immediate_executor.hpp"
#include "transformations/utils/utils.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

using AutoBatchRequestTestParams = std::tuple<uint32_t,             // batch_size
                                              ov::element::Type_t,  // data type
                                              uint32_t>;            // inference interval

class AutoBatchAsyncInferRequestTest : public ::testing::TestWithParam<AutoBatchRequestTestParams> {
public:
    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<ov::Model> m_batched_model;
    std::shared_ptr<NiceMock<ov::MockICore>> m_core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> m_auto_batch_plugin;

    std::shared_ptr<NiceMock<MockIPlugin>> m_hardware_plugin;

    std::shared_ptr<NiceMock<MockICompiledModel>> m_i_compile_model_without_batch;
    ov::SoPtr<ov::ICompiledModel> m_compile_model_without_batch;

    std::shared_ptr<NiceMock<MockICompiledModel>> m_i_compile_model_with_batch;
    ov::SoPtr<ov::ICompiledModel> m_compile_model_with_batch;

    ov::AnyMap m_config;
    DeviceInformation m_device_info;
    std::set<std::size_t> m_batched_inputs;
    std::set<std::size_t> m_batched_outputs;
    ov::SoPtr<ov::IRemoteContext> m_remote_context;

    std::shared_ptr<CompiledModel> m_auto_batch_compile_model;

    std::shared_ptr<NiceMock<MockISyncInferRequest>> m_sync_infer_request_with_batch;

    std::shared_ptr<NiceMock<MockIAsyncInferRequest>> m_async_infer_request_with_batch;

    std::shared_ptr<NiceMock<MockISyncInferRequest>> m_sync_infer_request_without_batch;

    std::shared_ptr<NiceMock<MockIAsyncInferRequest>> m_async_infer_request_without_batch;

    std::shared_ptr<ov::threading::ImmediateExecutor> m_executor;

    std::shared_ptr<CompiledModel::WorkerInferRequest> workerRequestPtr;

    uint32_t m_batch_size;
    ov::element::Type_t m_element_type;
    uint32_t m_infer_interval;

    std::vector<std::shared_ptr<AsyncInferRequest>> m_auto_batch_async_infer_requests;

    std::vector<ov::ProfilingInfo> m_profiling_info;

    bool m_terminate;

    static std::string getTestCaseName(testing::TestParamInfo<AutoBatchRequestTestParams> obj) {
        uint32_t batch_size, infer_interval;
        ov::element::Type_t element_type;
        std::tie(batch_size, element_type, infer_interval) = obj.param;

        std::string res;
        res = "batch_size_" + std::to_string(batch_size);
        res += "_element_type_" + std::to_string(static_cast<int>(element_type));
        if (infer_interval > 0)
            res += "_infer_interval_" + std::to_string(infer_interval);
        return res;
    }

    void TearDown() override {
        m_terminate = true;
        m_profiling_info.clear();
        m_auto_batch_async_infer_requests.clear();
        m_auto_batch_plugin.reset();
        m_model.reset();
        m_batched_model.reset();
        m_core.reset();
        m_i_compile_model_without_batch.reset();
        m_compile_model_without_batch = {};
        m_i_compile_model_with_batch.reset();
        m_compile_model_with_batch = {};
        m_auto_batch_compile_model.reset();
        m_sync_infer_request_without_batch.reset();
        m_async_infer_request_without_batch.reset();
        m_executor.reset();
        clear_worker();
        workerRequestPtr.reset();
        m_sync_infer_request_with_batch.reset();
        m_async_infer_request_with_batch.reset();
    }

    void SetUp() override {
        std::tie(m_batch_size, m_element_type, m_infer_interval) = this->GetParam();
        m_terminate = false;
        std::vector<size_t> inputShape = {1, 3, 24, 24};
        m_model = ov::test::utils::make_multi_single_conv(inputShape, m_element_type);

        prepare_input(m_model, m_batch_size);

        m_core = std::shared_ptr<NiceMock<ov::MockICore>>(new NiceMock<ov::MockICore>());

        m_auto_batch_plugin =
            std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());

        m_hardware_plugin = std::shared_ptr<NiceMock<MockIPlugin>>(new NiceMock<MockIPlugin>());

        m_auto_batch_plugin->set_core(m_core);
        m_i_compile_model_without_batch = std::make_shared<NiceMock<MockICompiledModel>>(m_model, m_hardware_plugin);
        m_compile_model_without_batch = {m_i_compile_model_without_batch, {}};

        m_config = {{ov::auto_batch_timeout.name(), "200"}};

        m_device_info = {"CPU", {}, m_batch_size};

        auto reshaped = m_model->clone();
        auto inputs = reshaped->inputs();
        std::map<std::size_t, ov::PartialShape> partial_shapes;
        for (size_t input_id = 0; input_id < inputs.size(); input_id++) {
            auto input_shape = inputs[input_id].get_shape();
            if (m_batched_inputs.find(input_id) != m_batched_inputs.end()) {
                input_shape[0] = m_batch_size;
            }
            partial_shapes.insert({input_id, ov::PartialShape(input_shape)});
        }

        reshaped->reshape(partial_shapes);

        m_i_compile_model_with_batch = std::make_shared<NiceMock<MockICompiledModel>>(reshaped, m_hardware_plugin);
        m_compile_model_with_batch = {m_i_compile_model_with_batch, {}};

        OV_ASSERT_NO_THROW(m_auto_batch_compile_model = std::make_shared<CompiledModel>(m_model->clone(),
                                                                                     m_auto_batch_plugin,
                                                                                     m_config,
                                                                                     m_device_info,
                                                                                     m_batched_inputs,
                                                                                     m_batched_outputs,
                                                                                     m_compile_model_with_batch,
                                                                                     m_compile_model_without_batch,
                                                                                     m_remote_context));

        m_sync_infer_request_with_batch =
            std::make_shared<NiceMock<MockISyncInferRequest>>(m_i_compile_model_with_batch);

        m_executor = std::make_shared<ov::threading::ImmediateExecutor>();

        m_async_infer_request_with_batch =
            std::make_shared<NiceMock<MockIAsyncInferRequest>>(m_sync_infer_request_with_batch, m_executor, nullptr);

        m_sync_infer_request_without_batch =
            std::make_shared<NiceMock<MockISyncInferRequest>>(m_i_compile_model_without_batch);

        m_async_infer_request_without_batch =
            std::make_shared<NiceMock<MockIAsyncInferRequest>>(m_sync_infer_request_without_batch, m_executor, nullptr);

        m_profiling_info = {};
    }

    void create_worker(int batch_size) {
        workerRequestPtr = std::make_shared<CompiledModel::WorkerInferRequest>();

        workerRequestPtr->_infer_request_batched = {m_async_infer_request_with_batch, {}};
        workerRequestPtr->_batch_size = batch_size;
        workerRequestPtr->_completion_tasks.resize(workerRequestPtr->_batch_size);
        workerRequestPtr->_infer_request_batched->set_callback([this](std::exception_ptr exceptionPtr) mutable {
            if (exceptionPtr)
                workerRequestPtr->_exception_ptr = exceptionPtr;
        });

        ON_CALL(*m_async_infer_request_with_batch, start_async()).WillByDefault([this]() {
            OPENVINO_ASSERT(workerRequestPtr->_completion_tasks.size() == (size_t)workerRequestPtr->_batch_size);
            for (int c = 0; c < workerRequestPtr->_batch_size; c++) {
                workerRequestPtr->_completion_tasks[c]();
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
                if (m_terminate) {
                    break;
                } else {
                    // as we pop the tasks from the queue only here
                    // it is ok to call size() (as the _tasks can only grow in parallel)
                    const int sz = static_cast<int>(workerRequestPtr->_tasks.size());
                    if (sz == workerRequestPtr->_batch_size) {
                        std::pair<AsyncInferRequest*, ov::threading::Task> t;
                        for (int n = 0; n < sz; n++) {
                            OPENVINO_ASSERT(workerRequestPtr->_tasks.try_pop(t));
                            workerRequestPtr->_completion_tasks[n] = std::move(t.second);
                            t.first->m_sync_request->copy_inputs_if_needed();
                            t.first->m_sync_request->m_batched_request_status =
                                SyncInferRequest::eExecutionFlavor::BATCH_EXECUTED;
                        }
                        workerRequestPtr->_infer_request_batched->start_async();
                    } else if ((status == std::cv_status::timeout) && sz) {
                        std::pair<AsyncInferRequest*, ov::threading::Task> t;
                        for (int n = 0; n < sz; n++) {
                            OPENVINO_ASSERT(workerRequestPtr->_tasks.try_pop(t));
                            t.first->m_sync_request->m_batched_request_status =
                                SyncInferRequest::eExecutionFlavor::TIMEOUT_EXECUTED;
                            t.first->m_request_without_batch->start_async();
                            t.second();
                        }
                    }
                }
            }
        });
        return;
    }

    void clear_worker() {
        workerRequestPtr->_infer_request_batched = {};
        workerRequestPtr->_completion_tasks.clear();
        workerRequestPtr->_thread.join();
    }

    void prepare_input(std::shared_ptr<ov::Model>& model, int batch_size) {
        const auto& params = model->get_parameters();
        for (size_t input_id = 0; input_id < params.size(); input_id++) {
            m_batched_inputs.insert(input_id);
        }
        const auto& results = model->get_results();
        for (size_t output_id = 0; output_id < results.size(); output_id++) {
            m_batched_outputs.insert(output_id);
        }
    }
};

TEST_P(AutoBatchAsyncInferRequestTest, AutoBatchRequestCreateTestCase) {
    prepare_input(m_model, m_batch_size);
    create_worker(m_batch_size);

    for (uint32_t batch_id = 0; batch_id < m_batch_size; batch_id++) {
        auto req = std::make_shared<SyncInferRequest>(m_auto_batch_compile_model,
                                                      workerRequestPtr,
                                                      batch_id,
                                                      m_batch_size,
                                                      m_batched_inputs,
                                                      m_batched_outputs);
        EXPECT_NE(req, nullptr);

        auto asyncInferRequest = std::make_shared<AsyncInferRequest>(req, m_async_infer_request_without_batch, nullptr);
        EXPECT_NE(asyncInferRequest, nullptr);
        m_auto_batch_async_infer_requests.emplace_back(asyncInferRequest);
    }
}

TEST_P(AutoBatchAsyncInferRequestTest, AutoBatchAsyncInferRequestStartAsyncTest) {
    prepare_input(m_model, m_batch_size);
    create_worker(m_batch_size);

    for (uint32_t batch_id = 0; batch_id < m_batch_size; batch_id++) {
        auto req = std::make_shared<SyncInferRequest>(m_auto_batch_compile_model,
                                                      workerRequestPtr,
                                                      batch_id,
                                                      m_batch_size,
                                                      m_batched_inputs,
                                                      m_batched_outputs);
        EXPECT_NE(req, nullptr);

        auto asyncInferRequest = std::make_shared<AsyncInferRequest>(req, m_async_infer_request_without_batch, nullptr);
        EXPECT_NE(asyncInferRequest, nullptr);
        m_auto_batch_async_infer_requests.emplace_back(asyncInferRequest);
    }

    for (auto& req : m_auto_batch_async_infer_requests) {
        if (m_infer_interval > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(m_infer_interval));
        EXPECT_NO_THROW(req->start_async());
    }

    for (auto& req : m_auto_batch_async_infer_requests) {
        EXPECT_NO_THROW(req->wait());
    }
}

std::vector<ov::element::Type_t> element_type_param{ov::element::Type_t::f16,
                                                    ov::element::Type_t::f32,
                                                    ov::element::Type_t::f64,
                                                    ov::element::Type_t::i8,
                                                    ov::element::Type_t::i16,
                                                    ov::element::Type_t::i32,
                                                    ov::element::Type_t::i64,
                                                    ov::element::Type_t::u8,
                                                    ov::element::Type_t::u16,
                                                    ov::element::Type_t::u32,
                                                    ov::element::Type_t::u64};
const std::vector<uint32_t> batch_size_param{1, 8, 16, 32, 64, 128};
const std::vector<uint32_t> infer_interval_timeout_param{0, 10};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         AutoBatchAsyncInferRequestTest,
                         ::testing::Combine(::testing::ValuesIn(batch_size_param),
                                            ::testing::ValuesIn(element_type_param),
                                            ::testing::ValuesIn(infer_interval_timeout_param)),
                         AutoBatchAsyncInferRequestTest::getTestCaseName);
