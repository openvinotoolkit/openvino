// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "compiled_model.hpp"

#include "async_infer_request.hpp"
#include "ie_performance_hints.hpp"
#include "sync_infer_request.hpp"

namespace ov {
namespace autobatch_plugin {
CompiledModel::CompiledModel(const InferenceEngine::SoExecutableNetworkInternal& networkWithBatch,
                             const InferenceEngine::SoExecutableNetworkInternal& networkWithoutBatch,
                             const DeviceInformation& networkDevice,
                             const std::unordered_map<std::string, InferenceEngine::Parameter>& config,
                             const std::set<std::string>& batchedInputs,
                             const std::set<std::string>& batchedOutputs)
    : InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr,
                                                          std::make_shared<InferenceEngine::ImmediateExecutor>()),
      m_model_with_batch{networkWithBatch},
      m_model_without_batch{networkWithoutBatch},
      m_config{config},
      m_batched_inputs(batchedInputs),
      m_batched_outputs(batchedOutputs) {
    // WA for gcc 4.8 ( fails compilation with member init-list)
    m_device_info = networkDevice;
    auto time_out = config.find(CONFIG_KEY(AUTO_BATCH_TIMEOUT));
    IE_ASSERT(time_out != config.end());
    m_timeout = ParseTimeoutValue(time_out->second.as<std::string>());
}

CompiledModel::~CompiledModel() {
    m_terminate = true;
    for (const auto& w : m_worker_requests) {
        w->_thread.join();
    }
    m_worker_requests.clear();
}

unsigned int CompiledModel::ParseTimeoutValue(const std::string& s) {
    auto val = std::stoi(s);
    if (val < 0)
        IE_THROW(ParameterMismatch) << "Value for the " << CONFIG_KEY(AUTO_BATCH_TIMEOUT) << " should be unsigned int";
    return val;
}

std::shared_ptr<InferenceEngine::RemoteContext> CompiledModel::GetContext() const {
    return m_model_without_batch->GetContext();
}

InferenceEngine::IInferRequestInternal::Ptr CompiledModel::CreateInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs,
    InferenceEngine::OutputsDataMap networkOutputs) {
    auto workerRequestPtrAndId = GetWorkerInferRequest();
    return std::make_shared<SyncInferRequest>(networkInputs,
                                              networkOutputs,
                                              workerRequestPtrAndId.first,
                                              workerRequestPtrAndId.second,
                                              m_device_info.batch_for_device,
                                              m_batched_inputs,
                                              m_batched_outputs);
}

InferenceEngine::IInferRequestInternal::Ptr CompiledModel::CreateInferRequestImpl(
    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    if (!this->_plugin || !_plugin->IsNewAPI())
        return nullptr;
    auto workerRequestPtrAndId = GetWorkerInferRequest();
    return std::make_shared<SyncInferRequest>(inputs,
                                              outputs,
                                              workerRequestPtrAndId.first,
                                              workerRequestPtrAndId.second,
                                              m_device_info.batch_for_device,
                                              m_batched_inputs,
                                              m_batched_outputs);
}

std::pair<CompiledModel::WorkerInferRequest&, int> CompiledModel::GetWorkerInferRequest() {
    auto num = m_num_requests_created++;
    std::lock_guard<std::mutex> lock(m_worker_requests_mutex);
    auto batch_id = num % m_device_info.batch_for_device;
    if (!batch_id) {  // need new request
        m_worker_requests.push_back(std::make_shared<WorkerInferRequest>());
        auto workerRequestPtr = m_worker_requests.back().get();
        workerRequestPtr->_inferRequestBatched = {m_model_with_batch->CreateInferRequest(), m_model_with_batch._so};
        workerRequestPtr->_batchSize = m_device_info.batch_for_device;
        workerRequestPtr->_completionTasks.resize(workerRequestPtr->_batchSize);
        workerRequestPtr->_inferRequestBatched->SetCallback(
            [workerRequestPtr](std::exception_ptr exceptionPtr) mutable {
                if (exceptionPtr)
                    workerRequestPtr->m_exceptionPtr = exceptionPtr;
                IE_ASSERT(workerRequestPtr->_completionTasks.size() == (size_t)workerRequestPtr->_batchSize);
                // notify the individual requests on the completion
                for (int c = 0; c < workerRequestPtr->_batchSize; c++) {
                    workerRequestPtr->_completionTasks[c]();
                }
                // reset the timeout
                workerRequestPtr->_cond.notify_one();
            });

        workerRequestPtr->_thread = std::thread([workerRequestPtr, this] {
            while (1) {
                std::cv_status status;
                {
                    std::unique_lock<std::mutex> lock(workerRequestPtr->_mutex);
                    status = workerRequestPtr->_cond.wait_for(lock, std::chrono::milliseconds(m_timeout));
                }
                if (m_terminate) {
                    break;
                } else {
                    // as we pop the tasks from the queue only here
                    // it is ok to call size() (as the _tasks can only grow in parallel)
                    const int sz = static_cast<int>(workerRequestPtr->_tasks.size());
                    if (sz == workerRequestPtr->_batchSize) {
                        std::pair<AsyncInferRequest*, InferenceEngine::Task> t;
                        for (int n = 0; n < sz; n++) {
                            IE_ASSERT(workerRequestPtr->_tasks.try_pop(t));
                            workerRequestPtr->_completionTasks[n] = std::move(t.second);
                            t.first->m_sync_infer_request->CopyInputsIfNeeded();
                            t.first->m_sync_infer_request->m_batched_request_status =
                                SyncInferRequest::eExecutionFlavor::BATCH_EXECUTED;
                        }
                        workerRequestPtr->_inferRequestBatched->StartAsync();
                    } else if ((status == std::cv_status::timeout) && sz) {
                        // timeout to collect the batch is over, have to execute the requests in the batch1 mode
                        std::pair<AsyncInferRequest*, InferenceEngine::Task> t;
                        // popping all tasks collected by the moment of the time-out and execute each with batch1
                        std::atomic<int> arrived = {0};
                        std::promise<void> all_completed;
                        auto all_completed_future = all_completed.get_future();
                        for (int n = 0; n < sz; n++) {
                            IE_ASSERT(workerRequestPtr->_tasks.try_pop(t));
                            t.first->m_infer_request_without_batch->SetCallback(
                                [t, sz, &arrived, &all_completed](std::exception_ptr p) {
                                    if (p)
                                        t.first->m_sync_infer_request->m_exceptionPtr = p;
                                    t.second();
                                    if (sz == ++arrived)
                                        all_completed.set_value();
                                });
                            t.first->m_sync_infer_request->m_batched_request_status =
                                SyncInferRequest::eExecutionFlavor::TIMEOUT_EXECUTED;
                            t.first->m_sync_infer_request->SetBlobsToAnotherRequest(
                                t.first->m_infer_request_without_batch);
                            t.first->m_infer_request_without_batch->StartAsync();
                        }
                        all_completed_future.get();
                        // now when all the tasks for this batch are completed, start waiting for the timeout again
                    }
                }
            }
        });
    }
    return {*m_worker_requests.back(), static_cast<int>(batch_id)};
}

InferenceEngine::IInferRequestInternal::Ptr CompiledModel::CreateInferRequest() {
    if (!m_model_with_batch) {
        auto res = m_model_without_batch->CreateInferRequest();
        res->setPointerToExecutableNetworkInternal(shared_from_this());
        res->setPointerToSo(m_model_without_batch._so);
        _so = m_model_without_batch._so;
        return res;
    }
    // trying to create the new API request first
    InferenceEngine::IInferRequestInternal::Ptr syncRequestImpl = CreateInferRequestImpl(_parameters, _results);
    if (!syncRequestImpl)
        syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    InferenceEngine::SoIInferRequestInternal inferRequestWithoutBatch = {m_model_without_batch->CreateInferRequest(),
                                                                         m_model_without_batch._so};
    return std::make_shared<AsyncInferRequest>(std::static_pointer_cast<SyncInferRequest>(syncRequestImpl),
                                               inferRequestWithoutBatch,
                                               _callbackExecutor);
}

std::shared_ptr<ngraph::Function> CompiledModel::GetExecGraphInfo() {
    return m_model_with_batch && m_model_with_batch->GetExecGraphInfo() ? m_model_with_batch->GetExecGraphInfo()
                                                                        : m_model_without_batch->GetExecGraphInfo();
}

void CompiledModel::SetConfig(const std::map<std::string, InferenceEngine::Parameter>& user_config) {
    auto timeout = user_config.find(CONFIG_KEY(AUTO_BATCH_TIMEOUT));
    if (timeout == user_config.end() || user_config.size() > 1) {
        IE_THROW() << "The only config that can be changed on the fly for the AutoBatching the is the "
                   << CONFIG_KEY(AUTO_BATCH_TIMEOUT);
    } else {
        m_timeout = ParseTimeoutValue(timeout->second.as<std::string>());
    }
}

InferenceEngine::Parameter CompiledModel::GetConfig(const std::string& name) const {
    auto it = m_config.find(name);
    if (it != m_config.end()) {
        return it->second;
    } else {
        // find config key among networks config keys
        auto param = m_model_without_batch->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        for (auto&& configKey : param.as<std::vector<std::string>>()) {
            if (configKey == name) {
                return m_model_without_batch->GetConfig(configKey);
            }
        }
        IE_THROW(NotFound) << name << " not found in the ExecutableNetwork config";
    }
}

InferenceEngine::Parameter CompiledModel::GetMetric(const std::string& name) const {
    if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        auto reqs = 0;
        try {
            auto hint = m_model_without_batch->GetConfig(CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS)).as<std::string>();
            reqs = InferenceEngine::PerfHintsConfig::CheckPerformanceHintRequestValue(hint);
            if (!reqs)  // no limitations from user, let's deduce the full blown #requests
                // (multiplied by the devices capabilities to run multiple <batched> requests for further perf)
                reqs =
                    m_device_info.batch_for_device *
                    m_model_without_batch->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
        } catch (const InferenceEngine::Exception&) {
        }
        reqs = std::max(reqs, m_device_info.batch_for_device);  // round up to the possible  user's value
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, reqs);
    } else if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_SET_METRIC_RETURN(NETWORK_NAME,
                             m_model_without_batch->GetMetric(METRIC_KEY(NETWORK_NAME)).as<std::string>());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS,
                             {METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
                              METRIC_KEY(SUPPORTED_METRICS),
                              METRIC_KEY(NETWORK_NAME),
                              METRIC_KEY(SUPPORTED_CONFIG_KEYS),
                              ov::execution_devices.name()});
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS,
                             {CONFIG_KEY(AUTO_BATCH_TIMEOUT)});  // only timeout can be changed on the fly
    } else if (name == ov::execution_devices) {
        return m_model_without_batch->GetMetric(name);
    } else {
        IE_THROW() << "Unsupported Network metric: " << name;
    }
}

}  // namespace autobatch_plugin
}  // namespace ov
