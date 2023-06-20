// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "compiled_model.hpp"

#include "async_infer_request.hpp"
#include "ie_performance_hints.hpp"
#include "sync_infer_request.hpp"

namespace AutoBatchPlugin {
using namespace InferenceEngine;
AutoBatchExecutableNetwork::AutoBatchExecutableNetwork(
    const InferenceEngine::SoExecutableNetworkInternal& networkWithBatch,
    const InferenceEngine::SoExecutableNetworkInternal& networkWithoutBatch,
    const DeviceInformation& networkDevice,
    const std::unordered_map<std::string, InferenceEngine::Parameter>& config,
    const std::set<std::string>& batchedInputs,
    const std::set<std::string>& batchedOutputs)
    : InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr,
                                                          std::make_shared<InferenceEngine::ImmediateExecutor>()),
      _network{networkWithBatch},
      _networkWithoutBatch{networkWithoutBatch},
      _config{config},
      _batchedInputs(batchedInputs),
      _batchedOutputs(batchedOutputs) {
    // WA for gcc 4.8 ( fails compilation with member init-list)
    _device = networkDevice;
    auto time_out = config.find(CONFIG_KEY(AUTO_BATCH_TIMEOUT));
    IE_ASSERT(time_out != config.end());
    _timeOut = ParseTimeoutValue(time_out->second.as<std::string>());
}

AutoBatchExecutableNetwork::~AutoBatchExecutableNetwork() {
    _terminate = true;
    for (const auto& w : _workerRequests) {
        w->_thread.join();
    }
    _workerRequests.clear();
}

unsigned int AutoBatchExecutableNetwork::ParseTimeoutValue(const std::string& s) {
    auto val = std::stoi(s);
    if (val < 0)
        IE_THROW(ParameterMismatch) << "Value for the " << CONFIG_KEY(AUTO_BATCH_TIMEOUT) << " should be unsigned int";
    return val;
}

std::shared_ptr<InferenceEngine::RemoteContext> AutoBatchExecutableNetwork::GetContext() const {
    return _networkWithoutBatch->GetContext();
}

InferenceEngine::IInferRequestInternal::Ptr AutoBatchExecutableNetwork::CreateInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs,
    InferenceEngine::OutputsDataMap networkOutputs) {
    auto workerRequestPtrAndId = GetWorkerInferRequest();
    return std::make_shared<AutoBatchInferRequest>(networkInputs,
                                                   networkOutputs,
                                                   workerRequestPtrAndId.first,
                                                   workerRequestPtrAndId.second,
                                                   _device.batchForDevice,
                                                   _batchedInputs,
                                                   _batchedOutputs);
}

InferenceEngine::IInferRequestInternal::Ptr AutoBatchExecutableNetwork::CreateInferRequestImpl(
    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    if (!this->_plugin || !_plugin->IsNewAPI())
        return nullptr;
    auto workerRequestPtrAndId = GetWorkerInferRequest();
    return std::make_shared<AutoBatchInferRequest>(inputs,
                                                   outputs,
                                                   workerRequestPtrAndId.first,
                                                   workerRequestPtrAndId.second,
                                                   _device.batchForDevice,
                                                   _batchedInputs,
                                                   _batchedOutputs);
}

std::pair<AutoBatchExecutableNetwork::WorkerInferRequest&, int> AutoBatchExecutableNetwork::GetWorkerInferRequest() {
    auto num = _numRequestsCreated++;
    std::lock_guard<std::mutex> lock(_workerRequestsMutex);
    auto batch_id = num % _device.batchForDevice;
    if (!batch_id) {  // need new request
        _workerRequests.push_back(std::make_shared<WorkerInferRequest>());
        auto workerRequestPtr = _workerRequests.back().get();
        workerRequestPtr->_inferRequestBatched = {_network->CreateInferRequest(), _network._so};
        workerRequestPtr->_batchSize = _device.batchForDevice;
        workerRequestPtr->_completionTasks.resize(workerRequestPtr->_batchSize);
        workerRequestPtr->_inferRequestBatched->SetCallback(
            [workerRequestPtr](std::exception_ptr exceptionPtr) mutable {
                if (exceptionPtr)
                    workerRequestPtr->_exceptionPtr = exceptionPtr;
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
                    status = workerRequestPtr->_cond.wait_for(lock, std::chrono::milliseconds(_timeOut));
                }
                if (_terminate) {
                    break;
                } else {
                    // as we pop the tasks from the queue only here
                    // it is ok to call size() (as the _tasks can only grow in parallel)
                    const int sz = static_cast<int>(workerRequestPtr->_tasks.size());
                    if (sz == workerRequestPtr->_batchSize) {
                        std::pair<AutoBatchAsyncInferRequest*, InferenceEngine::Task> t;
                        for (int n = 0; n < sz; n++) {
                            IE_ASSERT(workerRequestPtr->_tasks.try_pop(t));
                            workerRequestPtr->_completionTasks[n] = std::move(t.second);
                            t.first->_inferRequest->CopyInputsIfNeeded();
                            t.first->_inferRequest->_wasBatchedRequestUsed =
                                AutoBatchInferRequest::eExecutionFlavor::BATCH_EXECUTED;
                        }
                        workerRequestPtr->_inferRequestBatched->StartAsync();
                    } else if ((status == std::cv_status::timeout) && sz) {
                        // timeout to collect the batch is over, have to execute the requests in the batch1 mode
                        std::pair<AutoBatchAsyncInferRequest*, InferenceEngine::Task> t;
                        // popping all tasks collected by the moment of the time-out and execute each with batch1
                        std::atomic<int> arrived = {0};
                        std::promise<void> all_completed;
                        auto all_completed_future = all_completed.get_future();
                        for (int n = 0; n < sz; n++) {
                            IE_ASSERT(workerRequestPtr->_tasks.try_pop(t));
                            t.first->_inferRequestWithoutBatch->SetCallback(
                                [t, sz, &arrived, &all_completed](std::exception_ptr p) {
                                    if (p)
                                        t.first->_inferRequest->_exceptionPtr = p;
                                    t.second();
                                    if (sz == ++arrived)
                                        all_completed.set_value();
                                });
                            t.first->_inferRequest->_wasBatchedRequestUsed =
                                AutoBatchInferRequest::eExecutionFlavor::TIMEOUT_EXECUTED;
                            t.first->_inferRequest->SetBlobsToAnotherRequest(t.first->_inferRequestWithoutBatch);
                            t.first->_inferRequestWithoutBatch->StartAsync();
                        }
                        all_completed_future.get();
                        // now when all the tasks for this batch are completed, start waiting for the timeout again
                    }
                }
            }
        });
    }
    return {*_workerRequests.back(), static_cast<int>(batch_id)};
}

InferenceEngine::IInferRequestInternal::Ptr AutoBatchExecutableNetwork::CreateInferRequest() {
    if (!_network) {
        auto res = _networkWithoutBatch->CreateInferRequest();
        res->setPointerToExecutableNetworkInternal(shared_from_this());
        res->setPointerToSo(_networkWithoutBatch._so);
        _so = _networkWithoutBatch._so;
        return res;
    }
    // trying to create the new API request first
    IInferRequestInternal::Ptr syncRequestImpl = CreateInferRequestImpl(_parameters, _results);
    if (!syncRequestImpl)
        syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    InferenceEngine::SoIInferRequestInternal inferRequestWithoutBatch = {_networkWithoutBatch->CreateInferRequest(),
                                                                         _networkWithoutBatch._so};
    return std::make_shared<AutoBatchAsyncInferRequest>(
        std::static_pointer_cast<AutoBatchInferRequest>(syncRequestImpl),
        inferRequestWithoutBatch,
        _callbackExecutor);
}

std::shared_ptr<ngraph::Function> AutoBatchExecutableNetwork::GetExecGraphInfo() {
    return _network && _network->GetExecGraphInfo() ? _network->GetExecGraphInfo()
                                                    : _networkWithoutBatch->GetExecGraphInfo();
}

void AutoBatchExecutableNetwork::SetConfig(const std::map<std::string, InferenceEngine::Parameter>& user_config) {
    auto timeout = user_config.find(CONFIG_KEY(AUTO_BATCH_TIMEOUT));
    if (timeout == user_config.end() || user_config.size() > 1) {
        IE_THROW() << "The only config that can be changed on the fly for the AutoBatching the is the "
                   << CONFIG_KEY(AUTO_BATCH_TIMEOUT);
    } else {
        _timeOut = ParseTimeoutValue(timeout->second.as<std::string>());
    }
}

InferenceEngine::Parameter AutoBatchExecutableNetwork::GetConfig(const std::string& name) const {
    auto it = _config.find(name);
    if (it != _config.end()) {
        return it->second;
    } else {
        // find config key among networks config keys
        auto param = _networkWithoutBatch->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        for (auto&& configKey : param.as<std::vector<std::string>>()) {
            if (configKey == name) {
                return _networkWithoutBatch->GetConfig(configKey);
            }
        }
        IE_THROW(NotFound) << name << " not found in the ExecutableNetwork config";
    }
}

InferenceEngine::Parameter AutoBatchExecutableNetwork::GetMetric(const std::string& name) const {
    if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        auto reqs = 0;
        try {
            auto hint = _networkWithoutBatch->GetConfig(CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS)).as<std::string>();
            reqs = InferenceEngine::PerfHintsConfig::CheckPerformanceHintRequestValue(hint);
            if (!reqs)  // no limitations from user, let's deduce the full blown #requests
                // (multiplied by the devices capabilities to run multiple <batched> requests for further perf)
                reqs = _device.batchForDevice *
                       _networkWithoutBatch->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
        } catch (const InferenceEngine::Exception&) {
        }
        reqs = std::max(reqs, _device.batchForDevice);  // round up to the possible  user's value
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, reqs);
    } else if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_SET_METRIC_RETURN(NETWORK_NAME, _networkWithoutBatch->GetMetric(METRIC_KEY(NETWORK_NAME)).as<std::string>());
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
        return _networkWithoutBatch->GetMetric(name);
    } else {
        IE_THROW() << "Unsupported Network metric: " << name;
    }
}
}  // namespace AutoBatchPlugin