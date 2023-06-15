// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "auto_batch_compiled_model.hpp"

#include "auto_batch_async_infer_request.hpp"

ov::autobatch_plugin::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                                   const std::shared_ptr<const ov::IPlugin>& plugin,
                                                   const ov::AnyMap& config,
                                                   const DeviceInformation& deviceInfo,
                                                   const std::set<std::string>& batchedInputs,
                                                   const std::set<std::string>& batchedOutputs,
                                                   const ov::SoPtr<ov::ICompiledModel>& compiledModelWithBatch,
                                                   const ov::SoPtr<ov::ICompiledModel>& compiledModelWithoutBatch)
    : ov::ICompiledModel(model, plugin),
      m_config(config),
      m_batchedInputs(batchedInputs),
      m_batchedOutputs(batchedOutputs),
      m_compiledModelWithBatch(compiledModelWithBatch),
      m_compiledModelWithoutBatch(compiledModelWithoutBatch) {
    // WA for gcc 4.8 ( fails compilation with member init-list)
    m_deviceInfo = deviceInfo;
    auto time_out = config.find(ov::auto_batch_timeout.name());
    OPENVINO_ASSERT(time_out != config.end());
    m_timeOut = time_out->second.as<std::uint32_t>();
}

void ov::autobatch_plugin::CompiledModel::set_property(const ov::AnyMap& properties) {
    auto time_out = properties.find(ov::auto_batch_timeout.name());
    if (time_out == properties.end() || properties.size() > 1) {
        OPENVINO_THROW("The only config that can be changed on the fly for the AutoBatching is the ",
                       ov::auto_batch_timeout.name());
    } else {
        m_timeOut = time_out->second.as<std::uint32_t>();
    }
}

ov::Any ov::autobatch_plugin::CompiledModel::get_property(const std::string& name) const {
    auto it = m_config.find(name);
    if (it != m_config.end()) {
        return it->second;
    } else {
        // find config key among networks config keys
        auto modelSupportedProperties = m_compiledModelWithoutBatch->get_property(ov::supported_properties.name());
        for (auto&& property : modelSupportedProperties.as<std::vector<ov::PropertyName>>()) {
            if (property == name) {
                return m_compiledModelWithoutBatch->get_property(property);
            }
        }
        if (name == ov::optimal_number_of_infer_requests.name()) {
            uint32_t numRequest = 0;
            try {
                numRequest =
                    m_compiledModelWithoutBatch->get_property(ov::hint::num_requests.name()).as<std::uint32_t>();
                if (numRequest == 0)  // no limitations from user, let's deduce the full blown #requests
                    // (multiplied by the devices capabilities to run multiple <batched> requests for further perf)
                    numRequest = m_deviceInfo.batchForDevice *
                                 m_compiledModelWithoutBatch->get_property(ov::optimal_number_of_infer_requests.name())
                                     .as<uint32_t>();
            } catch (const InferenceEngine::Exception&) {
            }
            numRequest = std::max(numRequest, m_deviceInfo.batchForDevice);  // round up to the possible  user's value
            return ov::Any(numRequest);
        } else if (name == ov::model_name.name()) {
            return m_compiledModelWithoutBatch->get_property(name);
        } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
            return std::vector<std::string>{ov::optimal_number_of_infer_requests.name(),
                                            METRIC_KEY(SUPPORTED_METRICS),
                                            ov::model_name.name(),
                                            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
                                            ov::execution_devices.name()};
        } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
            return std::vector<std::string>{ov::auto_batch_timeout.name()};
        } else if (name == ov::execution_devices) {
            return m_compiledModelWithoutBatch->get_property(name);
        } else {
            OPENVINO_THROW("Unsupported Compiled Model Property: ", name);
        }
    }
}

std::shared_ptr<ov::IRemoteContext> ov::autobatch_plugin::CompiledModel::get_context() const {
    return m_compiledModelWithoutBatch->get_context();
}

std::shared_ptr<const ov::Model> ov::autobatch_plugin::CompiledModel::get_runtime_model() const {
    return m_compiledModelWithBatch && m_compiledModelWithBatch->get_runtime_model()
               ? m_compiledModelWithBatch->get_runtime_model()
               : m_compiledModelWithoutBatch->get_runtime_model();
}

void ov::autobatch_plugin::CompiledModel::export_model(std::ostream& model) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::IAsyncInferRequest> ov::autobatch_plugin::CompiledModel::create_infer_request() const {
    if (!m_compiledModelWithBatch) {
        return m_compiledModelWithoutBatch->create_infer_request();
    }

    auto sync_res = create_sync_infer_request();
    return std::make_shared<ov::autobatch_plugin::AsyncInferRequest>(
        std::static_pointer_cast<ov::autobatch_plugin::SyncInferRequest>(sync_res),
        m_compiledModelWithoutBatch->create_infer_request(),
        get_callback_executor());
}

std::shared_ptr<ov::ISyncInferRequest> ov::autobatch_plugin::CompiledModel::create_sync_infer_request() const {
    if (!get_plugin())
        return nullptr;
    auto workerRequestPtrAndId = GetWorkerInferRequest();
    auto async_infer_request = std::make_shared<ov::autobatch_plugin::SyncInferRequest>(
        std::static_pointer_cast<const ov::autobatch_plugin::CompiledModel>(shared_from_this()),
        workerRequestPtrAndId.first,
        workerRequestPtrAndId.second,
        m_deviceInfo.batchForDevice,
        m_batchedInputs,
        m_batchedOutputs);
    return async_infer_request;
}

ov::autobatch_plugin::CompiledModel::~CompiledModel() {
    _terminate = true;
    for (const auto& w : _workerRequests) {
        w->_thread.join();
    }
    _workerRequests.clear();
}

std::pair<std::shared_ptr<ov::autobatch_plugin::CompiledModel::WorkerInferRequest>, int>
ov::autobatch_plugin::CompiledModel::GetWorkerInferRequest() const {
    auto num = _numRequestsCreated++;
    std::lock_guard<std::mutex> lock(_workerRequestsMutex);
    auto batch_id = num % m_deviceInfo.batchForDevice;
    if (!batch_id) {  // need new request
        _workerRequests.push_back(std::make_shared<WorkerInferRequest>());
        auto workerRequestPtr = _workerRequests.back().get();
        workerRequestPtr->_inferRequestBatched = {m_compiledModelWithBatch->create_infer_request(), m_compiledModelWithBatch._so};
        workerRequestPtr->_batchSize = m_deviceInfo.batchForDevice;
        workerRequestPtr->_completionTasks.resize(workerRequestPtr->_batchSize);
        workerRequestPtr->_inferRequestBatched->set_callback(
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
                    status = workerRequestPtr->_cond.wait_for(lock, std::chrono::milliseconds(m_timeOut));
                }
                if (_terminate) {
                    break;
                } else {
                    // as we pop the tasks from the queue only here
                    // it is ok to call size() (as the _tasks can only grow in parallel)
                    const int sz = static_cast<int>(workerRequestPtr->_tasks.size());
                    if (sz == workerRequestPtr->_batchSize) {
                        std::pair<ov::autobatch_plugin::AsyncInferRequest*, ov::threading::Task> t;
                        for (int n = 0; n < sz; n++) {
                            IE_ASSERT(workerRequestPtr->_tasks.try_pop(t));
                            workerRequestPtr->_completionTasks[n] = std::move(t.second);
                            t.first->m_sync_request->CopyInputsIfNeeded();
                            t.first->m_sync_request->_wasBatchedRequestUsed =
                                ov::autobatch_plugin::SyncInferRequest::eExecutionFlavor::BATCH_EXECUTED;
                        }
                        workerRequestPtr->_inferRequestBatched->start_async();
                    } else if ((status == std::cv_status::timeout) && sz) {
                        // timeout to collect the batch is over, have to execute the requests in the batch1 mode
                        std::pair<ov::autobatch_plugin::AsyncInferRequest*, ov::threading::Task> t;
                        // popping all tasks collected by the moment of the time-out and execute each with batch1
                        std::atomic<int> arrived = {0};
                        std::promise<void> all_completed;
                        auto all_completed_future = all_completed.get_future();
                        for (int n = 0; n < sz; n++) {
                            IE_ASSERT(workerRequestPtr->_tasks.try_pop(t));
                            t.first->m_inferRequestWithoutBatch->set_callback(
                                [t, sz, &arrived, &all_completed](std::exception_ptr p) {
                                    if (p)
                                        t.first->m_sync_request->_exceptionPtr = p;
                                    t.second();
                                    if (sz == ++arrived) {
                                        all_completed.set_value();
                                    }
                                });
                            t.first->m_sync_request->_wasBatchedRequestUsed =
                                ov::autobatch_plugin::SyncInferRequest::eExecutionFlavor::TIMEOUT_EXECUTED;
                            t.first->m_sync_request->SetBlobsToAnotherRequest(t.first->m_inferRequestWithoutBatch);
                            t.first->m_inferRequestWithoutBatch->start_async();
                        }
                        all_completed_future.get();
                        // now when all the tasks for this batch are completed, start waiting for the timeout again
                    }
                }
            }
        });
    }
    return {_workerRequests.back(), static_cast<int>(batch_id)};
}
