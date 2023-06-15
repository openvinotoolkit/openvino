// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "compiled_model.hpp"

#include "async_infer_request.hpp"

ov::autobatch_plugin::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                                   const std::shared_ptr<const ov::IPlugin>& plugin,
                                                   const ov::AnyMap& config,
                                                   const DeviceInformation& device_info,
                                                   const std::set<std::string>& batched_inputs,
                                                   const std::set<std::string>& batched_outputs,
                                                   const ov::SoPtr<ov::ICompiledModel>& compiled_Model_with_batch,
                                                   const ov::SoPtr<ov::ICompiledModel>& compiled_Model_Without_batch)
    : ov::ICompiledModel(model, plugin),
      m_config(config),
      m_batched_inputs(batched_inputs),
      m_batched_outputs(batched_outputs),
      m_compiled_model_with_batch(compiled_Model_with_batch),
      m_compiled_model_without_batch(compiled_Model_Without_batch) {
    // WA for gcc 4.8 ( fails compilation with member init-list)
    m_device_info = device_info;
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
        auto modelSupportedProperties = m_compiled_model_without_batch->get_property(ov::supported_properties.name());
        for (auto&& property : modelSupportedProperties.as<std::vector<ov::PropertyName>>()) {
            if (property == name) {
                return m_compiled_model_without_batch->get_property(property);
            }
        }
        if (name == ov::optimal_number_of_infer_requests.name()) {
            uint32_t numRequest = 0;
            try {
                numRequest =
                    m_compiled_model_without_batch->get_property(ov::hint::num_requests.name()).as<std::uint32_t>();
                if (numRequest == 0)  // no limitations from user, let's deduce the full blown #requests
                    // (multiplied by the devices capabilities to run multiple <batched> requests for further perf)
                    numRequest =
                        m_device_info.device_batch_size *
                        m_compiled_model_without_batch->get_property(ov::optimal_number_of_infer_requests.name())
                            .as<uint32_t>();
            } catch (const ov::Exception&) {
            }
            numRequest =
                std::max(numRequest, m_device_info.device_batch_size);  // round up to the possible  user's value
            return ov::Any(numRequest);
        } else if (name == ov::model_name.name()) {
            return m_compiled_model_without_batch->get_property(name);
        } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
            return std::vector<std::string>{ov::optimal_number_of_infer_requests.name(),
                                            METRIC_KEY(SUPPORTED_METRICS),
                                            ov::model_name.name(),
                                            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
                                            ov::execution_devices.name()};
        } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
            return std::vector<std::string>{ov::auto_batch_timeout.name()};
        } else if (name == ov::execution_devices) {
            return m_compiled_model_without_batch->get_property(name);
        } else {
            OPENVINO_THROW("Unsupported Compiled Model Property: ", name);
        }
    }
}

std::shared_ptr<ov::IRemoteContext> ov::autobatch_plugin::CompiledModel::get_context() const {
    return m_compiled_model_without_batch->get_context();
}

std::shared_ptr<const ov::Model> ov::autobatch_plugin::CompiledModel::get_runtime_model() const {
    return m_compiled_model_with_batch && m_compiled_model_with_batch->get_runtime_model()
               ? m_compiled_model_with_batch->get_runtime_model()
               : m_compiled_model_without_batch->get_runtime_model();
}

void ov::autobatch_plugin::CompiledModel::export_model(std::ostream& model) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::IAsyncInferRequest> ov::autobatch_plugin::CompiledModel::create_infer_request() const {
    if (!m_compiled_model_with_batch) {
        return m_compiled_model_without_batch->create_infer_request();
    }

    auto sync_res = create_sync_infer_request();
    return std::make_shared<ov::autobatch_plugin::AsyncInferRequest>(
        std::static_pointer_cast<ov::autobatch_plugin::SyncInferRequest>(sync_res),
        m_compiled_model_without_batch->create_infer_request(),
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
        m_device_info.device_batch_size,
        m_batched_inputs,
        m_batched_outputs);
    return async_infer_request;
}

ov::autobatch_plugin::CompiledModel::~CompiledModel() {
    m_terminate = true;
    for (const auto& w : m_worker_requests) {
        w->_thread.join();
    }
    m_worker_requests.clear();
}

std::pair<std::shared_ptr<ov::autobatch_plugin::CompiledModel::WorkerInferRequest>, int>
ov::autobatch_plugin::CompiledModel::GetWorkerInferRequest() const {
    auto num = m_num_requests_created++;
    std::lock_guard<std::mutex> lock(m_worker_requests_mutex);
    auto batch_id = num % m_device_info.device_batch_size;
    if (!batch_id) {  // need new request
        m_worker_requests.push_back(std::make_shared<WorkerInferRequest>());
        auto workerRequestPtr = m_worker_requests.back().get();
        workerRequestPtr->_inferRequestBatched = {m_compiled_model_with_batch->create_infer_request(),
                                                  m_compiled_model_with_batch._so};
        workerRequestPtr->_batchSize = m_device_info.device_batch_size;
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
                if (m_terminate) {
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
                            t.first->m_sync_request->copy_inputs_if_needed();
                            t.first->m_sync_request->m_batched_req_used =
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
                            t.first->m_request_without_batch->set_callback(
                                [t, sz, &arrived, &all_completed](std::exception_ptr p) {
                                    if (p)
                                        t.first->m_sync_request->m_exception_ptr = p;
                                    t.second();
                                    if (sz == ++arrived) {
                                        all_completed.set_value();
                                    }
                                });
                            t.first->m_sync_request->m_batched_req_used =
                                ov::autobatch_plugin::SyncInferRequest::eExecutionFlavor::TIMEOUT_EXECUTED;
                            t.first->m_sync_request->set_tensors_to_another_request(t.first->m_request_without_batch);
                            t.first->m_request_without_batch->start_async();
                        }
                        all_completed_future.get();
                        // now when all the tasks for this batch are completed, start waiting for the timeout again
                    }
                }
            }
        });
    }
    return {m_worker_requests.back(), static_cast<int>(batch_id)};
}
