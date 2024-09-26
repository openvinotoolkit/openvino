// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "compiled_model.hpp"

#include "async_infer_request.hpp"

namespace ov {
namespace autobatch_plugin {
CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const ov::AnyMap& config,
                             const DeviceInformation& device_info,
                             const std::set<std::size_t>& batched_inputs,
                             const std::set<std::size_t>& batched_outputs,
                             const ov::SoPtr<ov::ICompiledModel>& compiled_model_with_batch,
                             const ov::SoPtr<ov::ICompiledModel>& compiled_model_without_batch,
                             const ov::SoPtr<ov::IRemoteContext>& context)
    : ov::ICompiledModel(model, plugin, context),
      m_config(config),
      m_batched_inputs(batched_inputs),
      m_batched_outputs(batched_outputs),
      m_compiled_model_with_batch(compiled_model_with_batch),
      m_compiled_model_without_batch(compiled_model_without_batch) {
    // WA for gcc 4.8 ( fails compilation with member init-list)
    m_device_info = device_info;
    auto time_out = config.find(ov::auto_batch_timeout.name());
    OPENVINO_ASSERT(time_out != config.end(), "No timeout property be set in config, default will be used!");
    m_time_out = time_out->second.as<std::uint32_t>();
}

CompiledModel::~CompiledModel() {
    m_terminate = true;
    for (const auto& w : m_worker_requests) {
        w->_thread.join();
    }
    m_worker_requests.clear();
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    auto workerRequestPtrAndId = GetWorkerInferRequest();
    auto async_infer_request = std::make_shared<ov::autobatch_plugin::SyncInferRequest>(
        std::dynamic_pointer_cast<const ov::autobatch_plugin::CompiledModel>(shared_from_this()),
        workerRequestPtrAndId.first,
        workerRequestPtrAndId.second,
        m_device_info.device_batch_size,
        m_batched_inputs,
        m_batched_outputs);
    return async_infer_request;
}

std::pair<std::shared_ptr<ov::autobatch_plugin::CompiledModel::WorkerInferRequest>, int>
CompiledModel::GetWorkerInferRequest() const {
    auto num = m_num_requests_created++;
    std::lock_guard<std::mutex> lock(m_worker_requests_mutex);
    auto batch_id = num % m_device_info.device_batch_size;
    if (!batch_id) {  // need new request
        m_worker_requests.push_back(std::make_shared<WorkerInferRequest>());
        auto workerRequestPtr = m_worker_requests.back().get();
        workerRequestPtr->_infer_request_batched._ptr = m_compiled_model_with_batch->create_infer_request();
        if (workerRequestPtr->_infer_request_batched._so == nullptr)
            workerRequestPtr->_infer_request_batched._so = m_compiled_model_with_batch._so;
        workerRequestPtr->_batch_size = m_device_info.device_batch_size;
        workerRequestPtr->_completion_tasks.resize(workerRequestPtr->_batch_size);
        workerRequestPtr->_is_wakeup = false;
        workerRequestPtr->_infer_request_batched->set_callback(
            [workerRequestPtr](std::exception_ptr exceptionPtr) mutable {
                if (exceptionPtr)
                    workerRequestPtr->_exception_ptr = exceptionPtr;
                OPENVINO_ASSERT(workerRequestPtr->_completion_tasks.size() == (size_t)workerRequestPtr->_batch_size);
                // notify the individual requests on the completion
                for (int c = 0; c < workerRequestPtr->_batch_size; c++) {
                    workerRequestPtr->_completion_tasks[c]();
                }
                // reset the timeout
                workerRequestPtr->_is_wakeup = true;
                workerRequestPtr->_cond.notify_one();
            });

        workerRequestPtr->_thread = std::thread([workerRequestPtr, this] {
            while (1) {
                std::cv_status status;
                {
                    std::unique_lock<std::mutex> lock(workerRequestPtr->_mutex);
                    status = workerRequestPtr->_cond.wait_for(lock, std::chrono::milliseconds(m_time_out));
                    if ((status != std::cv_status::timeout) && (workerRequestPtr->_is_wakeup == false))
                        continue;
                    workerRequestPtr->_is_wakeup = false;
                }
                if (m_terminate) {
                    break;
                } else {
                    // as we pop the tasks from the queue only here
                    // it is ok to call size() (as the _tasks can only grow in parallel)
                    const int sz = static_cast<int>(workerRequestPtr->_tasks.size());
                    if (sz == workerRequestPtr->_batch_size) {
                        std::pair<ov::autobatch_plugin::AsyncInferRequest*, ov::threading::Task> t;
                        for (int n = 0; n < sz; n++) {
                            OPENVINO_ASSERT(workerRequestPtr->_tasks.try_pop(t));
                            workerRequestPtr->_completion_tasks[n] = std::move(t.second);
                            t.first->m_sync_request->copy_inputs_if_needed();
                            t.first->m_sync_request->m_batched_request_status =
                                ov::autobatch_plugin::SyncInferRequest::eExecutionFlavor::BATCH_EXECUTED;
                        }
                        workerRequestPtr->_infer_request_batched->start_async();
                    } else if ((status == std::cv_status::timeout) && sz) {
                        // timeout to collect the batch is over, have to execute the requests in the batch1 mode
                        std::pair<ov::autobatch_plugin::AsyncInferRequest*, ov::threading::Task> t;
                        // popping all tasks collected by the moment of the time-out and execute each with batch1
                        std::atomic<int> arrived = {0};
                        std::promise<void> all_completed;
                        auto all_completed_future = all_completed.get_future();
                        for (int n = 0; n < sz; n++) {
                            OPENVINO_ASSERT(workerRequestPtr->_tasks.try_pop(t));
                            t.first->m_request_without_batch->set_callback(
                                [t, sz, &arrived, &all_completed](std::exception_ptr p) {
                                    if (p)
                                        t.first->m_sync_request->m_exception_ptr = p;
                                    t.second();
                                    if (sz == ++arrived) {
                                        all_completed.set_value();
                                    }
                                });
                            t.first->m_sync_request->m_batched_request_status =
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

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    ov::SoPtr<ov::IAsyncInferRequest> infer_request_without_batch = {
        m_compiled_model_without_batch->create_infer_request(),
        m_compiled_model_without_batch._so};
    // simpler wrapper if m_compiled_model_with_batch is empty
    std::shared_ptr<ov::ISyncInferRequest> sync_res;
    if (m_compiled_model_with_batch)
        sync_res = create_sync_infer_request();
    else
        sync_res = std::make_shared<ov::autobatch_plugin::SyncInferRequest>(
            std::dynamic_pointer_cast<const ov::autobatch_plugin::CompiledModel>(shared_from_this()),
            nullptr,
            0,
            0);
    return std::make_shared<ov::autobatch_plugin::AsyncInferRequest>(
        std::dynamic_pointer_cast<ov::autobatch_plugin::SyncInferRequest>(sync_res),
        infer_request_without_batch,
        get_callback_executor());
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    auto& compiled_model = m_compiled_model_with_batch ? m_compiled_model_with_batch : m_compiled_model_without_batch;
    auto model = compiled_model->get_runtime_model();
    set_model_shared_object(const_cast<ov::Model&>(*model), compiled_model._so);
    return model;
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    for (const auto& property : properties) {
        if (property.first == ov::auto_batch_timeout.name()) {
            m_time_out = property.second.as<std::uint32_t>();
            m_config[ov::auto_batch_timeout.name()] = property.second.as<std::uint32_t>();
        } else {
            OPENVINO_THROW("AutoBatching Compiled Model dosen't support property",
                           property.first,
                           ". The only property that can be changed on the fly is the ",
                           ov::auto_batch_timeout.name());
        }
    }
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    auto it = m_config.find(name);
    if (it != m_config.end()) {
        return it->second;
    } else {
        if (name == ov::optimal_number_of_infer_requests.name()) {
            uint32_t num_request = 0;
            try {
                num_request =
                    m_compiled_model_without_batch->get_property(ov::hint::num_requests.name()).as<std::uint32_t>();
                if (num_request == 0)  // no limitations from user, let's deduce the full blown #requests
                    // (multiplied by the devices capabilities to run multiple <batched> requests for further perf)
                    num_request =
                        m_device_info.device_batch_size *
                        m_compiled_model_without_batch->get_property(ov::optimal_number_of_infer_requests.name())
                            .as<uint32_t>();
            } catch (const ov::Exception&) {
            }
            num_request =
                std::max(num_request, m_device_info.device_batch_size);  // round up to the possible  user's value
            return num_request;
        } else if (name == ov::model_name.name()) {
            return m_compiled_model_without_batch->get_property(name);
        } else if (name == ov::execution_devices) {
            return m_compiled_model_without_batch->get_property(name);
        } else if (name == ov::loaded_from_cache) {
            return m_compiled_model_without_batch->get_property(ov::loaded_from_cache.name());
        } else if (name == ov::supported_properties) {
            return std::vector<ov::PropertyName>{
                ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
                ov::PropertyName{ov::optimal_number_of_infer_requests.name(), ov::PropertyMutability::RO},
                ov::PropertyName{ov::model_name.name(), ov::PropertyMutability::RO},
                ov::PropertyName{ov::execution_devices.name(), ov::PropertyMutability::RO},
                ov::PropertyName{ov::auto_batch_timeout.name(), ov::PropertyMutability::RW}};
        } else if (name == ov::auto_batch_timeout) {
            uint32_t time_out = m_time_out;
            return time_out;
        } else if (name == ov::device::properties) {
            ov::AnyMap all_devices = {};
            ov::AnyMap device_properties = {};
            auto device_supported_props = m_compiled_model_without_batch->get_property(ov::supported_properties.name());
            for (auto&& property_name : device_supported_props.as<std::vector<ov::PropertyName>>())
                device_properties[property_name] = m_compiled_model_without_batch->get_property(property_name);
            all_devices[m_device_info.device_name] = device_properties;
            return all_devices;
        } else {
            // find config key among networks config keys
            auto modelSupportedProperties =
                m_compiled_model_without_batch->get_property(ov::supported_properties.name());
            for (auto&& property : modelSupportedProperties.as<std::vector<ov::PropertyName>>()) {
                if (property == name) {
                    return m_compiled_model_without_batch->get_property(property);
                }
            }
            OPENVINO_THROW("Unsupported Compiled Model Property: ", name);
        }
    }
}

const std::vector<ov::Output<const ov::Node>>& CompiledModel::outputs() const {
    return m_compiled_model_without_batch->outputs();
}

const std::vector<ov::Output<const ov::Node>>& CompiledModel::inputs() const {
    return m_compiled_model_without_batch->inputs();
}

void CompiledModel::export_model(std::ostream& model) const {
    OPENVINO_NOT_IMPLEMENTED;
}

}  // namespace autobatch_plugin
}  // namespace ov
