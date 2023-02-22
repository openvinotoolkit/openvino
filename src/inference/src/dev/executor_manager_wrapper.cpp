// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "executor_manager_wrapper.hpp"

#include "openvino/runtime/properties.hpp"

ov::ExecutorManagerWrapper::ExecutorManagerWrapper(const std::shared_ptr<ov::ExecutorManager>& manager)
    : m_manager(manager) {}

InferenceEngine::ITaskExecutor::Ptr ov::ExecutorManagerWrapper::getExecutor(const std::string& id) {
    return m_manager->get_executor(id);
}

InferenceEngine::IStreamsExecutor::Ptr ov::ExecutorManagerWrapper::getIdleCPUStreamsExecutor(
    const InferenceEngine::IStreamsExecutor::Config& config) {
    return m_manager->get_idle_cpu_streams_executor(config);
}

size_t ov::ExecutorManagerWrapper::getExecutorsNumber() const {
    return m_manager->get_executors_number();
}

size_t ov::ExecutorManagerWrapper::getIdleCPUStreamsExecutorsNumber() const {
    return m_manager->get_idle_cpu_streams_executors_number();
}

void ov::ExecutorManagerWrapper::clear(const std::string& id) {
    return m_manager->clear(id);
}

void ov::ExecutorManagerWrapper::setTbbFlag(bool flag) {
    m_manager->set_property({{ov::force_tbb_terminate.name(), flag}});
}
bool ov::ExecutorManagerWrapper::getTbbFlag() {
    return m_manager->get_property(ov::force_tbb_terminate.name()).as<bool>();
}
