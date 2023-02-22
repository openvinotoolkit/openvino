// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/threading/executor_manager.hpp"
#include "threading/ie_executor_manager.hpp"

namespace ov {

class ExecutorManagerWrapper : public InferenceEngine::ExecutorManager {
private:
    std::shared_ptr<ov::ExecutorManager> m_manager;

public:
    ExecutorManagerWrapper(const std::shared_ptr<ov::ExecutorManager>& manager);

    InferenceEngine::ITaskExecutor::Ptr getExecutor(const std::string& id) override;

    InferenceEngine::IStreamsExecutor::Ptr getIdleCPUStreamsExecutor(
        const InferenceEngine::IStreamsExecutor::Config& config) override;

    size_t getExecutorsNumber() const override;

    size_t getIdleCPUStreamsExecutorsNumber() const override;

    void clear(const std::string& id = {}) override;

    void setTbbFlag(bool flag) override;
    bool getTbbFlag() override;
};

}  // namespace ov
