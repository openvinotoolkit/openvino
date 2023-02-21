// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <threading/ie_cpu_streams_executor.hpp>
#include <functional>
#include <memory>

namespace cldnn {

class ICompilationContext {
public:
    using Task = std::function<void()>;
    virtual void push_task(size_t key, Task&& task) = 0;
    virtual void remove_keys(std::vector<size_t>&& keys) = 0;
    virtual ~ICompilationContext() = default;
    virtual bool is_stopped() = 0;
    virtual void cancel() = 0;

    static std::unique_ptr<ICompilationContext> create(InferenceEngine::CPUStreamsExecutor::Config task_executor_config);
};

}  // namespace cldnn
