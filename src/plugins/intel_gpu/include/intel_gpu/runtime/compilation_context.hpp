// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"

namespace cldnn {

class ICompilationContext {
public:
    using Task = std::function<void()>;
    virtual void push_task(kernel_impl_params key, Task&& task) = 0;
    virtual void remove_keys(std::vector<kernel_impl_params>&& keys) = 0;
    virtual ~ICompilationContext() = default;
    virtual bool is_stopped() = 0;
    virtual void cancel() = 0;
    virtual void wait_all() = 0;

    static std::shared_ptr<ICompilationContext> create(ov::threading::IStreamsExecutor::Config task_executor_config);
};

}  // namespace cldnn
