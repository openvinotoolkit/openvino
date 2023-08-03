// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "compilation_context.hpp"
#include <mutex>
#include <atomic>
#include <unordered_set>
#include "intel_gpu/runtime/utils.hpp"

namespace cldnn {
class CompilationContext : public ICompilationContext {
public:
    CompilationContext(ov::threading::IStreamsExecutor::Config task_executor_config) : _task_executor_config(task_executor_config) {
        _task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(_task_executor_config);
    }

    void push_task(kernel_impl_params key, Task&& task) override {
        if (_stop_compilation)
            return;

        std::lock_guard<std::mutex> lock(_mutex);
        if (_task_keys.find(key) == _task_keys.end()) {
            _task_keys.insert(key);
            if (_task_executor != nullptr)
                _task_executor->run(task);
        }
    }

    void remove_keys(std::vector<kernel_impl_params>&& keys) override {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_task_keys.empty()) {
            for (auto key : keys) {
                if (_task_keys.find(key) != _task_keys.end()) {
                    _task_keys.erase(key);
                }
            }
        }
    }

    ~CompilationContext() noexcept {
        cancel();
    }

    bool is_stopped() override {
        return _stop_compilation;
    }

    void cancel() noexcept override {
        if (_stop_compilation)
            return;

        _stop_compilation = true;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_task_executor != nullptr)
                _task_executor.reset();
            _task_keys.clear();
        }
    }

private:
    ov::threading::IStreamsExecutor::Config _task_executor_config;
    std::shared_ptr<ov::threading::IStreamsExecutor> _task_executor;
    std::mutex _mutex;
    std::unordered_set<kernel_impl_params, kernel_impl_params::Hasher> _task_keys;
    std::atomic_bool _stop_compilation{false};
};

std::unique_ptr<ICompilationContext> ICompilationContext::create(ov::threading::IStreamsExecutor::Config task_executor_config) {
    return cldnn::make_unique<CompilationContext>(task_executor_config);
}

}  // namespace cldnn
