// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mutex>
#include <atomic>
#include <unordered_set>
#include <future>
#include "intel_gpu/runtime/utils.hpp"
#include "intel_gpu/runtime/compilation_context.hpp"

#include "openvino/runtime/threading/cpu_streams_executor.hpp"

namespace cldnn {
class CompilationContext : public ICompilationContext {
public:
    CompilationContext(ov::threading::IStreamsExecutor::Config task_executor_config) : _task_executor_config(task_executor_config) {
        _task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(_task_executor_config);
    }

    void push_task(kernel_impl_params key, Task&& task) override {
        if (_stop_compilation)
            return;

        auto promise = std::make_shared<std::promise<void>>();

        std::lock_guard<std::recursive_mutex> lock(_mutex);

        if (_task_keys.find(key) == _task_keys.end()) {
            if (_task_executor != nullptr) {
                _task_keys.insert({key, promise->get_future()});
                _task_executor->run([task, promise] {
                    task();
                    promise->set_value();
                });
            }
        }
    }

    void remove_keys(std::vector<kernel_impl_params>&& keys) override {
        std::lock_guard<std::recursive_mutex> lock(_mutex);
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

        // Flush all remaining tasks.
        wait_all();

        {
            std::lock_guard<std::recursive_mutex> lock(_mutex);
            if (_task_executor != nullptr)
                _task_executor.reset();
            _task_keys.clear();
        }
    }

    void wait_all() override {
        for (auto&& key_future : _task_keys) {
            if (key_future.second.valid()) {
                key_future.second.wait();
            }
        }
    }

private:
    ov::threading::IStreamsExecutor::Config _task_executor_config;
    std::shared_ptr<ov::threading::IStreamsExecutor> _task_executor;
    std::recursive_mutex _mutex;
    std::unordered_map<kernel_impl_params, std::future<void>, kernel_impl_params::Hasher> _task_keys;
    std::atomic_bool _stop_compilation{false};
};

std::shared_ptr<ICompilationContext> ICompilationContext::create(ov::threading::IStreamsExecutor::Config task_executor_config) {
    return std::make_unique<CompilationContext>(task_executor_config);
}

}  // namespace cldnn
