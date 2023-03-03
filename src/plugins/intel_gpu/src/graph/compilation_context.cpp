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
<<<<<<< HEAD
    CompilationContext(InferenceEngine::CPUStreamsExecutor::Config task_executor_config) : _task_executor_config(task_executor_config) {
        _task_executor_config._streams = 4;
        _task_executor = std::make_shared<InferenceEngine::CPUStreamsExecutor>(_task_executor_config);
=======
    CompilationContext(cldnn::engine& engine, const ExecutionConfig& config, size_t program_id) {
        _kernels_cache =
            cldnn::make_unique<kernels_cache>(engine,
                                              config,
                                              static_cast<uint32_t>(program_id),
                                              nullptr,
                                              kernel_selector::KernelBase::get_db().get_batch_header_str());
        _worker = std::thread([this](){
            while (!_stop_compilation) {
                CompilationContext::Task task;
                size_t task_key;
                bool success = _queue.pop_front_task(task_key, task);
                if (success) {
                    task(*_kernels_cache);
                    _queue.erase_task_key(task_key);
                } else {
                    std::chrono::milliseconds ms{1};
                    std::this_thread::sleep_for(ms);
                }
            }
        });
>>>>>>> 5c64774442 (4th part of fixed warnings)
    }

    void push_task(size_t key, Task&& task) override {
        if (_stop_compilation)
            return;

        std::lock_guard<std::mutex> lock(_mutex);
        if (_task_keys.find(key) == _task_keys.end()) {
            _task_keys.insert(key);
            if (_task_executor != nullptr)
                _task_executor->run(task);
        }
    }

    void remove_keys(std::vector<size_t>&& keys) override {
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
    InferenceEngine::CPUStreamsExecutor::Config _task_executor_config;
    InferenceEngine::CPUStreamsExecutor::Ptr _task_executor;
    std::mutex _mutex;
    std::unordered_set<size_t> _task_keys;
    std::atomic_bool _stop_compilation{false};
};

std::unique_ptr<ICompilationContext> ICompilationContext::create(InferenceEngine::CPUStreamsExecutor::Config task_executor_config) {
    return cldnn::make_unique<CompilationContext>(task_executor_config);
}

}  // namespace cldnn
