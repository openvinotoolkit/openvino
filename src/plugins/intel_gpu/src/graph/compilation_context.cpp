// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <threading/ie_cpu_streams_executor.hpp>
#include "compilation_context.hpp"
#include <mutex>
#include <unordered_set>
#include "intel_gpu/runtime/utils.hpp"

namespace cldnn {
class CompilationContext : public ICompilationContext {
public:
    CompilationContext(InferenceEngine::CPUStreamsExecutor::Ptr task_executor) : _task_executor(task_executor) { }

    void push_task(size_t key, Task&& task) override {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_task_keys.find(key) == _task_keys.end()) {
            _task_keys.insert(key);
            _task_executor->run(task);
        }
    }

    void remove_keys(std::vector<size_t>&& keys) override {
        std::lock_guard<std::mutex> lock(_mutex);
        for (auto key : keys) {
            _task_keys.erase(key);
        }
    }

    ~CompilationContext() noexcept {
        cancel();
        _task_executor = nullptr;
        _task_keys.clear();
    }

    bool is_stopped() override {
        std::lock_guard<std::mutex> lock(_mutex);
        return _is_stopped;
    }

    void cancel() override {
        if (_is_stopped)
            return;

        {
            std::lock_guard<std::mutex> lock(_mutex);
            _is_stopped = true;
        }
    }

private:
    InferenceEngine::CPUStreamsExecutor::Ptr _task_executor;
    std::mutex _mutex;
    std::unordered_set<size_t> _task_keys;
    bool _is_stopped = false;
};

std::unique_ptr<ICompilationContext> ICompilationContext::create(InferenceEngine::CPUStreamsExecutor::Ptr task_executor) {
    return cldnn::make_unique<CompilationContext>(task_executor);
}

}  // namespace cldnn