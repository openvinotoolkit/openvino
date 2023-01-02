// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compilation_context.hpp"
#include "threading/ie_thread_safe_containers.hpp"
#include "kernel_selector/kernel_base.h"

namespace cldnn {

class CompilationContext : public ICompilationContext {
public:
    using compilation_queue_t = InferenceEngine::ThreadSafeQueue<ICompilationContext::Task>;

    CompilationContext(cldnn::engine& engine, size_t program_id) {
        _kernels_cache = cldnn::make_unique<kernels_cache>(engine, program_id, kernel_selector::KernelBase::get_db().get_batch_header_str());
        _worker = std::thread([this](){
            while (!_stop_compilation) {
                CompilationContext::Task task;
                bool success = _queue.try_pop(task);
                if (success) {
                    task(*_kernels_cache);
                } else {
                    std::chrono::milliseconds ms{1};
                    std::this_thread::sleep_for(ms);
                }
            }
        });
    }

    void push_task(ICompilationContext::Task&& task) override {
        _queue.push(task);
    }

    void cancel() noexcept override {
        _stop_compilation = true;
        if (_worker.joinable())
            _worker.join();
    }

    ~CompilationContext() noexcept { cancel(); }

private:
    std::unique_ptr<kernels_cache> _kernels_cache;
    compilation_queue_t _queue;
    std::thread _worker;
    std::atomic_bool _stop_compilation{false};
};

std::unique_ptr<ICompilationContext> ICompilationContext::create(cldnn::engine& engine, size_t program_id) {
    return cldnn::make_unique<CompilationContext>(engine, program_id);
}

}  // namespace cldnn
