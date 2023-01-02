// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compilation_context.hpp"
#include "threading/ie_thread_safe_containers.hpp"
#include "kernel_selector/kernel_base.h"

namespace cldnn {

class CompilationContext : public ICompilationContext {
public:
    using data_type = std::pair<size_t, ICompilationContext::Task>;
    using data_list_type = std::list<data_type>;
    using data_list_iter = typename data_list_type::iterator;

    CompilationContext(cldnn::engine& engine, const ExecutionConfig& config, size_t program_id) {
        _kernels_cache = cldnn::make_unique<kernels_cache>(engine, config, program_id, nullptr, kernel_selector::KernelBase::get_db().get_batch_header_str());
        _worker = std::thread([this](){
            while (!_stop_compilation) {
                CompilationContext::Task task;
                bool success = get_front_task(task);
                if (success) {
                    task(*_kernels_cache);
                    pop_front_task();
                } else {
                    std::chrono::milliseconds ms{1};
                    std::this_thread::sleep_for(ms);
                }
            }
        });
    }

    void push_task(size_t key, ICompilationContext::Task&& task) override {
        std::lock_guard<std::mutex> lock(_mutex);
        auto iter = _compile_queue_keymap.find(key);
        if (iter == _compile_queue_keymap.end()) {
            auto insert_it = _compile_queue.insert(_compile_queue.end(), {key, task});
            _compile_queue_keymap.insert({key, insert_it});
        }
    }

    void cancel() noexcept override {
        _stop_compilation = true;
        if (_worker.joinable())
            _worker.join();
    }

    ~CompilationContext() noexcept { cancel(); }

private:
    bool get_front_task(ICompilationContext::Task& task) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_compile_queue.empty()) {
            auto front = _compile_queue.front();
            task = front.second;
            return true;
        }
        return false;
    }

    void pop_front_task() {
        std::lock_guard<std::mutex> lock(_mutex);
        auto front = _compile_queue.front();
        _compile_queue_keymap.erase(front.first);
        _compile_queue.pop_front();
    }

private:
    std::unique_ptr<kernels_cache> _kernels_cache;
    std::thread _worker;
    std::atomic_bool _stop_compilation{false};

    data_list_type _compile_queue;
    std::unordered_map<size_t, data_list_iter> _compile_queue_keymap;
    std::mutex _mutex;
};

std::unique_ptr<ICompilationContext> ICompilationContext::create(cldnn::engine& engine, const ExecutionConfig& config, size_t program_id) {
    return cldnn::make_unique<CompilationContext>(engine, config, program_id);
}

}  // namespace cldnn
