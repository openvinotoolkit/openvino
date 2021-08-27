// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/kernel.hpp"

#include <map>
#include <mutex>
#include <vector>
#include <memory>
#include <atomic>
#include <string>
#include <set>

#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
#include <tbb/task_arena.h>
#elif(CLDNN_THREADING == CLDNN_THREADING_THREADPOOL)
#include <queue>
#include <future>
#include <functional>
#include <condition_variable>
#endif

namespace cldnn {

#if (CLDNN_THREADING == CLDNN_THREADING_THREADPOOL)
class thread_pool {
public:
    thread_pool(size_t num_threads) : _stop_pool(false) {
        _workers.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            _workers.emplace_back(std::thread(&thread_pool::worker_thread, this));
        }
    }

    ~thread_pool() {
        {
            std::lock_guard<std::mutex> lock(_q_m);
            _stop_pool = true;
        }
        this->wait_all();
    }

    template <class F, class... Args>
    std::future<typename std::result_of<F(Args...)>::type> enqueue(F&& f, Args&&... args) {
        if (_stop_pool) {
            throw std::runtime_error("Thread pool is stoped");
        }

        using return_type = typename std::result_of<F(Args...)>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>> (std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<return_type> result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(_q_m);
            _tasks.push([task]() {(*task)();});
        }
        _cv.notify_one();
        return result;
    }

    void wait_all() {
        _cv.notify_all();
        for (auto& w : _workers) {
            w.join();
        }
    }

private:
    std::vector<std::thread> _workers;
    std::queue<std::function<void()>> _tasks;
    std::condition_variable _cv;
    std::mutex _q_m;
    bool _stop_pool;

    void worker_thread() {
        while (true) {
            std::unique_lock<std::mutex> lock(this->_q_m);
            _cv.wait(lock, [this]() { return (!this->_tasks.empty()) || (_stop_pool); });
            if ((_stop_pool) && (this->_tasks.empty())) return;
            auto task = std::move(_tasks.front());
            this->_tasks.pop();
            lock.unlock();
            task();
        }
    }
};
#endif

class kernels_cache {
public:
    using source_code = std::vector<std::string>;
    struct batch_program {
        int32_t bucket_id;
        int32_t batch_id;
        size_t hash_value;
        uint32_t kernels_counter;
        source_code source;
        std::string options;
        bool dump_custom_program;
        std::map<std::string, std::string> entry_point_to_id;

        explicit batch_program(int32_t _bucket_id, int32_t _batch_id, std::string _options, std::string batch_header_str)
            : bucket_id(_bucket_id),
              batch_id(_batch_id),
              hash_value(0),
              kernels_counter(0),
              source({batch_header_str}),
              options(_options),
              dump_custom_program(false),
              entry_point_to_id({}) {
        }
    };

    struct kernel_code {
        std::shared_ptr<kernel_string> kernel_strings;
        std::string id;
        bool dump_custom_program;
        size_t hash_value;

        kernel_code(const std::shared_ptr<kernel_string>& _kernel_strings,
                    const std::string& _id,
                    bool _dump_custom_program)
            : kernel_strings(_kernel_strings),
              id(_id),
              dump_custom_program(_dump_custom_program),
              hash_value(_kernel_strings->get_hash()) {}

        bool operator == (const kernel_code& rhs) const {
            return (hash_value == rhs.hash_value);
        }
    };

    struct cmp_kernel_code {
        bool operator()(const kernel_code& x1, const kernel_code& x2) const {
            return (x1.hash_value < x2.hash_value);
        }
    };

    using kernels_code = std::set<kernel_code, cmp_kernel_code>;

private:
    static std::mutex _mutex;
    engine& _engine;
    kernels_code _kernels_code;
    std::atomic<bool> _pending_compilation{false};
    std::map<const std::string, kernel::ptr> _kernels;
#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
    std::unique_ptr<tbb::task_arena> arena;
#elif(CLDNN_THREADING == CLDNN_THREADING_THREADPOOL)
    std::unique_ptr<thread_pool> pool;
#endif

    void get_program_source(const kernels_code& kernels_source_code, std::vector<batch_program>*) const;
    void build_batch(const engine& build_engine, const batch_program& batch);

    std::string get_cache_path() const;
    bool is_cache_enabled() const;
    size_t get_max_kernels_per_batch() const;

public:
    explicit kernels_cache(engine& engine);
    kernel_id set_kernel_source(const std::shared_ptr<kernel_string>& kernel_string,
                                bool dump_custom_program);
    kernel::ptr get_kernel(kernel_id id) const;
    // forces compilation of all pending kernels/programs
    void build_all();
    void reset();
};

}  // namespace cldnn
