// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <map>
#include <mutex>
#include <vector>
#include <memory>
#include <atomic>
#include <string>
#include <unordered_set>
#include <kernel_selector_common.h>

#define CLDNN_THREADING_SEQ 0
#define CLDNN_THREADING_TBB 1
#define CLDNN_THREADING_THREADPOOL 2

#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
#include <tbb/task_arena.h>
#elif(CLDNN_THREADING == CLDNN_THREADING_THREADPOOL)
#include <queue>
#include <future>
#include <functional>
#include <condition_variable>
#endif

namespace cl {
class Kernel;
class KernelIntel;
}

namespace kernel_selector {
using kernel_string = kernel_selector::KernelString;
}

namespace cldnn {
namespace gpu {

class gpu_toolkit;
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
        int32_t bucket_id = 0;
        int32_t batch_id = 0;
        source_code source;
        size_t hash_value;
        uint32_t kernels_counter = 0;
        std::string options;
        bool dump_custom_program = false;
        bool one_time = false;
        std::map<std::string, std::string> entry_point_to_id;
    };

    struct kernel_code {
        std::shared_ptr<kernel_selector::kernel_string> kernel_strings;
        std::string id;
        bool dump_custom_program;
        bool one_time_kernel;

        kernel_code(const std::shared_ptr<kernel_selector::kernel_string>& _kernel_strings,
                    const std::string& _id,
                    bool _dump_custom_program,
                    bool _one_time_kernel)
            : kernel_strings(_kernel_strings),
              id(_id),
              dump_custom_program(_dump_custom_program),
              one_time_kernel(_one_time_kernel) {}

        bool operator == (const kernel_code& c2) const {
            return kernel_strings->get_hash() == c2.kernel_strings->get_hash();
        }
    };

    struct hash_kernel_code {
        size_t operator()(const kernel_code& x) const {
            return std::hash<std::string>()(x.kernel_strings->get_hash());
        }
    };

    typedef std::string kernel_id;
    typedef cl::KernelIntel kernel_type;
    using kernels_map = std::map<std::string, kernel_type>;
    using kernels_code = std::unordered_set<kernel_code, hash_kernel_code>;

private:
    gpu_toolkit& _context;
    kernels_code _kernels_code;
    std::atomic<bool> _pending_compilation{false};
    std::map<const std::string, const kernel_type> _kernels;
    std::map<const std::string, const kernel_type> _one_time_kernels;  // These kernels are intended to be executed only once (can
                                                           // be removed later from the cache).
    uint32_t _prog_id;
#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
    std::unique_ptr<tbb::task_arena> arena;
#elif(CLDNN_THREADING == CLDNN_THREADING_THREADPOOL)
    std::unique_ptr<thread_pool> pool;
#endif


    void get_program_source(const kernels_code& kernels_source_code, std::vector<batch_program>*) const;
    void build_batch(const batch_program& batch);

    std::string get_cache_path() const;
    bool is_cache_enabled() const;
    size_t get_max_kernels_per_batch() const;

public:
    explicit kernels_cache(gpu_toolkit& context, uint32_t prog_id);
    kernel_id set_kernel_source(const std::shared_ptr<kernel_selector::kernel_string>& kernel_string,
                                bool dump_custom_program,
                                bool one_time_kernel);
    kernel_type get_kernel(kernel_id id, bool one_time_kernel) const;
    gpu_toolkit& get_context() { return _context; }
    // forces compilation of all pending kernels/programs
    void build_all();
    void reset();
};

}  // namespace gpu
}  // namespace cldnn
