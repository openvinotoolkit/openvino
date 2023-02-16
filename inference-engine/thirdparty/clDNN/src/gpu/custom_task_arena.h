// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Contains declarations and custom threading interfaces based on TBB info and task_arena APIs.
 *
 * @file cldnn_custom_arena.hpp
 */

#pragma once

#define CLDNN_THREADING_SEQ 0
#define CLDNN_THREADING_TBB 1
#define CLDNN_THREADING_THREADPOOL 2

#if (CLDNN_THREADING == CLDNN_THREADING_TBB)

#include <cstddef>
#include <type_traits>
#include <mutex>
#include <vector>
#include <memory>

#ifndef NOMINMAX
# define NOMINMAX
#endif
#ifndef TBB_PREVIEW_LOCAL_OBSERVER
# define TBB_PREVIEW_LOCAL_OBSERVER 1
#endif
#ifndef TBB_PREVIEW_NUMA_SUPPORT
# define TBB_PREVIEW_NUMA_SUPPORT 1
#endif
#ifndef TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION
# define TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION 1
#endif

#include "tbb/blocked_range.h"
#include "tbb/blocked_range2d.h"
#include "tbb/blocked_range3d.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/parallel_sort.h"
#include "tbb/task_arena.h"
#include "tbb/task_scheduler_observer.h"

namespace cldnn {
namespace custom {

using numa_node_id = int;
using core_type_id = int;

namespace detail {
struct constraints {
    constraints(numa_node_id id = tbb::task_arena::automatic, int maximal_concurrency = tbb::task_arena::automatic)
        : numa_id{id}
        , max_concurrency{maximal_concurrency}
        , core_type{tbb::task_arena::automatic}
        , max_threads_per_core{tbb::task_arena::automatic}
    {}

    constraints& set_numa_id(numa_node_id id) {
        numa_id = id;
        return *this;
    }
    constraints& set_max_concurrency(int maximal_concurrency) {
        max_concurrency = maximal_concurrency;
        return *this;
    }
    constraints& set_core_type(core_type_id id) {
        core_type = id;
        return *this;
    }
    constraints& set_max_threads_per_core(int threads_number) {
        max_threads_per_core = threads_number;
        return *this;
    }

    numa_node_id numa_id = tbb::task_arena::automatic;
    int max_concurrency = tbb::task_arena::automatic;
    core_type_id core_type = tbb::task_arena::automatic;
    int max_threads_per_core = tbb::task_arena::automatic;
};

class binding_handler;

class binding_observer : public tbb::task_scheduler_observer {
    binding_handler* my_binding_handler;
public:
    binding_observer(tbb::task_arena& ta, int num_slots, const constraints& c);
    ~binding_observer();

    void on_scheduler_entry(bool) override;
    void on_scheduler_exit(bool) override;
};

struct binding_observer_deleter {
    void operator()(binding_observer* observer) const {
        observer->observe(false);
        delete observer;
    }
};

using binding_oberver_ptr = std::unique_ptr<binding_observer, binding_observer_deleter>;

} // namespace detail

class task_arena {
    tbb::task_arena my_task_arena;
    std::once_flag my_initialization_state;
    detail::constraints my_constraints;
    detail::binding_oberver_ptr my_binding_observer;

public:
    using constraints = detail::constraints;
    static const int automatic = tbb::task_arena::automatic;

    task_arena(int max_concurrency_ = automatic, unsigned reserved_for_masters = 1);
    task_arena(const constraints& constraints_, unsigned reserved_for_masters = 1);

    void initialize();
    void initialize(int max_concurrency_, unsigned reserved_for_masters = 1);
    void initialize(constraints constraints_, unsigned reserved_for_masters = 1);

    explicit operator tbb::task_arena&();

    int max_concurrency();

    template<typename F>
    void enqueue(F&& f) {
        initialize();
        my_task_arena.enqueue(std::forward<F>(f));
    }
    template<typename F>
    auto execute(F&& f) -> decltype(f()) {
        initialize();
        return my_task_arena.execute(std::forward<F>(f));
    }
};

struct task_scheduler_observer: public tbb::task_scheduler_observer {
    task_scheduler_observer(custom::task_arena& arena) :
        tbb::task_scheduler_observer(static_cast<tbb::task_arena&>(arena)),
        my_arena(arena)
    {}

    void observe(bool state = true) {
        if (state) {
            my_arena.initialize();
        }
        tbb::task_scheduler_observer::observe(state);
    }

    custom::task_arena& my_arena;
};

namespace info {
    std::vector<numa_node_id> numa_nodes();
    std::vector<core_type_id> core_types();

    int default_concurrency(numa_node_id id = task_arena::automatic);
    int default_concurrency(task_arena::constraints c);

    int get_num_big_cores();
} // namespace info
} // namespace custom
} // namespace cldnn
#endif /*(IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)*/
