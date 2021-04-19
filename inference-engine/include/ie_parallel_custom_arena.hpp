// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Contains declarations and custom threading interfaces based on TBB info and task_arena APIs.
 *
 * @file ie_parallel_custom_arena.hpp
 */

#pragma once

#include <cstddef>
#include <type_traits>

#define IE_THREAD_TBB 0
#define IE_THREAD_OMP 1
#define IE_THREAD_SEQ 2
#define IE_THREAD_TBB_AUTO 3

#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
#ifndef NOMINMAX
# define NOMINMAX
#endif
#ifndef TBB_PREVIEW_LOCAL_OBSERVER
# define TBB_PREVIEW_LOCAL_OBSERVER 1
#endif
#ifndef TBB_PREVIEW_NUMA_SUPPORT
# define TBB_PREVIEW_NUMA_SUPPORT 1
#endif
#ifndef TBBBIND_2_4_AVAILABLE
# define TBBBIND_2_4_AVAILABLE 0
#endif

#include <mutex>

#include "tbb/task_arena.h"
#include "tbb/task_scheduler_observer.h"

namespace custom {

using numa_node_id = int;
using core_type_id = int;

namespace detail {
struct constraints {
    constraints() = default;
    constraints(numa_node_id id, int maximal_concurrency)
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

class binding_observer;
} // namespace detail

class task_arena : public tbb::task_arena {
    std::once_flag my_initialization_state;
    detail::constraints my_constraints;
    detail::binding_observer* my_binding_observer;

public:
    using constraints = detail::constraints;
    static const int automatic = tbb::task_arena::automatic;

    task_arena(int max_concurrency_ = automatic, unsigned reserved_for_masters = 1);
    task_arena(const constraints& constraints_, unsigned reserved_for_masters = 1);
    task_arena(const task_arena &s);

    void initialize();
    void initialize(int max_concurrency_, unsigned reserved_for_masters = 1);
    void initialize(constraints constraints_, unsigned reserved_for_masters = 1);

    int max_concurrency();

    template<typename F>
    void enqueue(F&& f) {
        initialize();
        tbb::task_arena::enqueue(std::forward<F>(f));
    }
    template<typename F>
    auto execute(F&& f) -> decltype(f()) {
        initialize();
        return tbb::task_arena::execute(std::forward<F>(f));
    }

    ~task_arena();
};

namespace info {
    std::vector<numa_node_id> numa_nodes();
    int default_concurrency(numa_node_id id = task_arena::automatic);
    std::vector<core_type_id> core_types();
    int default_concurrency(task_arena::constraints c);
} // namespace info
} // namespace custom
#endif /*(IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)*/
