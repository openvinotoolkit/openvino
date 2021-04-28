// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Contains declarations and custom threading interfaces based on TBB info and task_arena APIs.
 *
 * @file ie_parallel_custom.hpp
 */

#pragma once

#include "ie_parallel.hpp"

#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)

#include <cstddef>
#include <type_traits>
#include <mutex>
#include <vector>

namespace custom {

using numa_node_id = int;
using core_type_id = int;

namespace detail {
struct constraints {
    constraints(numa_node_id id = -1, int maximal_concurrency = -1)
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
    std::vector<core_type_id> core_types();

    int default_concurrency(numa_node_id id = task_arena::automatic);
    int default_concurrency(task_arena::constraints c);
} // namespace info
} // namespace custom
#endif /*(IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)*/
