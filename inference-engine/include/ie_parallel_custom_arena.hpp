// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Contains declarations and custom threading interfaces based on TBB info and task_arena APIs.
 *
 * @file ie_parallel_custom_arena.hpp
 */

#pragma once

#define IE_THREAD_TBB 0
#define IE_THREAD_OMP 1
#define IE_THREAD_SEQ 2
#define IE_THREAD_TBB_AUTO 3

#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
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
#include <cstddef>
#include <type_traits>
#include <vector>

#include "tbb/task_arena.h"
#include "tbb/task_scheduler_observer.h"

#define TBB_NUMA_SUPPORT_PRESENT (TBB_INTERFACE_VERSION >= 11100 || TBBBIND_2_4_AVAILABLE)
#define TBB_HYBRID_CPUS_SUPPORT_PRESENT (TBB_INTERFACE_VERSION >= 12020 || TBBBIND_2_4_AVAILABLE)

namespace custom {

#if TBBBIND_2_4_AVAILABLE && TBB_INTERFACE_VERSION < 12020
using numa_node_id = int;
using core_type_id = int;

namespace detail {
struct constraints {
    constraints() = default;
    constraints(numa_node_id id, int maximal_concurrency)
        : numa_id{id}
        , max_concurrency{maximal_concurrency}
        , core_type{-1}
        , max_threads_per_core{-1}
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

    numa_node_id numa_id = -1;
    int max_concurrency = -1;
    core_type_id core_type = -1;
    int max_threads_per_core = -1;
};

class binding_handler;

class binding_observer : public tbb::task_scheduler_observer {
    binding_handler* my_binding_handler;
public:
    binding_observer(tbb::task_arena& ta, int num_slots, const constraints& c);
    void on_scheduler_entry(bool) override;
    void on_scheduler_exit(bool) override;
    ~binding_observer();
};
} // namespace detail

class task_arena {
    tbb::task_arena my_task_arena;
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

    //TODO: Make custom::task_arena inherited from tbb::task_arena
    operator tbb::task_arena&() { return my_task_arena; }

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

    ~task_arena();

    friend tbb::task_arena& get_arena_ref(task_arena& ta);
};

namespace info {
    std::vector<numa_node_id> numa_nodes();
    int default_concurrency(numa_node_id id = task_arena::automatic);
    std::vector<core_type_id> core_types();
    int default_concurrency(task_arena::constraints c);
} // namespace info

inline tbb::task_arena& get_arena_ref(task_arena& ta) { return ta.my_task_arena; }

#else /*TBBBIND_2_4_AVAILABLE && TBB_INTERFACE_VERSION < 12020*/

using task_arena = tbb::task_arena;
inline tbb::task_arena& get_arena_ref(task_arena& ta) { return ta; }

#if TBB_NUMA_SUPPORT_PRESENT
using numa_node_id = tbb::numa_node_id;
namespace info {
    using namespace tbb::info;
} // namespace info
#endif /*TBB_NUMA_SUPPORT_PRESENT*/

#endif /*TBBBIND_2_4_AVAILABLE && TBB_INTERFACE_VERSION < 12020*/
} // namespace custom
#else
#define TBB_NUMA_SUPPORT_PRESENT 0
#define TBB_HYBRID_CPUS_SUPPORT_PRESENT 0
#endif /*(IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)*/
