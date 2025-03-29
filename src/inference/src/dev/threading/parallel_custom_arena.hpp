// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Contains declarations and custom threading interfaces based on TBB info and task_arena APIs.
 *
 * @file dev/threading/parallel_custom_arena.hpp
 */

#pragma once

#include "openvino/core/parallel.hpp"

#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)

#    include <cstddef>
#    include <memory>
#    include <mutex>
#    include <type_traits>
#    include <vector>

#    ifndef TBBBIND_2_5_AVAILABLE
#        define TBBBIND_2_5_AVAILABLE 0
#    endif

// On Ubuntu22.04, system tbb is 2021.5 oneTBB and tbbbind dynamic library doesn't exist.
// In this case, tbbbind static library is needed.
#    define USE_TBBBIND_2_5 TBBBIND_2_5_AVAILABLE
#    if USE_TBBBIND_2_5
#        pragma message("USE_TBBBIND_2_5 is enabled")
#    else
#        pragma message("USE_TBBBIND_2_5 is disabled")
#    endif

namespace custom {

using numa_node_id = int;
using core_type_id = int;

namespace detail {
struct constraints {
    constraints(numa_node_id id = tbb::task_arena::automatic, int maximal_concurrency = tbb::task_arena::automatic)
        : numa_id{id},
          max_concurrency{maximal_concurrency},
          core_type{tbb::task_arena::automatic},
          max_threads_per_core{tbb::task_arena::automatic} {}

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

#if USE_TBBBIND_2_5
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
#endif
}  // namespace detail

class task_arena {
    tbb::task_arena my_task_arena;
    std::once_flag my_initialization_state;
    detail::constraints my_constraints;
#if USE_TBBBIND_2_5
    detail::binding_oberver_ptr my_binding_observer;
#endif

public:
    using constraints = detail::constraints;
    static const int automatic = tbb::task_arena::automatic;

    task_arena(int max_concurrency_ = automatic, unsigned reserved_for_masters = 1);
    task_arena(const constraints& constraints_, unsigned reserved_for_masters = 1);
    task_arena(const task_arena& s);

    void initialize();
    void initialize(int max_concurrency_, unsigned reserved_for_masters = 1);
    void initialize(constraints constraints_, unsigned reserved_for_masters = 1);

    explicit operator tbb::task_arena&();

    int max_concurrency();

    template <typename F>
    void enqueue(F&& f) {
        initialize();
        my_task_arena.enqueue(std::forward<F>(f));
    }
    template <typename F>
    auto execute(F&& f) -> decltype(f()) {
        initialize();
        return my_task_arena.execute(std::forward<F>(f));
    }
};

struct task_scheduler_observer : public tbb::task_scheduler_observer {
    task_scheduler_observer(custom::task_arena& arena)
        : tbb::task_scheduler_observer(static_cast<tbb::task_arena&>(arena)),
          my_arena(arena) {}

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
}  // namespace info
}  // namespace custom
#endif /*(OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)*/

