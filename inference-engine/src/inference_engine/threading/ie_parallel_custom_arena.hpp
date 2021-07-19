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
#include <memory>

#if defined(_WIN32) || defined(_WIN64)
#include "cpusets_api_observer.hpp"
#include <Windows.h>
#endif

namespace custom {

#if defined(_WIN32) || defined(_WIN64)

using numa_node_id = int;
using core_type_id = BYTE;

namespace detail {
struct constraints {
    constraints(numa_node_id id = tbb::task_arena::automatic, int maximal_concurrency = tbb::task_arena::automatic)
        : numa_id{id}
        , max_concurrency{maximal_concurrency}
        , core_type{static_cast<core_type_id>(tbb::task_arena::automatic)}
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

struct soft_affinity_observer_deleter {
    void operator()(soft_affinity_observer* observer) const {
        observer->observe(false);
        delete observer;
    }
};

using binding_oberver_ptr = std::unique_ptr<soft_affinity_observer, soft_affinity_observer_deleter>;


inline binding_oberver_ptr construct_binding_observer(tbb::task_arena& ta, const constraints& c) {
    binding_oberver_ptr observer{};
    if (c.core_type >= 0 && win::core_types().size() > 1) {
        observer.reset(new soft_affinity_observer{ta, c.core_type});
        observer->observe(true);
    }
    return observer;
}

} // namespace detail


namespace info {
    inline std::vector<numa_node_id> numa_nodes() { return {-1}; }
    inline std::vector<core_type_id> core_types() { return win::core_types();}

    inline int default_concurrency(numa_node_id id = tbb::task_arena::automatic) {
        return tbb::this_task_arena::max_concurrency();
    }
    inline int default_concurrency(detail::constraints c) {
        if (c.core_type != tbb::task_arena::automatic) {
            return win::default_concurrency(c.core_type);
        }
        return tbb::this_task_arena::max_concurrency();
    }
} // namespace info

class task_arena {
    tbb::task_arena my_task_arena;
    std::once_flag my_initialization_state;
    detail::constraints my_constraints;
    detail::binding_oberver_ptr my_binding_observer;

public:
    using constraints = detail::constraints;
    static const int automatic = tbb::task_arena::automatic;

    task_arena(int max_concurrency_ = automatic, unsigned reserved_for_masters = 1)
        : my_task_arena{max_concurrency_, reserved_for_masters}
        , my_initialization_state{}
        , my_constraints{}
        , my_binding_observer{}
    {}

    task_arena(const constraints& constraints_, unsigned reserved_for_masters = 1)
        : my_task_arena {info::default_concurrency(constraints_), reserved_for_masters}
        , my_initialization_state{}
        , my_constraints{constraints_}
        , my_binding_observer{}
    {}

    task_arena(const task_arena &s)
        : my_task_arena{s.my_task_arena}
        , my_initialization_state{}
        , my_constraints{s.my_constraints}
        , my_binding_observer{}
    {}

    void initialize() {
        my_task_arena.initialize();
        std::call_once(my_initialization_state, [this] {
            my_binding_observer = detail::construct_binding_observer(
                my_task_arena, my_constraints);
        });
    }
    void initialize(int max_concurrency_, unsigned reserved_for_masters = 1) {
        my_task_arena.initialize(max_concurrency_, reserved_for_masters);
        std::call_once(my_initialization_state, [this] {
            my_binding_observer = detail::construct_binding_observer(
                my_task_arena, my_constraints);
        });
    }
    void initialize(constraints constraints_, unsigned reserved_for_masters = 1) {
        my_constraints = constraints_;
        my_task_arena.initialize(info::default_concurrency(constraints_), reserved_for_masters);
        std::call_once(my_initialization_state, [this] {
            my_binding_observer = detail::construct_binding_observer(
                my_task_arena, my_constraints);
        });
    }

    explicit operator tbb::task_arena&() {
        return my_task_arena;
    }

    int max_concurrency() {
        initialize();
        return my_task_arena.max_concurrency();
    }

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

#else

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
    task_arena(const task_arena &s);

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
} // namespace info

#endif

} // namespace custom

#endif /*(IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)*/
