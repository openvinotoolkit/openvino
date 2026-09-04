// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Contains declarations and custom threading interfaces based on TBB info and task_arena APIs.
 *
 * @file dev/threading/parallel_custom_arena.hpp
 */

#pragma once

#include "openvino/core/parallel.hpp"
#include "openvino/runtime/common.hpp"

#if OV_THREAD_USE_TBB

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

constexpr int TBB_VERSION_MAJOR_CORE_TYPES_WINDOWS = 1202;
constexpr int TBB_VERSION_PATCH_CORE_TYPES_WINDOWS = 6;
constexpr int TBB_VERSION_MAJOR_CORE_TYPES_LINUX = 1213;
constexpr int TBB_VERSION_PATCH_CORE_TYPES_LINUX = 1;

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
    constraints& set_core_types(const std::vector<core_type_id>& ids) {
        if (ids.empty()) {
            core_type = -1;
        } else if (ids.size() == 1) {
            core_type = ids[0];
        } else {
            // Set a marker bit to indicate multiple core type format
            core_type = (1 << core_type_id_bits);

            for (core_type_id id : ids) {
                OPENVINO_ASSERT((0 <= id) && (id < static_cast<core_type_id>(core_type_id_bits)), "Wrong core type id");
                core_type |= (1 << id);
            }
        }

        return *this;
    }
    constraints& set_max_threads_per_core(int threads_number) {
        max_threads_per_core = threads_number;
        return *this;
    }
    constraints& set_processor_group_base(int base) {
        processor_group_base = base;
        return *this;
    }

    numa_node_id numa_id = tbb::task_arena::automatic;
    int max_concurrency = tbb::task_arena::automatic;
    core_type_id core_type = tbb::task_arena::automatic;
    // Upper 4 bits reserved for format marker (single vs multiple core types)
    static constexpr size_t core_type_id_bits = sizeof(core_type_id) * CHAR_BIT - 4;
    int max_threads_per_core = tbb::task_arena::automatic;
    // Per-stream offset for soft (non-pinning) distribution of the arena's threads across Windows
    // processor groups (>64 logical processors). Negative means no group policy.
    int processor_group_base = -1;
};

#    if USE_TBBBIND_2_5
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
#    endif

#    if defined(_WIN32)
// Distributes an arena's worker threads across Windows processor groups without core pinning: each
// entering thread is soft-bound (full active group mask) to group (base + slot) % group_count, and its
// previous group affinity is restored on scheduler exit. This lets a single stream span multiple groups.
class group_affinity_observer : public tbb::task_scheduler_observer {
    // Per-stream group offset stored as a plain type so this header stays free of <windows.h>.
    int my_group_base = -1;
    bool my_valid = false;

public:
    group_affinity_observer(tbb::task_arena& ta, int group_base);

    bool valid() const {
        return my_valid;
    }

    void on_scheduler_entry(bool) override;
    void on_scheduler_exit(bool) override;
};

struct group_affinity_observer_deleter {
    void operator()(group_affinity_observer* observer) const {
        observer->observe(false);
        delete observer;
    }
};

using group_affinity_observer_ptr = std::unique_ptr<group_affinity_observer, group_affinity_observer_deleter>;
#    endif
}  // namespace detail

class task_arena {
    tbb::task_arena my_task_arena;
    std::once_flag my_initialization_state;
    detail::constraints my_constraints;
#    if USE_TBBBIND_2_5
    detail::binding_oberver_ptr my_binding_observer;
#    endif
#    if defined(_WIN32)
    detail::group_affinity_observer_ptr my_group_affinity_observer;
    std::once_flag my_group_observer_state;
#    endif

    void init_group_affinity_observer();

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
#endif /*(OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO || OV_THREAD == OV_THREAD_TBB_ADAPTIVE)*/
