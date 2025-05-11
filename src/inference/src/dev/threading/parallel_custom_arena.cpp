// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "dev/threading/parallel_custom_arena.hpp"

#include <cstring>

#include "dev/threading/itt.hpp"

#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO

#    define TBB_NUMA_SUPPORT_PRESENT (TBB_INTERFACE_VERSION >= 11100)
#    if defined(__APPLE__)
// 2021.2 TBB doesn't export for macOS symbol:
//     _ZN3tbb6detail2r131constraints_default_concurrencyERKNS0_2d111constraintsEl
#        define TBB_HYBRID_CPUS_SUPPORT_PRESENT (TBB_INTERFACE_VERSION > 12020)
#    else
#        define TBB_HYBRID_CPUS_SUPPORT_PRESENT (TBB_INTERFACE_VERSION >= 12020)
#    endif

#    if defined(_WIN32) || defined(_WIN64)
#        ifndef NOMINMAX
#            define NOMINMAX
#        endif
#        include <windows.h>
#    endif

namespace custom {
namespace detail {

#    if USE_TBBBIND_2_5

extern "C" {
void __TBB_internal_initialize_system_topology(std::size_t groups_num,
                                               int& numa_nodes_count,
                                               int*& numa_indexes_list,
                                               int& core_types_count,
                                               int*& core_types_indexes_list);
void __TBB_internal_destroy_system_topology();
binding_handler* __TBB_internal_allocate_binding_handler(int number_of_slots,
                                                         int numa_id,
                                                         int core_type_id,
                                                         int max_threads_per_core);
void __TBB_internal_deallocate_binding_handler(binding_handler* handler_ptr);
void __TBB_internal_apply_affinity(binding_handler* handler_ptr, int slot_num);
void __TBB_internal_restore_affinity(binding_handler* handler_ptr, int slot_num);
int __TBB_internal_get_default_concurrency(int numa_id, int core_type_id, int max_threads_per_core);
}

static bool is_binding_environment_valid() {
#        if defined(_WIN32) && !defined(_WIN64)
    static bool result = [] {
        // For 32-bit Windows applications, process affinity masks can only support up to 32 logical CPUs.
        SYSTEM_INFO si;
        GetNativeSystemInfo(&si);
        if (si.dwNumberOfProcessors > 32)
            return false;
        return true;
    }();
    return result;
#        else
    return true;
#        endif /* _WIN32 && !_WIN64 */
}

#    elif TBB_NUMA_SUPPORT_PRESENT || TBB_HYBRID_CPUS_SUPPORT_PRESENT

static tbb::task_arena::constraints convert_constraints(const custom::task_arena::constraints& c) {
    tbb::task_arena::constraints result{};
#        if TBB_HYBRID_CPUS_SUPPORT_PRESENT
    result.core_type = c.core_type;
    result.max_threads_per_core = c.max_threads_per_core;
#        endif
    result.numa_id = c.numa_id;
    result.max_concurrency = c.max_concurrency;
    return result;
}

#    endif  // USE_TBBBIND_2_5

class TBBbindSystemTopology {
    TBBbindSystemTopology() {
#    if USE_TBBBIND_2_5
        if (is_binding_environment_valid()) {
            TBB_BIND_SCOPE(TBBbindSystemTopology)
            __TBB_internal_initialize_system_topology(get_processors_group_num(),
                                                      numa_nodes_count,
                                                      numa_nodes_indexes,
                                                      core_types_count,
                                                      core_types_indexes);
        }
        if (numa_nodes_count > 1 || core_types_count > 1) {
            TBB_BIND_NUMA_ENABLED;
        }
#    endif
    }

public:
    ~TBBbindSystemTopology() {
#    if USE_TBBBIND_2_5
        if (is_binding_environment_valid()) {
            TBB_BIND_SCOPE(TBBbindSystemTopology)
            __TBB_internal_destroy_system_topology();
        }
#    endif
    }

    std::vector<numa_node_id> numa_nodes() const {
#    if USE_TBBBIND_2_5
        std::vector<numa_node_id> node_indexes(numa_nodes_count);
        std::memcpy(node_indexes.data(), numa_nodes_indexes, numa_nodes_count * sizeof(int));
        return node_indexes;
#    elif TBB_NUMA_SUPPORT_PRESENT
        return tbb::info::numa_nodes();
#    else
        return {tbb::task_arena::automatic};
#    endif
    }

    std::vector<core_type_id> core_types() const {
#    if USE_TBBBIND_2_5
        std::vector<numa_node_id> core_type_indexes(core_types_count);
        std::memcpy(core_type_indexes.data(), core_types_indexes, core_types_count * sizeof(int));
        return core_type_indexes;
#    elif TBB_HYBRID_CPUS_SUPPORT_PRESENT
        return tbb::info::core_types();
#    else
        return {tbb::task_arena::automatic};
#    endif
    }

    int default_concurrency(task_arena::constraints c) const {
        if (c.max_concurrency > 0) {
            return c.max_concurrency;
        }
#    if USE_TBBBIND_2_5
        if (is_binding_environment_valid()) {
            TBB_BIND_SCOPE(default_concurrency)
            return __TBB_internal_get_default_concurrency(c.numa_id, c.core_type, c.max_threads_per_core);
        }
        return tbb::this_task_arena::max_concurrency();
#    elif TBB_HYBRID_CPUS_SUPPORT_PRESENT
        return tbb::info::default_concurrency(convert_constraints(c));
#    elif TBB_NUMA_SUPPORT_PRESENT
        return tbb::info::default_concurrency(c.numa_id);
#    else
        return tbb::this_task_arena::max_concurrency();
#    endif
    }

    friend const TBBbindSystemTopology& system_topology();

private:
    int get_processors_group_num() const {
#    if defined(_WIN32)
        SYSTEM_INFO si;
        GetNativeSystemInfo(&si);

        DWORD_PTR pam, sam, m = 1;
        GetProcessAffinityMask(GetCurrentProcess(), &pam, &sam);
        int nproc = 0;
        for (std::size_t i = 0; i < sizeof(DWORD_PTR) * CHAR_BIT; ++i, m <<= 1) {
            if (pam & m)
                ++nproc;
        }
        if (nproc == static_cast<int>(si.dwNumberOfProcessors)) {
            return GetActiveProcessorGroupCount();
        }
#    endif
        return 1;
    }

private:
#    if USE_TBBBIND_2_5
    int dummy_index = task_arena::automatic;

    int numa_nodes_count = 1;
    int* numa_nodes_indexes = &dummy_index;

    int core_types_count = 1;
    int* core_types_indexes = &dummy_index;
#    endif
};

const TBBbindSystemTopology& system_topology() {
    static TBBbindSystemTopology topology;
    return topology;
}

#    if USE_TBBBIND_2_5

binding_observer::binding_observer(tbb::task_arena& ta, int num_slots, const constraints& c)
    : task_scheduler_observer(ta) {
    detail::system_topology();
    TBB_BIND_SCOPE(binding_observer)
    my_binding_handler =
        detail::__TBB_internal_allocate_binding_handler(num_slots, c.numa_id, c.core_type, c.max_threads_per_core);
}

binding_observer::~binding_observer() {
    TBB_BIND_SCOPE(binding_observer)
    detail::__TBB_internal_deallocate_binding_handler(my_binding_handler);
}

void binding_observer::on_scheduler_entry(bool) {
    TBB_BIND_SCOPE(on_scheduler_entry)
    detail::__TBB_internal_apply_affinity(my_binding_handler, tbb::this_task_arena::current_thread_index());
}

void binding_observer::on_scheduler_exit(bool) {
    TBB_BIND_SCOPE(on_scheduler_exit)
    detail::__TBB_internal_restore_affinity(my_binding_handler, tbb::this_task_arena::current_thread_index());
}

static binding_oberver_ptr construct_binding_observer(tbb::task_arena& ta, int num_slots, const constraints& c) {
    binding_oberver_ptr observer{};
    if (detail::is_binding_environment_valid() &&
        ((c.core_type >= 0 && info::core_types().size() > 1) || (c.numa_id >= 0 && info::numa_nodes().size() > 1) ||
         c.max_threads_per_core > 0)) {
        observer.reset(new binding_observer{ta, num_slots, c});
        observer->observe(true);
    }
    return observer;
}

#    endif  // USE_TBBBIND_2_5
}  // namespace detail

task_arena::task_arena(int max_concurrency_, unsigned reserved_for_masters)
    : my_task_arena{max_concurrency_, reserved_for_masters},
      my_initialization_state{},
      my_constraints{}
#    if USE_TBBBIND_2_5
      ,
      my_binding_observer{}
#    endif
{
}

task_arena::task_arena(const constraints& constraints_, unsigned reserved_for_masters)
#    if USE_TBBBIND_2_5
    : my_task_arena{info::default_concurrency(constraints_), reserved_for_masters}
#    elif TBB_NUMA_SUPPORT_PRESENT || TBB_HYBRID_CPUS_SUPPORT_PRESENT
    : my_task_arena{convert_constraints(constraints_), reserved_for_masters}
#    else
    : my_task_arena{constraints_.max_concurrency, reserved_for_masters}
#    endif
      ,
      my_initialization_state{},
      my_constraints{constraints_}
#    if USE_TBBBIND_2_5
      ,
      my_binding_observer{}
#    endif
{
}

task_arena::task_arena(const task_arena& s)
    : my_task_arena{s.my_task_arena},
      my_initialization_state{},
      my_constraints{s.my_constraints}
#    if USE_TBBBIND_2_5
      ,
      my_binding_observer{}
#    endif
{
}

void task_arena::initialize() {
    my_task_arena.initialize();
#    if USE_TBBBIND_2_5
    TBB_BIND_SCOPE(task_arena_initialize)
    std::call_once(my_initialization_state, [this] {
        my_binding_observer =
            detail::construct_binding_observer(my_task_arena, my_task_arena.max_concurrency(), my_constraints);
    });
#    endif
}

void task_arena::initialize(int max_concurrency_, unsigned reserved_for_masters) {
    my_task_arena.initialize(max_concurrency_, reserved_for_masters);
#    if USE_TBBBIND_2_5
    TBB_BIND_SCOPE(task_arena_initialize_max_concurrency)
    std::call_once(my_initialization_state, [this] {
        my_binding_observer =
            detail::construct_binding_observer(my_task_arena, my_task_arena.max_concurrency(), my_constraints);
    });
#    endif
}

void task_arena::initialize(constraints constraints_, unsigned reserved_for_masters) {
    my_constraints = constraints_;
#    if USE_TBBBIND_2_5
    TBB_BIND_SCOPE(task_arena_initialize_constraints)
    my_task_arena.initialize(info::default_concurrency(constraints_), reserved_for_masters);
    std::call_once(my_initialization_state, [this] {
        my_binding_observer =
            detail::construct_binding_observer(my_task_arena, my_task_arena.max_concurrency(), my_constraints);
    });
#    elif TBB_NUMA_SUPPORT_PRESENT || TBB_HYBRID_CPUS_SUPPORT_PRESENT
    my_task_arena.initialize(convert_constraints(my_constraints), reserved_for_masters);
#    else
    my_task_arena.initialize(my_constraints.max_concurrency, reserved_for_masters);
#    endif
}

task_arena::operator tbb::task_arena&() {
    return my_task_arena;
}

int task_arena::max_concurrency() {
    initialize();
    return my_task_arena.max_concurrency();
}

namespace info {
std::vector<numa_node_id> numa_nodes() {
    return detail::system_topology().numa_nodes();
}

std::vector<core_type_id> core_types() {
    return detail::system_topology().core_types();
}

int default_concurrency(task_arena::constraints c) {
    return detail::system_topology().default_concurrency(c);
}

int default_concurrency(numa_node_id id) {
    return detail::system_topology().default_concurrency(task_arena::constraints{}.set_numa_id(id));
}

}  // namespace info
}  // namespace custom
#endif /*OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO*/
