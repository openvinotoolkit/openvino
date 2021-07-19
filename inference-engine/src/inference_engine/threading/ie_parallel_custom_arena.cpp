// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_parallel_custom_arena.hpp"

#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO

#if !defined(_WIN32) && !defined(_WIN64)

#ifndef TBBBIND_2_4_AVAILABLE
# define TBBBIND_2_4_AVAILABLE 0
#endif

#define USE_TBBBIND_2_4 (TBBBIND_2_4_AVAILABLE && TBB_INTERFACE_VERSION < 12020)
#define TBB_NUMA_SUPPORT_PRESENT (TBB_INTERFACE_VERSION >= 11100)
#define TBB_HYBRID_CPUS_SUPPORT_PRESENT (TBB_INTERFACE_VERSION >= 12020)

namespace custom {
namespace detail {

#if USE_TBBBIND_2_4
extern "C" {
void __TBB_internal_initialize_system_topology(
    std::size_t groups_num,
    int& numa_nodes_count, int*& numa_indexes_list,
    int& core_types_count, int*& core_types_indexes_list
);
binding_handler* __TBB_internal_allocate_binding_handler(int number_of_slots, int numa_id, int core_type_id, int max_threads_per_core);
void __TBB_internal_deallocate_binding_handler(binding_handler* handler_ptr);
void __TBB_internal_apply_affinity(binding_handler* handler_ptr, int slot_num);
void __TBB_internal_restore_affinity(binding_handler* handler_ptr, int slot_num);
int __TBB_internal_get_default_concurrency(int numa_id, int core_type_id, int max_threads_per_core);
}

int get_processors_group_num() {
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO si;
    GetNativeSystemInfo(&si);

    DWORD_PTR pam, sam, m = 1;
    GetProcessAffinityMask(GetCurrentProcess(), &pam, &sam);
    int nproc = 0;
    for (std::size_t i = 0; i < sizeof(DWORD_PTR) * CHAR_BIT; ++i, m <<= 1) {
        if ( pam & m )
            ++nproc;
    }
    if (nproc == static_cast<int>(si.dwNumberOfProcessors)) {
        return GetActiveProcessorGroupCount();
    }
#endif
    return 1;
}

bool is_binding_environment_valid() {
#if defined(_WIN32) && !defined(_WIN64)
    static bool result = [] {
        // For 32-bit Windows applications, process affinity masks can only support up to 32 logical CPUs.
        SYSTEM_INFO si;
        GetNativeSystemInfo(&si);
        if (si.dwNumberOfProcessors > 32) return false;
        return true;
    }();
    return result;
#else
    return true;
#endif /* _WIN32 && !_WIN64 */
}

static int  numa_nodes_count = 0;
static int* numa_nodes_indexes = nullptr;

static int  core_types_count = 0;
static int* core_types_indexes = nullptr;

void initialize_system_topology() {
    static std::once_flag is_topology_initialized;

    std::call_once(is_topology_initialized, [&]{
        if (is_binding_environment_valid()) {
            __TBB_internal_initialize_system_topology(
                get_processors_group_num(),
                numa_nodes_count, numa_nodes_indexes,
                core_types_count, core_types_indexes);
        } else {
            static int dummy_index = task_arena::automatic;

            numa_nodes_count = 1;
            numa_nodes_indexes = &dummy_index;

            core_types_count = 1;
            core_types_indexes = &dummy_index;
        }
    });
}

binding_observer::binding_observer(tbb::task_arena& ta, int num_slots, const constraints& c)
    : task_scheduler_observer(ta) {
    detail::initialize_system_topology();
    my_binding_handler = detail::__TBB_internal_allocate_binding_handler(num_slots, c.numa_id, c.core_type, c.max_threads_per_core);
}

binding_observer::~binding_observer() {
    detail::__TBB_internal_deallocate_binding_handler(my_binding_handler);
}

void binding_observer::on_scheduler_entry(bool) {
    detail::__TBB_internal_apply_affinity(my_binding_handler, tbb::this_task_arena::current_thread_index());
}

void binding_observer::on_scheduler_exit(bool) {
    detail::__TBB_internal_restore_affinity(my_binding_handler, tbb::this_task_arena::current_thread_index());
}

binding_oberver_ptr construct_binding_observer(tbb::task_arena& ta, int num_slots, const constraints& c) {
    binding_oberver_ptr observer{};
    if (detail::is_binding_environment_valid() &&
      ((c.core_type >= 0 && info::core_types().size() > 1) || (c.numa_id >= 0 && info::numa_nodes().size() > 1) || c.max_threads_per_core > 0)) {
        observer.reset(new binding_observer{ta, num_slots, c});
        observer->observe(true);
    }
    return observer;
}

#endif /*USE_TBBBIND_2_4*/

#if TBB_NUMA_SUPPORT_PRESENT
tbb::task_arena::constraints convert_constraints(const custom::task_arena::constraints& c) {
    tbb::task_arena::constraints result{};
#if TBB_HYBRID_CPUS_SUPPORT_PRESENT
    result.core_type = c.core_type;
    result.max_threads_per_core = c.max_threads_per_core;
#endif
    result.numa_id = c.numa_id;
    result.max_concurrency = c.max_concurrency;
    return result;
}
#endif
} // namespace detail

task_arena::task_arena(int max_concurrency_, unsigned reserved_for_masters)
    : my_task_arena{max_concurrency_, reserved_for_masters}
    , my_initialization_state{}
    , my_constraints{}
    , my_binding_observer{}
{}

task_arena::task_arena(const constraints& constraints_, unsigned reserved_for_masters)
#if USE_TBBBIND_2_4
    : my_task_arena {info::default_concurrency(constraints_), reserved_for_masters}
#elif TBB_NUMA_SUPPORT_PRESENT || TBB_HYBRID_CPUS_SUPPORT_PRESENT
    : my_task_arena {convert_constraints(constraints_), reserved_for_masters}
#else
    : my_task_arena {constraints_.max_concurrency, reserved_for_masters}
#endif
    , my_initialization_state{}
    , my_constraints{constraints_}
    , my_binding_observer{}
{}

task_arena::task_arena(const task_arena &s)
    : my_task_arena{s.my_task_arena}
    , my_initialization_state{}
    , my_constraints{s.my_constraints}
    , my_binding_observer{}
{}

void task_arena::initialize() {
    my_task_arena.initialize();
#if USE_TBBBIND_2_4
    std::call_once(my_initialization_state, [this] {
        my_binding_observer = detail::construct_binding_observer(
            my_task_arena, my_task_arena.max_concurrency(), my_constraints);
    });
#endif
}

void task_arena::initialize(int max_concurrency_, unsigned reserved_for_masters) {
    my_task_arena.initialize(max_concurrency_, reserved_for_masters);
#if USE_TBBBIND_2_4
    std::call_once(my_initialization_state, [this] {
        my_binding_observer = detail::construct_binding_observer(
            my_task_arena, my_task_arena.max_concurrency(), my_constraints);
    });
#endif
}

void task_arena::initialize(constraints constraints_, unsigned reserved_for_masters) {
        my_constraints = constraints_;
#if USE_TBBBIND_2_4
        my_task_arena.initialize(info::default_concurrency(constraints_), reserved_for_masters);
        std::call_once(my_initialization_state, [this] {
            my_binding_observer = detail::construct_binding_observer(
                my_task_arena, my_task_arena.max_concurrency(), my_constraints);
        });
#elif TBB_NUMA_SUPPORT_PRESENT || TBB_HYBRID_CPUS_SUPPORT_PRESENT
        my_task_arena.initialize(convert_constraints(my_constraints), reserved_for_masters);
#else
        my_task_arena.initialize(my_constraints.max_concurrency, reserved_for_masters);
#endif
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
#if USE_TBBBIND_2_4
    detail::initialize_system_topology();
    std::vector<numa_node_id> node_indexes(detail::numa_nodes_count);
    std::memcpy(node_indexes.data(), detail::numa_nodes_indexes, detail::numa_nodes_count * sizeof(int));
    return node_indexes;
#elif TBB_NUMA_SUPPORT_PRESENT
    return tbb::info::numa_nodes();
#else
    return {tbb::task_arena::automatic};
#endif
}

std::vector<core_type_id> core_types() {
#if USE_TBBBIND_2_4
    detail::initialize_system_topology();
    std::vector<numa_node_id> core_type_indexes(detail::core_types_count);
    std::memcpy(core_type_indexes.data(), detail::core_types_indexes, detail::core_types_count * sizeof(int));
    return core_type_indexes;
#elif TBB_HYBRID_CPUS_SUPPORT_PRESENT
    return tbb::info::core_types();
#else
    return {tbb::task_arena::automatic};
#endif
}

int default_concurrency(task_arena::constraints c) {
    if (c.max_concurrency > 0) {
        return c.max_concurrency;
    }
#if USE_TBBBIND_2_4
    if (detail::is_binding_environment_valid()) {
        detail::initialize_system_topology();
        return detail::__TBB_internal_get_default_concurrency(c.numa_id, c.core_type, c.max_threads_per_core);
    }
    return tbb::this_task_arena::max_concurrency();
#elif TBB_HYBRID_CPUS_SUPPORT_PRESENT
    return tbb::info::default_concurrency(convert_constraints(c));
#elif TBB_NUMA_SUPPORT_PRESENT
    return tbb::info::default_concurrency(c.numa_id);
#else
    return tbb::this_task_arena::max_concurrency();
#endif
}

int default_concurrency(numa_node_id id) {
    return default_concurrency(task_arena::constraints{}.set_numa_id(id));
}

} // namespace info
} // namespace custom

#endif

#endif /*IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO*/
