#pragma once

#include <memory>
#include <map>
#include <vector>
#include <Windows.h>

#include <tbb/task_scheduler_observer.h>
#include <tbb/task_arena.h>

#if defined(_WIN32) || defined(_WIN64)

namespace win {

constexpr bool debug_mode = false;

extern std::map<BYTE, std::vector<DWORD>> core_types_map;
extern std::vector<BYTE> core_types_ids_vector;
extern std::vector<DWORD> whole_system_ids;

inline void report_cpu_configuration(SYSTEM_CPU_SET_INFORMATION* cpuSetInfo, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        printf("CPU ID: %i\n\tGroup: %i"
            "\n\tLogical index: %i\n\tCore index: %i\n\tCache ID: %i"
            "\n\tNUMA ID: %i\n\tEfficiency class: %i\n\tAll flags: %i"
            "\n\tAllocated: %i\n\tAllocated to target: %i"
            "\n\tParked: %i\n\tRealtime: %i\n\tReserved flags: %i\n\tReserved: %i\n",
            cpuSetInfo[i].CpuSet.Id,                    // Unique ID for every CPU core. This is the value to use with SetProcessDefaultCpuSets
            cpuSetInfo[i].CpuSet.Group,                 // Some PCs (mostly servers) have groups of CPU cores
            cpuSetInfo[i].CpuSet.LogicalProcessorIndex, // Index of the logical core of the CPU, relative to this CPU group
            cpuSetInfo[i].CpuSet.CoreIndex,             // Index of the home core any logical core is associated with, relative to this CPU group 
            cpuSetInfo[i].CpuSet.LastLevelCacheIndex,   // ID of the memory cache this core uses, relative to this CPU group
            cpuSetInfo[i].CpuSet.NumaNodeIndex,         // ID of the NUMA group for this core, relative to this CPU group
            cpuSetInfo[i].CpuSet.EfficiencyClass,
            cpuSetInfo[i].CpuSet.AllFlags,
            cpuSetInfo[i].CpuSet.Allocated,
            cpuSetInfo[i].CpuSet.AllocatedToTargetProcess,
            cpuSetInfo[i].CpuSet.Parked,
            cpuSetInfo[i].CpuSet.RealTime,
            cpuSetInfo[i].CpuSet.ReservedFlags,
            cpuSetInfo[i].CpuSet.Reserved);
    }
}

inline void report_core_types_map() {
    printf("Size: %zd\n", core_types_map.size());
    for (auto it = core_types_map.begin(); it != core_types_map.end(); ++it) {
        printf("Effeciency class: %d\n", it->first);
        printf("IDs: ");
        for (auto id_it = it->second.begin(); id_it != it->second.end(); ++id_it) {
            printf("%d ", *id_it);
        }
        printf("\n");
    }
}

inline std::vector<BYTE> core_types() {
    static bool initialization_state{false};
    if (initialization_state) return core_types_ids_vector;

    ULONG size;
    HANDLE curProc = GetCurrentProcess();
    (void)GetSystemCpuSetInformation(nullptr, 0, &size, curProc, 0);

    // Get CPUs information
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[size]);
    PSYSTEM_CPU_SET_INFORMATION cpuSets = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.get());
    if (!GetSystemCpuSetInformation(cpuSets, size, &size, curProc, 0))
    {
        DWORD err = GetLastError();
        if (err == ERROR_INSUFFICIENT_BUFFER)
            throw std::exception("Insufficient buffer size for querying CPU set information");
        else
            throw std::exception("An unexpected error occured attempting to query CPU set information");
    }

    // Get CPUs Count
    size_t count = 0;
    while (size > 0)
    {
        size -= cpuSets[count].Size;
        ++count;
    }

    if (debug_mode) {
        report_cpu_configuration(cpuSets, count);
    }

    // Parse CPUs info
    for (size_t i = 0; i < count; ++i) {
        whole_system_ids.emplace_back(cpuSets[i].CpuSet.Id);
        core_types_map[cpuSets[i].CpuSet.EfficiencyClass].emplace_back(cpuSets[i].CpuSet.Id);
    }

    for (auto it = core_types_map.begin(); it != core_types_map.end(); ++it) {
        core_types_ids_vector.emplace_back(it->first);
    }

    if (debug_mode) {
        report_core_types_map();
    }

    initialization_state = true;
    return core_types_ids_vector;
}

inline int default_concurrency(BYTE core_type_id) {
    return static_cast<int>(core_types_map[core_type_id].size());
}

} // namespace info

class soft_affinity_observer : public tbb::task_scheduler_observer {
    BYTE my_core_type_id;
public:
    soft_affinity_observer( tbb::task_arena &a, BYTE core_type_id )
        : tbb::task_scheduler_observer(a), my_core_type_id(core_type_id)
    {}

    void on_scheduler_entry( bool ) override {
        // for (auto& el: win::core_types_map[my_core_type_id]) {
        //     std::cout << "pin to core: " << el << std::endl;
        // }
        SetThreadSelectedCpuSets(
            GetCurrentThread(),
            win::core_types_map[my_core_type_id].data(),
            win::core_types_map[my_core_type_id].size()
        );
    }

    void on_scheduler_exit( bool ) override {
        SetThreadSelectedCpuSets(
            GetCurrentThread(),
            win::whole_system_ids.data(),
            win::whole_system_ids.size()
        );
    }
};

#endif /*defined(_WIN32) || defined(_WIN64)*/
