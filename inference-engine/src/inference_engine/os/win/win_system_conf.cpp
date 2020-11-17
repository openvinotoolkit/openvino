// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <windows.h>
#include <memory>
#include <vector>
#include "ie_system_conf.h"
#include "ie_parallel.hpp"

namespace InferenceEngine {
int getNumberOfCPUCores(bool bigCoresOnly) {
    const int fallback_val = parallel_get_max_threads();
    DWORD sz = 0;
    // querying the size of the resulting structure, passing the nullptr for the buffer
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &sz) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        return fallback_val;

    std::unique_ptr<uint8_t[]> ptr(new uint8_t[sz]);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr.get()), &sz))
        return fallback_val;

    int phys_cores = 0;
    size_t offset = 0;
    do {
        offset += reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr.get() + offset)->Size;
        phys_cores++;
    } while (offset < sz);

    printf("original getNumberOfCPUCores: %d \n", phys_cores);
    #if TBB_INTERFACE_VERSION >= 12010// TBB has hybrid CPU aware task_arena api
    auto core_types = oneapi::tbb::info::core_types();
    if (bigCoresOnly && core_types.size() > 1) /*Hybrid CPU*/ {
        const auto little_cores = *core_types.begin();
        // assuming the Little cores feature no hyper-threading
        phys_cores -= oneapi::tbb::info::default_concurrency(little_cores);
        printf("patched getNumberOfCPUCores: %d \n", phys_cores);
    }
    #endif
    return phys_cores;
}

#if !(IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
// OMP/SEQ threading on the Windows doesn't support NUMA
std::vector<int> getAvailableNUMANodes() { return std::vector<int>(1, 0); }
#endif

}  // namespace InferenceEngine
