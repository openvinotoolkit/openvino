// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#include "mem_usage.hpp"

#if defined _WIN32

#    include <windows.h>
#    include <psapi.h>

#    include <cmath>
#    include <stdexcept>

int64_t intel_npu::get_peak_memory_usage() {
    PROCESS_MEMORY_COUNTERS mem_counters;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &mem_counters, sizeof(mem_counters))) {
        throw std::runtime_error("Can't get system memory values");
    }

    // Linux tracks memory usage in pages and then converts them to kB.
    // Thus, there is always some room for inaccuracy as pages are not guaranteed to be fully used.
    // In Windows, the situation is different: the system returns the memory usage in bytes, not in pages.
    // To align the output between the two operating systems as closely as possible, we have two options:
    //     1. Use rounding to the nearest integer.
    //     2. Try to estimate the number of pages used in Windows. However,
    //         this approach is likely to be inaccurate as well, so option 1 was chosen.
    static constexpr double bytes_in_kilobyte = 1024.0;

    // please note then we calculate difference
    // to get peak memory increment value, so we return int64, not size_t
    return static_cast<int64_t>(std::round(mem_counters.PeakWorkingSetSize / bytes_in_kilobyte));
}

#else

#    include <fstream>
#    include <regex>
#    include <sstream>

// clang-format on

int64_t intel_npu::get_peak_memory_usage() {
    std::size_t peak_mem_usage_kB = 0;

    std::ifstream status_file("/proc/self/status");
    std::string line;
    std::regex vm_peak_regex("VmPeak:");
    std::smatch vm_match;
    bool mem_values_found = false;
    while (std::getline(status_file, line)) {
        if (std::regex_search(line, vm_match, vm_peak_regex)) {
            std::istringstream iss(vm_match.suffix());
            iss >> peak_mem_usage_kB;
            mem_values_found = true;
        }
    }

    if (!mem_values_found) {
        throw std::runtime_error("Can't get system memory values");
    }

    // please note then we calculate difference
    // to get peak memory increment value, so we return int64, not size_t
    return static_cast<int64_t>(peak_mem_usage_kB);
}

#endif
