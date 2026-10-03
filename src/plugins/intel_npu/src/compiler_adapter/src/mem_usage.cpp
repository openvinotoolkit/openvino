// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#include "mem_usage.hpp"
#include "openvino/core/except.hpp"

#include <algorithm>

#if defined _WIN32

#    ifndef NOMINMAX
#        define NOMINMAX
#    endif

#    include <windows.h>
#    include <psapi.h>

#    include <cmath>
#    include <stdexcept>

int64_t intel_npu::get_current_memory_usage() {
    PROCESS_MEMORY_COUNTERS mem_counters;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &mem_counters, sizeof(mem_counters))) {
        OPENVINO_THROW("Can't get system memory values");
    }

    static constexpr double bytes_in_kilobyte = 1024.0;
    return static_cast<int64_t>(std::round(mem_counters.WorkingSetSize / bytes_in_kilobyte));
}

#else

#    include <fstream>
#    include <regex>
#    include <sstream>

// clang-format on

int64_t intel_npu::get_current_memory_usage() {
    std::size_t mem_usage_kB = 0;

    std::ifstream status_file("/proc/self/status");
    std::string line;
    std::regex vm_rss_regex("VmRSS:");
    std::smatch vm_match;
    bool mem_values_found = false;
    while (std::getline(status_file, line)) {
        if (std::regex_search(line, vm_match, vm_rss_regex)) {
            std::istringstream iss(vm_match.suffix());
            iss >> mem_usage_kB;
            mem_values_found = true;
        }
    }

    if (!mem_values_found) {
        OPENVINO_THROW("Can't get system memory values");
    }

    return static_cast<int64_t>(mem_usage_kB);
}

#endif

namespace intel_npu {

MemoryPeakTracker::MemoryPeakTracker(std::chrono::milliseconds pollInterval)
    : _baselineKb(get_current_memory_usage()),
      _peakKb(_baselineKb) {
    _pollThread = std::thread(&MemoryPeakTracker::poll_loop, this, pollInterval);
}

MemoryPeakTracker::~MemoryPeakTracker() {
    stop();
}

void MemoryPeakTracker::poll_loop(std::chrono::milliseconds interval) {
    while (!_stopRequested.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(interval);
        const int64_t sample = get_current_memory_usage();
        if (sample > _peakKb.load(std::memory_order_relaxed)) {
            _peakKb.store(sample, std::memory_order_relaxed);
        }
    }
}

void MemoryPeakTracker::stop() {
    if (!_stopped) {
        _stopRequested.store(true, std::memory_order_relaxed);
        if (_pollThread.joinable()) {
            _pollThread.join();
        }
        _stopped = true;
    }
}

int64_t MemoryPeakTracker::get_peak_increase_kb() {
    // One last sample to catch any spike between the previous poll and now.
    const int64_t finalSample = get_current_memory_usage();
    stop();
    const int64_t peakKb = std::max(_peakKb.load(std::memory_order_relaxed), finalSample);
    return peakKb - _baselineKb;
}

}  // namespace intel_npu
