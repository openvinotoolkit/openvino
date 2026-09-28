// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

namespace intel_npu {

// Current resident memory usage of this process, in KB.
int64_t get_current_memory_usage();

// Tracks the peak resident memory reached while an instance is alive, by polling
// get_current_memory_usage() from a background thread.
//
// Unlike the OS-level "peak since process start" counters (VmHWM / PeakWorkingSetSize), this is
// not contaminated by memory peaks left over from earlier, unrelated work in the same process:
// the peak it reports is only ever computed from samples taken during this instance's lifetime.
class MemoryPeakTracker {
public:
    explicit MemoryPeakTracker(std::chrono::milliseconds pollInterval = std::chrono::milliseconds(1));
    ~MemoryPeakTracker();

    MemoryPeakTracker(const MemoryPeakTracker&) = delete;
    MemoryPeakTracker& operator=(const MemoryPeakTracker&) = delete;

    // Stops sampling and returns how far above the construction-time baseline the observed peak
    // got, in KB.
    int64_t get_peak_increase_kb();

private:
    void poll_loop(std::chrono::milliseconds interval);
    void stop();

    int64_t _baselineKb;
    std::atomic<int64_t> _peakKb;
    std::atomic<bool> _stopRequested{false};
    std::thread _pollThread;
    bool _stopped = false;
};

}  // namespace intel_npu
