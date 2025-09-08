// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <cstdint>
#include <ratio>

namespace ov::intel_cpu {

class PerfCount {
    uint64_t total_duration = 0;
    uint32_t num = 0;

    std::chrono::high_resolution_clock::time_point _start;
    std::chrono::high_resolution_clock::time_point _finish;

public:
    PerfCount() = default;

    [[nodiscard]] std::chrono::duration<double, std::milli> duration() const {
        return _finish - _start;
    }

    [[nodiscard]] uint64_t avg() const {
        return (num == 0) ? 0 : total_duration / num;
    }
    [[nodiscard]] uint32_t count() const {
        return num;
    }

private:
    void start_itr() {
        _start = std::chrono::high_resolution_clock::now();
    }

    void finish_itr() {
        _finish = std::chrono::high_resolution_clock::now();
        total_duration += std::chrono::duration_cast<std::chrono::microseconds>(_finish - _start).count();
        num++;
    }

    friend class PerfHelper;
};

class PerfHelper {
    PerfCount& counter;

public:
    explicit PerfHelper(PerfCount& count) : counter(count) {
        counter.start_itr();
    }

    ~PerfHelper() {
        counter.finish_itr();
    }
};

}  // namespace ov::intel_cpu

#define GET_PERF(_node)    std::unique_ptr<PerfHelper>(new PerfHelper((_node)->PerfCounter()))
#define PERF(_node, _need) auto pc = (_need) ? GET_PERF(_node) : nullptr;
