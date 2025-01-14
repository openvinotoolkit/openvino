// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <ratio>

namespace ov {
namespace intel_cpu {

class PerfCount {
    uint64_t total_duration;
    uint32_t num;

    std::chrono::high_resolution_clock::time_point __start = {};
    std::chrono::high_resolution_clock::time_point __finish = {};

public:
    PerfCount() : total_duration(0), num(0) {}

    std::chrono::duration<double, std::milli> duration() const {
        return __finish - __start;
    }

    uint64_t avg() const {
        return (num == 0) ? 0 : total_duration / num;
    }
    uint32_t count() const {
        return num;
    }

private:
    void start_itr() {
        __start = std::chrono::high_resolution_clock::now();
    }

    void finish_itr() {
        __finish = std::chrono::high_resolution_clock::now();
        total_duration += std::chrono::duration_cast<std::chrono::microseconds>(__finish - __start).count();
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

}  // namespace intel_cpu
}  // namespace ov

#define GET_PERF(_node)    std::unique_ptr<PerfHelper>(new PerfHelper(_node->PerfCounter()))
#define PERF(_node, _need) auto pc = _need ? GET_PERF(_node) : nullptr;
