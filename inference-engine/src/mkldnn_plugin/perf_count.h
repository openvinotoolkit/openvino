// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>

namespace MKLDNNPlugin {

class PerfCount {
    uint64_t duration;
    uint32_t num;

    std::chrono::high_resolution_clock::time_point __start = {};
    std::chrono::high_resolution_clock::time_point __finish = {};

public:
    PerfCount(): duration(0), num(0) {}

    uint64_t avg() { return (num == 0) ? 0 : duration / num; }

private:
    void start_itr() {
        __start = std::chrono::high_resolution_clock::now();
    }

    void finish_itr() {
        __finish = std::chrono::high_resolution_clock::now();

        duration += std::chrono::duration_cast<std::chrono::microseconds>(__finish - __start).count();
        num++;
    }

    friend class PerfHelper;
};

class PerfHelper {
    PerfCount &counter;

public:
    explicit PerfHelper(PerfCount &count): counter(count) { counter.start_itr(); }

    ~PerfHelper() { counter.finish_itr(); }
};

}  // namespace MKLDNNPlugin

#define PERF(_counter) PerfHelper __helper##__counter (_counter->PerfCounter());
