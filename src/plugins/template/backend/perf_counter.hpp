// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <memory>
#include <ratio>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace runtime {
namespace interpreter {

static const char PERF_COUNTER_NAME[] = "template_perf_counter";

class PerfCounter {
    uint64_t total_duration;
    uint32_t num;

    std::chrono::high_resolution_clock::time_point __start = {};
    std::chrono::high_resolution_clock::time_point __finish = {};

public:
    PerfCounter() : total_duration(0), num(0) {}

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
    std::shared_ptr<PerfCounter> counter;

public:
    explicit PerfHelper(const std::shared_ptr<ov::Node>& node) {
        auto info = node->get_rt_info();
        const auto& it = info.find(ov::runtime::interpreter::PERF_COUNTER_NAME);
        OPENVINO_ASSERT(it != info.end(), "Operation ", node, " doesn't contain performance counter");
        counter = it->second.as<std::shared_ptr<ov::runtime::interpreter::PerfCounter>>();
        OPENVINO_ASSERT(counter, "Performance counter is empty");
        counter->start_itr();
    }

    ~PerfHelper() {
        counter->finish_itr();
    }
};

}  // namespace interpreter
}  // namespace runtime
}  // namespace ov

#define GET_PERF(node)   std::unique_ptr<PerfHelper>(new PerfHelper(node))
#define PERF(node, need) auto pc = need ? GET_PERF(node) : nullptr
