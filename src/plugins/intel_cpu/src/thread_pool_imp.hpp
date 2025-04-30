// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/dnnl_thread.hpp>
#include <iostream>
#include <oneapi/dnnl/dnnl_threadpool_iface.hpp>

#include "openvino/core/parallel.hpp"
#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"
#include "cpu_parallel.hpp"

namespace ov::intel_cpu {

int dnnl_get_multiplier();

class ThreadPool : public dnnl::threadpool_interop::threadpool_iface {
private:
    Partitioner m_partitoner;

public:
    ThreadPool() = default;
    ThreadPool(const Partitioner& partitioner) : m_partitoner(partitioner) {}
    int get_num_threads() const override {
        int num = parallel_get_max_threads();
        return num;
    }
    bool get_in_parallel() const override {
        return 0;
    }
    uint64_t get_flags() const override {
        return 0;
    }
    void parallel_for(int n, const std::function<void(int, int)>& fn) override {
        if (m_partitoner == Partitioner::STATIC) {
            tbb::parallel_for(
                0,
                n,
                [&](int i) {
                    fn(i, n);
                },
                tbb::static_partitioner());
        } else {
            tbb::parallel_for(0, n, [&](int i) {
                fn(i, n);
            });
        }
    }
};

}  // namespace ov::intel_cpu
