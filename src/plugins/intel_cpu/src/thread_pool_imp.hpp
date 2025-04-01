// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl_threadpool_iface.hpp>
#include <common/dnnl_thread.hpp>

#include <iostream>

#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"
#include "openvino/core/parallel.hpp"

namespace ov::intel_cpu {

class threadpool_t : public dnnl::threadpool_interop::threadpool_iface {
private:
    int _num_threads;
public:
    explicit threadpool_t(int num_threads) {
        _num_threads = num_threads;
    }
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
        tbb::parallel_for(
            0,
            n,
            [&](int i) {
                fn(i, n);
            },
            tbb::static_partitioner());
    }
};

threadpool_t* get_thread_pool();

}  // namespace ov::intel_cpu