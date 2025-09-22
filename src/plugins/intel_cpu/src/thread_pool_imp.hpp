// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/dnnl_thread.hpp>
#include <iostream>
#include <oneapi/dnnl/dnnl_threadpool.hpp>
#include <oneapi/dnnl/dnnl_threadpool_iface.hpp>

#include "openvino/core/parallel.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"

namespace ov::intel_cpu {

class ThreadPool : public dnnl::threadpool_interop::threadpool_iface {
private:
    ov::intel_cpu::TbbPartitioner m_partitioner = ov::intel_cpu::TbbPartitioner::STATIC;
    size_t m_multiplier = 0;

public:
    ThreadPool() = default;
    ThreadPool(const ov::intel_cpu::TbbPartitioner partitioner, const size_t multiplier)
        : m_partitioner(partitioner),
          m_multiplier(multiplier) {}
    [[nodiscard]] int get_num_threads() const override {
        int num = m_partitioner == ov::intel_cpu::TbbPartitioner::STATIC ? parallel_get_max_threads()
                                                                         : parallel_get_max_threads() * m_multiplier;
        return num;
    }
    [[nodiscard]] bool get_in_parallel() const override {
        return false;
    }
    [[nodiscard]] uint64_t get_flags() const override {
        return 0;
    }
    void parallel_for(int n, const std::function<void(int, int)>& fn) override {
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
        if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
            tbb::parallel_for(0, n, [&](int ithr) {
                fn(ithr, n);
            });
        } else {
            tbb::parallel_for(
                0,
                n,
                [&](int ithr) {
                    fn(ithr, n);
                },
                tbb::static_partitioner());
        }
    }
#endif
};

dnnl::stream make_stream(const dnnl::engine& engine, const std::shared_ptr<ThreadPool>& thread_pool = nullptr);

}  // namespace ov::intel_cpu
