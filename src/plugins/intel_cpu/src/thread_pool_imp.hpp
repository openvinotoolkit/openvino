// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/dnnl_thread.hpp>
#include <oneapi/dnnl/dnnl_threadpool.hpp>
#include <oneapi/dnnl/dnnl_threadpool_iface.hpp>
#include <openvino/core/parallel.hpp>

#include "cpu_parallel.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"

namespace ov::intel_cpu {

class ThreadPool : public dnnl::threadpool_interop::threadpool_iface {
public:
    ThreadPool() = delete;
    ThreadPool(ThreadPool&) = delete;
    ThreadPool& operator=(ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    explicit ThreadPool(const CpuParallel& cpu_parallel) : m_cpu_parallel(cpu_parallel) {}

    [[nodiscard]] int get_num_threads() const override {
        return m_cpu_parallel.get_num_threads();
    }
    [[nodiscard]] bool get_in_parallel() const override {
        return false;
    }
    [[nodiscard]] uint64_t get_flags() const override {
        return 0;
    }
    void parallel_for(int n, const std::function<void(int, int)>& fn) override {
        m_cpu_parallel.parallel_simple(n, fn);
    }

private:
    const CpuParallel& m_cpu_parallel;
};

dnnl::stream make_stream(const dnnl::engine& engine, const std::shared_ptr<ThreadPool>& thread_pool = nullptr);

}  // namespace ov::intel_cpu
