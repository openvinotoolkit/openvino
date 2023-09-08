// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"
#include <cpu/x64/jit_generator.hpp>

// if CHRONO_CALL is defined, use std::chrono::high_resolution_clock as timer
// otherwise uncomment below line to read tsc as cycle counters
#define CHRONO_CALL

namespace ov {
namespace intel_cpu {

class jit_perf_count_start_emitter : public jit_emitter {
public:
    jit_perf_count_start_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                            const std::shared_ptr<ov::Node>& n);
    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const override;
#ifdef CHRONO_CALL
    mutable std::chrono::high_resolution_clock::time_point* m_current_time = nullptr;
#else
    mutable uint64_t* m_current_count = nullptr;
#endif
};

class jit_perf_count_end_emitter : public jit_emitter {
public:
    jit_perf_count_end_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                            const std::shared_ptr<ov::Node>& n);
    size_t get_inputs_num() const override;

private:
    // use start in emit_impl
    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const override;
#ifdef CHRONO_CALL
    mutable std::chrono::high_resolution_clock::time_point* m_start = nullptr;
#else
    mutable uint64_t* m_start_count = nullptr;
#endif
    mutable uint64_t* m_accumulation = nullptr;
    mutable uint32_t* m_iteration = nullptr;
};

}   // namespace intel_cpu
}   // namespace ov
