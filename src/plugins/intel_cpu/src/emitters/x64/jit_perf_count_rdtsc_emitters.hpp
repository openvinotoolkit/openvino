// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"
#include <cpu/x64/jit_generator.hpp>
#include "transformations/snippets/x64/op/perf_count_rdtsc.hpp"

// if CHRONO_CALL is defined, use std::chrono::high_resolution_clock as timer
// otherwise uncomment below line to read tsc as cycle counters
#define CHRONO_CALL

namespace ov {
namespace intel_cpu {

class jit_perf_count_rdtsc_start_emitter : public jit_emitter {
public:
    jit_perf_count_rdtsc_start_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                            const std::shared_ptr<ov::Node>& n);
    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const override;
    mutable uint64_t* m_current_count = nullptr;
};

class jit_perf_count_rdtsc_end_emitter : public jit_emitter {
public:
    jit_perf_count_rdtsc_end_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                            const std::shared_ptr<ov::Node>& n);
    size_t get_inputs_num() const override;

private:
    // use start in emit_impl
    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const override;
    mutable uint64_t* m_start_count = nullptr;
    mutable uint64_t* m_accumulation = nullptr;
    mutable uint32_t* m_iteration = nullptr;
};

}   // namespace intel_cpu
}   // namespace ov
