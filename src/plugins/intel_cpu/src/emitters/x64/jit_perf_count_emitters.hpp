// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"
#include <cpu/x64/jit_generator.hpp>

namespace ov {
namespace intel_cpu {

class jit_perf_count_start_emitter : public jit_emitter {
public:
    jit_perf_count_start_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                            const std::shared_ptr<ov::Node>& n);
    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const override;

    mutable std::chrono::high_resolution_clock::time_point* m_current_time = nullptr;

    void align_rsp() const;

    void restore_rsp() const;
};

class jit_perf_count_end_emitter : public jit_emitter {
public:
    jit_perf_count_end_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                            const std::shared_ptr<ov::Node>& n);
    size_t get_inputs_num() const override;

private:
    // use start in emit_impl
    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const override;

    mutable std::chrono::high_resolution_clock::time_point* m_start = nullptr;
    mutable uint64_t* m_accumulation = nullptr;
    mutable uint32_t* m_iteration = nullptr;

    void align_rsp() const;

    void restore_rsp() const;
};

}   // namespace intel_cpu
}   // namespace ov
