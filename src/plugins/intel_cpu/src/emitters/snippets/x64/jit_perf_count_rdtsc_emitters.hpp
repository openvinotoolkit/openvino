// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#    pragma once

#    include "emitters/plugin/x64/jit_emitter.hpp"
#    include "transformations/snippets/x64/op/perf_count_rdtsc.hpp"

namespace ov::intel_cpu {

class jit_perf_count_rdtsc_start_emitter : public jit_emitter {
public:
    jit_perf_count_rdtsc_start_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                                       dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                       const std::shared_ptr<ov::Node>& n);
    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override;

    std::shared_ptr<ov::intel_cpu::PerfCountRdtscBegin> m_start_node = nullptr;
};

class jit_perf_count_rdtsc_end_emitter : public jit_emitter {
public:
    jit_perf_count_rdtsc_end_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                                     dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& n);
    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override;

    std::shared_ptr<ov::intel_cpu::PerfCountRdtscEnd> m_end_node = nullptr;
};

}  // namespace ov::intel_cpu

#endif  // SNIPPETS_DEBUG_CAPS
