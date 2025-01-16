// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"

namespace ov {
namespace intel_cpu {

// The class emit register spills for the possible call of external binary code
class EmitABIRegSpills {
public:
    EmitABIRegSpills(dnnl::impl::cpu::x64::jit_generator* h);
    ~EmitABIRegSpills();

    // push (save) all registers on the stack
    void preamble();
    // pop (take) all registers from the stack
    void postamble();

    // align stack on 16-byte and allocate shadow space as ABI reqiures
    // callee is responsible to save and restore `rbx`. `rbx` must not be changed after call callee.
    void rsp_align();
    void rsp_restore();

private:
    EmitABIRegSpills() = default;

    static dnnl::impl::cpu::x64::cpu_isa_t get_isa();

    inline size_t get_max_vecs_count() const { return dnnl::impl::cpu::x64::isa_num_vregs(isa); }
    inline size_t get_vec_length() const { return dnnl::impl::cpu::x64::isa_max_vlen(isa); }

    dnnl::impl::cpu::x64::jit_generator* h {nullptr};
    const dnnl::impl::cpu::x64::cpu_isa_t isa {dnnl::impl::cpu::x64::cpu_isa_t::isa_undef};

    static constexpr int k_mask_size = 8;
    static constexpr int k_mask_num = 8;
    static constexpr int gpr_size = 8;

    bool spill_status = true;
    bool rsp_status = true;
};

}   // namespace intel_cpu
}   // namespace ov
