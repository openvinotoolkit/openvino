// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/jit_generator.hpp>
#include <cstddef>

namespace ov::intel_cpu {

using namespace dnnl::impl::cpu::x64;

struct brgemm_decomp_kernel_params_t {
    const void* ptr_B;
    const void* scratch_buf;
    const void* bitmask_ptr;
};

struct jit_brgemm_decompress_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_decompress_kernel_t)

    jit_brgemm_decompress_kernel_t(int ic, int oc_block) : jit_generator_t(jit_name()), blocks_(ic * oc_block / 4096) {
        create_kernel();
    }

    void operator()(const brgemm_decomp_kernel_params_t* args) {
        jit_generator_t::operator()(args);
    }

private:
    int blocks_;

    Xbyak::Reg64 wei_ptr = Xbyak::Reg64(14);
    Xbyak::Reg64 dst_ptr = Xbyak::Reg64(13);  // NOLINT(clang-diagnostic-unused-private-field)

    Xbyak::Zmm zmm_comp2 = Xbyak::Zmm(28);
    Xbyak::Zmm zmm_comp1 = Xbyak::Zmm(27);
    Xbyak::Zmm zmm_comp4 = Xbyak::Zmm(26);
    Xbyak::Zmm zmm_comp3 = Xbyak::Zmm(25);

    const Xbyak::Reg64 reg_ptr_decomp_src = Xbyak::Reg64(9);
    const Xbyak::Reg64 reg_ptr_decomp_dst = Xbyak::Reg64(8);
    const Xbyak::Reg64 reg_ptr_decomp_mask = Xbyak::Reg64(0);
    const Xbyak::Reg64 reg_popcnt = Xbyak::Reg64(6);

    const Xbyak::Reg64 reg_comp_mask_tmp1 = Xbyak::Reg64(10);
    const Xbyak::Reg64 reg_comp_mask_tmp2 = Xbyak::Reg64(12);
    const Xbyak::Reg64 reg_comp_mask_tmp3 = Xbyak::Reg64(3);
    const Xbyak::Reg64 reg_comp_mask_tmp4 = Xbyak::Reg64(2);

    const Xbyak::Reg64 reg_ptr_decomp_src_align = Xbyak::Reg64(10);

    const Xbyak::Opmask reg_comp_mask1 = Xbyak::Opmask(1);
    const Xbyak::Opmask reg_comp_mask2 = Xbyak::Opmask(2);
    const Xbyak::Opmask reg_comp_mask3 = Xbyak::Opmask(3);
    const Xbyak::Opmask reg_comp_mask4 = Xbyak::Opmask(4);

    void generate() override;
};

}  // namespace ov::intel_cpu
