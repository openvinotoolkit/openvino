// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_decompress_kernel.hpp"

#include <cstddef>

#define GET_OFF(field) offsetof(brgemm_decomp_kernel_params_t, field)

// NOLINTBEGIN(*)
namespace ov::intel_cpu {

using namespace Xbyak;

void jit_brgemm_decompress_kernel_t::generate() {
    preamble();
    mov(wei_ptr, ptr[param1 + GET_OFF(ptr_B)]);
    mov(reg_ptr_decomp_mask, ptr[param1 + GET_OFF(bitmask_ptr)]);
    mov(reg_ptr_decomp_dst, ptr[param1 + GET_OFF(scratch_buf)]);
    lea(reg_ptr_decomp_src, ptr[wei_ptr]);

    for (int block = 0; block < blocks_; block++) {
        int wei_offset = block * 4096;
        int bitmask_off = wei_offset / (1 * 8);
        for (int cl = 0; cl < 64; cl = cl + 4) {
            mov(reg_comp_mask_tmp1, ptr[reg_ptr_decomp_mask + cl * 8 + bitmask_off]);
            kmovq(reg_comp_mask1, reg_comp_mask_tmp1);
            mov(reg_comp_mask_tmp2, ptr[reg_ptr_decomp_mask + (cl + 1) * 8 + bitmask_off]);
            kmovq(reg_comp_mask2, reg_comp_mask_tmp2);
            mov(reg_comp_mask_tmp3, ptr[reg_ptr_decomp_mask + (cl + 2) * 8 + bitmask_off]);
            kmovq(reg_comp_mask3, reg_comp_mask_tmp3);
            mov(reg_comp_mask_tmp4, ptr[reg_ptr_decomp_mask + (cl + 3) * 8 + bitmask_off]);
            kmovq(reg_comp_mask4, reg_comp_mask_tmp4);

            vmovdqu8(zmm_comp1, ptr[reg_ptr_decomp_src]);
            popcnt(reg_popcnt, reg_comp_mask_tmp1);
            add(reg_ptr_decomp_src, reg_popcnt);

            vmovdqu8(zmm_comp2, ptr[reg_ptr_decomp_src]);
            popcnt(reg_popcnt, reg_comp_mask_tmp2);
            add(reg_ptr_decomp_src, reg_popcnt);

            vmovdqu8(zmm_comp3, ptr[reg_ptr_decomp_src]);
            popcnt(reg_popcnt, reg_comp_mask_tmp3);
            add(reg_ptr_decomp_src, reg_popcnt);

            vmovdqu8(zmm_comp4, ptr[reg_ptr_decomp_src]);
            popcnt(reg_popcnt, reg_comp_mask_tmp4);
            add(reg_ptr_decomp_src, reg_popcnt);

            vpexpandb(zmm_comp1 | reg_comp_mask1 | T_z, zmm_comp1);
            vmovdqu8(ptr[reg_ptr_decomp_dst + wei_offset + cl * 64], zmm_comp1);

            vpexpandb(zmm_comp2 | reg_comp_mask2 | T_z, zmm_comp2);
            vmovdqu8(ptr[reg_ptr_decomp_dst + wei_offset + (cl + 1) * 64], zmm_comp2);

            vpexpandb(zmm_comp3 | reg_comp_mask3 | T_z, zmm_comp3);
            vmovdqu8(ptr[reg_ptr_decomp_dst + wei_offset + (cl + 2) * 64], zmm_comp3);

            vpexpandb(zmm_comp4 | reg_comp_mask4 | T_z, zmm_comp4);
            vmovdqu8(ptr[reg_ptr_decomp_dst + wei_offset + (cl + 3) * 64], zmm_comp4);
        }

        mov(reg_ptr_decomp_src_align, reg_ptr_decomp_src);
        not_(reg_ptr_decomp_src_align);
        and_(reg_ptr_decomp_src_align, 0x3f);
        add(reg_ptr_decomp_src_align, 0x1);
        and_(reg_ptr_decomp_src_align, 0x3f);
        add(reg_ptr_decomp_src, reg_ptr_decomp_src_align);
    }
    postamble();
}

}  // namespace ov::intel_cpu
// NOLINTEND(*)
