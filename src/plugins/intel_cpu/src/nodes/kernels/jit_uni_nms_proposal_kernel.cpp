// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "jit_uni_nms_proposal_kernel.hpp"

using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {

template<cpu_isa_t isa>
jit_uni_nms_proposal_kernel_impl<isa>::jit_uni_nms_proposal_kernel_impl(const jit_nms_conf &jcp) :
    jit_uni_nms_proposal_kernel{jcp}, jit_generator{}, reg_is_dead_ptr{abi_not_param1}, reg_params{abi_param1} {
}

template <cpu_isa_t isa>
void jit_uni_nms_proposal_kernel_impl<isa>::generate() {
    preamble();
    Label box_loop_label;
    Label box_loop_end_label;
    Label box_loop_continue_label;
    Label vc_coordinates_offset_const;
    Label vc_ione_const;
    Label vc_fzero_const;
    Label vc_nms_thresh_const;
    xor_(reg_box_idx, reg_box_idx);
    xor_(reg_count, reg_count);
    xor_(reg_pre_nms_topn, reg_pre_nms_topn);
    mov(reg_pre_nms_topn.cvt32(), dword[reg_params + offsetof(jit_nms_call_args, pre_nms_topn)]);
    mov(reg_simd_tail_len, reg_pre_nms_topn);
    sub(reg_simd_tail_len, simd_width);
    mov(reg_x0_ptr, ptr[reg_params + offsetof(jit_nms_call_args, x0)]);
    mov(reg_y0_ptr, ptr[reg_params + offsetof(jit_nms_call_args, y0)]);
    mov(reg_x1_ptr, ptr[reg_params + offsetof(jit_nms_call_args, x1)]);
    mov(reg_y1_ptr, ptr[reg_params + offsetof(jit_nms_call_args, y1)]);
    mov(reg_is_dead_ptr, ptr[reg_params + offsetof(jit_nms_call_args, is_dead)]);
    Vmm reg_nms_thresh {13};
    uni_vbroadcastss(reg_nms_thresh, ptr[rip + vc_nms_thresh_const]);
    Vmm reg_coordinates_offset {14};
    uni_vbroadcastss(reg_coordinates_offset, ptr[rip + vc_coordinates_offset_const]);
    Vmm reg_ione {15};
    uni_vbroadcastss(reg_ione, ptr[rip + vc_ione_const]);
    L(box_loop_label);
    {
        cmp(reg_box_idx, reg_pre_nms_topn);
        jge(box_loop_end_label, T_NEAR);

        // is dead
        cmp(dword[reg_is_dead_ptr + sizeof(int) * reg_box_idx], 0);
        jne(box_loop_continue_label, T_NEAR);

        // index_out[count++] = base_index + box;
        mov(rax, ptr[reg_params + offsetof(jit_nms_call_args, index_out)]);
        mov(dword[rax + sizeof(int) * reg_count], /*base_index + */reg_box_idx.cvt32());
        inc(reg_count);

        // if (count == max_num_out) break;
        cmp(reg_count, jcp_.max_num_out);
        je(box_loop_end_label, T_NEAR);

        // int tail = box + 1;
        mov(reg_tail, reg_box_idx);
        inc(reg_tail);

        Vmm vx0i {9};
        Vmm vy0i {1};
        Vmm vx1i {2};
        Vmm vy1i {3};
        uni_vbroadcastss(vx0i, ptr[reg_x0_ptr + sizeof(float) * reg_box_idx]);
        uni_vbroadcastss(vy0i, ptr[reg_y0_ptr + sizeof(float) * reg_box_idx]);
        uni_vbroadcastss(vx1i, ptr[reg_x1_ptr + sizeof(float) * reg_box_idx]);
        uni_vbroadcastss(vy1i, ptr[reg_y1_ptr + sizeof(float) * reg_box_idx]);

        Vmm vA_area {4};
        {
            Vmm vA_width {5};
            Vmm vA_height {6};
            uni_vsubps(vA_width, vx1i, vx0i);
            uni_vsubps(vA_height, vy1i, vy0i);
            uni_vaddps(vA_width, vA_width, reg_coordinates_offset);
            uni_vaddps(vA_height, vA_height, reg_coordinates_offset);
            uni_vmulps(vA_area, vA_width, vA_height);
        }

        Label tail_loop_label;
        Label tail_loop_end_label;
        L(tail_loop_label);
        {
            cmp(reg_tail, reg_simd_tail_len);
            jg(tail_loop_end_label, T_NEAR);

            Vmm vx0j {5};
            Vmm vy0j {6};
            Vmm vx1j {7};
            Vmm vy1j {8};
            uni_vmovups(vx0j, ptr[reg_x0_ptr + sizeof(float) * reg_tail]);
            uni_vmovups(vy0j, ptr[reg_y0_ptr + sizeof(float) * reg_tail]);
            uni_vmovups(vx1j, ptr[reg_x1_ptr + sizeof(float) * reg_tail]);
            uni_vmovups(vy1j, ptr[reg_y1_ptr + sizeof(float) * reg_tail]);

            if (isa == avx512_core) {
                // Hint to pre-fetch data for the next iteration
                mov(rax, reg_tail);
                add(rax, simd_width);
                prefetcht2(ptr[reg_x0_ptr + sizeof(float) * rax]);
                prefetcht2(ptr[reg_y0_ptr + sizeof(float) * rax]);
                prefetcht2(ptr[reg_x1_ptr + sizeof(float) * rax]);
                prefetcht2(ptr[reg_y1_ptr + sizeof(float) * rax]);
            }

            Vmm varea {0};
            {
                Vmm vx0 {10};
                Vmm vx1 {11};
                Vmm vy1 {12};

                Vmm vwidth {vx1};
                uni_vmaxps(vx0, vx0i, vx0j);
                uni_vminps(vx1, vx1i, vx1j);
                uni_vsubps(vwidth, vx1, vx0);
                uni_vaddps(vwidth, vwidth, reg_coordinates_offset);
                Vmm reg_fzero {vx0};
                uni_vpxor(reg_fzero, reg_fzero, reg_fzero);
                uni_vmaxps(vwidth, vwidth, reg_fzero);

                Vmm vy0 {vx0};
                Vmm vheight {vy1};
                uni_vmaxps(vy0, vy0i, vy0j);
                uni_vminps(vy1, vy1i, vy1j);
                uni_vsubps(vheight, vy1, vy0);
                uni_vaddps(vheight, vheight, reg_coordinates_offset);
                uni_vpxor(reg_fzero, reg_fzero, reg_fzero);
                uni_vmaxps(vheight, vheight, reg_fzero);

                uni_vmulps(varea, vwidth, vheight);
            }
            Vmm vB_area {10};
            {
                Vmm vB_width {11};
                uni_vsubps(vB_width, vx1j, vx0j);
                uni_vsubps(vB_area, vy1j, vy0j);
                uni_vaddps(vB_width, vB_width, reg_coordinates_offset);
                uni_vaddps(vB_area, vB_area, reg_coordinates_offset);
                uni_vmulps(vB_area, vB_area, vB_width);
            }
            // vintersection_area calculation
            uni_vaddps(vB_area, vB_area, vA_area);
            uni_vsubps(vB_area, vB_area, varea);
            uni_vdivps(varea, varea, vB_area);

            if (isa == avx512_core) {
                // uni_* wrappers don't allow passing opmask registers
                vcmpps(k0, vx0i, vx1j, VCMPPS_LE);
                vcmpps(k1, vy0i, vy1j, VCMPPS_LE);
                vcmpps(k2, vx0j, vx1i, VCMPPS_LE);
                vcmpps(k3, vy0j, vy1i, VCMPPS_LE);
                vcmpps(k4, varea, reg_nms_thresh, VCMPPS_GT);
                vcmpps(k5, reg_nms_thresh, varea, VCMPPS_ORD);

                kandw(k0, k0, k1);
                kandw(k2, k2, k3);
                kandw(k4, k4, k0);
                kandw(k4, k4, k2);
                kandw(k4, k4, k5);

                uni_vmovdqu(ptr[reg_is_dead_ptr + sizeof(int) * reg_tail]|k4, reg_ione);
            } else {
                Vmm not_nan { vB_area };
                uni_vcmpps(not_nan, reg_nms_thresh, varea, VCMPPS_ORD);
                uni_vcmpps(vx1j, vx1j, vx0i, VCMPPS_GE);
                uni_vcmpps(vy1j, vy1j, vy0i, VCMPPS_GE);
                uni_vcmpps(vx0j, vx0j, vx1i, VCMPPS_LE);
                uni_vcmpps(vy0j, vy0j, vy1i, VCMPPS_LE);
                uni_vcmpps(varea, varea, reg_nms_thresh, VCMPPS_GT);

                uni_vandps(vx1j, vx1j, vy1j);
                uni_vandps(vx0j, vx0j, vy0j);
                uni_vandps(varea, varea, vx1j);
                uni_vandps(varea, varea, vx0j);
                uni_vandps(varea, varea, not_nan);

                Vmm reg_is_dead_val {vB_area};
                uni_vmovdqu(reg_is_dead_val, ptr[reg_is_dead_ptr + sizeof(int) * reg_tail]);
                uni_vblendvps(reg_is_dead_val, reg_is_dead_val, reg_ione, varea);
                uni_vmovdqu(ptr[reg_is_dead_ptr + sizeof(int) * reg_tail], reg_is_dead_val);
            }

            add(reg_tail, simd_width);
            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);

        // Scalar tail processing
        Label scalar_tail_loop_label;
        Label scalar_tail_loop_end_label;
        L(scalar_tail_loop_label);
        {
            cmp(reg_tail, reg_pre_nms_topn);
            jge(scalar_tail_loop_end_label, T_NEAR);

            // Use low 128 bits of the precalculated values
            Xmm vx0i {9};
            Xmm vy0i {1};
            Xmm vx1i {2};
            Xmm vy1i {3};
            Xmm vA_area {4};
            Xmm reg_nms_thresh {13};
            Xmm reg_coordinates_offset {14};
            Xmm reg_ione {15};

            Xmm vx0j {5};
            Xmm vy0j {6};
            Xmm vx1j {7};
            Xmm vy1j {8};
            uni_vmovss(vx0j, ptr[reg_x0_ptr + sizeof(float) * reg_tail]);
            uni_vmovss(vy0j, ptr[reg_y0_ptr + sizeof(float) * reg_tail]);
            uni_vmovss(vx1j, ptr[reg_x1_ptr + sizeof(float) * reg_tail]);
            uni_vmovss(vy1j, ptr[reg_y1_ptr + sizeof(float) * reg_tail]);

            Xmm varea {0};
            {
                Xmm vx0 {10};
                Xmm vx1 {11};
                Xmm vy1 {12};

                Xmm vwidth { vx1 };
                uni_vmaxss(vx0, vx0i, vx0j);
                uni_vminss(vx1, vx1i, vx1j);
                uni_vsubss(vwidth, vx1, vx0);
                uni_vaddss(vwidth, vwidth, reg_coordinates_offset);
                Xmm reg_fzero {vx0};
                uni_vpxor(reg_fzero, reg_fzero, reg_fzero);
                uni_vmaxss(vwidth, vwidth, reg_fzero);

                Xmm vy0 {vx0};
                Xmm vheight {vy1};
                uni_vmaxss(vy0, vy0i, vy0j);
                uni_vminss(vy1, vy1i, vy1j);
                uni_vsubss(vheight, vy1, vy0);
                uni_vaddss(vheight, vheight, reg_coordinates_offset);
                uni_vpxor(reg_fzero, reg_fzero, reg_fzero);
                uni_vmaxss(vheight, vheight, reg_fzero);

                uni_vmulss(varea, vwidth, vheight);
            }

            Xmm reg_is_dead_val {10}; // vB_area
            uni_vmovss(reg_is_dead_val, ptr[reg_is_dead_ptr + sizeof(int) * reg_tail]);

            Xmm vB_area {11};
            {
                Xmm vB_width {12};
                uni_vsubss(vB_width, vx1j, vx0j);
                uni_vsubss(vB_area, vy1j, vy0j);
                uni_vaddss(vB_width, vB_width, reg_coordinates_offset);
                uni_vaddss(vB_area, vB_area, reg_coordinates_offset);
                uni_vmulss(vB_area, vB_area, vB_width);
            }
            // vintersection_area calculation
            uni_vaddss(vB_area, vB_area, vA_area);
            uni_vsubss(vB_area, vB_area, varea);
            uni_vdivss(varea, varea, vB_area);

            Xmm not_nan { vB_area };
            movss(not_nan, reg_nms_thresh);
            cmpss(not_nan, varea, VCMPPS_ORD);
            cmpss(vx1j, vx0i, VCMPPS_GE);
            cmpss(vy1j, vy0i, VCMPPS_GE);
            cmpss(vy0j, vy1i, VCMPPS_LE);
            cmpss(vx0j, vx1i, VCMPPS_LE);
            cmpss(varea, reg_nms_thresh, VCMPPS_GT);

            uni_vandps(vx1j, vx1j, vy1j);
            uni_vandps(vx0j, vx0j, vy0j);
            uni_vandps(varea, varea, vx1j);
            uni_vandps(varea, varea, vx0j);
            uni_vandps(varea, varea, not_nan);

            uni_vblendvps(reg_is_dead_val, reg_is_dead_val, reg_ione, varea);
            uni_vmovss(ptr[reg_is_dead_ptr + sizeof(int) * reg_tail], reg_is_dead_val);

            inc(reg_tail);
            jmp(scalar_tail_loop_label, T_NEAR);
        }
        L(scalar_tail_loop_end_label);

        L(box_loop_continue_label);
        inc(reg_box_idx);
        jmp(box_loop_label, T_NEAR);
    }
    L(box_loop_end_label);

    mov(rax, ptr[reg_params + offsetof(jit_nms_call_args, num_out)]);
    mov(ptr[rax], reg_count);
    postamble();

    // Constants
    std::uint32_t mem_placeholder;

    L_aligned(vc_coordinates_offset_const);
    {
        memcpy(&mem_placeholder, &jcp_.coordinates_offset, sizeof(float));
        dd(mem_placeholder);
    }

    L_aligned(vc_ione_const);
    {
        dd(1);
    }

    L_aligned(vc_fzero_const);
    {
        float vc_fzero {0.f};
        memcpy(&mem_placeholder, &vc_fzero, sizeof(float));
        dd(mem_placeholder);
    }

    L_aligned(vc_nms_thresh_const);
    {
        memcpy(&mem_placeholder, &jcp_.nms_threshold, sizeof(float));
        dd(mem_placeholder);
    }
}

template <cpu_isa_t isa>
void jit_uni_nms_proposal_kernel_impl<isa>::uni_vsubss(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                                                       const Xbyak::Operand &op2) {
    if (!x.isEqualIfNotInherited(op1)) {
        assert(!x.isEqualIfNotInherited(op2));
        movss(x, op1);
    }
    subss(x, op2);
}

template <cpu_isa_t isa>
void jit_uni_nms_proposal_kernel_impl<isa>::uni_vmulss(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                                                       const Xbyak::Operand &op2) {
    if (!x.isEqualIfNotInherited(op1) && !x.isEqualIfNotInherited(op2)) {
        movss(x, op1);
        mulss(x, op2);
    } else if (!x.isEqualIfNotInherited(op1) && x.isEqualIfNotInherited(op2)) {
        mulss(x, op1);
    } else {
        mulss(x, op2);
    }
}

template class jit_uni_nms_proposal_kernel_impl<sse41>;
template class jit_uni_nms_proposal_kernel_impl<avx2>;
template class jit_uni_nms_proposal_kernel_impl<avx512_core>;

} // namespace node
} // namespace intel_cpu
} // namespace ov
