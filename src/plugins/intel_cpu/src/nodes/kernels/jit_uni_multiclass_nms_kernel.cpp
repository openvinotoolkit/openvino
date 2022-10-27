// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "jit_uni_multiclass_nms_kernel.hpp"

using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {

template<cpu_isa_t isa>
jit_uni_multiclass_nms_kernel_impl<isa>::jit_uni_multiclass_nms_kernel_impl() :
    jit_uni_multiclass_nms_kernel{}, jit_generator{jit_name()}, reg_params_{abi_param1} {
}

template <cpu_isa_t isa>
void jit_uni_multiclass_nms_kernel_impl<isa>::generate() {
    preamble();

    Label box_loop_label;
    Label box_loop_continue_label;
    Label box_loop_end_label;

    Label iou_loop_label;
    Label iou_loop_continue_label;
    Label iou_loop_end_label;

    Label const_fhalf;

    // float iou_threshold;
    PVmm reg_iou_threshold {reg_pool_};
    uni_vbroadcastss(reg_iou_threshold, ptr[reg_params_ + offsetof(jit_nms_call_args, iou_threshold)]);
    // float score_threshold, used as integer for equality checks
    PReg32 reg_score_threshold {reg_pool_};
    mov(reg_score_threshold, dword[reg_params_ + offsetof(jit_nms_call_args, score_threshold)]);
    // float nms_eta;
    PVmm reg_nms_eta {reg_pool_};
    uni_vbroadcastss(reg_nms_eta, ptr[reg_params_ + offsetof(jit_nms_call_args, nms_eta)]);
    // float coordinates_offset;
    PVmm reg_coordinates_offset {reg_pool_};
    uni_vbroadcastss(reg_coordinates_offset, ptr[reg_params_ + offsetof(jit_nms_call_args, coordinates_offset)]);

    // Box* boxes_ptr;
    PReg64 reg_boxes_ptr {reg_pool_};
    mov(reg_boxes_ptr, qword[reg_params_ + offsetof(jit_nms_call_args, boxes_ptr)]);
    // int num_boxes;
    PReg64 reg_num_boxes {reg_pool_};
    mov(static_cast<Reg64>(reg_num_boxes).cvt32(), dword[reg_params_ + offsetof(jit_nms_call_args, num_boxes)]);
    // const float* coords_ptr;
    PReg64 reg_coords_array_ptr {reg_pool_};
    mov(reg_coords_array_ptr, qword[reg_params_ + offsetof(jit_nms_call_args, coords_ptr)]);

    PVmm reg_halfs {reg_pool_};
    uni_vbroadcastss(reg_halfs, ptr[rip + const_fhalf]);

    // int num_boxes_selected = 0;
    PReg64 reg_num_boxes_selected {reg_pool_};
    xor_(reg_num_boxes_selected, reg_num_boxes_selected);

    // for (size_t i = 0; i < args->num_boxes; i++) {
    PReg64 reg_i {reg_pool_};
    xor_(reg_i, reg_i);
    L(box_loop_label);
    {
        cmp(reg_i, reg_num_boxes);
        jge(box_loop_end_label, T_NEAR);

        PReg32 box_score {reg_pool_};

        PVmm xminI {reg_pool_};
        PVmm yminI {reg_pool_};
        PVmm xmaxI {reg_pool_};
        PVmm ymaxI {reg_pool_};
        {
            // const float* box_coords_ptr = &args->coords_ptr[args->boxes_ptr[i].box_idx * 4];
            PReg64 box_ptr {reg_pool_};
            inline_get_box_ptr(reg_boxes_ptr, reg_i, box_ptr);
            mov(box_score, dword[box_ptr + offsetof(Box, score)]);
            const PReg64& box_coords_ptr = box_ptr;
            inline_get_box_coords_ptr(box_ptr, reg_coords_array_ptr, box_coords_ptr);

            uni_vbroadcastss(xminI, ptr[box_coords_ptr]);
            uni_vbroadcastss(yminI, ptr[box_coords_ptr + 4]);
            uni_vbroadcastss(xmaxI, ptr[box_coords_ptr + 8]);
            uni_vbroadcastss(ymaxI, ptr[box_coords_ptr + 12]);
        }

        // float areaI = (ymaxI - yminI + norm) * (xmaxI - xminI + norm);
        PVmm areaI {reg_pool_};
        {
            PVmm width {reg_pool_};
            PVmm height {reg_pool_};
            uni_vsubps(width, xmaxI, xminI);
            uni_vsubps(height, ymaxI, yminI);
            uni_vaddps(width, width, reg_coordinates_offset);
            uni_vaddps(height, height, reg_coordinates_offset);
            uni_vmulps(areaI, width, height);
        }

        // for (int j = num_boxes_selected - simd_width - 1; (j + simd_width) >= 0; j -= simd_width)
        PReg64 reg_j {reg_pool_};
        mov(reg_j, reg_num_boxes_selected);
        sub(reg_j, simd_width + 1);
        L(iou_loop_label);
        {
            PReg64 reg_k {reg_pool_};
            mov(reg_k, reg_j);
            add(reg_k, simd_width);
            js(iou_loop_end_label, T_NEAR);

            // &args->coords_ptr[args->boxes_ptr[j].box_idx * 4]
            PVmm xminJ {reg_pool_};
            PVmm yminJ {reg_pool_};
            PVmm xmaxJ {reg_pool_};
            PVmm ymaxJ {reg_pool_};
            for (int i = 0; i < simd_width; ++i) {
                PReg64 zero {reg_pool_};
                xor_(zero, zero);
                mov(rax, reg_k);
                sub(rax, i);
                cmovs(rax, zero);
                inline_get_box_coords_ptr(reg_boxes_ptr, rax, reg_coords_array_ptr, rax);
                inline_pinsrd(xminJ, dword[rax], i);
                inline_pinsrd(yminJ, dword[rax + 4], i);
                inline_pinsrd(xmaxJ, dword[rax + 8], i);
                inline_pinsrd(ymaxJ, dword[rax + 12], i);
            }

            // const float iou = intersection_over_union();
            PVmm reg_iou;
            {
                // float areaJ = (ymaxJ - yminJ + norm) * (xmaxJ - xminJ + norm);
                PVmm areaJ;
                {
                    PVmm width {reg_pool_};
                    PVmm height {reg_pool_};
                    uni_vsubps(width, xmaxJ, xminJ);
                    uni_vsubps(height, ymaxJ, yminJ);
                    uni_vaddps(width, width, reg_coordinates_offset);
                    uni_vaddps(height, height, reg_coordinates_offset);
                    uni_vmulps(width, width, height);
                    areaJ = std::move(width);
                }

                PVmm intersection_area;
                {
                    PVmm tmp0 {reg_pool_};

                    // std::max(std::min(ymaxI, ymaxJ) - std::max(yminI, yminJ) + norm, 0.f)
                    PVmm height {reg_pool_};
                    uni_vminps(height, ymaxI, ymaxJ);
                    uni_vmaxps(tmp0, yminI, yminJ);
                    uni_vsubps(height, height, tmp0);
                    uni_vaddps(height, height, reg_coordinates_offset);
                    uni_vpxor(tmp0, tmp0, tmp0);
                    uni_vmaxps(height, height, tmp0);
                    ymaxJ.release();
                    yminJ.release();

                    // std::max(std::min(xmaxI, xmaxJ) - std::max(xminI, xminJ) + norm, 0.f)
                    PVmm width {reg_pool_};
                    uni_vminps(width, xmaxI, xmaxJ);
                    uni_vmaxps(tmp0, xminI, xminJ);
                    uni_vsubps(width, width, tmp0);
                    uni_vaddps(width, width, reg_coordinates_offset);
                    uni_vpxor(tmp0, tmp0, tmp0);
                    uni_vmaxps(width, width, tmp0);
                    xmaxJ.release();
                    xminJ.release();

                    uni_vmulps(width, width, height);
                    intersection_area = std::move(width);
                }

                // iou_denominator = (areaI + areaJ - intersection_area)
                PVmm iou_denominator {reg_pool_};
                uni_vaddps(iou_denominator, areaJ, areaI);
                uni_vsubps(iou_denominator, iou_denominator, intersection_area);

                // if (areaI <= 0.f || areaJ <= 0.f)
                //     reg_iou = 0.f;
                PVmm zero {reg_pool_};
                uni_vpxor(zero, zero, zero);
                if (isa == avx512_core) {
                    vcmpps(k1, areaI, zero, VCMPPS_GT);
                    vcmpps(k2, areaJ, zero, VCMPPS_GT);
                    kandw(k1, k1, k2);
                    reg_iou = PVmm {reg_pool_};
                    vdivps(static_cast<Vmm>(reg_iou)|k1|T_z, intersection_area, iou_denominator);
                } else {
                    PVmm maskI {reg_pool_};
                    PVmm maskJ {reg_pool_};
                    uni_vcmpps(maskI, areaI, zero, VCMPPS_GT);
                    uni_vcmpps(maskJ, areaJ, zero, VCMPPS_GT);
                    uni_vandps(maskI, maskI, maskJ);
                    // reg_iou = intersection_area / (areaI + areaJ - intersection_area)
                    reg_iou = PVmm {reg_pool_};
                    uni_vdivps(reg_iou, intersection_area, iou_denominator);
                    uni_vandps(reg_iou, reg_iou, maskI);
                }
            }

#undef BUG_IN_REFERENCE_IMPL_IS_FIXED
#ifdef BUG_IN_REFERENCE_IMPL_IS_FIXED
            // if (iou >= iou_threshold) {
            //     box_is_selected = false;
            //     break;
            // }
            if (isa == avx512_core) {
                vcmpps(k1, reg_iou, reg_iou_threshold, VCMPPS_GE);
                ktestw(k1, k1);
                jnz(box_loop_continue_label, T_NEAR);
            } else {
                uni_vcmpps(reg_iou, reg_iou, reg_iou_threshold, VCMPPS_GE);
                uni_vtestps(reg_iou, reg_iou);
                jnz(box_loop_continue_label, T_NEAR);
            }
#else // BUG_IN_REFERENCE_IMPL_IS_FIXED
            // box_is_selected = (iou < iou_threshold);
            // if (!box_is_selected || scoreI_equal_to_threshold) // TODO: scoreI_equal_to_threshold - bug in reference impl?
            //     break;
            Label scoreI_equal_to_threshold_label;
            cmp(box_score, reg_score_threshold);
            je(scoreI_equal_to_threshold_label, T_NEAR);
            {
                if (isa == avx512_core) {
                    vcmpps(k1, reg_iou, reg_iou_threshold, VCMPPS_GE);
                    ktestw(k1, k1);
                    jnz(box_loop_continue_label, T_NEAR);
                } else {
                    uni_vcmpps(reg_iou, reg_iou, reg_iou_threshold, VCMPPS_GE);
                    uni_vtestps(reg_iou, reg_iou);
                    jnz(box_loop_continue_label, T_NEAR);
                }
                jmp(iou_loop_continue_label, T_NEAR);
            }
            L(scoreI_equal_to_threshold_label);
            {
                if (isa == avx512_core) {
                    vcmpps(k1, reg_iou, reg_iou_threshold, VCMPPS_GE);
                    PReg64 mask {reg_pool_};
                    mov(mask, 1);
                    kmovq(k2, mask);
                    kandw(k1, k1, k2);
                    ktestw(k1, k1);
                    jnz(box_loop_continue_label, T_NEAR);
                } else {
                    uni_vcmpps(reg_iou, reg_iou, reg_iou_threshold, VCMPPS_GE);
                    uni_vpextrd(eax, Xmm {reg_iou.getIdx()}, 0);
                    test(eax, eax);
                    jnz(box_loop_continue_label, T_NEAR);
                }
                jmp(iou_loop_end_label, T_NEAR);
            }
            L(iou_loop_continue_label);
#endif // BUG_IN_REFERENCE_IMPL_IS_FIXED
            sub(reg_j, simd_width);
            jmp(iou_loop_label, T_NEAR);
        }
        L(iou_loop_end_label);

        // if (iou_threshold > 0.5f) {
        //     iou_threshold *= args->nms_eta;
        // }
        // args->boxes[num_boxes_selected++] = args->boxes[i];
        {
            Label copy_box_label;
            if (isa == avx512_core) {
                vcmpps(k1, reg_iou_threshold, reg_halfs, VCMPPS_GT);
                ktestw(k1, k1);
                jz(copy_box_label, T_NEAR);
            } else {
                PVmm tmp {reg_pool_};
                uni_vcmpps(tmp, reg_iou_threshold, reg_halfs, VCMPPS_GT);
                uni_vtestps(tmp, tmp);
                jz(copy_box_label, T_NEAR);
            }
            uni_vmulps(reg_iou_threshold, reg_iou_threshold, reg_nms_eta);
            L(copy_box_label);
            {
                PReg64 src {reg_pool_};
                PReg64 dst {reg_pool_};
                assert(sizeof(Box) == 16);
                mov(src, reg_i);
                sal(src, 4);
                lea(src, ptr[reg_boxes_ptr + src]);
                mov(dst, reg_num_boxes_selected);
                inc(reg_num_boxes_selected);
                sal(dst, 4);
                lea(dst, ptr[reg_boxes_ptr + dst]);
                mov(rax, qword[src]);
                mov(qword[dst], rax);
                mov(rax, qword[src + 8]);
                mov(qword[dst + 8], rax);
            }
        }

        L(box_loop_continue_label);
        inc(reg_i);
        jmp(box_loop_label, T_NEAR);
    }
    L(box_loop_end_label);

    // *args->num_boxes_selected = num_boxes_selected;
    mov(rax, qword[reg_params_ + offsetof(jit_nms_call_args, num_boxes_selected_ptr)]);
    mov(qword[rax], reg_num_boxes_selected);
    postamble();

    // Constants
    std::uint32_t mem_placeholder;

    L_aligned(const_fhalf);
    {
        float vc_fhalf {0.5f};
        memcpy(&mem_placeholder, &vc_fhalf, sizeof(float));
        dd(mem_placeholder);
    }
}

/*
    boxes_ptr[box_idx]
 */
template <cpu_isa_t isa>
void jit_uni_multiclass_nms_kernel_impl<isa>::inline_get_box_ptr(
    Reg64 boxes_ptr, Reg64 box_idx, Reg64 result) {
    assert(sizeof(Box) == 16);
    mov(result, box_idx);
    sal(result, 4);
    lea(result, ptr[boxes_ptr + result]);
}

/*
    coords_array_ptr[box_idx * 4];
 */
template <cpu_isa_t isa>
void jit_uni_multiclass_nms_kernel_impl<isa>::inline_get_box_coords_ptr(
    Reg64 box_ptr, Reg64 coords_array_ptr, Reg64 result) {
    mov(result.cvt32(), dword[box_ptr + offsetof(Box, box_idx)]);
    sal(result, 4);
    lea(result, ptr[coords_array_ptr + result]);
}

/*
    rax = coords_array_ptr[boxes_ptr[box_idx].box_idx * 4];
 */
template <cpu_isa_t isa>
void jit_uni_multiclass_nms_kernel_impl<isa>::inline_get_box_coords_ptr(
    Reg64 boxes_ptr, Reg64 box_idx, Reg64 coords_array_ptr, Reg64 result) {
    inline_get_box_ptr(boxes_ptr, box_idx, result);
    inline_get_box_coords_ptr(result, coords_array_ptr, result);
}

template <>
void jit_uni_multiclass_nms_kernel_impl<sse41>::inline_pinsrd(const Vmm& x1, const Operand& op, const int imm) {
    pinsrd(x1, op, imm);
}

template <>
void jit_uni_multiclass_nms_kernel_impl<avx2>::inline_pinsrd(const Vmm& x1, const Operand& op, const int imm) {
    if (imm < 4) {
        pinsrd(Xmm {x1.getIdx()}, op, imm);
    } else {
        PXmm tmp {reg_pool_};
        vextracti128(tmp, x1, 1);
        pinsrd(tmp, op, imm % 4);
        vinserti128(x1, x1, tmp, 1);
    }
}

template <>
void jit_uni_multiclass_nms_kernel_impl<avx512_core>::inline_pinsrd(const Vmm& x1, const Operand& op, const int imm) {
    PReg64 mask {reg_pool_};
    mov(mask, 1);
    sal(mask, imm);
    kmovq(k1, mask);
    vpbroadcastd(x1|k1, op);
}

template class jit_uni_multiclass_nms_kernel_impl<sse41>;
template class jit_uni_multiclass_nms_kernel_impl<avx2>;
template class jit_uni_multiclass_nms_kernel_impl<avx512_core>;

} // namespace node
} // namespace intel_cpu
} // namespace ov
