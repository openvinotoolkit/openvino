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
    jit_uni_multiclass_nms_kernel{},
    jit_generator{jit_name(), nullptr, MAX_CODE_SIZE, true, isa},
    reg_pool_{RegistersPool::create<isa>({Reg64(Operand::RAX), Reg64(Operand::RCX), Reg64(Operand::RBP), param1})} {
}

template <cpu_isa_t isa>
void jit_uni_multiclass_nms_kernel_impl<isa>::generate() {
    preamble();

    Label box_loop_label;
    Label box_loop_continue_label;
    Label box_loop_end_label;

    Label iou_loop_label;
    Label iou_loop_end_label;

    Label const_fhalf;

    PVmm reg_iou_threshold {reg_pool_};
    uni_vbroadcastss(reg_iou_threshold, ptr[param1 + offsetof(jit_nms_call_args, iou_threshold)]);

    PVmm reg_nms_eta {reg_pool_};
    uni_vbroadcastss(reg_nms_eta, ptr[param1 + offsetof(jit_nms_call_args, nms_eta)]);

    PVmm reg_coordinates_offset {reg_pool_};
    uni_vbroadcastss(reg_coordinates_offset, ptr[param1 + offsetof(jit_nms_call_args, coordinates_offset)]);

    PReg64 reg_boxes_ptr {reg_pool_};
    mov(reg_boxes_ptr, qword[param1 + offsetof(jit_nms_call_args, boxes_ptr)]);

    PReg64 reg_num_boxes {reg_pool_};
    mov(static_cast<Reg64>(reg_num_boxes).cvt32(), dword[param1 + offsetof(jit_nms_call_args, num_boxes)]);

    PReg64 reg_coords_array_ptr {reg_pool_};
    mov(reg_coords_array_ptr, qword[param1 + offsetof(jit_nms_call_args, coords_ptr)]);

    PReg64 reg_xmin_ptr {reg_pool_};
    PReg64 reg_ymin_ptr {reg_pool_};
    PReg64 reg_xmax_ptr {reg_pool_};
    PReg64 reg_ymax_ptr {reg_pool_};
    mov(reg_xmin_ptr, qword[param1 + offsetof(jit_nms_call_args, xmin_ptr)]);
    mov(reg_ymin_ptr, qword[param1 + offsetof(jit_nms_call_args, ymin_ptr)]);
    mov(reg_xmax_ptr, qword[param1 + offsetof(jit_nms_call_args, xmax_ptr)]);
    mov(reg_ymax_ptr, qword[param1 + offsetof(jit_nms_call_args, ymax_ptr)]);

    PVmm reg_halfs {reg_pool_};
    uni_vbroadcastss(reg_halfs, ptr[rip + const_fhalf]);

    // int num_boxes_selected = 0;
    PReg64 reg_num_boxes_selected {reg_pool_};
    xor_(reg_num_boxes_selected, reg_num_boxes_selected);

    // for (int i = 0; i < args->num_boxes; ++i)
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
            // const Box& candidate_box = args->boxes_ptr[i];
            PReg64 box_ptr {reg_pool_};
            get_box_ptr(reg_boxes_ptr, reg_i, box_ptr);

            mov(box_score, dword[box_ptr + offsetof(Box, score)]);

            // const float* const candidate_box_coords = &args->coords_ptr[candidate_box.box_idx * 4];
            const PReg64& box_coords_ptr = box_ptr;
            get_box_coords_ptr(box_ptr, reg_coords_array_ptr, box_coords_ptr);

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

#undef BUG_IN_REFERENCE_IMPL_IS_FIXED
#ifdef BUG_IN_REFERENCE_IMPL_IS_FIXED
        // for (int reg_j = 0; reg_j < num_boxes_selected; reg_j += simd_width)
        PReg64 reg_j {reg_pool_};
        xor_(reg_j, reg_j);
#else
        // const bool scoreI_equal_to_threshold = (candidate_box.score == args->score_threshold);
        // const int from_idx = scoreI_equal_to_threshold ? std::max(num_boxes_selected - 1, 0) : 0;
        // for (int reg_j = from_idx; reg_j < num_boxes_selected; reg_j += simd_width)
        PReg64 reg_j {reg_pool_};
        {
            Reg32 reg_score_threshold {Operand::ECX};
            mov(reg_score_threshold, dword[param1 + offsetof(jit_nms_call_args, score_threshold)]);
            Reg64 reg_zero {Operand::RAX};
            xor_(reg_zero, reg_zero);
            mov(reg_j, reg_num_boxes_selected);
            dec(reg_j);
            cmp(box_score, reg_score_threshold);
            cmovne(reg_j, reg_zero);
            test(reg_num_boxes_selected, reg_num_boxes_selected);
            cmovz(reg_j, reg_zero);
            box_score.release();
        }
#endif
        L(iou_loop_label);
        {
            cmp(reg_j, reg_num_boxes_selected);
            jge(iou_loop_end_label, T_NEAR);

            PVmm xminJ {reg_pool_};
            PVmm yminJ {reg_pool_};
            PVmm xmaxJ {reg_pool_};
            PVmm ymaxJ {reg_pool_};
            load_simd_register(xminJ, reg_xmin_ptr, reg_num_boxes_selected, reg_j);
            load_simd_register(yminJ, reg_ymin_ptr, reg_num_boxes_selected, reg_j);
            load_simd_register(xmaxJ, reg_xmax_ptr, reg_num_boxes_selected, reg_j);
            load_simd_register(ymaxJ, reg_ymax_ptr, reg_num_boxes_selected, reg_j);

            // const float reg_iou = intersection_over_union();
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
                // reg_iou = intersection_area / iou_denominator;
                PVmm zero {reg_pool_};
                uni_vpxor(zero, zero, zero);
                if (isa == avx512_core) {
                    vcmpps(k1, areaI, zero, VCMPPS_GT);
                    vcmpps(k2, areaJ, zero, VCMPPS_GT);
                    kandw(k1, k1, k2);
                    vdivps(static_cast<Vmm>(intersection_area)|k1|T_z, intersection_area, iou_denominator);
                    reg_iou = std::move(intersection_area);
                } else {
                    PVmm maskI {reg_pool_};
                    PVmm maskJ {reg_pool_};
                    uni_vcmpps(maskI, areaI, zero, VCMPPS_GT);
                    uni_vcmpps(maskJ, areaJ, zero, VCMPPS_GT);
                    uni_vandps(maskI, maskI, maskJ);
                    uni_vdivps(intersection_area, intersection_area, iou_denominator);
                    reg_iou = std::move(intersection_area);
                    uni_vandps(reg_iou, reg_iou, maskI);
                }
            }

            // candidate_box_selected = (iou < iou_threshold);
            // if (!candidate_box_selected)
            //     break;
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
            }

            add(reg_j, simd_width);
            jmp(iou_loop_label, T_NEAR);
        }
        L(iou_loop_end_label);

        {
            // if (iou_threshold > 0.5f) {
            //     iou_threshold *= args->nms_eta;
            // }
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
                // args->boxes_ptr[num_boxes_selected] = candidate_box;
                {
                    PReg64 tmp {reg_pool_};
                    Reg64 src {Operand::RCX};
                    Reg64 dst {Operand::RAX};
                    assert(sizeof(Box) == 16);
                    mov(src, reg_i);
                    shl(src, 4);
                    lea(src, ptr[reg_boxes_ptr + src]);
                    mov(dst, reg_num_boxes_selected);
                    shl(dst, 4);
                    lea(dst, ptr[reg_boxes_ptr + dst]);
                    mov(tmp, qword[src]);
                    mov(qword[dst], tmp);
                    mov(tmp, qword[src + 8]);
                    mov(qword[dst + 8], tmp);
                }

                // args->xmin_ptr[num_boxes_selected] = candidate_box_coords[0];
                // args->ymin_ptr[num_boxes_selected] = candidate_box_coords[1];
                // args->xmax_ptr[num_boxes_selected] = candidate_box_coords[2];
                // args->ymax_ptr[num_boxes_selected] = candidate_box_coords[3];
                uni_vmovss(dword[reg_xmin_ptr + sizeof(float)*reg_num_boxes_selected], xminI);
                uni_vmovss(dword[reg_ymin_ptr + sizeof(float)*reg_num_boxes_selected], yminI);
                uni_vmovss(dword[reg_xmax_ptr + sizeof(float)*reg_num_boxes_selected], xmaxI);
                uni_vmovss(dword[reg_ymax_ptr + sizeof(float)*reg_num_boxes_selected], ymaxI);

                inc(reg_num_boxes_selected);
            }
        }

        L(box_loop_continue_label);
        inc(reg_i);
        jmp(box_loop_label, T_NEAR);
    }
    L(box_loop_end_label);

    // *args->num_boxes_selected = num_boxes_selected;
    mov(rax, qword[param1 + offsetof(jit_nms_call_args, num_boxes_selected_ptr)]);
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

template <cpu_isa_t isa>
void jit_uni_multiclass_nms_kernel_impl<isa>::get_box_ptr(
    const Reg64& boxes_ptr, const Reg64& box_idx, const Reg64& result) {
    assert(sizeof(Box) == 16);
    mov(result, box_idx);
    shl(result, 4);
    lea(result, ptr[boxes_ptr + result]);
}

template <cpu_isa_t isa>
void jit_uni_multiclass_nms_kernel_impl<isa>::get_box_coords_ptr(
    const Reg64& box_ptr, const Reg64& coords_array_ptr, const Reg64& result) {
    mov(result.cvt32(), dword[box_ptr + offsetof(Box, box_idx)]);
    shl(result, 4);
    lea(result, ptr[coords_array_ptr + result]);
}

template <>
void jit_uni_multiclass_nms_kernel_impl<sse41>::load_simd_register(
    const Xbyak::Xmm& reg, const Reg64& buff_ptr, const Reg64& buff_size, const Reg64& index) {
    movups(reg, ptr[buff_ptr + sizeof(float) * index]);

    PReg64 num_elements {reg_pool_};
    get_simd_tail_length(buff_size, index, num_elements);

    static const uint32_t mask[simd_width * 2] STRUCT_ALIGN(16) = {
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0, 0
    };
    Reg64 reg_mask_ptr {Operand::RAX};
    mov(reg_mask_ptr, reinterpret_cast<uint64_t>(&mask[simd_width]));
    shl(num_elements, 2);
    sub(reg_mask_ptr, num_elements);
    PVmm reg_mask {reg_pool_};
    movups(reg_mask, ptr[reg_mask_ptr]);
    andps(reg, reg_mask);
}

template <>
void jit_uni_multiclass_nms_kernel_impl<avx2>::load_simd_register(
    const Xbyak::Ymm& reg, const Reg64& buff_ptr, const Reg64& buff_size, const Reg64& index) {
    PReg64 num_elements {reg_pool_};
    get_simd_tail_length(buff_size, index, num_elements);

    static const uint32_t mask[simd_width * 2] STRUCT_ALIGN(32) = {
        0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000,
        0, 0, 0, 0, 0, 0, 0, 0
    };
    Reg64 reg_mask_ptr {Operand::RAX};
    mov(reg_mask_ptr, reinterpret_cast<uint64_t>(&mask[simd_width]));
    PVmm reg_mask {reg_pool_};
    shl(num_elements, 2);
    sub(reg_mask_ptr, num_elements);
    vmovups(reg_mask, ptr[reg_mask_ptr]);
    vmaskmovps(reg, reg_mask, ptr[buff_ptr + sizeof(float) * index]);
}

template <>
void jit_uni_multiclass_nms_kernel_impl<avx512_core>::load_simd_register(
    const Xbyak::Zmm& reg, const Reg64& buff_ptr, const Reg64& buff_size, const Reg64& index) {
#ifdef _WIN32
    push(rcx);
#endif
    Reg64 num_elements {Operand::RCX};
    get_simd_tail_length(buff_size, index, num_elements);

    // mask = 2^num_elements - 1
    PReg64 mask {reg_pool_};
    mov(mask, 1);
    shl(mask, num_elements.cvt8());
    dec(mask);
    kmovq(k1, mask);

    vmovups(reg | k1 | T_z, ptr[buff_ptr + sizeof(float) * index]);
#ifdef _WIN32
    pop(rcx);
#endif
}

template <cpu_isa_t isa>
void jit_uni_multiclass_nms_kernel_impl<isa>::get_simd_tail_length(
    const Reg64& buff_size, const Reg64& index, const Reg64& result) {
    mov(result, buff_size);
    sub(result, index);
    Reg64 reg_simd_width {Operand::RAX};
    mov(reg_simd_width, simd_width);
    cmp(result, reg_simd_width);
    cmovg(result, reg_simd_width);
}

template class jit_uni_multiclass_nms_kernel_impl<sse41>;
template class jit_uni_multiclass_nms_kernel_impl<avx2>;
template class jit_uni_multiclass_nms_kernel_impl<avx512_core>;

} // namespace node
} // namespace intel_cpu
} // namespace ov
