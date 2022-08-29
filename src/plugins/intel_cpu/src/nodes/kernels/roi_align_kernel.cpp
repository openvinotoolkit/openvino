// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roi_align_kernel.h"
#include <ie_common.h>
#include <utils/bfloat16.hpp>

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {
namespace node {

#define GET_OFF(field) offsetof(jit_roi_align_call_args, field)

template <cpu_isa_t isa>
jit_uni_roi_align_kernel_f32<isa>::jit_uni_roi_align_kernel_f32(jit_roi_align_params jcp) : jit_uni_roi_align_kernel(jcp), jit_generator(jit_name()) {}

template <cpu_isa_t isa>
void jit_uni_roi_align_kernel_f32<isa>::create_ker() {
    jit_generator::create_kernel();
    ker_ = (decltype(ker_))jit_ker();
}

template <cpu_isa_t isa>
void jit_uni_roi_align_kernel_f32<isa>::generate() {
    this->preamble();

    uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

    load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};
    store_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx())};
    store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};

    if (jcp_.layout == ROIAlignLayoutType::ncsp) {
        roi_align_planar();
    } else {
        roi_align_cgather();
    }

    this->postamble();

    emit_emitters_data();
}

template <cpu_isa_t isa>
void jit_uni_roi_align_kernel_f32<isa>::emit_emitters_data() {
        for (const auto& emitter : emitters) {
            emitter.second->emit_data();
        }
    }

template <cpu_isa_t isa>
inline void jit_uni_roi_align_kernel_f32<isa>::load(Xbyak::Reg64 reg_src, Vmm vmm_src, const int elt_num, const int offset) {
    emit_load(reg_src, vmm_src, jcp_.data_prc, Precision::FP32, elt_num, offset);
}

template <cpu_isa_t isa>
inline void jit_uni_roi_align_kernel_f32<isa>::load_buffer(Xbyak::Reg64 reg_src, Vmm vmm_src, const int elt_num, const int offset) {
    emit_load(reg_src, vmm_src, Precision::FP32, Precision::FP32, elt_num, offset);
}

template <cpu_isa_t isa>
inline void jit_uni_roi_align_kernel_f32<isa>::load_idx(Xbyak::Reg64 reg_src, Vmm vmm_src, const int elt_num, const int offset) {
    emit_load(reg_src, vmm_src, Precision::I32, Precision::I32, elt_num, offset);
}

template <cpu_isa_t isa>
inline void jit_uni_roi_align_kernel_f32<isa>::store(Vmm vmm_dst, Xbyak::Reg64 reg_dst, const int elt_num, const int offset) {
    emit_store(vmm_dst, reg_dst, Precision::FP32, jcp_.data_prc, elt_num, offset);
}

template <cpu_isa_t isa>
inline void jit_uni_roi_align_kernel_f32<isa>::store_buffer(Vmm vmm_dst, Xbyak::Reg64 reg_dst, const int elt_num, const int offset) {
    emit_store(vmm_dst, reg_dst, Precision::FP32, Precision::FP32, elt_num, offset);
}

template <cpu_isa_t isa>
inline void jit_uni_roi_align_kernel_f32<isa>::emit_load(Xbyak::Reg64 reg_src, Vmm vmm_src, Precision src_prc, Precision dst_prc,
    const int elt_num, const int offset) {
    const auto seed = load_emitter_params(src_prc, dst_prc, elt_num).hash();
    if (!emitters[seed]) {
        emitters[seed].reset(new jit_load_emitter(this, isa, src_prc, dst_prc, elt_num));
    }

    emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), static_cast<size_t>(offset)},
                                {static_cast<size_t>(vmm_src.getIdx())}, {}, {load_pool_gpr_idxs});
}

template <cpu_isa_t isa>
inline void jit_uni_roi_align_kernel_f32<isa>::emit_store(Vmm vmm_dst, Xbyak::Reg64 reg_dst, Precision src_prc, Precision dst_prc,
    const int elt_num, const int offset) {
    const auto seed = store_emitter_params(src_prc, dst_prc, elt_num).hash();
    if (!emitters[seed]) {
        emitters[seed].reset(new jit_store_emitter(this, isa, src_prc, dst_prc, elt_num));
    }

    // for cases when Store emitter need 2 aux vmm we can use vmm_dst as second aux vmm
    std::vector<size_t> local_store_pool_vec_idxs = { static_cast<size_t>(vmm_dst.getIdx()) };
    local_store_pool_vec_idxs.insert(local_store_pool_vec_idxs.begin(), store_pool_vec_idxs.begin(), store_pool_vec_idxs.end());

    emitters[seed]->emit_code({static_cast<size_t>(vmm_dst.getIdx()), static_cast<size_t>(offset)},
                                {static_cast<size_t>(reg_dst.getIdx())},
                                {local_store_pool_vec_idxs}, {store_pool_gpr_idxs});
}

template <cpu_isa_t isa>
void jit_uni_roi_align_kernel_f32<isa>::roi_align_cgather() {
    mov(reg_src_address, ptr[reg_params + GET_OFF(src)]);
    mov(reg_weights, ptr[reg_params + GET_OFF(weights)]);

    mov(reg_num_samples, ptr[reg_params + GET_OFF(num_samples)]);
    mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
    if (jcp_.alg == Algorithm::ROIAlignAvg) {
        mov(reg_tmp_64, ptr[reg_params + GET_OFF(scale)]);
        uni_vbroadcastss(vmm_scale, ptr[reg_tmp_64]);
    }
    mov(reg_tmp_64, ptr[reg_params + GET_OFF(buffer)]);
    uni_vpinsrq(xmm_args_pool, xmm_args_pool, reg_tmp_64, 0);
    mov(reg_tmp_64, ptr[reg_params + GET_OFF(dst)]);
    uni_vpinsrq(xmm_args_pool, xmm_args_pool, reg_tmp_64, 1);

    if (jcp_.layout == ROIAlignLayoutType::nspc) {
        int src_stride = v_step * jcp_.data_size;
        mov(reg_src_stride, src_stride);
    } else if (jcp_.layout == ROIAlignLayoutType::blk) {
        mov(reg_src_stride, ptr[reg_params + GET_OFF(src_stride)]);
        imul(reg_src_stride, reg_src_stride, jcp_.data_size);
    }

    // out loop for samples in bin
    Xbyak::Label out_loop_label;
    Xbyak::Label out_loop_end_label;

    L(out_loop_label);
    {
        cmp(reg_num_samples, 1);
        jl(out_loop_end_label, T_NEAR);

        // get 4 src address and 4 vmm_weights
        get_src();
        get_weights();

        // inner loop for channels of one sample
        Xbyak::Label in_loop_main_label;
        Xbyak::Label in_loop_main_end_label;
        Xbyak::Label in_loop_tail_label;
        Xbyak::Label in_loop_tail_end_label;

        uni_vpextrq(reg_buf, xmm_args_pool, 0);
        // do not spoil reg_work_amount(channels), which is needed in store rountine
        mov(reg_tmp_64, reg_work_amount);

        L(in_loop_main_label);
        {
            cmp(reg_tmp_64, v_step);
            jl(in_loop_main_end_label, T_NEAR);

            generate_samples(v_step);
            // now this sample value across channel reside in vmm_sample
            // compute with other samples in vmm_buf
            load_buffer(reg_buf, vmm_buf, v_step);
            if (jcp_.alg == Algorithm::ROIAlignAvg) {
                uni_vaddps(vmm_buf, vmm_buf, vmm_sample);
            } else {
                uni_vmaxps(vmm_buf, vmm_buf, vmm_sample);
            }
            store_buffer(vmm_buf, reg_buf, v_step);

            if ((isa == cpu::x64::sse41) && (jcp_.layout == ROIAlignLayoutType::blk)) {
                add(reg_src0, x_step * jcp_.data_size);
                add(reg_src1, x_step * jcp_.data_size);
                add(reg_src2, x_step * jcp_.data_size);
                add(reg_src3, x_step * jcp_.data_size);
                add(reg_buf, x_step * sizeof(float));

                generate_samples(x_step);
                load_buffer(reg_buf, vmm_buf, x_step);
                if (jcp_.alg == Algorithm::ROIAlignAvg) {
                    uni_vaddps(vmm_buf, vmm_buf, vmm_sample);
                } else {
                    uni_vmaxps(vmm_buf, vmm_buf, vmm_sample);
                }
                store_buffer(vmm_buf, reg_buf, x_step);

                sub(reg_src0, x_step * jcp_.data_size);
                sub(reg_src1, x_step * jcp_.data_size);
                sub(reg_src2, x_step * jcp_.data_size);
                sub(reg_src3, x_step * jcp_.data_size);
                // reg_buf no need reset back, buf is continious

                sub(reg_tmp_64, x_step);
            }

            add(reg_src0, reg_src_stride);
            add(reg_src1, reg_src_stride);
            add(reg_src2, reg_src_stride);
            add(reg_src3, reg_src_stride);
            add(reg_buf, v_step * sizeof(float));

            sub(reg_tmp_64, v_step);

            jmp(in_loop_main_label, T_NEAR);
        }
        L(in_loop_main_end_label);

        int tail_step = 1;
        L(in_loop_tail_label);
        {
            cmp(reg_tmp_64, tail_step);
            jl(in_loop_tail_end_label, T_NEAR);

            generate_samples(tail_step);
            load_buffer(reg_buf, vmm_buf, tail_step);
            if (jcp_.alg == Algorithm::ROIAlignAvg) {
                uni_vaddps(vmm_buf, vmm_buf, vmm_sample);
            } else {
                uni_vmaxps(vmm_buf, vmm_buf, vmm_sample);
            }
            store_buffer(vmm_buf, reg_buf, tail_step);

            int tail_src_stride = tail_step * jcp_.data_size;
            add(reg_src0, tail_src_stride);
            add(reg_src1, tail_src_stride);
            add(reg_src2, tail_src_stride);
            add(reg_src3, tail_src_stride);
            add(reg_buf, tail_step * sizeof(float));

            sub(reg_tmp_64, tail_step);

            jmp(in_loop_tail_label, T_NEAR);
        }
        L(in_loop_tail_end_label);

        sub(reg_num_samples, 1);

        jmp(out_loop_label, T_NEAR);
    }
    L(out_loop_end_label);

    // store
    Xbyak::Label store_loop_main_label;
    Xbyak::Label store_loop_main_end_label;
    Xbyak::Label store_loop_tail_label;
    Xbyak::Label store_loop_tail_end_label;

    // EOL for reg_src0-reg_src3
    reg64_t reg_dst = reg_src0;
    uni_vpextrq(reg_dst, xmm_args_pool, 1);
    uni_vpextrq(reg_buf, xmm_args_pool, 0);

    reg64_t reg_dst_stride = reg_src1;
    if (jcp_.layout == ROIAlignLayoutType::nspc) {
        int dst_stride = v_step * jcp_.data_size;
        mov(reg_dst_stride, dst_stride);
    } else if (jcp_.layout == ROIAlignLayoutType::blk) {
        int blk_size = (isa == cpu::x64::sse41) ? v_step * 2 : v_step;
        int dst_stride = blk_size * jcp_.pooled_h * jcp_.pooled_w * jcp_.data_size;
        mov(reg_dst_stride, dst_stride);
    }

    L(store_loop_main_label);
    {
        cmp(reg_work_amount, v_step);
        jl(store_loop_main_end_label, T_NEAR);

        load_buffer(reg_buf, vmm_buf, v_step);
        if (jcp_.alg == Algorithm::ROIAlignAvg) {
            uni_vmulps(vmm_buf, vmm_buf, vmm_scale);
        }
        store(vmm_buf, reg_dst, v_step);

        if ((isa == cpu::x64::sse41) && (jcp_.layout == ROIAlignLayoutType::blk)) {
            add(reg_buf, x_step * sizeof(float));
            add(reg_dst, x_step * jcp_.data_size);

            load_buffer(reg_buf, vmm_buf, x_step);
            if (jcp_.alg == Algorithm::ROIAlignAvg) {
                uni_vmulps(vmm_buf, vmm_buf, vmm_scale);
            }
            store(vmm_buf, reg_dst, x_step);

            sub(reg_dst, x_step * jcp_.data_size);

            sub(reg_work_amount, x_step);
        }

        add(reg_buf, v_step * sizeof(float));
        add(reg_dst, reg_dst_stride);

        sub(reg_work_amount, v_step);

        jmp(store_loop_main_label, T_NEAR);
    }
    L(store_loop_main_end_label);

    int tail_step = 1;
    L(store_loop_tail_label);
    {
        cmp(reg_work_amount, tail_step);
        jl(store_loop_tail_end_label, T_NEAR);

        load_buffer(reg_buf, vmm_buf, tail_step);
        if (jcp_.alg == Algorithm::ROIAlignAvg) {
            uni_vmulps(vmm_buf, vmm_buf, vmm_scale);
        }
        store(vmm_buf, reg_dst, tail_step);

        add(reg_buf, tail_step * sizeof(float));
        add(reg_dst, tail_step * jcp_.data_size);

        sub(reg_work_amount, tail_step);

        jmp(store_loop_tail_label, T_NEAR);
    }
    L(store_loop_tail_end_label);
}

template <cpu_isa_t isa>
void jit_uni_roi_align_kernel_f32<isa>::get_src() {
    mov(reg_src0, ptr[reg_src_address + 0 * sizeof(void*)]);
    mov(reg_src1, ptr[reg_src_address + 1 * sizeof(void*)]);
    mov(reg_src2, ptr[reg_src_address + 2 * sizeof(void*)]);
    mov(reg_src3, ptr[reg_src_address + 3 * sizeof(void*)]);
    add(reg_src_address, 4 * sizeof(void*));
}

template <cpu_isa_t isa>
void jit_uni_roi_align_kernel_f32<isa>::get_weights() {
    uni_vbroadcastss(vmm_weights0, ptr[reg_weights + 0 * sizeof(float)]);
    uni_vbroadcastss(vmm_weights1, ptr[reg_weights + 1 * sizeof(float)]);
    uni_vbroadcastss(vmm_weights2, ptr[reg_weights + 2 * sizeof(float)]);
    uni_vbroadcastss(vmm_weights3, ptr[reg_weights + 3 * sizeof(float)]);
    add(reg_weights, 4 * sizeof(float));
}

template <cpu_isa_t isa>
void jit_uni_roi_align_kernel_f32<isa>::generate_samples(int num) {
    uni_vpxor(vmm_sample, vmm_sample, vmm_sample);
    load(reg_src0, vmm_src0, num);
    uni_vfmadd231ps(vmm_sample, vmm_src0, vmm_weights0);
    load(reg_src1, vmm_src1, num);
    uni_vfmadd231ps(vmm_sample, vmm_src1, vmm_weights1);
    load(reg_src2, vmm_src2, num);
    uni_vfmadd231ps(vmm_sample, vmm_src2, vmm_weights2);
    load(reg_src3, vmm_src3, num);
    uni_vfmadd231ps(vmm_sample, vmm_src3, vmm_weights3);
}

template <cpu_isa_t isa>
void jit_uni_roi_align_kernel_f32<isa>::roi_align_planar() {
    mov(reg_src, ptr[this->reg_params + GET_OFF(src)]);
    mov(reg_buf, ptr[this->reg_params + GET_OFF(buffer)]);
    mov(reg_weights, ptr[this->reg_params + GET_OFF(weights)]);

    mov(reg_dst, ptr[this->reg_params + GET_OFF(dst)]);
    mov(reg_num_samples, ptr[reg_params + GET_OFF(num_samples)]);

    if (jcp_.alg == Algorithm::ROIAlignAvg) {
        mov(reg_tmp_64, ptr[reg_params + GET_OFF(scale)]);
        uni_vbroadcastss(vmm_scale, ptr[reg_tmp_64]);
    }

    Xbyak::Label main_loop_label;
    Xbyak::Label main_loop_end_label;
    Xbyak::Label tail_loop_label;
    Xbyak::Label tail_loop_end_label;

    int lane = v_len / cpu_isa_traits<sse41>::vlen;
    uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
    L(main_loop_label);
    {
        cmp(reg_num_samples, lane);
        jl(main_loop_end_label, T_NEAR);

        load_idx(reg_buf, vmm_buf, v_step);

        if (jcp_.data_prc == Precision::FP32)
            gather_f32(vmm_src, reg_src, vmm_buf);
        else if (jcp_.data_prc == Precision::BF16)
            gather_bf16_to_f32_zmm(vmm_src, reg_src, vmm_buf);

        uni_vmovups(vmm_weights, ptr[reg_weights]);

        if (jcp_.alg == Algorithm::ROIAlignAvg) {
            uni_vfmadd231ps(vmm_dst, vmm_src, vmm_weights);
        } else {
            uni_vmulps(vmm_src, vmm_src, vmm_weights);
            // horizontal add for each lane
            // xmm_dst[0] hold the max
            if (isa == cpu::x64::avx512_core) {
                for (int i = 0; i < lane; i++) {
                    vextractf32x4(xmm_temp1, Xbyak::Zmm(vmm_src.getIdx()), i);
                    horizontal_add_xmm(xmm_temp1, xmm_temp2);
                    uni_vmaxps(xmm_dst, xmm_dst, xmm_temp1);
                }
            } else if (isa == cpu::x64::avx2) {
                for (int i = 0; i < lane; i++) {
                    vextractf128(xmm_temp1, Xbyak::Ymm(vmm_src.getIdx()), i);
                    horizontal_add_xmm(xmm_temp1, xmm_temp2);
                    uni_vmaxps(xmm_dst, xmm_dst, xmm_temp1);
                }
            } else {
                horizontal_add_xmm(xmm_src, xmm_temp2);
                uni_vmaxps(xmm_dst, xmm_dst, xmm_src);
            }
        }

        add(reg_buf, v_len);
        add(reg_weights, v_len);
        sub(reg_num_samples, lane);

        jmp(main_loop_label, T_NEAR);
    }
    L(main_loop_end_label);

    if (jcp_.alg == Algorithm::ROIAlignAvg)
        uni_vpxor(vmm_dst_tail, vmm_dst_tail, vmm_dst_tail);

    lane = 1;
    L(tail_loop_label);
    {
        cmp(reg_num_samples, 1);
        jl(tail_loop_end_label, T_NEAR);

        load_idx(reg_buf, vmm_buf, x_step);

        if (jcp_.data_prc == Precision::FP32)
            gather_f32_xmm(xmm_src, reg_src, xmm_buf);
        else if (jcp_.data_prc == Precision::BF16)
            gather_bf16_to_f32_xmm(xmm_src, reg_src, xmm_buf);

        uni_vmovups(xmm_weights, ptr[reg_weights]);
        if (jcp_.alg == Algorithm::ROIAlignAvg) {
            // as vex instruction will zero upper bit for xmm version, store result in seperate xmm_dst_tail
            uni_vfmadd231ps(xmm_dst_tail, xmm_src, xmm_weights);
        } else {
            uni_vmulps(xmm_src, xmm_src, xmm_weights);
            horizontal_add_xmm(xmm_src, xmm_temp2);
            uni_vmaxps(xmm_dst, xmm_dst, xmm_src);
        }

        add(reg_buf, x_len);
        add(reg_weights, x_len);
        sub(reg_num_samples, lane);

        jmp(tail_loop_label, T_NEAR);
    }
    L(tail_loop_end_label);

    if (jcp_.alg == Algorithm::ROIAlignAvg) {
        uni_vaddps(vmm_dst, vmm_dst, vmm_dst_tail);
        horizontal_add();  // xmm_dst[0] is the dst value
        uni_vmulps(vmm_dst, vmm_dst, vmm_scale);
    }

    // xmm_dst[0] of f32 is the dst value
    if (jcp_.data_prc == Precision::FP32)
        uni_vpextrd(ptr[reg_dst], xmm_dst, 0);
    else if (jcp_.data_prc == Precision::BF16)
        uni_vpextrw(ptr[reg_dst], xmm_dst, 1);
}

template <cpu_isa_t isa>
inline void jit_uni_roi_align_kernel_f32<isa>::gather_f32(Vmm &vmm_src, const reg64_t &reg_src, const Vmm &vmm_idx) {
    constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
    constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

    if (is_zmm) {
        kxnord(k_mask, k_mask, k_mask);
        vgatherdps(vmm_src | k_mask, ptr[reg_src + vmm_idx * jcp_.data_size]);
    } else if (is_ymm) {
        uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
        vgatherdps(vmm_src, ptr[reg_src + vmm_idx * jcp_.data_size], vmm_mask);
    } else {
        gather_f32_xmm(Xbyak::Xmm(vmm_src.getIdx()), reg_src, Xbyak::Xmm(vmm_idx.getIdx()));
    }
}

template <cpu_isa_t isa>
inline void jit_uni_roi_align_kernel_f32<isa>::gather_f32_xmm(Xbyak::Xmm xmm_src, const reg64_t reg_src, const Xbyak::Xmm xmm_idx) {
    sub(rsp, x_len);
    uni_vmovdqu(ptr[rsp], xmm_idx);
    for (int i = 0; i < x_step; i++) {
        mov(reg_tmp_32, ptr[rsp + i * sizeof(int)]);       // sizeof(int)  index_size
        mov(reg_tmp_32, ptr[reg_src + reg_tmp_64 * jcp_.data_size]);  // scale: sizeof(float)   value_size
        mov(ptr[rsp + i * sizeof(int)], reg_tmp_32);
    }
    uni_vmovups(xmm_src, ptr[rsp]);
    add(rsp, x_len);
}

template <cpu_isa_t isa>
inline void jit_uni_roi_align_kernel_f32<isa>::gather_bf16_to_f32_zmm(Vmm vmm_src, const reg64_t reg_src, const Vmm vmm_idx) {
    if (!std::is_same<Vmm, Xbyak::Zmm>::value)
        IE_THROW() << "bf16 is only supported from avx512_core platform for ROIAlign node.";
    sub(rsp, v_len);
    uni_vmovdqu(ptr[rsp], vmm_idx);
    for (int i = 0; i < v_step; i++) {
        mov(reg_tmp_32, ptr[rsp + i * sizeof(int)]);       // sizeof(int)  index_size
        mov(reg_tmp_16, word[reg_src + reg_tmp_64 * jcp_.data_size]);  // scale: sizeof(bf16)   value_size
        mov(ptr[rsp + i * sizeof(int)], reg_tmp_16);
    }
    uni_vmovups(vmm_src, ptr[rsp]);    // |_ x|_ x|_ x|_ x|
    uni_vpslld(vmm_src, vmm_src, 16);  // |x 0|x 0|x 0|x 0|

    add(rsp, v_len);
}

template <cpu_isa_t isa>
inline void jit_uni_roi_align_kernel_f32<isa>::gather_bf16_to_f32_xmm(Xbyak::Xmm xmm_src, const reg64_t reg_src, const Xbyak::Xmm xmm_idx) {
    sub(rsp, x_len);
    uni_vmovdqu(ptr[rsp], xmm_idx);
    for (int i = 0; i < x_step; i++) {
        mov(reg_tmp_32, ptr[rsp + i * sizeof(int)]);
        mov(reg_tmp_16, ptr[reg_src + reg_tmp_64 * jcp_.data_size]);
        mov(ptr[rsp + i * sizeof(int)], reg_tmp_16);
    }
    uni_vmovups(xmm_src, ptr[rsp]);    // |_ x|_ x|_ x|_ x|
    uni_vpslld(xmm_src, xmm_src, 16);  // |x 0|x 0|x 0|x 0|

    add(rsp, x_len);
}

template <cpu_isa_t isa>
inline void jit_uni_roi_align_kernel_f32<isa>::horizontal_add_xmm(const Xbyak::Xmm &xmm_dst, const Xbyak::Xmm &xmm_aux) {
    uni_vmovshdup(xmm_aux, xmm_dst);              //  dst:1,2,3,4; aux:2,2,4,4
    uni_vaddps(xmm_dst, xmm_dst, xmm_aux);        //  dst:1+2,2+2,3+4,4+4
    uni_vmovhlps(xmm_aux, xmm_aux, xmm_dst);      //  aux:3+4,4+4,4,4
    uni_vaddps(xmm_dst, xmm_dst, xmm_aux);        //  dst:1+2+3+4,...
}

// horizontal add for vmm_dst, temp1 and temp2 as aux
template <cpu_isa_t isa>
inline void jit_uni_roi_align_kernel_f32<isa>::horizontal_add() {
    Xbyak::Xmm xmm_dst = Xbyak::Xmm(vmm_dst.getIdx());
    Xbyak::Xmm xmm_temp1 = Xbyak::Xmm(vmm_temp1.getIdx());
    Xbyak::Xmm xmm_temp2 = Xbyak::Xmm(vmm_temp2.getIdx());
    if (isa == cpu::x64::sse41) {
        horizontal_add_xmm(xmm_dst, xmm_temp1);
    } else if (isa == cpu::x64::avx2) {
        Xbyak::Ymm ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());
        vextractf128(xmm_temp1, ymm_dst, 0);
        vextractf128(xmm_temp2, ymm_dst, 1);
        uni_vaddps(xmm_dst, xmm_temp1, xmm_temp2);
        horizontal_add_xmm(xmm_dst, xmm_temp1);
    } else {
        Xbyak::Zmm zmm_dst = Xbyak::Zmm(vmm_dst.getIdx());
        vextractf32x4(xmm_temp1, zmm_dst, 0);
        vextractf32x4(xmm_temp2, zmm_dst, 1);
        uni_vaddps(xmm_temp1, xmm_temp1, xmm_temp2);
        vextractf32x4(xmm_temp2, zmm_dst, 2);
        vextractf32x4(xmm_dst, zmm_dst, 3);
        uni_vaddps(xmm_dst, xmm_dst, xmm_temp2);
        uni_vaddps(xmm_dst, xmm_dst, xmm_temp1);
        horizontal_add_xmm(xmm_dst, xmm_temp1);
    }
}

template struct jit_uni_roi_align_kernel_f32<x64::sse41>;
template struct jit_uni_roi_align_kernel_f32<x64::avx2>;
template struct jit_uni_roi_align_kernel_f32<x64::avx512_core>;

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
