// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_topk_node.h"

#include <mkldnn.hpp>

#include <string>
#include <vector>
#include <set>
#include <mkldnn_extension_utils.h>
#include "emitters/jit_load_store_emitters.hpp"
#include "ie_parallel.hpp"
#include <ngraph/op/topk.hpp>
#include <algorithm>

#include <cpu/x64/jit_generator.hpp>
#include <cpu/x64/jit_uni_eltwise.hpp>
#include "common/cpu_memcpy.h"

#include <ngraph/opsets/opset1.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_topk_call_args, field)

#define vmm_tmp     Vmm(0)
#define vmm_mask    Vmm(1)
#define vmm_val(i)  Vmm(2 * (i) + 2)
#define vmm_idx(i)  Vmm(2 * (i) + 3)
#define vmm_val_l   Vmm(2)
#define vmm_idx_l   Vmm(3)
#define vmm_val_r   Vmm(4)
#define vmm_idx_r   Vmm(5)

#define xmm_tmp     Xmm(0)
#define xmm_mask    Xmm(1)
#define xmm_val(i)  Xmm(2 * (i) + 2)
#define xmm_idx(i)  Xmm(2 * (i) + 3)
#define xmm_val_l   Xmm(2)
#define xmm_idx_l   Xmm(3)
#define xmm_val_r   Xmm(4)
#define xmm_idx_r   Xmm(5)

#define xmm_val_p   Xmm(6)
#define xmm_idx_p   Xmm(7)

#define JMP_TO_LABEL(label)                  \
    if (isa == avx512_common) {              \
        kmovw(reg_tmp_32, k_mask);           \
    } else {                                 \
        uni_vmovmskps(reg_tmp_32, xmm_mask); \
    }                                        \
    and_(reg_tmp_32, 0x1);                   \
    cmp(reg_tmp_32, 0x1);                    \
    je(label, T_NEAR);

static inline bool isFloatCompatible(memory::data_type type) {
    return memory::data_type::f32 == type || memory::data_type::bf16 == type;
}

template <cpu_isa_t isa>
struct jit_uni_topk_kernel_f32 : public jit_uni_topk_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_topk_kernel_f32)

    explicit jit_uni_topk_kernel_f32(jit_topk_config_params jcp)
        : jit_uni_topk_kernel(jcp), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        load_emitter.reset(new jit_load_emitter(this, isa, nullptr));
        store_emitter.reset(new jit_store_emitter(this, isa, nullptr));
        if (!mayiuse(avx512_core_bf16) && mayiuse(avx512_core))
            emu_vcvtneps2bf16.reset(new jit_emu_vcvtneps2bf16(this, isa, nullptr));

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_dst_idx, ptr[reg_params + GET_OFF(index)]);
        mov(reg_prc, ptr[reg_params + GET_OFF(process)]);
        mov(reg_prc_idx, ptr[reg_params + GET_OFF(process_index)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        mov(reg_table, l_table);

        if (jcp_.mode_max) {
            cmp_flg = _cmp_nlt_us;      // if val[left] >= val[right], set mask 1, no swap
            heap_cmp_flg = _cmp_le_os;  // min heap is used for max topk, if a <= b, set mask 1, no swap
        } else {
            cmp_flg = _cmp_le_os;       // if val[left] <= val[right], set mask 1, no swap
            heap_cmp_flg = _cmp_nlt_us; // max heap is used for min topk, if a >= b, set mask 1, no swap
        }

        if (isa == cpu::x64::avx512_common)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};

        topk_loop();

        this->postamble();

        load_emitter->emit_data();
        store_emitter->emit_data();
        if (!mayiuse(avx512_core_bf16) && mayiuse(avx512_core))
            emu_vcvtneps2bf16->emit_data();

        prepare_idx_table();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Address table_val(int index) { return ptr[reg_table + index * vlen]; }
    Xbyak::Address table_seq_val(int index) { return ptr[reg_table + jcp_.axis_dim * vlen + index * sizeof(float)]; }

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_dst_idx = r10;
    Xbyak::Reg64 reg_prc = r11;
    Xbyak::Reg64 reg_prc_idx = r12;
    Xbyak::Reg64 reg_work_amount = r13;
    Xbyak::Reg64 reg_table = r14;
    Xbyak::Reg64 reg_params = abi_param1;
    Xbyak::Reg64 reg_j = rax;
    Xbyak::Reg64 reg_aux = rdx;
    Xbyak::Reg64 reg_aux_idx = rbx;

    Xbyak::Reg8 reg_tmp_8 = r15b;
    Xbyak::Reg32 reg_tmp_32 = r15d;
    Xbyak::Reg64 reg_tmp_64 = r15;

    Xbyak::Reg64 reg_load_table = rbp;
    Xbyak::Reg64 reg_load_store_mask = rsi;

    Vmm vmm_zero = Vmm(1); // vmm_zero represents Vmm(1) when isa is avx512_common, otherwise vmm_mask represents Vmm(1)

    const Xbyak::Opmask k_mask = Xbyak::Opmask(1);
    const int step = vlen / sizeof(float);
    const int tail = jcp_.work_amout % step;
    const int topk_tail = jcp_.top_k % step;

    unsigned char cmp_flg;
    unsigned char heap_cmp_flg;

    Xbyak::Label l_table;

    std::unique_ptr<jit_emu_vcvtneps2bf16> emu_vcvtneps2bf16 = nullptr;
    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;

    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> load_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;

    inline void topk_loop() {
        if (jcp_.algorithm == TopKAlgorithm::topk_bubble_sort) {
            if (jcp_.layout == TopKLayoutType::topk_blocked && jcp_.topk_innermost) {
                if (jcp_.top_k == 1) {
                    topk_bubble_horiz_blocked_innermost();
                } else {
                    topk_bubble_scalar_blocked_innermost();
                }
            } else {
                topk_bubble_vector();
            }
        } else if (jcp_.algorithm == TopKAlgorithm::topk_bitonic_sort) {
            if (jcp_.layout == TopKLayoutType::topk_blocked && jcp_.topk_innermost) {
                topk_bitonic_blocked_innermost();
            } else {
                topk_bitonic_vector();
            }
        } else if (jcp_.algorithm == TopKAlgorithm::topk_heap_sort) {
            topk_heap_scalar();
        }
    }

    inline void topk_bitonic_vector() {
        Xbyak::Label topk_main_loop_label;
        Xbyak::Label topk_main_loop_end_label;
        L(topk_main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(topk_main_loop_end_label, T_NEAR);

            topk_bitonic(step);

            add(reg_src, step * jcp_.data_size);
            add(reg_dst, step * jcp_.data_size);
            add(reg_dst_idx, step * sizeof(int));
            sub(reg_work_amount, step);

            jmp(topk_main_loop_label, T_NEAR);
        }
        L(topk_main_loop_end_label);

        if (tail) {
            topk_bitonic(tail);
        }
    }

    inline void topk_bitonic(int elt_num) {
        // src => prc
        for (int i = 0; i < jcp_.axis_dim; i++) {
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                          {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_prc.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});

            load_emitter->emit_code({static_cast<size_t>(reg_table.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                          std::make_shared<load_emitter_context>(Precision::I32, Precision::I32, elt_num, i * vlen),
                          {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_prc_idx.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::I32, Precision::I32, elt_num, i * jcp_.sort_stride * sizeof(int)),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        }

        // sort
        bitonic_sort_vector(jcp_.bitonic_size, jcp_.axis_dim, elt_num);
        if (jcp_.sort_index) {
            bitonic_sort_vector(jcp_.bitonic_k_size, jcp_.top_k, elt_num, false);
        }

        // prc => dst
        for (int i = 0; i < jcp_.top_k; i++) {
            load_emitter->emit_code({static_cast<size_t>(reg_prc.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                          {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});

            load_emitter->emit_code({static_cast<size_t>(reg_prc_idx.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                          std::make_shared<load_emitter_context>(Precision::I32, Precision::I32, elt_num, i * jcp_.sort_stride * sizeof(int)),
                          {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_dst_idx.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::I32, Precision::I32, elt_num, i * jcp_.sort_stride * sizeof(int)),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        }
    }

    // src memory layout: (N) * (CB * H * W * blk_size)
    // prc memory layout: (C) * (N * H * W)
    // topk_bitonic_vector_blocked_innermost: sort (C) * (N * H * W / blk_size * blk_size) elements
    //                                        sort (C) * (N * H * W % blk_size) elements in the rear
    inline void topk_bitonic_blocked_innermost() {
        Xbyak::Label topk_main_loop_label;
        Xbyak::Label topk_main_loop_end_label;
        L(topk_main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(topk_main_loop_end_label, T_NEAR);

            // src => prc
            blocked_innermost_load(step);

            // sort
            bitonic_sort_vector(jcp_.bitonic_size, jcp_.axis_dim, step);
            if (jcp_.sort_index) {
                bitonic_sort_vector(jcp_.bitonic_k_size, jcp_.top_k, step, false);
            }

            // prc => dst
            blocked_innermost_store(step);

            add(reg_src, step * jcp_.blk_size * jcp_.data_size);
            add(reg_dst, step * jcp_.blk_size * jcp_.data_size);
            add(reg_dst_idx, step * jcp_.blk_size * sizeof(int));
            sub(reg_work_amount, step);

            jmp(topk_main_loop_label, T_NEAR);
        }
        L(topk_main_loop_end_label);

        // tail exists because working buffer has planar layout, though source buffer has blocked layout)
        // src => prc
        blocked_innermost_load(tail);

        bitonic_sort_vector(jcp_.bitonic_size, jcp_.axis_dim, tail);
        if (jcp_.sort_index) {
            bitonic_sort_vector(jcp_.bitonic_k_size, jcp_.top_k, tail, false);
        }

        // prc => dst
        blocked_innermost_store(tail);
    }

    // bitonic_size: length of the total array for sorting, power of 2
    //          len: length of the array being sorted
    //        start: start index of the array being sorted
    //      sub_len: half of len
    //    sub_start: start index of the sub array being sorted
    //    minor_len: half of sub_len
    //            n: number of valid elements in bitonic sort
    //            p: pow of 2 number, so that p/2 < n <= p
    //   empty tail: p-n elements in the rear don't need sorting,
    inline void bitonic_sort_vector(int p, int n, int elt_num, bool cmp_val = true) {
        for (int len = 2; len < p; len <<= 1) {
            for (int start = 0; start < p; start += len) {
                int sub_len = len >> 1;
                // empty tail
                for (int i = sub_len - 1; start + len - i - 1 < n && i >= 0; i--) {
                    bitonic_swap_vector(start + i, start + len - i - 1, elt_num, cmp_val);
                }
                for (; sub_len > 0; sub_len >>= 1) {
                    for (int sub_start = start; sub_start < start + len; sub_start += sub_len) {
                        int minor_len = sub_len >> 1;
                        // empty tail
                        for (int j = 0; sub_start + j + minor_len < n && j < minor_len; j++) {
                            bitonic_swap_vector(sub_start + j, sub_start + j + minor_len, elt_num, cmp_val);
                        }
                    }
                }
            }
        }

        // last round sort
        int sub_p = p >> 1;
        for (int i = sub_p - 1; p - i - 1 < n && i >= 0; i--) {
            bitonic_swap_vector(i, p - i - 1, elt_num, cmp_val);
        }
        for (; sub_p > 0; sub_p >>= 1) {
            // support partial sort as well as full sort
            for (int sub_start = 0; (!cmp_val || (cmp_val && sub_start < n)) && sub_start < p;
                 sub_start += sub_p) {
                int minor_p = sub_p >> 1;
                for (int j = 0; sub_start + j + minor_p < n && j < minor_p; j++) {
                    bitonic_swap_vector(sub_start + j, sub_start + j + minor_p, elt_num, cmp_val);
                }
            }
        }
    }

    inline void blocked_innermost_load(int elt_num) {
        for (int i = 0; i < jcp_.axis_dim; i++) {
            for (int j = 0; j < elt_num; j++) {
                int offset = i / jcp_.blk_size * jcp_.blk_stride + i % jcp_.blk_size + j * jcp_.blk_size;

                load_scalar(xmm_tmp, ptr[reg_src + offset * jcp_.data_size], jcp_.data_type);
                store_scalar(ptr[reg_prc + (i * jcp_.sort_stride + j) * jcp_.data_size], xmm_tmp, jcp_.data_type);

                uni_vmovdqu(xmm_tmp, table_val(i));
                store_scalar(ptr[reg_prc_idx + (i * jcp_.sort_stride + j) * sizeof(int)], xmm_tmp, memory::data_type::s32, false);
            }
        }
    }

    inline void blocked_innermost_store(int elt_num) {
        for (int i = 0; i < jcp_.top_k; i++) {
            for (int j = 0; j < elt_num; j++) {
                int offset = i / jcp_.blk_size * jcp_.blk_stride + i % jcp_.blk_size + j * jcp_.blk_size;

                load_scalar(xmm_tmp, ptr[reg_prc + (i * jcp_.sort_stride + j) * jcp_.data_size], jcp_.data_type);
                store_scalar(ptr[reg_dst + offset * jcp_.data_size], xmm_tmp, jcp_.data_type);

                load_scalar(xmm_tmp, ptr[reg_prc_idx + (i * jcp_.sort_stride + j) * sizeof(int)], memory::data_type::s32);
                store_scalar(ptr[reg_dst_idx + offset * sizeof(int)], xmm_tmp, memory::data_type::s32);
            }
        }
    }

    inline void bitonic_swap_vector(int l, int r, int elt_num, bool cmp_val = true) {
        load_emitter->emit_code({static_cast<size_t>(reg_prc.getIdx())}, {static_cast<size_t>(vmm_val_l.getIdx())},
                      std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num, l * jcp_.sort_stride * jcp_.data_size),
                      {}, {load_pool_gpr_idxs});
        load_emitter->emit_code({static_cast<size_t>(reg_prc.getIdx())}, {static_cast<size_t>(vmm_val_r.getIdx())},
                      std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num, r * jcp_.sort_stride * jcp_.data_size),
                      {}, {load_pool_gpr_idxs});
        load_emitter->emit_code({static_cast<size_t>(reg_prc_idx.getIdx())}, {static_cast<size_t>(vmm_idx_l.getIdx())},
                      std::make_shared<load_emitter_context>(Precision::I32, Precision::FP32, elt_num, l * jcp_.sort_stride * sizeof(int)),
                      {}, {load_pool_gpr_idxs});
        load_emitter->emit_code({static_cast<size_t>(reg_prc_idx.getIdx())}, {static_cast<size_t>(vmm_idx_r.getIdx())},
                      std::make_shared<load_emitter_context>(Precision::I32, Precision::FP32, elt_num, r * jcp_.sort_stride * sizeof(int)),
                      {}, {load_pool_gpr_idxs});

        if (isa == avx512_common) {
            if (cmp_val)
                vcmpps(k_mask, vmm_val_l, vmm_val_r, cmp_flg);
            else
                vcmpps(k_mask, vmm_idx_l, vmm_idx_r, _cmp_le_os);

            uni_vmovups(vmm_tmp, vmm_val_l);
            vblendmps(vmm_val_l | k_mask, vmm_val_r, vmm_val_l);
            vblendmps(vmm_val_r | k_mask, vmm_tmp, vmm_val_r);

            uni_vmovups(vmm_tmp, vmm_idx_l);
            vblendmps(vmm_idx_l | k_mask, vmm_idx_r, vmm_idx_l);
            vblendmps(vmm_idx_r | k_mask, vmm_tmp, vmm_idx_r);
        } else {
            if (cmp_val) {
                if (cmp_flg == _cmp_nlt_us)
                    vcmpge_oqps(vmm_mask, vmm_val_l, vmm_val_r);
                else
                    vcmple_oqps(vmm_mask, vmm_val_l, vmm_val_r);
            } else {
                vcmple_oqps(vmm_mask, vmm_idx_l, vmm_idx_r);
            }

            uni_vmovups(vmm_tmp, vmm_val_l);
            vblendvps(vmm_val_l, vmm_val_r, vmm_val_l, vmm_mask);
            vblendvps(vmm_val_r, vmm_tmp, vmm_val_r, vmm_mask);

            uni_vmovups(vmm_tmp, vmm_idx_l);
            vblendvps(vmm_idx_l, vmm_idx_r, vmm_idx_l, vmm_mask);
            vblendvps(vmm_idx_r, vmm_tmp, vmm_idx_r, vmm_mask);
        }

        store_emitter->emit_code({static_cast<size_t>(vmm_val_l.getIdx())}, {static_cast<size_t>(reg_prc.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, elt_num, l * jcp_.sort_stride * jcp_.data_size),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        store_emitter->emit_code({static_cast<size_t>(vmm_val_r.getIdx())}, {static_cast<size_t>(reg_prc.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, elt_num, r * jcp_.sort_stride * jcp_.data_size),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        store_emitter->emit_code({static_cast<size_t>(vmm_idx_l.getIdx())}, {static_cast<size_t>(reg_prc_idx.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, Precision::I32, elt_num, l * jcp_.sort_stride * sizeof(int)),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        store_emitter->emit_code({static_cast<size_t>(vmm_idx_r.getIdx())}, {static_cast<size_t>(reg_prc_idx.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, Precision::I32, elt_num, r * jcp_.sort_stride * sizeof(int)),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
    }

    inline void topk_heap_scalar() {
        // init dst
        int i = 0;
        for (; i + step <= jcp_.top_k; i += step) {
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                                std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, step, i * jcp_.data_size),
                                {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, step, i * jcp_.data_size),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});

            uni_vmovdqu(vmm_tmp, table_seq_val(i));
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_dst_idx.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::I32, Precision::I32, step, i * sizeof(int)),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        }
        if (topk_tail) {
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, topk_tail, i * jcp_.data_size),
                          {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, topk_tail, i * jcp_.data_size),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});

            load_emitter->emit_code({static_cast<size_t>(reg_table.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                          std::make_shared<load_emitter_context>(Precision::I32, Precision::I32, topk_tail, jcp_.axis_dim * vlen + i * sizeof(float)),
                          {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_dst_idx.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::I32, Precision::I32, topk_tail, i * sizeof(int)),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        }

        // heapify
        int end = (jcp_.top_k - 2) / 2;
        for (int i = end; i >= 0; i--) {
            heapipy_sub_tree(i, jcp_.top_k - 1);
        }

        // update
        for (int i = jcp_.top_k; i < jcp_.axis_dim; i++) {
            Xbyak::Label topk_update_loop_end_label;
            load_scalar(xmm_val_p, ptr[reg_src + i * jcp_.data_size], jcp_.data_type);
            uni_vmovdqu(xmm_idx_p, table_val(i));
            uni_vcvtdq2ps(xmm_idx_p, xmm_idx_p);
            load_scalar(xmm_val_l, ptr[reg_dst], jcp_.data_type);
            load_scalar(xmm_idx_l, ptr[reg_dst_idx], memory::data_type::s32);

            heap_cmp_node(xmm_val_p, xmm_idx_p, xmm_val_l, xmm_idx_l);
            JMP_TO_LABEL(topk_update_loop_end_label);

            store_scalar(ptr[reg_dst], xmm_val_p, jcp_.data_type);
            store_scalar(ptr[reg_dst_idx], xmm_idx_p, memory::data_type::s32);
            heapipy_sub_tree(0, jcp_.top_k - 1);

            L(topk_update_loop_end_label);
        }

        // extract topk
        if (jcp_.sort_index) {
            // reheapify by index
            for (int i = end; i >= 0; i--) {
                heapipy_sub_tree(i, jcp_.top_k - 1, false);
            }

            // extract by index
            for (int i = jcp_.top_k - 1; i > 0; i--) {
                heap_swap_root(i);
                heapipy_sub_tree(0, i - 1, false);
            }
        } else {
            // extract by value
            for (int i = jcp_.top_k - 1; i > 0; i--) {
                heap_swap_root(i);
                heapipy_sub_tree(0, i - 1);
            }
        }
    }

    inline void heapipy_sub_tree(int i, int valid, bool cmp_val = true) {
        Xbyak::Label topk_heapify_loop_label;
        Xbyak::Label topk_heapify_loop_end_label;
        Xbyak::Label topk_lchild_loop_label;
        Xbyak::Label topk_rchild_loop_label;

        if (valid > 0) {
            int end = (valid - 1) / 2;
            mov(reg_j, i);
            mov(reg_aux, reg_dst);
            mov(reg_aux_idx, reg_dst_idx);
            add(reg_aux, i * jcp_.data_size);
            add(reg_aux_idx, i * sizeof(int));
            mov(reg_work_amount, (2 * i + 1) * jcp_.data_size);
            L(topk_heapify_loop_label);
            {
                cmp(reg_j, end);
                jg(topk_heapify_loop_end_label, T_NEAR);

                load_scalar(xmm_val_p, ptr[reg_aux], jcp_.data_type);
                load_scalar(xmm_idx_p, ptr[reg_aux_idx], memory::data_type::s32);

                // compare lchild-rchild
                mov(reg_prc, reg_dst);
                add(reg_prc, reg_work_amount);
                mov(reg_prc_idx, reg_dst_idx);
                add(reg_prc_idx, reg_work_amount);
                load_scalar(xmm_val_l, ptr[reg_prc], jcp_.data_type);
                load_scalar(xmm_idx_l, ptr[reg_prc_idx], memory::data_type::s32);
                add(reg_prc, jcp_.sort_stride * jcp_.data_size);
                add(reg_prc_idx, jcp_.sort_stride * jcp_.data_size);

                // if last valid parent has no rchild
                cmp(reg_j, valid / 2);
                jge(topk_lchild_loop_label, T_NEAR);

                load_scalar(xmm_val_r, ptr[reg_prc], jcp_.data_type);
                load_scalar(xmm_idx_r, ptr[reg_prc_idx], memory::data_type::s32);

                heap_cmp_node(xmm_val_l, xmm_idx_l, xmm_val_r, xmm_idx_r, cmp_val);
                JMP_TO_LABEL(topk_lchild_loop_label);

                // compare node-rchild
                L(topk_rchild_loop_label);
                {
                    heap_cmp_node(xmm_val_p, xmm_idx_p, xmm_val_r, xmm_idx_r, cmp_val);
                    JMP_TO_LABEL(topk_heapify_loop_end_label);

                    heap_swap_node(xmm_val_p, xmm_idx_p, xmm_val_r, xmm_idx_r);
                    mov(reg_aux, reg_prc);
                    mov(reg_aux_idx, reg_prc_idx);
                    add(reg_work_amount, jcp_.sort_stride * jcp_.data_size);
                    shl(reg_work_amount, 1);
                    add(reg_work_amount, jcp_.sort_stride * jcp_.data_size);
                    shl(reg_j, 1);
                    add(reg_j, 2);
                    jmp(topk_heapify_loop_label, T_NEAR);
                }

                // compare node-lchild
                L(topk_lchild_loop_label);
                {
                    heap_cmp_node(xmm_val_p, xmm_idx_p, xmm_val_l, xmm_idx_l, cmp_val);
                    JMP_TO_LABEL(topk_heapify_loop_end_label);

                    sub(reg_prc, jcp_.sort_stride * jcp_.data_size);
                    sub(reg_prc_idx, jcp_.sort_stride * jcp_.data_size);
                    heap_swap_node(xmm_val_p, xmm_idx_p, xmm_val_l, xmm_idx_l);
                    mov(reg_aux, reg_prc);
                    mov(reg_aux_idx, reg_prc_idx);
                    shl(reg_work_amount, 1);
                    add(reg_work_amount, jcp_.sort_stride * jcp_.data_size);
                    shl(reg_j, 1);
                    add(reg_j, 1);
                    jmp(topk_heapify_loop_label, T_NEAR);
                }
            }
            L(topk_heapify_loop_end_label);
        }
    }

    inline void heap_cmp_node(Xmm xmm_val_a, Xmm xmm_idx_a, Xmm xmm_val_b, Xmm xmm_idx_b, bool cmp_val = true) {
        if (isa == avx512_common) {
            if (cmp_val)
                vcmpps(k_mask, xmm_val_a, xmm_val_b, heap_cmp_flg);
            else
                vcmpps(k_mask, xmm_idx_a, xmm_idx_b, _cmp_le_os);
        } else {
            if (cmp_val) {
                if (heap_cmp_flg == _cmp_nlt_us)
                    vcmpge_oqps(xmm_mask, xmm_val_a, xmm_val_b);
                else
                    vcmple_oqps(xmm_mask, xmm_val_a, xmm_val_b);
            } else {
                vcmpge_oqps(xmm_mask, xmm_idx_a, xmm_idx_b);
            }
        }
    }

    // n: node, c: child
    inline void heap_swap_node(Xmm xmm_val_n, Xmm xmm_idx_n, Xmm xmm_val_c, Xmm xmm_idx_c) {
        // swap store
        store_scalar(ptr[reg_aux], xmm_val_c, jcp_.data_type);
        store_scalar(ptr[reg_aux_idx], xmm_idx_c, memory::data_type::s32);
        store_scalar(ptr[reg_prc], xmm_val_n, jcp_.data_type);
        store_scalar(ptr[reg_prc_idx], xmm_idx_n, memory::data_type::s32);
    }

    inline void heap_swap_root(int i) {
        load_scalar(xmm_val_p, ptr[reg_dst], jcp_.data_type);
        load_scalar(xmm_idx_p, ptr[reg_dst_idx], memory::data_type::s32);
        load_scalar(xmm_val_l, ptr[reg_dst + i * jcp_.data_size], jcp_.data_type);
        load_scalar(xmm_idx_l, ptr[reg_dst_idx + i * sizeof(int)], memory::data_type::s32);
        store_scalar(ptr[reg_dst], xmm_val_l, jcp_.data_type);
        store_scalar(ptr[reg_dst_idx], xmm_idx_l, memory::data_type::s32);
        store_scalar(ptr[reg_dst + i * jcp_.data_size], xmm_val_p, jcp_.data_type);
        store_scalar(ptr[reg_dst_idx + i * sizeof(int)], xmm_idx_p, memory::data_type::s32);
    }

    inline void heap_swap_scalar(bool cmp_val) {
        if (isa == avx512_common) {
            if (cmp_val)
                vcmpps(k_mask, xmm_val_l, xmm_val_r, cmp_flg);
            else
                vcmpps(k_mask, xmm_idx_l, xmm_idx_r, _cmp_le_os);

            uni_vmovups(xmm_tmp, xmm_val_l);
            vblendmps(xmm_val_l | k_mask, xmm_val_r, xmm_val_l);
            vblendmps(xmm_val_r | k_mask, xmm_tmp, xmm_val_r);

            uni_vmovups(xmm_tmp, xmm_idx_l);
            vblendmps(xmm_idx_l | k_mask, xmm_idx_r, xmm_idx_l);
            vblendmps(xmm_idx_r | k_mask, xmm_tmp, xmm_idx_r);
        } else {
            if (cmp_val) {
                if (cmp_flg == _cmp_nlt_us)
                    vcmpge_oqps(xmm_mask, xmm_val_l, xmm_val_r);
                else
                    vcmple_oqps(xmm_mask, xmm_val_l, xmm_val_r);
            } else {
                vcmple_oqps(xmm_mask, xmm_idx_l, xmm_idx_r);
            }

            uni_vmovups(xmm_tmp, xmm_val_l);
            vblendvps(xmm_val_l, xmm_val_r, xmm_val_l, xmm_mask);
            vblendvps(xmm_val_r, xmm_tmp, xmm_val_r, xmm_mask);

            uni_vmovups(xmm_tmp, xmm_idx_l);
            vblendvps(xmm_idx_l, xmm_idx_r, xmm_idx_l, xmm_mask);
            vblendvps(xmm_idx_r, xmm_tmp, xmm_idx_r, xmm_mask);
        }
    }

    inline void topk_bubble_vector() {
        Xbyak::Label topk_main_loop_label;
        Xbyak::Label topk_main_loop_end_label;
        L(topk_main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(topk_main_loop_end_label, T_NEAR);

            topk_bubble(step);

            add(reg_src, step * jcp_.data_size);
            add(reg_dst, step * jcp_.data_size);
            add(reg_dst_idx, step * sizeof(int));
            sub(reg_work_amount, step);

            jmp(topk_main_loop_label, T_NEAR);
        }
        L(topk_main_loop_end_label);

        // tail
        if (tail) {
            topk_bubble(tail);
        }
    }

    inline void topk_bubble(int elt_num) {
        // load
        for (int i = 0; i < jcp_.top_k; i++) {
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val(i).getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                          {}, {load_pool_gpr_idxs});
            uni_vmovdqu(vmm_idx(i), table_val(i));
            uni_vcvtdq2ps(vmm_idx(i), vmm_idx(i));
        }
        // sort
        for (int i = 0; i < jcp_.top_k - 1; i++) {
            for (int j = jcp_.top_k - 1; j > i; j--) {
                swap_vector(j - 1, j);
            }
        }
        for (int i = jcp_.top_k; i < jcp_.axis_dim; i++) {
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val(jcp_.top_k).getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                          {}, {load_pool_gpr_idxs});
            uni_vmovdqu(vmm_idx(jcp_.top_k), table_val(i));
            uni_vcvtdq2ps(vmm_idx(jcp_.top_k), vmm_idx(jcp_.top_k));
            for (int j = jcp_.top_k; j > 0; j--) {
                swap_vector(j - 1, j);
            }
        }
        if (jcp_.sort_index) {
            for (int i = 0; i < jcp_.top_k - 1; i++) {
                for (int j = jcp_.top_k - 1; j > i; j--) {
                    swap_vector(j - 1, j, false);
                }
            }
        }
        // store
        for (int i = 0; i < jcp_.top_k; i++) {
            store_emitter->emit_code({static_cast<size_t>(vmm_val(i).getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_idx(i).getIdx())}, {static_cast<size_t>(reg_dst_idx.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, Precision::I32, elt_num, i * jcp_.sort_stride * sizeof(int)),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        }
    }

    inline void topk_bubble_horiz_blocked_innermost() {
        // load and sort
        int i = 0;
        if (jcp_.axis_dim < jcp_.blk_size) {
            load_scalar(xmm_val(0), ptr[reg_src], jcp_.data_type);
            uni_vmovdqu(xmm_idx(0), table_val(0));
            uni_vcvtdq2ps(xmm_idx(0), xmm_idx(0));
            i = 1;
        } else {
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val(0).getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, step, 0),
                          {}, {load_pool_gpr_idxs});
            uni_vmovdqu(vmm_idx(0), table_seq_val(0));
            uni_vcvtdq2ps(vmm_idx(0), vmm_idx(0));
            if (isa == cpu::x64::sse41) {
                load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val(1).getIdx())},
                              std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, step, 4 * jcp_.data_size),
                              {}, {load_pool_gpr_idxs});
                uni_vmovdqu(vmm_idx(1), table_seq_val(4));
                uni_vcvtdq2ps(vmm_idx(1), vmm_idx(1));
                swap_vector(0, 1);
            }
            i = jcp_.blk_size;
            for (; i + jcp_.blk_size <= jcp_.axis_dim; i += jcp_.blk_size) {
                int offset = i / jcp_.blk_size * jcp_.blk_stride;
                load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val(1).getIdx())},
                              std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, step, offset * jcp_.data_size),
                              {}, {load_pool_gpr_idxs});
                uni_vmovdqu(vmm_idx(1), table_seq_val(i));
                uni_vcvtdq2ps(vmm_idx(1), vmm_idx(1));
                swap_vector(0, 1);
                if (isa == cpu::x64::sse41) {
                    load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val(1).getIdx())},
                                  std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, step, (offset + 4) * jcp_.data_size),
                                  {}, {load_pool_gpr_idxs});
                    uni_vmovdqu(vmm_idx(1), table_seq_val(i + 4));
                    uni_vcvtdq2ps(vmm_idx(1), vmm_idx(1));
                    swap_vector(0, 1);
                }
            }
            horiz_process();
        }
        for (; i < jcp_.axis_dim; i++) {
            int offset = i / jcp_.blk_size * jcp_.blk_stride + i % jcp_.blk_size;
            load_scalar(xmm_val(1), ptr[reg_src + offset * jcp_.data_size], jcp_.data_type);
            uni_vmovdqu(xmm_idx(1), table_val(i));
            uni_vcvtdq2ps(xmm_idx(1), xmm_idx(1));
            swap_scalar(0, 1);
        }
        // store
        store_scalar(ptr[reg_dst], xmm_val(0), jcp_.data_type);
        store_scalar(ptr[reg_dst_idx], xmm_idx(0), memory::data_type::s32);
    }

    // dst: xmm_val(0) and xmm_idx(0)
    // aux: xmm_val(2/3/4) and xmm_idx(2/3/4)
    inline void horiz_process() {
        if (isa == cpu::x64::sse41) {
            horize_top1();
        } else if (isa == cpu::x64::avx2) {
            Xbyak::Ymm ymm_val_dst = Xbyak::Ymm(vmm_val(0).getIdx());
            vextractf128(xmm_val(2), ymm_val_dst, 0);
            vextractf128(xmm_val(3), ymm_val_dst, 1);
            Xbyak::Ymm ymm_idx_dst = Xbyak::Ymm(vmm_idx(0).getIdx());
            vextractf128(xmm_idx(2), ymm_idx_dst, 0);
            vextractf128(xmm_idx(3), ymm_idx_dst, 1);
            swap_scalar(2, 3);
            movups(xmm_val(0), xmm_val(2));
            movups(xmm_idx(0), xmm_idx(2));
            horize_top1();
        } else {
            Xbyak::Zmm zmm_val_dst = Xbyak::Zmm(vmm_val(0).getIdx());
            vextractf32x4(xmm_val(2), zmm_val_dst, 0);
            vextractf32x4(xmm_val(3), zmm_val_dst, 1);
            Xbyak::Zmm zmm_idx_dst = Xbyak::Zmm(vmm_idx(0).getIdx());
            vextractf32x4(xmm_idx(2), zmm_idx_dst, 0);
            vextractf32x4(xmm_idx(3), zmm_idx_dst, 1);
            swap_scalar(2, 3);
            vextractf32x4(xmm_val(3), zmm_val_dst, 2);
            vextractf32x4(xmm_val(4), zmm_val_dst, 3);
            vextractf32x4(xmm_idx(3), zmm_idx_dst, 2);
            vextractf32x4(xmm_idx(4), zmm_idx_dst, 3);
            swap_scalar(3, 4);
            swap_scalar(2, 3);
            movups(xmm_val(0), xmm_val(2));
            movups(xmm_idx(0), xmm_idx(2));
            horize_top1();
        }
    }

    // dst: xmm_val(0) and xmm_idx(0)
    // aux: xmm_val(3) and xmm_idx(3)
    inline void horize_top1() {
        movshdup(xmm_val(3), xmm_val(0)); // dst:1,2,3,4; aux:2,2,4,4
        movshdup(xmm_idx(3), xmm_idx(0));
        swap_scalar(0, 3);                // dst:f(1,2),f(2,2),f(3,4),f(4,4)
        movhlps(xmm_val(3), xmm_val(0));  // aux:f(3,4),f(4,4),4,4
        movhlps(xmm_idx(3), xmm_idx(0));
        swap_scalar(0, 3);                // dst:f(1,2,3,4),...
    }

    inline void topk_bubble_scalar_blocked_innermost() {
        // load
        for (int i = 0; i < jcp_.top_k; i++) {
            int offset = i / jcp_.blk_size * jcp_.blk_stride + i % jcp_.blk_size;
            load_scalar(xmm_val(i), ptr[reg_src + offset * jcp_.data_size], jcp_.data_type);
            uni_vmovdqu(xmm_idx(i), table_val(i));
            uni_vcvtdq2ps(xmm_idx(i), xmm_idx(i));
        }
        // sort
        for (int i = 0; i < jcp_.top_k - 1; i++) {
            for (int j = jcp_.top_k - 1; j > i; j--) {
                swap_scalar(j - 1, j);
            }
        }
        for (int i = jcp_.top_k; i < jcp_.axis_dim; i++) {
            int offset = i / jcp_.blk_size * jcp_.blk_stride + i % jcp_.blk_size;
            load_scalar(xmm_val(jcp_.top_k), ptr[reg_src + offset * jcp_.data_size], jcp_.data_type);
            uni_vmovdqu(xmm_idx(jcp_.top_k), table_val(i));
            uni_vcvtdq2ps(xmm_idx(jcp_.top_k), xmm_idx(jcp_.top_k));
            for (int j = jcp_.top_k; j > 0; j--) {
                swap_scalar(j - 1, j);
            }
        }
        if (jcp_.sort_index) {
            for (int i = 0; i < jcp_.top_k - 1; i++) {
                for (int j = jcp_.top_k - 1; j > i; j--) {
                    swap_scalar(j - 1, j, false);
                }
            }
        }
        // store
        for (int i = 0; i < jcp_.top_k; i++) {
            int offset = i / jcp_.blk_size * jcp_.blk_stride + i % jcp_.blk_size;
            store_scalar(ptr[reg_dst + offset * jcp_.data_size], xmm_val(i), jcp_.data_type);
            store_scalar(ptr[reg_dst_idx + offset * sizeof(int)], xmm_idx(i), memory::data_type::s32);
        }
    }

    inline void swap_vector(int l, int r, bool cmp_val = true) {
        if (isa == avx512_common) {
            if (cmp_val)
                vcmpps(k_mask, vmm_val(l), vmm_val(r), cmp_flg);
            else
                vcmpps(k_mask, vmm_idx(l), vmm_idx(r), _cmp_le_os);

            uni_vmovups(vmm_tmp, vmm_val(l));
            vblendmps(vmm_val(l) | k_mask, vmm_val(r), vmm_val(l));
            vblendmps(vmm_val(r) | k_mask, vmm_tmp, vmm_val(r));

            uni_vmovups(vmm_tmp, vmm_idx(l));
            vblendmps(vmm_idx(l) | k_mask, vmm_idx(r), vmm_idx(l));
            vblendmps(vmm_idx(r) | k_mask, vmm_tmp, vmm_idx(r));
        } else {
            if (cmp_val) {
                if (cmp_flg == _cmp_nlt_us)
                    vcmpge_oqps(vmm_mask, vmm_val(l), vmm_val(r));
                else
                    vcmple_oqps(vmm_mask, vmm_val(l), vmm_val(r));
            } else {
                vcmple_oqps(vmm_mask, vmm_idx(l), vmm_idx(r));
            }

            uni_vmovups(vmm_tmp, vmm_val(l));
            vblendvps(vmm_val(l), vmm_val(r), vmm_val(l), vmm_mask);
            vblendvps(vmm_val(r), vmm_tmp, vmm_val(r), vmm_mask);

            uni_vmovups(vmm_tmp, vmm_idx(l));
            vblendvps(vmm_idx(l), vmm_idx(r), vmm_idx(l), vmm_mask);
            vblendvps(vmm_idx(r), vmm_tmp, vmm_idx(r), vmm_mask);
        }
    }

    inline void swap_scalar(int l, int r, bool cmp_val = true) {
        if (isa == avx512_common) {
            if (cmp_val)
                vcmpps(k_mask, xmm_val(l), xmm_val(r), cmp_flg);
            else
                vcmpps(k_mask, xmm_idx(l), xmm_idx(r), _cmp_le_os);

            uni_vmovups(xmm_tmp, xmm_val(l));
            vblendmps(xmm_val(l) | k_mask, xmm_val(r), xmm_val(l));
            vblendmps(xmm_val(r) | k_mask, xmm_tmp, xmm_val(r));

            uni_vmovups(xmm_tmp, xmm_idx(l));
            vblendmps(xmm_idx(l) | k_mask, xmm_idx(r), xmm_idx(l));
            vblendmps(xmm_idx(r) | k_mask, xmm_tmp, xmm_idx(r));
        } else {
            if (cmp_val) {
                if (cmp_flg == _cmp_nlt_us)
                    vcmpge_oqps(xmm_mask, xmm_val(l), xmm_val(r));
                else
                    vcmple_oqps(xmm_mask, xmm_val(l), xmm_val(r));
            } else {
                vcmple_oqps(xmm_mask, xmm_idx(l), xmm_idx(r));
            }

            uni_vmovups(xmm_tmp, xmm_val(l));
            vblendvps(xmm_val(l), xmm_val(r), xmm_val(l), xmm_mask);
            vblendvps(xmm_val(r), xmm_tmp, xmm_val(r), xmm_mask);

            uni_vmovups(xmm_tmp, xmm_idx(l));
            vblendvps(xmm_idx(l), xmm_idx(r), xmm_idx(l), xmm_mask);
            vblendvps(xmm_idx(r), xmm_tmp, xmm_idx(r), xmm_mask);
        }
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, memory::data_type src_dt, bool cvt_dt = true) {
        switch (src_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                movss(xmm_src, op);
                break;
            case memory::data_type::bf16:
                pinsrw(xmm_src, op, 0x0);
                uni_vpslld(xmm_src, xmm_src, 16);
                break;
            case memory::data_type::s8:
                movsx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            case memory::data_type::u8:
                movzx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            default:
                assert(!"unknown src_dt");
        }

        if (cvt_dt && !isFloatCompatible(src_dt)) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, memory::data_type dst_dt, bool cvt_dt = true) {
        if (cvt_dt && !isFloatCompatible(dst_dt)) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                movss(op, xmm_dst);
                break;
            case memory::data_type::bf16:
                uni_vpsrld(xmm_dst, xmm_dst, 16);
                pextrw(op, xmm_dst, 0x0);
                break;
            case memory::data_type::s8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case memory::data_type::u8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    void prepare_idx_table() {
        auto broadcast_int = [&](int val) {
            for (size_t d = 0; d < vlen / sizeof(float); ++d) {
                dd(val);
            }
        };

        align(64);
        L(l_table);

        // 00000000 11111111 22222222 ...
        for (int i = 0; i < jcp_.axis_dim; i++) {
            broadcast_int(i);
        }

        // 01234567 89...
        for (int i = 0; i < jcp_.axis_dim; i++) {
            dd(i);
        }
    }
};

bool MKLDNNTopKNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto topKOp = ngraph::as_type_ptr<const ngraph::op::v1::TopK>(op);
        if (!topKOp) {
            errorMessage = "Node is not an instance of the TopK from the operations set v1 or v3";
            return false;
        }
        if (topKOp->get_mode() != ngraph::op::TopKMode::MAX &&
                topKOp->get_mode() != ngraph::op::TopKMode::MIN) {
            errorMessage = "Unsupported mode.";
            return false;
        }
        if (!one_of(topKOp->get_sort_type(), ngraph::op::TopKSortType::NONE,
                                  ngraph::op::TopKSortType::SORT_VALUES,
                                  ngraph::op::TopKSortType::SORT_INDICES)) {
            errorMessage = "Unsupported sort type.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNTopKNode::MKLDNNTopKNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "TopK layer with name '" + getName() + "'";

        auto topKOp = ngraph::as_type_ptr<ngraph::op::v1::TopK>(op);

        src_dims = topKOp->get_input_shape(TOPK_DATA);
        dst_dims = topKOp->get_output_shape(TOPK_DATA);
        dst_idx_dims = topKOp->get_output_shape(TOPK_INDEX);
        src_dims_size = src_dims.size();
        dst_dims_size = dst_dims.size();

        top_k = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(topKOp->get_input_node_shared_ptr(TOPK_K))->cast_vector<int>()[0];

        axis = topKOp->get_axis();

        if (topKOp->get_mode() == ngraph::op::TopKMode::MAX)
            mode_max = true;
        else
            mode_max = false;

        if (topKOp->get_sort_type() == ngraph::op::TopKSortType::SORT_INDICES)
            sort_index = true;
        else
            sort_index = false;
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNTopKNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 2 || getChildEdges().size() < 2)
        IE_THROW() << errorPrefix << " gets incorrect number of input/output edges!";

    if (getParentEdgeAt(TOPK_DATA)->getDims().ndims() != getChildEdgeAt(TOPK_DATA)->getDims().ndims()) {
        IE_THROW() << errorPrefix << " gets incorrect number of input/output dimensions!";
    }
    if (getParentEdgeAt(TOPK_K)->getDims().ndims() != 1) {
        IE_THROW() << errorPrefix << " gets incorrect index vector dimension! Index vector should be 1 dimension.";
    }

    if (dst_dims != dst_idx_dims)
        IE_THROW() << errorPrefix << " gets incorrect output tensor dimension sizes!";

    if (axis < 0)
        axis += src_dims_size;
    if (axis < 0 || axis >= static_cast<int>(src_dims_size))
        IE_THROW() << errorPrefix << " gets incorrect input parameters dimensions and axis number!";
    axis_dim = src_dims[axis];

    if (top_k > src_dims[axis])
        IE_THROW() << errorPrefix << " gets top_k out of range!";
}

void MKLDNNTopKNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    static const Precision supportedPrecision[] = {
        Precision::FP32,
        Precision::BF16,
        Precision::I32,
        Precision::I8,
        Precision::U8
    };

    Precision inputPrecision = getOriginalInputPrecisionAtPort(TOPK_DATA);
    Precision outputPrecision = getOriginalOutputPrecisionAtPort(TOPK_DATA);
    if (outputPrecision == Precision::BF16)
        assert(mayiuse(avx512_core));

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    // topk shouldn't change the data precision itself
    if (inputDataType != outputDataType)
        inputDataType = outputDataType;

    jit_mode = mayiuse(cpu::x64::sse41) && std::find(std::begin(supportedPrecision), std::end(supportedPrecision), inputPrecision)
                                        != std::end(supportedPrecision);

    data_type = inputDataType;
    data_size = MKLDNNExtensionUtils::sizeOfDataType(data_type);

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(2);
    config.outConfs.resize(2);
    config.inConfs[TOPK_DATA].constant = false;
    config.inConfs[TOPK_K].constant = false;
    config.outConfs[TOPK_DATA].constant = false;
    config.outConfs[TOPK_INDEX].constant = false;
    config.inConfs[TOPK_DATA].inPlace = -1;
    config.inConfs[TOPK_K].inPlace = -1;
    config.outConfs[TOPK_DATA].inPlace = -1;
    config.outConfs[TOPK_INDEX].inPlace = -1;

    auto pushDesc = [&](memory::format_tag inFormat, memory::format_tag outFormat, memory::data_type dataType, impl_desc_type impl_type) {
        config.inConfs[TOPK_DATA].desc = MKLDNNMemoryDesc(getParentEdgeAt(TOPK_DATA)->getDims(), dataType, inFormat);
        config.inConfs[TOPK_K].desc = MKLDNNMemoryDesc(getParentEdgeAt(TOPK_K)->getDims(), memory::data_type::s32, memory::format_tag::x);
        config.outConfs[TOPK_DATA].desc = MKLDNNMemoryDesc(getChildEdgeAt(TOPK_DATA)->getDims(), dataType, outFormat);
        config.outConfs[TOPK_INDEX].desc = MKLDNNMemoryDesc(getChildEdgeAt(TOPK_INDEX)->getDims(), memory::data_type::s32, outFormat);
        supportedPrimitiveDescriptors.push_back({config, impl_type, outFormat});
    };

    if (jit_mode) {
        impl_desc_type impl_type = impl_desc_type::jit_sse42;
        if (mayiuse(cpu::x64::avx512_common)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (mayiuse(cpu::x64::avx2)) {
            impl_type = impl_desc_type::jit_avx2;
        }

        pushDesc(MKLDNNMemory::GetPlainFormat(memory::dims(getParentEdgeAt(TOPK_DATA)->getDims().ndims())),
             MKLDNNMemory::GetPlainFormat(memory::dims(getChildEdgeAt(TOPK_DATA)->getDims().ndims())), data_type, impl_type);
        if (getParentEdgeAt(TOPK_DATA)->getDims().ndims() == 4) {
            if (mayiuse(cpu::x64::avx512_common)) {
                pushDesc(memory::format_tag::nhwc, memory::format_tag::nhwc, data_type, impl_type);
                pushDesc(memory::format_tag::nChw16c, memory::format_tag::nChw16c, data_type, impl_type);
            } else if (mayiuse(cpu::x64::sse41)) {
                pushDesc(memory::format_tag::nhwc, memory::format_tag::nhwc, data_type, impl_type);
                pushDesc(memory::format_tag::nChw8c, memory::format_tag::nChw8c, data_type, impl_type);
            }
        } else if (getParentEdgeAt(TOPK_DATA)->getDims().ndims() == 5) {
            if (mayiuse(cpu::x64::avx512_common)) {
                pushDesc(memory::format_tag::ndhwc, memory::format_tag::ndhwc, data_type, impl_type);
                pushDesc(memory::format_tag::nCdhw16c, memory::format_tag::nCdhw16c, data_type, impl_type);
            } else if (mayiuse(cpu::x64::sse41)) {
                pushDesc(memory::format_tag::ndhwc, memory::format_tag::ndhwc, data_type, impl_type);
                pushDesc(memory::format_tag::nCdhw8c, memory::format_tag::nCdhw8c, data_type, impl_type);
            }
        }
    } else {
        pushDesc(MKLDNNMemory::GetPlainFormat(memory::dims(getParentEdgeAt(TOPK_DATA)->getDims().ndims())),
             MKLDNNMemory::GetPlainFormat(memory::dims(getChildEdgeAt(TOPK_DATA)->getDims().ndims())), memory::data_type::f32, impl_desc_type::ref);
    }
}

void MKLDNNTopKNode::createPrimitive() {
    auto &dstMemPtr = getChildEdgeAt(TOPK_DATA)->getMemoryPtr();
    auto &srcDataMemPtr = getParentEdgeAt(TOPK_DATA)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix << " has not allocated destination memory.";
    if (!srcDataMemPtr || !srcDataMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix << " has not allocate input memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << errorPrefix << " has nullable preferable primitive descriptor";

    auto selectedPD = getSelectedPrimitiveDescriptor();
    Layout selected_layout = selectedPD->getConfig().inConfs[TOPK_DATA].desc.getLayout();
    if (MKLDNNMemory::GetPlainLayout(getParentEdgeAt(TOPK_DATA)->getDims()) == selected_layout) {
        layout = TopKLayoutType::topk_ncsp;
    } else if ((selected_layout == NHWC) || (selected_layout == NDHWC)) {
        layout = TopKLayoutType::topk_nspc;
    } else {
        layout = TopKLayoutType::topk_blocked;
    }

    topk_innermost = false;
    if ((layout == TopKLayoutType::topk_ncsp && axis == dst_dims_size - 1) ||
       ((layout == TopKLayoutType::topk_nspc || layout == TopKLayoutType::topk_blocked) && axis == 1))
        topk_innermost = true;

    if (mayiuse(cpu::x64::avx512_common)) {
        blk_size = 16;
        count_xmm = 16; // only 16 vector registers are valid in sse instructions
    } else if (mayiuse(cpu::x64::sse41)) {
        blk_size = 8;
        count_xmm = 16;
    }

    auto ceil_pow_2 = [&](size_t n, size_t &p) {
        int m = n - 1;
        while (m) {
            p <<= 1;
            m >>= 1;
        }
    };

    // [case 1]: if 2 * (top_k + 1) + 2 <= count_vec, thus top_k is relatively small, and vector registers are sufficient
    //           to keep all necessary data for sorting, no need to load and store frequently, use bubble sort;
    // [case 2]: otherwise, only when topk is imposed on innermost dimsension of planar(ncsp/nspc) layout, should heap sort be used;
    // [case 3]: bitonic sort is used as default
    if (top_k <= count_xmm / 2 - 2) {
        algorithm = TopKAlgorithm::topk_bubble_sort;
    } else if ((layout == TopKLayoutType::topk_ncsp || layout == TopKLayoutType::topk_nspc) && topk_innermost) {
        algorithm = TopKAlgorithm::topk_heap_sort;
    } else {
        algorithm = TopKAlgorithm::topk_bitonic_sort;
    }

    if (algorithm == TopKAlgorithm::topk_bitonic_sort) {
        ceil_pow_2(axis_dim, bitonic_size);
        ceil_pow_2(top_k, bitonic_k_size);
    }

    if (jit_mode) {
        O = 1, I = 1;
        A = src_dims[axis];
        if (layout == TopKLayoutType::topk_ncsp) {
            for (int i = 0; i < axis; i++)
                O *= src_dims[i];
            for (size_t i = axis + 1; i < src_dims.size(); i++)
                I *= src_dims[i];
        } else if (layout == TopKLayoutType::topk_nspc) {
            if (axis != 1) {
                for (int i = 0; i < axis; i++) {
                    if (i != 1)
                        O *= src_dims[i];
                }
                I = src_dims[1];
                for (size_t i = axis + 1; i < src_dims.size(); i++)
                    if (i != 1)
                        I *= src_dims[i];
            } else {
                for (int i = 0; i < src_dims_size; i++) {
                    if (i != 1)
                        O *= src_dims[i];
                }
            }
        } else if (layout == TopKLayoutType::topk_blocked) {
            N = src_dims[0];
            ICB = div_up(src_dims[1], blk_size);
            OCB = div_up(dst_dims[1], blk_size);
            D = 1;
            if (src_dims_size == 5)
                D = src_dims[2];
            H = src_dims[src_dims_size - 2];
            W = src_dims[src_dims_size - 1];
            if (axis == 0) { //topk on N
                I = ICB * D * H * W * blk_size;
            } else if (axis == 1) { //topk on C
                if (algorithm != TopKAlgorithm::topk_bitonic_sort) {
                    O = N * D * H * W;
                } else {
                    I = D * H * W;
                }
            } else if (src_dims_size == 5 && axis == 2) { //topk on D
                O = N * ICB;
                I = H * W * blk_size;
            } else if (axis == src_dims_size - 2) { //topk on H
                O = N * ICB * D;
                I = W * blk_size;
            } else if (axis == src_dims_size - 1) { //topk on W
                O = N * ICB * D * H;
                I = blk_size;
            }
        }

        auto jcp = jit_topk_config_params();
        jcp.data_type = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().inConfs[TOPK_DATA].desc.getPrecision());
        jcp.precision = selectedPD->getConfig().inConfs[TOPK_DATA].desc.getPrecision();
        jcp.data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.data_type);
        jcp.blk_size = blk_size;
        jcp.layout = layout;
        jcp.top_k = top_k;
        jcp.axis_dim = axis_dim;
        jcp.mode_max = mode_max;
        jcp.sort_index = sort_index;
        jcp.topk_innermost = topk_innermost;
        jcp.algorithm = algorithm;
        jcp.bitonic_k_size = bitonic_k_size;
        jcp.bitonic_size = bitonic_size;
        jcp.sort_stride = static_cast<int>(I);
        jcp.work_amout = static_cast<int>(I);
        if (layout == TopKLayoutType::topk_blocked && topk_innermost) {
            jcp.blk_stride = D * H * W * blk_size;
            if (algorithm == TopKAlgorithm::topk_bubble_sort) {
                jcp.work_amout = static_cast<int>(axis_dim);
            }
        }

        if (mayiuse(cpu::x64::avx512_common)) {
            topk_kernel.reset(new jit_uni_topk_kernel_f32<cpu::x64::avx512_common>(jcp));
        } else if (mayiuse(cpu::x64::avx2)) {
            topk_kernel.reset(new jit_uni_topk_kernel_f32<cpu::x64::avx2>(jcp));
        } else if (mayiuse(cpu::x64::sse41)) {
            topk_kernel.reset(new jit_uni_topk_kernel_f32<cpu::x64::sse41>(jcp));
        }

        if (topk_kernel)
            topk_kernel->create_ker();
    } else { //reference mode
        int j;
        for (j = src_dims.size() - 1; j >= 0; j--) {
            if (src_dims[j] != 1)
                break;
        }
        if (static_cast<size_t>(j) == axis)
            is_last_dim = true;
        dim = static_cast<int>(src_dims[axis]);
        before_num = count(src_dims, 0, axis);
    }
}

void MKLDNNTopKNode::execute(mkldnn::stream strm) {
    auto &srcMemPtr = getParentEdgeAt(TOPK_DATA)->getMemoryPtr();
    auto &dstMemPtr = getChildEdgeAt(TOPK_DATA)->getMemoryPtr();
    auto &dstIndexesMemPtr = getChildEdgeAt(TOPK_INDEX)->getMemoryPtr();

    const uint8_t *src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetPtr());
    uint8_t *dst_data = reinterpret_cast<uint8_t *>(dstMemPtr->GetPtr());
    uint8_t *dst_idx = reinterpret_cast<uint8_t *>(dstIndexesMemPtr->GetPtr());

    if (jit_mode) {
        uint8_t *process_data = NULL;
        uint8_t *process_idx = NULL;
        if (algorithm == TopKAlgorithm::topk_bitonic_sort) {
            size_t src_count = srcMemPtr->GetElementsCount();
            process_data = reinterpret_cast<uint8_t *>(std::malloc(src_count * data_size));
            process_idx = reinterpret_cast<uint8_t *>(std::malloc(src_count * sizeof(int32_t)));
            if (!process_data || !process_idx)
                IE_THROW() << errorPrefix << " has not allocated process memory.";
        }

        if (layout == TopKLayoutType::topk_ncsp || layout == TopKLayoutType::topk_nspc) {
            topk_PLN(src_data, dst_data, dst_idx, process_data, process_idx);
        } else {
            topk_BLK(src_data, dst_data, dst_idx, process_data, process_idx);
        }

        if (algorithm == TopKAlgorithm::topk_bitonic_sort) {
            std::free(process_data);
            std::free(process_idx);
        }
    } else {
        if (layout == TopKLayoutType::topk_ncsp) {
            auto in_ptr = reinterpret_cast<const float *>(src_data);
            auto out_ptr = reinterpret_cast<float *>(dst_data);
            auto out_idx_ptr = reinterpret_cast<int32_t *>(dst_idx);
            topk_ref(in_ptr, out_ptr, out_idx_ptr);
        } else {
            IE_THROW() << errorPrefix <<  "only support plain layout on machine w/o sse42.";
        }
    }
}

void MKLDNNTopKNode::topk_PLN(const uint8_t *in_ptr, uint8_t *out_ptr, uint8_t *out_idx_ptr,
                                    uint8_t *process_ptr, uint8_t *process_idx_ptr) {
    parallel_for(O, [&](size_t o) {
        const uint8_t *in_ptr_a = in_ptr + o * A * I * data_size;
        uint8_t *process_ptr_a = process_ptr + o * A * I * data_size;
        uint8_t *process_idx_ptr_a = process_idx_ptr + o * A * I * sizeof(int32_t);
        uint8_t *out_ptr_a = out_ptr + o * top_k * I * data_size;
        uint8_t *out_idx_ptr_a = out_idx_ptr + o * top_k * I * sizeof(int32_t);
        size_t work_amount = I;
        topk_kernel_process(in_ptr_a, out_ptr_a, out_idx_ptr_a, process_ptr_a, process_idx_ptr_a, work_amount);
    });
}

void MKLDNNTopKNode::topk_BLK(const uint8_t *in_ptr, uint8_t *out_ptr, uint8_t *out_idx_ptr,
                                    uint8_t *process_ptr, uint8_t *process_idx_ptr) {
    if (!topk_innermost) {
        parallel_for(O, [&](size_t o) {
            const uint8_t *in_ptr_a = in_ptr + o * A * I * data_size;
            uint8_t *process_ptr_a = process_ptr + o * A * I * data_size;
            uint8_t *process_idx_ptr_a = process_idx_ptr + o * A * I * sizeof(int32_t);
            uint8_t *out_ptr_a = out_ptr + o * top_k * I * data_size;
            uint8_t *out_idx_ptr_a = out_idx_ptr + o * top_k * I * sizeof(int32_t);
            size_t work_amount = I;
            topk_kernel_process(in_ptr_a, out_ptr_a, out_idx_ptr_a, process_ptr_a, process_idx_ptr_a, work_amount);
        });
    } else { //topk on C
        if (algorithm == TopKAlgorithm::topk_bubble_sort) {
            parallel_for(O, [&](size_t o) {
                size_t n = o / (D * H * W);
                size_t s = o % (D * H * W);
                const uint8_t *in_ptr_a = in_ptr + (n * ICB * D * H * W + s) * blk_size * data_size;
                uint8_t *out_ptr_a = out_ptr + (n * OCB * D * H * W + s) * blk_size * data_size;
                uint8_t *out_idx_ptr_a = out_idx_ptr + (n * OCB * D * H * W + s) * blk_size * sizeof(int32_t);
                size_t work_amount = axis_dim;
                topk_kernel_process(in_ptr_a, out_ptr_a, out_idx_ptr_a, NULL, NULL, work_amount);
            });
        } else if (algorithm == TopKAlgorithm::topk_bitonic_sort) {
            parallel_for(N, [&](size_t n) {
                const uint8_t *in_ptr_a = in_ptr + n * ICB * I * blk_size * data_size;
                uint8_t *process_ptr_a = process_ptr + n * ICB * blk_size * I * data_size;
                uint8_t *process_idx_ptr_a = process_idx_ptr + n * ICB * blk_size * I * sizeof(int32_t);
                uint8_t *out_ptr_a = out_ptr + n * OCB * I * blk_size * data_size;
                uint8_t *out_idx_ptr_a = out_idx_ptr + n * OCB * I * blk_size * sizeof(int32_t);
                size_t work_amount = I;
                topk_kernel_process(in_ptr_a, out_ptr_a, out_idx_ptr_a, process_ptr_a, process_idx_ptr_a, work_amount);
            });
        }
    }
}

inline void MKLDNNTopKNode::topk_kernel_process(const uint8_t *in_p, uint8_t *out_p, uint8_t *out_idx_p,
                                                uint8_t *process_p, uint8_t *process_idx_p, size_t work_amount) {
    auto arg = jit_topk_call_args();
    arg.src = static_cast<const void *>(in_p);
    arg.process = static_cast<void *>(process_p);
    arg.process_index = static_cast<void *>(process_idx_p);
    arg.dst = static_cast<void *>(out_p);
    arg.index = static_cast<void *>(out_idx_p);
    arg.work_amount = work_amount;
    (*topk_kernel)(&arg);
}

void MKLDNNTopKNode::topk_ref(const float *in_ptr, float *out_ptr, int32_t *dst_idx) {
    if (top_k == 1) {
        if (is_last_dim) {
            if (mode_max)
                top1(in_ptr, out_ptr, dst_idx, src_dims, [](float x, float y)->float { return x > y; });
            else
                top1(in_ptr, out_ptr, dst_idx, src_dims, [](float x, float y)->float { return x < y; });
        } else {
            if (mode_max)
                top1_axis(in_ptr, out_ptr, dst_idx, src_dims, [](float x, float y)->float { return x > y; });
            else
                top1_axis(in_ptr, out_ptr, dst_idx, src_dims, [](float x, float y)->float { return x < y; });
        }
    } else {
        if (is_last_dim) {
            if (mode_max)
                topk(in_ptr, out_ptr, dst_idx, src_dims, [](float x, float y)->float { return x > y; });
            else
                topk(in_ptr, out_ptr, dst_idx, src_dims, [](float x, float y)->float { return x < y; });
        } else {
            if (mode_max)
                topk_axis(in_ptr, out_ptr, dst_idx, src_dims, [](float x, float y)->float { return x > y; });
            else
                topk_axis(in_ptr, out_ptr, dst_idx, src_dims, [](float x, float y)->float { return x < y; });
        }
    }
}

void MKLDNNTopKNode::top1_axis(const float* src_data, float* dst_data, int32_t* dst_idx, const SizeVector &in_dims,
                               std::function<float(float, float)> compare) const {
    int after_num = count(in_dims, axis + 1, in_dims.size());

    parallel_for2d(before_num, after_num, [&](int i0, int i1) {
        int index_max_val = 0;
        int s_index = i0 * dim * after_num + i1;
        float max_val = src_data[s_index];
        for (int i2 = 1; i2 < dim; i2++) {
            s_index += after_num;
            if (compare(src_data[s_index], max_val)) {
                max_val = src_data[s_index];
                index_max_val = i2;
            }
        }
        if (dst_data)
            dst_data[i0 * after_num + i1] = max_val;
        if (dst_idx)
            dst_idx[i0 * after_num + i1] = index_max_val;
    });
}

void MKLDNNTopKNode::top1(const float* src_data, float* dst_data, int32_t* dst_idx, const SizeVector &in_dims,
                          std::function<float(float, float)> compare) const {
    parallel_for(before_num, [&](int i0) {
        int index_max_val = 0;
        int s_index = i0 * dim;
        float max_val = src_data[s_index];
        for (int i1 = 1; i1 < dim; i1++) {
            s_index++;
            if (compare(src_data[s_index], max_val)) {
                max_val = src_data[s_index];
                index_max_val = i1;
            }
        }
        if (dst_data)
            dst_data[i0] = max_val;
        if (dst_idx)
            dst_idx[i0] = index_max_val;
    });
}

void MKLDNNTopKNode::topk_axis(const float* src_data, float* dst_data, int32_t* dst_idx, const SizeVector &in_dims,
                               std::function<float(float, float)> compare) const {
    int after_num = count(in_dims, axis + 1, in_dims.size());

    parallel_for2d(before_num, after_num, [&](int i0, int i1) {
        std::vector<float> max_values(top_k + 1);
        std::vector<int> max_indexes(top_k + 1);
        float tmp_value;
        int tmp_index;
        int s_index = i0 * dim * after_num + i1;

        auto swap_func = [&](int index1, int index2) {
            tmp_value = max_values[index1];
            max_values[index1] = max_values[index2];
            max_values[index2] = tmp_value;

            tmp_index = max_indexes[index1];
            max_indexes[index1] = max_indexes[index2];
            max_indexes[index2] = tmp_index;
        };

        for (int i2 = 0; i2 < top_k; i2++) {
            max_values[i2] = src_data[s_index];
            max_indexes[i2] = i2;
            s_index += after_num;
        }
        for (int i2 = 0; i2 < top_k - 1; i2++) {
            for (int i3 = top_k - 1; i3 > i2; i3--) {
                if (compare(max_values[i3], max_values[i3 - 1])) {
                    swap_func(i3, i3 - 1);
                }
            }
        }
        for (int i2 = top_k; i2 < dim; i2++) {
            max_values[top_k] = src_data[s_index];
            max_indexes[top_k] = i2;
            for (int i3 = top_k; i3 > 0; i3--) {
                if (compare(max_values[i3], max_values[i3 - 1]))
                    swap_func(i3, i3 - 1);
                else
                    break;
            }
            s_index += after_num;
        }
        if (sort_index) {
            for (int i2 = 0; i2 < top_k - 1; i2++) {
                for (int i3 = top_k - 1; i3 > i2; i3--) {
                    if (std::greater<int>()(max_indexes[i3 - 1], max_indexes[i3])) {
                        swap_func(i3, i3 - 1);
                    }
                }
            }
        }
        if (dst_data) {
            for (int i2 = 0; i2 < top_k; i2++)
                dst_data[i0 * top_k * after_num + i2 * after_num + i1] = max_values[i2];
        }
        if (dst_idx) {
            for (int i2 = 0; i2 < top_k; i2++)
                dst_idx[i0 * top_k * after_num + i2 * after_num + i1] = max_indexes[i2];
        }
    });
}

void MKLDNNTopKNode::topk(const float* src_data, float* dst_data, int32_t* dst_idx, const SizeVector &in_dims,
                          std::function<float(float, float)> compare) const {
    parallel_for(before_num, [&](int i0) {
        std::vector<float> max_values(top_k + 1);
        std::vector<int> max_indexes(top_k + 1);
        float tmp_value;
        int tmp_index;
        int s_index = i0 * dim;

        auto swap_func = [&](int index1, int index2) {
            tmp_value = max_values[index1];
            max_values[index1] = max_values[index2];
            max_values[index2] = tmp_value;

            tmp_index = max_indexes[index1];
            max_indexes[index1] = max_indexes[index2];
            max_indexes[index2] = tmp_index;
        };

        for (int i2 = 0; i2 < top_k; i2++) {
            max_values[i2] = src_data[s_index];
            max_indexes[i2] = i2;
            s_index++;
        }
        for (int i2 = 0; i2 < top_k - 1; i2++) {
            for (int i3 = top_k - 1; i3 > i2; i3--) {
                if (compare(max_values[i3], max_values[i3 - 1])) {
                    swap_func(i3, i3 - 1);
                }
            }
        }
        for (int i2 = top_k; i2 < dim; i2++) {
            max_values[top_k] = src_data[s_index];
            max_indexes[top_k] = i2;
            for (int i3 = top_k; i3 > 0; i3--) {
                if (compare(max_values[i3], max_values[i3 - 1]))
                    swap_func(i3, i3 - 1);
                else
                    break;
            }
            s_index++;
        }
        if (sort_index) {
            for (int i2 = 0; i2 < top_k - 1; i2++) {
                for (int i3 = top_k - 1; i3 > i2; i3--) {
                    if (std::greater<int>()(max_indexes[i3 - 1], max_indexes[i3])) {
                        swap_func(i3, i3 - 1);
                    }
                }
            }
        }
        if (dst_data) {
            for (int i2 = 0; i2 < top_k; i2++)
                dst_data[i0 * top_k + i2] = max_values[i2];
        }
        if (dst_idx) {
            for (int i2 = 0; i2 < top_k; i2++)
                dst_idx[i0 * top_k + i2] = max_indexes[i2];
        }
    });
}

inline int MKLDNNTopKNode::count(SizeVector dims, size_t start_ind, size_t end_ind) {
    size_t count = 1;
    for (size_t i = start_ind; i < end_ind; i++)
        count *= dims[i];
    return static_cast<int>(count);
}

inline int MKLDNNTopKNode::count(SizeVector dims, size_t start_ind) {
    return count(dims, start_ind, dims.size());
}

bool MKLDNNTopKNode::created() const {
    return getType() == TopK;
}

REG_MKLDNN_PRIM_FOR(MKLDNNTopKNode, TopK);
