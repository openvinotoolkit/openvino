/*******************************************************************************
* Copyright 2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include <float.h>
#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_uni_1x1_conv_utils.hpp"
#include "jit_avx512_core_bf16_1x1_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_1x1_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

using namespace Xbyak;

void jit_avx512_core_bf16_1x1_conv_kernel::bcast_loop(int load_loop_blk)
{
    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_bcast_data, reg_bcast_data);

    mov(aux_reg_output_data, reg_output_data);
    mov(bcast_loop_iter, EVEX_compress_addr(rsp, bcast_loop_work_offt));

    Label bcast_loop;
    Label bcast_loop_tail;

    cmp(bcast_loop_iter, jcp.ur);
    jl(bcast_loop_tail, T_NEAR);

    L(bcast_loop); {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            reduce_loop(load_loop_blk, jcp.ur, i, false);
            if (i < num_substeps - 1) {
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data, jcp.bcast_loop_output_substep);
            }
            else {
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_step
                    - (num_substeps - 1) * jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data, jcp.bcast_loop_output_step
                    - (num_substeps - 1) * jcp.bcast_loop_output_substep);
            }
        }
        sub(bcast_loop_iter, jcp.bcast_block);
        cmp(bcast_loop_iter, jcp.bcast_block);
        jge(bcast_loop, T_NEAR);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        cmp(bcast_loop_iter, 0);
        jz(bcast_loop_tail_out, T_NEAR);
        reduce_loop(load_loop_blk, jcp.ur_tail, 0, true);
        L(bcast_loop_tail_out);
    }
}

void jit_avx512_core_bf16_1x1_conv_kernel::reduce_loop(int load_loop_blk,
         int ur, int substep, bool wraparound)
{
    auto vreg_load = [=](int i_load) {
        int idx = ur * load_loop_blk + i_load;
        assert(idx < 31);
        return Zmm(idx);
    };
    auto ymm_store = [=]() {
        return Xbyak::Ymm(31);
    };
    auto zmm_store = [=]() {
        return Xbyak::Zmm(31);
    };
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    auto zmm_prm = [=]() {
        return Xbyak::Zmm(31);
    };
#endif
    auto vreg_accum = [=](int i_load, int i_ur) {
        int idx = i_ur * load_loop_blk + i_load;
        assert(idx < 31);
        return Zmm(idx);
    };

    auto bias_ptr = [=](int i_load) {
        return EVEX_compress_addr(reg_bias_data,
            sizeof(float) * jcp.oc_block * i_load);
    };

    auto bcast_ptr = [=](int i_reduce, int i_ur, bool bcast) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        int offt;
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                   backward_data)) {
            assert(jcp.reduce_loop_unroll == jcp.reduce_block);
            offt = (i_reduce == jcp.reduce_loop_unroll)
                    ? (jcp.bcast_dim + i_ur) * jcp.reduce_loop_unroll
                    : i_ur * jcp.reduce_loop_unroll + i_reduce;
        } else {
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
            offt = i_reduce * jcp.ic_block + i_ur;
#else
            offt = (i_reduce / 2) * 2 * jcp.ic_block + 2 * i_ur;
#endif
        }
        return EVEX_compress_addr(aux_reg_bcast_data, jcp.typesize_in * offt,
                                bcast);
    };

    auto load_ptr = [=](int i_reduce, int i_load) {
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;
        int offt = (i_load * jcp.reduce_dim + u0) * jcp.load_block;
        return EVEX_compress_addr(aux_reg_load_data,
                                  u1 * jcp.reduce_loop_load_step
                                  + jcp.typesize_in * offt);
    };

    auto output_ptr = [=](int i_load, int i_ur, int scale = 1) {
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                   backward_data))
            return EVEX_compress_addr(aux_reg_output_data,
                    (i_load * jcp.bcast_dim + i_ur) * jcp.load_block
                    * jcp.typesize_out * scale);
        else
            return ptr[aux_reg_output_data +
                       (i_load
                            ? reg_output_stride * i_load
                            : 0) // TODO: Xbyak should allow 0 scale
                       + jcp.typesize_out * jcp.load_block * i_ur];
    };

    auto init = [=]() {
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                vpxord(r, r, r);
            }
    };

    auto store = [=]() {
        if (!jcp.is_cpx)
            bf16_emu_->init_vcvtneps2bf16();
        if (jcp.prop_kind == backward_weights) {
        Label store_noadd;
        if (!jcp.with_sum) {
            test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            jnz(store_noadd, T_NEAR);
        }
        for (int i_ur = 0; i_ur < ur; ++i_ur)
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                auto r = vreg_accum(i_load, i_ur);
                vaddps(r, r, output_ptr(i_load, i_ur));
            }
        L(store_noadd);
        }
        auto post_ops = [=]() {
            /* Bias addition */
            if (jcp.with_bias && one_of(jcp.prop_kind,
                          forward_training, forward_inference)) {
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    for (int i_ur = 0; i_ur < ur; ++i_ur) {
                            vaddps(vreg_accum(i_load, i_ur), bias_ptr(i_load));
                    }
                }
            }
            /* Sum */
            if (jcp.with_sum) {
                auto zmm_prev_dst = Xbyak::Zmm(31);
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    for (int i_ur = 0; i_ur < ur; ++i_ur) {
                        if (jcp.dst_dt == data_type::bf16) {
                            vpmovzxwd(zmm_prev_dst, output_ptr(i_load, i_ur));
                            vpslld(zmm_prev_dst, zmm_prev_dst, 16);
                            vaddps(vreg_accum(i_load, i_ur), zmm_prev_dst);
                         } else {
                            vmovups(zmm_prev_dst, output_ptr(i_load, i_ur));
                            vaddps(vreg_accum(i_load, i_ur), zmm_prev_dst);
                         }
                    }
                }
            }
            /* Eltwise post-op */
            if (jcp.with_eltwise)
                eltwise_injector_->compute_vector_range(0, ur * load_loop_blk);
        };

        auto store_output = [=](bool output_is_aligned) {
            if (jcp.prop_kind == backward_weights || jcp.dst_dt == data_type::f32) {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                        vmovups(output_ptr(i_load, i_ur),
                                    vreg_accum(i_load, i_ur));
                    }
                }
            } else if (jcp.dst_dt == data_type::bf16) {
                if (jcp.is_cpx) {
                    for (int i_load = 0; i_load < load_loop_blk; i_load++) {
                        int n_2bf2ps = (ur / 2) * 2, i_ur = 0;
                        for (i_ur = 0; i_ur < n_2bf2ps; i_ur += 2) {
                            auto zmm = zmm_store();
                            vcvtne2ps2bf16(zmm, vreg_accum(i_load, i_ur + 1),
                                                vreg_accum(i_load, i_ur));
                            vmovups(output_ptr(i_load, i_ur), zmm);
                        }
                        if (i_ur < ur) {
                            auto ymm = ymm_store();
                            vcvtneps2bf16(ymm, vreg_accum(i_load, i_ur));
                            vmovups(output_ptr(i_load, i_ur), ymm);
                        }
                    }
                } else {
                    for (int i_load = 0; i_load < load_loop_blk; i_load++) {
                        for (int i_ur = 0; i_ur < ur; ++i_ur) {
                            auto ymm = ymm_store();
                            bf16_emu_->r_vcvtneps2bf16(ymm, vreg_accum(i_load, i_ur));
                            vmovups(output_ptr(i_load, i_ur), ymm);
                        }
                    }

                }
            } else {
                assert(!"unsupported destination type");
            }
        };

        Label unaligned_store, end_store;
        post_ops();
        test(aux_reg_output_data, cpu_isa_traits<avx512_common>::vlen - 1);
        jnz(unaligned_store, T_NEAR);
        store_output(true);
        jmp(end_store, T_NEAR);
        L(unaligned_store); {
            store_output(false);
        }
        L(end_store);
    };

    auto fma_block = [=](bool is_tail) {
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
        if (jcp.prop_kind == backward_weights)
            mov(EVEX_compress_addr(rsp, perm_reg_offset), reg_reduce_pos_flag);
        int n_reduce = jcp.reduce_loop_unroll;
#else
        int n_reduce_tail = jcp.reduce_dim % jcp.reduce_loop_unroll;
        int n_reduce = is_tail && n_reduce_tail > 0
                ? n_reduce_tail
                : jcp.reduce_loop_unroll;
#endif
        for (int i_reduce = 0; i_reduce < n_reduce; i_reduce += 2) {
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
            Opmask load_mask = Opmask(7);
            if (jcp.prop_kind == backward_weights) {
                if (jcp.reduce_loop_unroll % 2
                    && i_reduce + 2 >= jcp.reduce_loop_unroll) {
                    mov(reg_trans_tmp.cvt32(), 0x0000ffff);
                } else {
                    mov(reg_trans_tmp.cvt32(), 0xffffffff);
                }
                kmovd(load_mask, reg_trans_tmp.cvt32());

                mov(reg_trans_tmp, dst_prm_table);
                vmovups(zmm_prm(), ptr[reg_trans_tmp]);

                int offt = i_reduce * jcp.ic_block;
                auto bcast_values = vreg_load(load_loop_blk-1);

                vpxord(bcast_values, bcast_values, bcast_values);
                vmovdqu16(bcast_values | load_mask | T_z,
                        ptr[aux_reg_bcast_data + jcp.typesize_in * offt]);
                vpermw(bcast_values, zmm_prm(), bcast_values);
                vmovups(ptr[rsp + broadcast_space], bcast_values);
            }
#endif
            if (jcp.is_cpx) {
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
                    if (jcp.prop_kind == backward_weights) {
                        vmovdqu16(vreg_load(i_load) | load_mask | T_z,
                                    load_ptr(i_reduce, i_load));
                        vpermw(vreg_load(i_load), zmm_prm(), vreg_load(i_load));
                    } else
#endif
                        vmovups(vreg_load(i_load), load_ptr(i_reduce, i_load));
                }
            }
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
                    int offset = jcp.typesize_in * 2 * i_ur;
#endif
                if (jcp.ver == ver_avx512_core && jcp.expl_bcast
                        && load_loop_blk > 1) {
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
                    if (jcp.prop_kind == backward_weights)
                        vpbroadcastd(vreg_bcast,
                            ptr[rsp + broadcast_space + offset]);
                    else
#endif
                        vpbroadcastd(vreg_bcast,
                            bcast_ptr(i_reduce, i_ur, false));
                }
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    if (!jcp.is_cpx) {
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
                        if (jcp.prop_kind == backward_weights) {
                            vmovdqu16(vreg_load(i_load) | load_mask | T_z,
                                        load_ptr(i_reduce, i_load));
                            vpermw(vreg_load(i_load), zmm_prm(), vreg_load(i_load));
                        } else
#endif
                            vmovups(vreg_load(i_load), load_ptr(i_reduce, i_load));
                    }
                    if (jcp.ver == ver_avx512_core && jcp.expl_bcast
                            && load_loop_blk > 1) {
                            if (!jcp.is_cpx) {
                                auto acc = vreg_accum(i_load, i_ur);
                                auto wei = vreg_load(i_load);
                                bf16_emu_->r_vdpbf16ps(acc, wei, vreg_bcast);
                            } else
                                vdpbf16ps(vreg_accum(i_load, i_ur),
                                        vreg_load(i_load), vreg_bcast);
                    } else {
                        if (!jcp.is_cpx) {
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
                            if (jcp.prop_kind == backward_weights)
                                vpbroadcastd(zmm_tmp2,
                                    ptr[rsp + broadcast_space + offset]);
                            else
#endif
                                vpbroadcastd(zmm_tmp2,
                                    bcast_ptr(i_reduce, i_ur, false));
                            auto acc = vreg_accum(i_load, i_ur);
                            auto wei = vreg_load(i_load);
                            bf16_emu_->r_vdpbf16ps(acc, wei, zmm_tmp2);
                        } else
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
                            if (jcp.prop_kind == backward_weights)
                                vdpbf16ps(vreg_accum(i_load, i_ur),
                                        vreg_load(i_load),
                                        zword_b[rsp + broadcast_space + offset]);
                            else
#endif
                                vdpbf16ps(vreg_accum(i_load, i_ur),
                                        vreg_load(i_load),
                                        bcast_ptr(i_reduce, i_ur, true));
                    }
                }
            }
        }
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
        if (jcp.prop_kind == backward_weights)
            mov(reg_reduce_pos_flag, EVEX_compress_addr(rsp, perm_reg_offset));
#endif
    };

    Label reduce_loop;
    Label reduce_loop_tail;

    mov(aux_reg_load_data, reg_load_data);

    mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    init();

    mov(reduce_loop_iter, reg_reduce_loop_work);
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    sub(reduce_loop_iter, jcp.reduce_loop_unroll);
    jle(reduce_loop_tail, T_NEAR);

    L(reduce_loop); {
        fma_block(false);
        add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jg(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    fma_block(true);
#else
    Label reduce_loop_exit;
    cmp(reduce_loop_iter, jcp.reduce_loop_unroll);
    jl(reduce_loop_tail, T_NEAR);

    L(reduce_loop); {
        fma_block(false);
        add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        cmp(reduce_loop_iter, jcp.reduce_loop_unroll);
        jge(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    cmp(reduce_loop_iter, 0);
    jle(reduce_loop_exit, T_NEAR);

    fma_block(true);
    L(reduce_loop_exit);
#endif
    store();
}

void jit_avx512_core_bf16_1x1_conv_kernel::generate()
{
    preamble();

    mov(reg_bcast_data, ptr[param1 + GET_OFF(bcast_data)]);
    mov(reg_load_data, ptr[param1 + GET_OFF(load_data)]);
    mov(reg_output_data, ptr[param1 + GET_OFF(output_data)]);

    sub(rsp, stack_space_needed);

    if (jcp.with_bias)
        mov(reg_bias_data, ptr[param1 + GET_OFF(bias_data)]);

    mov(reg_load_loop_work, ptr[param1 + GET_OFF(load_dim)]);
    mov(reg_bcast_loop_work, ptr[param1 + GET_OFF(bcast_dim)]);
    mov(EVEX_compress_addr(rsp, bcast_loop_work_offt), reg_bcast_loop_work);
    mov(reg_reduce_loop_work, ptr[param1 + GET_OFF(reduce_dim)]);
    if (jcp.prop_kind == backward_weights) {
        mov(reg_reduce_pos_flag, ptr[param1 + GET_OFF(first_last_flag)]);
        mov(reg_output_stride, ptr[param1 + GET_OFF(output_stride)]);
    }

    auto load_loop_body = [=](int load_loop_blk) {
        bcast_loop(load_loop_blk);
        add(reg_load_data, load_loop_blk * jcp.load_loop_load_step);
        switch (jcp.prop_kind) {
        case forward_training:
        case forward_inference:
            add(reg_bias_data,
                load_loop_blk * jcp.load_block * sizeof(float) /* Fix me !!! */);
            add(reg_output_data,
                load_loop_blk * jcp.bcast_dim * jcp.load_block *
                    jcp.typesize_out);
            break;
        case backward_data:
            add(reg_output_data,
                load_loop_blk * jcp.bcast_dim * jcp.load_block *
                    jcp.typesize_out);
            break;
        case backward_weights:
            for (int i_load = 0; i_load < load_loop_blk; i_load++)
                add(reg_output_data, reg_output_stride);
            break;
        default:
            assert(!"invalid prop_kind");
        }
        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
    };

    const int simd_w = 16;

    Label load_loop_blk[7];

    int ur_cases_fma_embd_bcast[] = { 2, 4, 5, 8, 14, 32 };
    int ur_cases_fma_expl_bcast[] = { 2, 5, 6, 9, 14, 32 };
    if (jcp.prop_kind == backward_weights)
        for (int i = 1; i < 6; i++)
            ur_cases_fma_expl_bcast[i] /= 2;

    const int size_ur_cases_fma
            = (jcp.ver == ver_avx512_core && jcp.expl_bcast) ?
            sizeof(ur_cases_fma_expl_bcast) :
            sizeof(ur_cases_fma_embd_bcast);
    const int *ur_cases_fma = (jcp.ver == ver_avx512_core && jcp.expl_bcast) ?
            ur_cases_fma_expl_bcast :
            ur_cases_fma_embd_bcast;
    const int *ur_cases = ur_cases_fma;
    const int num_ur_cases = (size_ur_cases_fma) / sizeof(*ur_cases);

    for (int ur_idx = num_ur_cases - 1; ur_idx > 0; ur_idx--) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.ur <= ur_cases[ur_idx]) {
            cmp(reg_load_loop_work, simd_w * (label_idx + 1));
            jle(load_loop_blk[label_idx], T_NEAR);
        }
    }

    for (int ur_idx = 0; ur_idx < num_ur_cases; ur_idx++) {
        if (jcp.ur <= ur_cases[ur_idx]) {
            int label_idx = num_ur_cases - ur_idx - 1;
            L(load_loop_blk[label_idx]);
            {
                if (label_idx == 0) {
                    cmp(reg_load_loop_work, 0);
                    je(load_loop_blk[num_ur_cases], T_NEAR);
                }
                load_loop_body(label_idx + 1);
                if (label_idx - 1 > 0) {
                    cmp(reg_load_loop_work, 2 * label_idx * simd_w);
                    je(load_loop_blk[label_idx - 1], T_NEAR);
                }
                cmp(reg_load_loop_work, (label_idx + 1) * simd_w);
                jge(load_loop_blk[label_idx]);
            }
            for (int idx = label_idx - 1; idx > 0; --idx) {
                cmp(reg_load_loop_work, simd_w * (idx + 1));
                je(load_loop_blk[idx], T_NEAR);
            }
            if (ur_idx < num_ur_cases - 2) {
                cmp(reg_load_loop_work, simd_w);
                jle(load_loop_blk[0], T_NEAR);
            }
        }
    }
    L(load_loop_blk[num_ur_cases]);

    add(rsp, stack_space_needed);

    postamble();

    if (jcp.with_eltwise)
        eltwise_injector_->prepare_table();

    if (jcp.prop_kind == backward_weights) {
        const uint16_t dst_prm_array[32] =
            {0,16,  1,17,  2,18,  3,19,  4,20,  5,21,  6,22,  7,23,  8,24,
            9,25,  10,26,  11,27,  12,28,  13,29,  14,30,  15,31 };

        align(64);
        L(dst_prm_table);
        for (int i = 0; i < 32; ++i)
            dw(dst_prm_array[i]);
    }
}

bool jit_avx512_core_bf16_1x1_conv_kernel::post_ops_ok(
        jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

    switch (p.len_) {
    case 0: return true; // no post_ops
    case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
    case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
    default: return false;
    }

    return false;
}

status_t jit_avx512_core_bf16_1x1_conv_kernel::init_conf(
        jit_1x1_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const primitive_attr_t &attr,
        int nthreads, bool reduce_src)
{
    jcp.is_cpx = mayiuse(avx512_core_bf16);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);
    const int ndims = src_d.ndims();

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = src_d.dims()[1] / jcp.ngroups;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1
        && (src_d.data_type() == data_type::f32
                || src_d.data_type() == data_type::bf16);
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][0];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[0];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.src_fmt = src_d.format();
    jcp.with_bias = pick_by_prop_kind(jcp.prop_kind, cd.bias_desc.format,
            memory_format::undef, cd.diff_bias_desc.format)
        != memory_format::undef;

    jcp.os = jcp.oh * jcp.ow;
    jcp.is = jcp.ih * jcp.iw;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;
        if (dst_d.data_type() == data_type::s32) return status::unimplemented;
    }

    bool args_ok = true
        && jcp.ngroups == 1
        && everyone_is(pick(ndims - 3, nCw16c, nChw16c), src_d.format(),
            dst_d.format())
        && one_of(cd.bias_desc.format, memory_format::undef, any, x);
    if (!args_ok) return status::unimplemented;

    args_ok = true
        && jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0
        && jcp.t_pad == 0 && jcp.l_pad == 0
        && jcp.stride_w == 1 && jcp.stride_h == 1
        && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) return status::unimplemented;

    jcp.ic_block = jcp.oc_block = simd_w;

    const int is_bwd_d = jcp.prop_kind == backward_data;
    /*TODO: Introduce formats for 1D spatial */
    memory_format_t weights_format = memory_format::undef;
    if (one_of(jcp.prop_kind, forward_training, forward_inference,
                   backward_data)) {
        weights_format = with_groups
            ? pick(2 * ndims - 6 + is_bwd_d, gOIw8i16o2i, gIOw8o16i2o,
                gOIhw8i16o2i, gIOhw8o16i2o)
            : pick(2 * ndims - 6 + is_bwd_d, OIw8i16o2i, IOw8o16i2o,
                OIhw8i16o2i, IOhw8o16i2o);
    } else if (jcp.prop_kind == backward_weights) {
        weights_format = with_groups
            ? pick(ndims - 3, gOIw16i16o, gOIhw16i16o)
            : pick(ndims - 3, OIw16i16o, OIhw16i16o);
    }

    if (weights_d.format() != weights_format)
        return status::unimplemented;

    jcp.typesize_acc = sizeof(prec_traits<data_type::f32>::type);
    jcp.typesize_in = sizeof(prec_traits<data_type::bf16>::type);
    if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
        jcp.typesize_out = dst_d.data_type() == data_type::bf16
                        ? sizeof(prec_traits<data_type::bf16>::type)
                        : sizeof(prec_traits<data_type::f32>::type);
        jcp.dst_dt = dst_d.data_type();
    } else if (jcp.prop_kind == backward_data) {
        jcp.typesize_out = src_d.data_type() == data_type::bf16
                        ? sizeof(prec_traits<data_type::bf16>::type)
                        : sizeof(prec_traits<data_type::f32>::type);
        jcp.dst_dt = src_d.data_type();
    } else if (jcp.prop_kind == backward_weights) {
        jcp.typesize_out = sizeof(prec_traits<data_type::f32>::type);
        jcp.dst_dt = weights_d.data_type();
    }

    jcp.ver = ver_avx512_core;
    /* once all the formats are set, check the padding consistency */
    args_ok = true
        && jcp.ic <= src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= dst_d.blocking_desc().padding_dims[1]
        && jcp.ic <= weights_d.blocking_desc().padding_dims[with_groups + 1]
        && jcp.oc <= weights_d.blocking_desc().padding_dims[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    const int SMALL_SPATIAL = 10;
    const int BIG_SPATIAL = 28;
    const int BIG_REDUCE_DIM = 1024;
    const int BIG_LOAD_DIM = 256;

    int load_blocking{ 0 };
    int load_blocking_max{ 0 };
    int bcast_blocking{ 0 };
    int bcast_blocking_max{ 0 };
    int reduce_blocking{ 0 };
    int reduce_blocking_max{ 0 };

    jcp.load_grp_count = 1;

    const int L1_capacity = get_cache_size(1, true) / sizeof(mkldnn_bfloat16_t);
    const int L2_size = get_cache_size(2, true) / sizeof(mkldnn_bfloat16_t);
    const int L2_capacity = (L2_size * 3) / 4;

    if (one_of(jcp.prop_kind, forward_training, forward_inference,
                backward_data)) {
        if (one_of(jcp.prop_kind, forward_inference, forward_training)) {
            jcp.reduce_dim = jcp.ic;
            jcp.reduce_block = jcp.ic_block;

            jcp.load_dim = jcp.oc;
            jcp.load_block = jcp.oc_block;

            jcp.bcast_dim = jcp.is;
        } else {
            jcp.reduce_dim = jcp.oc;
            jcp.reduce_block = jcp.oc_block;

            jcp.load_dim = jcp.ic;
            jcp.load_block = jcp.ic_block;

            jcp.bcast_dim = jcp.os;
        }
        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
                = jcp.reduce_loop_unroll * jcp.bcast_dim * jcp.typesize_in;
        jcp.reduce_loop_load_step
                = jcp.reduce_loop_unroll * jcp.load_block * jcp.typesize_in;
        jcp.load_loop_load_step
                = jcp.reduce_dim * jcp.load_block * jcp.typesize_in;

        // adjusting registry blocking
        int max_regs, min_regs, size_treshold, ur_step;
        const int spatial
                = (one_of(jcp.prop_kind, forward_training, forward_inference)) ?
                jcp.oh :
                jcp.ih;
        if (jcp.ver == ver_avx512_core && (8 * jcp.mb) / nthreads >= 1)
        {
            max_regs = 9;
            min_regs = 6;
            size_treshold = 14;
            ur_step = 1;
            jcp.expl_bcast = true;

            if (jcp.load_dim > 128 && jcp.load_dim < BIG_LOAD_DIM
                    && spatial > SMALL_SPATIAL && spatial < BIG_SPATIAL) {
                max_regs = 6;
                min_regs = (jcp.is_cpx) ? 5 : 4;
            }
        }
        else {
            max_regs = 30;
            min_regs = 9;
            size_treshold = 14;
            ur_step = 1;
            jcp.expl_bcast = false;
        }
        jcp.ur = 1;
        if (!jcp.is_cpx) {
            int adj_max_regs = max_regs / 3;
            max_regs =  (adj_max_regs < min_regs) ? min_regs : adj_max_regs;
        }
        for (int ur_w = max_regs; ur_w >= min_regs; ur_w -= ur_step) {
            if ((spatial >= size_treshold && spatial % ur_w == 0)
                    || (spatial < size_treshold && jcp.os % ur_w == 0)) {
                jcp.ur = ur_w;
                break;
            }
        }
        if (jcp.ur == 1) {
            jcp.ur = nstl::min(max_regs, jcp.os);
            int os_tail = jcp.os % max_regs;
            for (int i = max_regs; i >= min_regs; i -= ur_step) {
                int i_tail = jcp.os % i;
                if (i_tail > os_tail || i_tail == 0) {
                    jcp.ur = i;
                    os_tail = i_tail;
                    if (i_tail == 0)
                        break;
                }
            }
        }

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
                = jcp.reduce_loop_unroll * jcp.bcast_dim * jcp.typesize_in;

        jcp.bcast_block = jcp.ur;

        jcp.bcast_loop_output_step = jcp.ur * jcp.load_block * jcp.typesize_out;
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur * jcp.reduce_block * jcp.typesize_in;
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_iter_step = jcp.load_block;

        if (jcp.prop_kind == backward_data)
            jcp.loop_order = loop_lbr;
        else
            jcp.loop_order = reduce_src ? loop_blr : loop_lbr;

        int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
        int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);
        int nb_load = div_up(jcp.load_dim, jcp.load_block);

        if (jcp.ver == ver_avx512_core && jcp.expl_bcast) {
            if (jcp.load_dim <= BIG_LOAD_DIM && spatial > SMALL_SPATIAL
                    && spatial < BIG_SPATIAL)
                reduce_blocking = nstl::min(jcp.reduce_dim, 80);
            else if (spatial > SMALL_SPATIAL)
                reduce_blocking = nstl::min(jcp.reduce_dim, 512);
            else
                reduce_blocking = nstl::min(jcp.reduce_dim, 256);
        } else {
            reduce_blocking = nb_reduce;
            if (spatial <= SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
                reduce_blocking = 16;
            else if (spatial > SMALL_SPATIAL
                    && jcp.reduce_dim >= BIG_REDUCE_DIM)
                reduce_blocking = 8;
            reduce_blocking = best_divider(nb_reduce, 1, reduce_blocking, true);
            reduce_blocking *= jcp.reduce_block;
        }

        // Check input data cache aliasing.
        // For other ISA constants may be updated.
        // 64 * 1024 is chosen due to 1MB L2 16-way cache.
        // 7 is empirical value. It is about half of 16.
        // So we leave about half of the set for other data - weights, dst
        int way_size = (64 * 1024) / jcp.typesize_in;
        int max_hits = 7;
        if (jcp.bcast_dim * reduce_blocking > way_size * max_hits) {
            int nrb = reduce_blocking / simd_w;
            int sp = jcp.bcast_dim;
            int wl = way_size / simd_w;
            for (int start_off = 0; start_off < jcp.ur; start_off++) {
                for (int off = start_off, hits = 0; off < sp * nrb; off += wl) {
                    if (off % sp >= jcp.ur || ++hits < max_hits)
                        continue;
                    int max_r_blocking = simd_w * nstl::max(1, (off + wl) / sp);
                    reduce_blocking
                            = nstl::min(reduce_blocking, max_r_blocking);
                    break;
                }
            }
        }

        if (reduce_blocking < jcp.reduce_dim) {
            if (jcp.prop_kind == backward_data)
                jcp.loop_order = reduce_src ? loop_lbr : loop_rlb;
            else
                jcp.loop_order = reduce_src ? loop_rbl : loop_rlb;
        }
        load_blocking = jcp.load_dim;

        int load_size = jcp.load_dim * jcp.reduce_dim;
        int bcast_size = jcp.mb * jcp.ngroups * jcp.bcast_dim * jcp.reduce_dim;

        if (jcp.ver == ver_avx512_core && nthreads <= 28 && jcp.mb < nthreads
                && nb_load * nb_bcast > nthreads) {
            // Some heuristic here
            float calc_koef = 0.01, best_cost = FLT_MAX;
            int n_lgc = nthreads;
            float ratio = (float)load_size / (float)bcast_size;
            int best_lgc = ratio > 1 ? n_lgc : 1;
            auto calc_job_cost = [&](int lb, int tg, float mem_k) {
                int bb_size = jcp.mb * div_up(nb_bcast, tg);
                float calc_size = (float)(bb_size * jcp.ur)
                        * (lb * jcp.load_block) * jcp.reduce_dim;
                float mem_size = (float)(bb_size * jcp.ur + lb * jcp.load_block)
                        * jcp.reduce_dim;
                return calc_koef * calc_size + mem_k * mem_size;
            };
            for (int lgc, ilgc = 0; ilgc < n_lgc; ilgc++) {
                lgc = ratio > 1 ? n_lgc - ilgc : ilgc + 1;
                int min_lb = nb_load / lgc;
                int max_lb = div_up(nb_load, lgc);
                int min_tg = nthreads / lgc;
                int max_tg = div_up(nthreads, lgc);
                // Some heuristic here
                float mem_koef = (max_tg == 1) ? 1.f : 1.3f;
                float job_cost = 0.;
                if (nthreads % lgc < nb_load % lgc) {
                    job_cost = calc_job_cost(max_lb, min_tg, mem_koef);
                } else {
                    auto job_cost1 = calc_job_cost(max_lb, max_tg, mem_koef);
                    auto job_cost2 = calc_job_cost(min_lb, min_tg, mem_koef);
                    job_cost = nstl::max(job_cost1, job_cost2);
                }

                if (job_cost < best_cost) {
                    best_lgc = lgc;
                    best_cost = job_cost;
                }
            }
            jcp.load_grp_count = best_lgc;
            load_blocking = div_up(nb_load, jcp.load_grp_count) * jcp.load_block;
        } else {
            jcp.load_grp_count = div_up(nthreads, jcp.mb * jcp.ngroups * nb_bcast);
            jcp.load_grp_count = best_divider(
                nthreads, jcp.load_grp_count, 2 * jcp.load_grp_count, false);
        }

        if (jcp.ver == ver_avx512_core && jcp.expl_bcast && jcp.bcast_dim <= 64
                && load_size >= L2_size) {
            jcp.load_grp_count = nstl::max(jcp.load_grp_count, 4);
        } else if (jcp.bcast_dim <= 49 && jcp.mb <= nthreads
                && jcp.load_dim > 512 && jcp.load_dim / jcp.reduce_dim >= 4) {
            jcp.load_grp_count = nstl::max(jcp.load_grp_count, 2);
            load_blocking = jcp.load_block;
        }

        bcast_blocking = div_up(jcp.mb * jcp.ngroups * nb_bcast,
                                 div_up(nthreads, jcp.load_grp_count))
                * jcp.bcast_block;
        bcast_blocking = nstl::min(jcp.bcast_dim, bcast_blocking);
        bcast_blocking = rnd_up(bcast_blocking, jcp.bcast_block);

        int space_for_bcast
                = (L2_capacity - /* kernel_size - */
                    2 * jcp.load_block * reduce_blocking
                        - jcp.ur * reduce_blocking - 3 * 1024);
        if (jcp.reduce_dim * jcp.bcast_dim > L2_capacity)
            space_for_bcast /= 2;

        int bcast_in_cache
                = nstl::max(jcp.bcast_block, space_for_bcast / reduce_blocking);
        bcast_blocking = nstl::min(
                bcast_blocking, rnd_dn(bcast_in_cache, jcp.bcast_block));

        load_blocking_max = load_blocking;
        bcast_blocking_max = bcast_blocking * 3 / 2;
        reduce_blocking_max = reduce_blocking;
    } else if (jcp.prop_kind == backward_weights) {
        jcp.use_vmovntps = false;
        jcp.reduce_dim = jcp.is;

#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
        jcp.reduce_block = best_divider(jcp.reduce_dim, 8, 16, true, 2);
        if (jcp.reduce_dim % jcp.reduce_block != 0)
            jcp.reduce_block = best_divider(jcp.iw, 4, jcp.iw, false, 2);
        if (jcp.reduce_block > 256) {
            jcp.reduce_block = 1;
        }
#else
        jcp.reduce_block = best_divider(jcp.reduce_dim, 8, 16, true, 2);
        if (jcp.reduce_dim % jcp.reduce_block != 0)
            jcp.reduce_block = best_divider(jcp.iw, 4, jcp.iw, false, 2);
        if (jcp.reduce_block > 256) {
            jcp.reduce_block = 1;
        }
       jcp.reduce_block = rnd_up(jcp.reduce_block, 2);
/*
        // Just another variant of reduce_block computation
        int rdim = rnd_up(jcp.reduce_dim, 2);
        jcp.reduce_block = best_divider(rdim, 8, 16, true, 2);
        if (rdim % jcp.reduce_block != 0) {
            iw = rnd_up(jcp.iw, 2);
            jcp.reduce_block = best_divider(iw, 4, iw, false, 2);
        }
        if (jcp.reduce_block > 256) {
            jcp.reduce_block = 2;
        }
*/
#endif

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.ic;
        jcp.bcast_block = jcp.ic_block;

        if (jcp.reduce_block <= 19) {
            // if reduce_block is big then generated JIT code may be big
            // for small values of ur because reduce_loop_unroll = reduce_block
            jcp.ur = jcp.bcast_block / 2; // Possible P-Tuning
            jcp.expl_bcast = true;
        } else {
            jcp.ur = jcp.bcast_block;
            jcp.expl_bcast = false;
        }

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
            = jcp.reduce_loop_unroll * jcp.ic_block * jcp.typesize_in;
        jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.oc_block * jcp.typesize_in;

        jcp.bcast_loop_output_step =
                                jcp.oc_block * jcp.ic_block * jcp.typesize_out;
        jcp.bcast_loop_output_substep =
            jcp.oc_block * jcp.ur * jcp.typesize_out;
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
        jcp.bcast_loop_bcast_step =
                jcp.ic_block * jcp.reduce_dim * jcp.typesize_in;
        jcp.bcast_loop_bcast_substep = jcp.ur * jcp.typesize_in;
        jcp.load_loop_load_step = jcp.oc_block * jcp.os * jcp.typesize_in;
#else
        jcp.bcast_loop_bcast_step =
                jcp.ic_block * rnd_up(jcp.reduce_dim, 2) * jcp.typesize_in;
        jcp.bcast_loop_bcast_substep = 2 * jcp.ur * jcp.typesize_in;
        jcp.load_loop_load_step =
                jcp.oc_block * rnd_up(jcp.reduce_dim, 2) * jcp.typesize_in;
#endif
        jcp.load_loop_iter_step = jcp.oc_block;

        /* --- */
        balance(jcp, nthreads);

        load_blocking = div_up(jcp.load_dim, jcp.load_block);
        load_blocking = best_divider(load_blocking, 16, load_blocking, false);
        load_blocking *= jcp.load_block;

        load_blocking_max = load_blocking;
        assert(jcp.load_dim % load_blocking == 0);

        int max_bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        int min_bcast_blocking = 5;

        bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        bcast_blocking = best_divider(
                bcast_blocking, min_bcast_blocking, max_bcast_blocking, false);
        bcast_blocking *= jcp.bcast_block;
        bcast_blocking_max = bcast_blocking;
        assert(jcp.bcast_dim % bcast_blocking == 0);

        // for reduction balance
        int max_reduce_blocking
                = nstl::min(L1_capacity / jcp.ur, jcp.reduce_dim);
        int min_reduce_blocking = nstl::min(
                L1_capacity / jcp.ur, nstl::max(jcp.iw, jcp.ih));
        reduce_blocking = best_divider(jcp.reduce_dim, min_reduce_blocking,
                max_reduce_blocking, true);
        reduce_blocking
                = nstl::max(rnd_dn(reduce_blocking, jcp.reduce_block),
                        jcp.reduce_block);

        reduce_blocking_max = rnd_dn(reduce_blocking * 3 / 2, jcp.reduce_block);
    } else
        return status::unimplemented;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);
    assert(reduce_blocking_max);
    assert(load_blocking % jcp.load_block == 0);
    assert(load_blocking_max % jcp.load_block == 0);
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    assert(reduce_blocking % jcp.reduce_block == 0);
    assert(reduce_blocking_max % jcp.reduce_block == 0);
#endif

    assert(jcp.bcast_block % jcp.ur == 0);
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    assert(jcp.reduce_dim % jcp.reduce_block == 0);
#endif

    jcp.ur_tail = jcp.bcast_dim % jcp.ur;

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = load_blocking / jcp.load_block;
    jcp.nb_load_blocking_max = load_blocking_max / jcp.load_block;
    jcp.nb_reduce_blocking = reduce_blocking / jcp.reduce_block;
    jcp.nb_reduce_blocking_max = reduce_blocking_max / jcp.reduce_block;

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    return status::success;
}

void jit_avx512_core_bf16_1x1_conv_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp) {
    using namespace mkldnn::impl::memory_tracking::names;

    if (jcp.prop_kind != backward_data && jcp.with_bias
            && jcp.oc != jcp.oc_without_padding)
        scratchpad.book(key_conv_padded_bias, jcp.typesize_acc * jcp.oc);

    if (jcp.prop_kind == backward_weights) {
        const size_t wei_size = (size_t)jcp.ngroups * jcp.oc * jcp.ic;
        const int n_buffers = jcp.dst_dt == data_type::bf16
                              ? jcp.nthr_mb : jcp.nthr_mb - 1;
        scratchpad.book(key_conv_wei_reduction,
                jcp.typesize_acc * wei_size * n_buffers);

#ifndef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
        const size_t dst_diff_tr_size_per_thr =
            (size_t)rnd_up(jcp.reduce_dim, 2) * jcp.oc_block
            * jcp.nb_load_blocking_max * jcp.typesize_in;
        scratchpad.book(key_conv_tr_diff_dst,
                 jcp.nthr * dst_diff_tr_size_per_thr);
        const size_t src_tr_size_per_thr =
            (size_t)rnd_up(jcp.reduce_dim, 2) * jcp.ic_block
            * jcp.nb_bcast_blocking_max * jcp.typesize_in;
        scratchpad.book(key_conv_tr_src, jcp.nthr * src_tr_size_per_thr);
#endif
        if (jcp.with_bias) {
            const size_t d_dst_f32_size = (size_t)jcp.typesize_acc
                * jcp.oh * jcp.ow * jcp.oc_block;
            scratchpad.book(key_conv_dst_bf16_convert_wsp, jcp.nthr * d_dst_f32_size);
        }
    }
}

void jit_avx512_core_bf16_1x1_conv_kernel::balance(jit_1x1_conv_conf_t &jcp,
        int nthreads)
{
    // initialize jcp reduction threading properties
    jcp.nthr = jcp.nthr_mb = jcp.nthr_g = jcp.nthr_oc_b = jcp.nthr_ic_b = 1;
    if (nthreads < jcp.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        return;
    }
    const int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    const int nb_load = div_up(jcp.load_dim, jcp.load_block);
    const int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    jcp.nthr_g = jcp.ngroups;
    const int nthr = nthreads / jcp.nthr_g;

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level
        * optimizer tries to minimize memory consumption. few notes: (n1)
        * unclear why, but that essentially helps first convolution...
        *  (n2) assuming the reduction over minibatch is always there:
        *    - instead of 8 it should be 5 here (write ~= 2 read):
        *      kernel: temporal workspace 1 write
        *      reduction: 1 read from workspace and 1 write to the diff_wei
        *    - but experiments showed 8 works better than 5 or 6... */
        int bcast_koeff = 1;
        int load_koeff = 1;
        int output_koeff = 12;

        return 0
            + (size_t)bcast_koeff * div_up(jcp.mb * nb_reduce, nthr_mb)
            * div_up(jcp.ngroups, jcp.nthr_g)
            * div_up(nb_bcast, nthr_ic_b) * jcp.ic_block * jcp.reduce_block
            / jcp.stride_h / jcp.stride_w /* (n1) */
            + (size_t)load_koeff * div_up(jcp.mb * nb_reduce, nthr_mb)
            * div_up(jcp.ngroups, jcp.nthr_g)
            * div_up(nb_load, nthr_oc_b) * jcp.oc_block * jcp.reduce_block
            + (size_t)output_koeff /* (n2) */
            * div_up(jcp.ngroups, jcp.nthr_g) * div_up(nb_load, nthr_oc_b)
            * div_up(nb_bcast, nthr_ic_b) * jcp.ic_block
            * jcp.oc_block;
    };

    int nthr_mb = 1, nthr_oc_b = 1, nthr_ic_b = 1;
    auto best_mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);

    /* step 1: find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = nstl::min(nthr, jcp.mb * nb_reduce);
    for (nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, nb_load);
        for (nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, nb_bcast);
            auto mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                jcp.nthr_mb = nthr_mb;
                jcp.nthr_oc_b = nthr_oc_b;
                jcp.nthr_ic_b = nthr_ic_b;
            }
        }

        if (!mkldnn_thr_syncable()) { assert(nthr_mb == 1); break; }
    }
    if (jcp.nthr_mb > nthreads / 2 && jcp.nthr_mb < nthreads)
        jcp.nthr_mb = nstl::min(jcp.mb, nthreads);

    jcp.nthr = jcp.nthr_mb * jcp.nthr_g * jcp.nthr_oc_b * jcp.nthr_ic_b;
    assert(jcp.nthr <= nthreads);
}

}
}
}
