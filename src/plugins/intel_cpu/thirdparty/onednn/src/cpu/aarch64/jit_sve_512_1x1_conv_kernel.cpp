/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2021 FUJITSU LIMITED
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

#include <assert.h>
#include <float.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_barrier.hpp"
#include "cpu/platform.hpp"

#include "cpu/aarch64/jit_sve_512_1x1_conv_kernel.hpp"

#include "cpu/aarch64/jit_uni_1x1_conv_utils.hpp"

#define GET_OFF(field) \
    static_cast<int32_t>(offsetof(jit_1x1_conv_call_s, field))

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::utils;

void jit_sve_512_1x1_conv_kernel::bcast_loop(int load_loop_blk) {

    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_output_data, reg_output_data);
    mov(reg_bcast_loop_iter, reg_bcast_loop_work);

    Label bcast_loop;
    Label bcast_loop_tail;
    Label large_tail;

    cmp_imm(reg_bcast_loop_iter, jcp.bcast_block, reg_tmp_imm);
    b(LT, bcast_loop_tail);

    L(bcast_loop);
    {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            if (i + 1 == num_substeps) L(large_tail);
            reduce_loop(load_loop_blk, jcp.ur, i, false);
            if (i < num_substeps - 1) {
                add_imm(aux1_reg_bcast_data, aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_substep, reg_tmp_imm);
                add_imm(aux_reg_output_data, aux_reg_output_data,
                        jcp.bcast_loop_output_substep, reg_tmp_imm);
            } else {
                add_imm(aux1_reg_bcast_data, aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_step
                                - (num_substeps - 1)
                                        * jcp.bcast_loop_bcast_substep,
                        reg_tmp_imm);
                add_imm(aux_reg_output_data, aux_reg_output_data,
                        jcp.bcast_loop_output_step
                                - (num_substeps - 1)
                                        * jcp.bcast_loop_output_substep,
                        reg_tmp_imm);
            }
            subs_imm(reg_bcast_loop_iter, reg_bcast_loop_iter, jcp.ur,
                    reg_tmp_imm);
        }
        cmp_imm(reg_bcast_loop_iter, jcp.bcast_block, reg_tmp_imm);
        b(GE, bcast_loop);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        if (jcp.ur_tail >= jcp.ur) {
            cmp_imm(reg_bcast_loop_iter, jcp.ur, reg_tmp_imm);
            b(GE, large_tail);
        }
        if (jcp.ur_tail % jcp.ur) {
            cmp(reg_bcast_loop_iter, 0);
            b(LE, bcast_loop_tail_out);
            reduce_loop(load_loop_blk, jcp.ur_tail % jcp.ur, 0, true);
            L(bcast_loop_tail_out);
        }
    }
}

void jit_sve_512_1x1_conv_kernel::reduce_loop(
        int load_loop_blk, int ur, int substep, bool wraparound) {

    const bool out_layout_nxc = is_out_layout_nxc(jcp);
    const bool load_layout_nxc = is_load_layout_nxc(jcp);
    const bool bcast_layout_nxc = is_bcast_layout_nxc(jcp);
    const int reduce_dim_tail = jcp.reduce_dim % jcp.reduce_block;

    auto vreg_sum = [=]() { return ZReg(31); };
    auto vreg_sum_s = [=]() { return ZRegS(31); };

    auto vreg_load = [=](int i_load, int i_fma) {
        return ZReg(utils::rnd_up(ur * load_loop_blk, jcp.fma_step)
                + jcp.fma_step * i_load + i_fma);
    };
    auto vreg_load_s = [=](int i_load, int i_fma) {
        return ZRegS(utils::rnd_up(ur * load_loop_blk, jcp.fma_step)
                + jcp.fma_step * i_load + i_fma);
    };

    auto vreg_accum = [=](int i_load, int i_ur) {
        return ZReg(i_ur * load_loop_blk + i_load);
    };
    auto vreg_accum_s = [=](int i_load, int i_ur) {
        return ZRegS(i_ur * load_loop_blk + i_load);
    };

    auto bias_load = [=](int i_load, int i_ur) {
        int ofs = jcp.typesize_out * jcp.oc_block * i_load;
        if (ldr_imm_check(ofs)) {
            ldr(vreg_accum(i_load, i_ur),
                    ptr(reg_bias_data, static_cast<int32_t>(VL64_OFS(ofs))));
        } else {
            add_imm(reg_tmp_ofs, reg_bias_data, ofs, reg_tmp_imm);
            ldr(vreg_accum(i_load, i_ur), ptr(reg_tmp_ofs));
        }
    };

    auto bcast_load = [=](int i_reduce, int i_ur, int prev_ofs, int bcast_idx) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        int ofs;
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                    backward_data)) {
            assert(jcp.reduce_loop_unroll == jcp.reduce_block);
            const int reduce_mul = bcast_layout_nxc ? jcp.reduce_dim
                                                    : jcp.reduce_loop_unroll;
            ofs = (i_reduce == jcp.reduce_loop_unroll)
                    ? (jcp.bcast_dim + i_ur) * reduce_mul
                    : i_ur * reduce_mul + i_reduce;
        } else {
            int rmul = bcast_layout_nxc ? jcp.ic : jcp.ic_block;
            ofs = i_reduce * rmul + i_ur;
        }

        ofs = jcp.typesize_in * ofs;
        int tmp_ofs = ofs;
        if (ld1rw_imm_check(ofs)) {
            ld1rw(ZRegS(bcast_idx), reg_p_all_ones,
                    ptr(aux_reg_bcast_data, static_cast<int32_t>(ofs)));
        } else {
            if ((prev_ofs != -1) && ld1rw_imm_check(ofs - prev_ofs)) {
                ld1rw(ZRegS(bcast_idx), reg_p_all_ones,
                        ptr(reg_prev_bcast_addr,
                                static_cast<int32_t>((ofs - prev_ofs))));
            } else {
                if ((prev_ofs != -1) && ((ofs - prev_ofs) >= 0)) {
                    ofs = ofs - prev_ofs;
                    add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs,
                            reg_tmp_imm);
                } else {
                    add_imm(reg_prev_bcast_addr, aux_reg_bcast_data, ofs,
                            reg_tmp_imm);
                }
                prev_ofs = tmp_ofs;

                ld1rw(ZRegS(bcast_idx), reg_p_all_ones,
                        ptr(reg_prev_bcast_addr));
            }
        }
        return prev_ofs;
    };

    auto load_load = [=](int i_reduce, int i_load, int i_fma) {
        int ofs;
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;
        int lmul = jcp.load_block
                * (load_layout_nxc ? 1
                                   : utils::rnd_up(
                                           jcp.reduce_dim, jcp.reduce_block));
        int rmul = load_layout_nxc ? jcp.load_dim : jcp.load_block;
        ofs = i_load * lmul + u0 * rmul;
        ofs = u1 * jcp.reduce_loop_load_step + jcp.typesize_in * ofs;

        if (ldr_imm_check(ofs)) {
            ofs = VL64_OFS(ofs);
            ldr(vreg_load(i_load, i_fma),
                    ptr(aux_reg_load_data, static_cast<int32_t>(ofs)));
        } else {
            add_imm(reg_tmp_ofs, aux_reg_load_data, ofs, reg_tmp_imm);
            ldr(vreg_load(i_load, i_fma), ptr(reg_tmp_ofs));
        }
    };

    auto out_load = [=](int i_load, int i_ur, int prev_ofs) {
        int ofs, ofs_tmp;
        int bwd_iload
                = (i_load != 0) && one_of(jcp.prop_kind, backward_weights);
        auto r = (bwd_iload) ? reg_tmp_ofs : aux_reg_output_data;

        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                    backward_data)) {
            int i_load_shift = out_layout_nxc
                    ? jcp.load_block
                    : (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim)
                            * jcp.load_block;
            int i_ur_shift = out_layout_nxc ? jcp.load_dim : jcp.load_block;
            ofs = (i_load * i_load_shift + i_ur * i_ur_shift)
                    * jcp.typesize_out;
        } else {
            ofs = jcp.typesize_out * jcp.load_block * i_ur;
        }

        ofs_tmp = ofs;

        if (bwd_iload) mov(r, i_load);
        if (ldr_imm_check(ofs)) {
            if (bwd_iload) madd(r, r, reg_output_stride, aux_reg_output_data);
            ldr(vreg_sum(), ptr(r, static_cast<int32_t>(VL64_OFS(ofs))));
        } else {
            if ((prev_ofs != -1) && ((ofs - prev_ofs) > 0)
                    && (VL64_OFS(ofs - prev_ofs) <= LDRMAX)) {
                if (bwd_iload)
                    madd(r, r, reg_output_stride, reg_prev_out_addr);
                else
                    r = reg_prev_out_addr;
                ldr(vreg_sum(),
                        ptr(r, static_cast<int32_t>(VL64_OFS(ofs - prev_ofs))));
            } else {
                if ((prev_ofs != -1) && ((ofs - prev_ofs) > 0)) {
                    ofs = ofs - prev_ofs;
                    add_imm(reg_prev_out_addr, reg_prev_out_addr, ofs,
                            reg_tmp_imm);
                } else {
                    add_imm(reg_prev_out_addr, aux_reg_output_data, ofs,
                            reg_tmp_imm);
                }
                if (bwd_iload)
                    madd(r, r, reg_output_stride, reg_prev_out_addr);
                else
                    r = reg_prev_out_addr;
                ldr(vreg_sum(), ptr(r));

                prev_ofs = ofs_tmp;
            }
        }
        return prev_ofs;
    };

    auto out_str = [=](int i_load, int i_ur, int prev_ofs) {
        int ofs, ofs_tmp;
        int bwd_iload
                = (i_load != 0) && one_of(jcp.prop_kind, backward_weights);
        auto r = (bwd_iload) ? reg_tmp_ofs : aux_reg_output_data;
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                    backward_data)) {
            ofs = (i_load * jcp.bcast_dim + i_ur) * jcp.load_block
                    * jcp.typesize_out;
        } else {
            ofs = jcp.typesize_out * jcp.load_block * i_ur;
        }
        ofs_tmp = ofs;

        if (bwd_iload) mov(r, i_load);
        if (str_imm_check(ofs)) {
            if (bwd_iload) madd(r, r, reg_output_stride, aux_reg_output_data);
            str(vreg_accum(i_load, i_ur),
                    ptr(r, static_cast<int32_t>(VL64_OFS(ofs))));
        } else {
            if ((prev_ofs != -1) && str_imm_check(ofs - prev_ofs)) {
                if (bwd_iload)
                    madd(r, r, reg_output_stride, reg_prev_out_addr);
                else
                    r = reg_prev_out_addr;
                str(vreg_accum(i_load, i_ur),
                        ptr(r, static_cast<int32_t>(VL64_OFS(ofs - prev_ofs))));
            } else {
                if ((prev_ofs != -1) && ((ofs - prev_ofs) > 0)) {
                    ofs = ofs - prev_ofs;
                    add_imm(reg_prev_out_addr, reg_prev_out_addr, ofs,
                            reg_tmp_imm);
                } else {
                    add_imm(reg_prev_out_addr, aux_reg_output_data, ofs,
                            reg_tmp_imm);
                }
                if (bwd_iload)
                    madd(r, r, reg_output_stride, reg_prev_out_addr);
                else
                    r = reg_prev_out_addr;
                str(vreg_accum(i_load, i_ur), ptr(r));

                prev_ofs = ofs_tmp;
            }
        }
        return prev_ofs;
    };

    auto prefetch_output = [=](int i_load, int i_ur) {
        int ofs;
        int bwd_iload
                = (i_load != 0) && one_of(jcp.prop_kind, backward_weights);
        auto r = (bwd_iload) ? reg_tmp_ofs : aux_reg_output_data;
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                    backward_data)) {
            ofs = (i_load * jcp.bcast_dim + i_ur) * jcp.load_block
                    * jcp.typesize_out;
        } else {
            ofs = jcp.typesize_out * jcp.load_block * i_ur;
        }
        std::string op = "LD";
        prefetch(op, 2, r, ofs);
    };

    auto init = [=]() {
        Label init_done;
        Label init_zero;

        if (jcp.with_sum) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    prefetch_output(i_load, i_ur);
                }
            }
        }

        if (jcp.with_bias
                && one_of(jcp.prop_kind, forward_training, forward_inference)) {

            tst(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            b(EQ, init_zero);

            for (int i_load = 0; i_load < load_loop_blk; i_load++)
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    bias_load(i_load, i_ur);
                }
            b(init_done);
        }

        L(init_zero);
        /* Zero clear */
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                fmov(vreg_accum_s(i_load, i_ur));
            }
        L(init_done);
    };

    auto store = [=]() {
        Label store_noadd;
        if (!jcp.with_sum) {
            tst(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            b(NE, store_noadd);
        }

        int prev_ofs = -1;
        for (int i_ur = 0; i_ur < ur; ++i_ur)
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                auto r = vreg_accum_s(i_load, i_ur);
                prev_ofs = out_load(i_load, i_ur, prev_ofs);
                fadd(r, r, vreg_sum_s());
            }

        L(store_noadd);
        if (jcp.with_eltwise) {
#ifndef DISABLE_ELTWISE
            Label store_noeltwise;
            tst(reg_reduce_pos_flag, FLAG_REDUCE_LAST);
            b(EQ, store_noeltwise);
            eltwise_injector_->compute_vector_range(0, ur * load_loop_blk);
            L(store_noeltwise);
#else
            assert(!"fused eltwise error!");
#endif
        }

        prev_ofs = -1;
        for (int i_ur = 0; i_ur < ur; ++i_ur) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                prev_ofs = out_str(i_load, i_ur, prev_ofs);
            }
        }
    };

    auto fma_block = [=](bool last_block) {
        assert(jcp.reduce_loop_unroll % jcp.fma_step == 0);

        int reduce_step = jcp.fma_step;
        int prev_bcast_ofs = -1;
        assert(reduce_dim_tail % reduce_step == 0);

        const int i_reduce_end = reduce_dim_tail && last_block
                ? reduce_dim_tail
                : jcp.reduce_loop_unroll;

        int bcast_reg_ofs = utils::rnd_up(ur * load_loop_blk, jcp.fma_step)
                + jcp.fma_step * load_loop_blk;
        int num_bcast_regs = 32 - bcast_reg_ofs;
        int bcast_reg_idx = 0;

        for (int i_reduce = 0; i_reduce < i_reduce_end;
                i_reduce += reduce_step) { // IC
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) { // OC
                for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                    load_load(i_reduce + i_fma, i_load, i_fma);
                }
            }

            int bcast_reg_startidx = bcast_reg_idx % num_bcast_regs;
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                if (i_ur >= num_bcast_regs) break;
                prev_bcast_ofs = bcast_load(i_reduce, i_ur, prev_bcast_ofs,
                        bcast_reg_ofs + (bcast_reg_idx % num_bcast_regs));
                bcast_reg_idx++;
            }

            for (int i_ur = 0; i_ur < ur; ++i_ur) {

                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    fmla(vreg_accum_s(i_load, i_ur), reg_p_all_ones,
                            vreg_load_s(i_load, 0),
                            ZRegS(bcast_reg_ofs
                                    + ((bcast_reg_startidx + i_ur)
                                            % num_bcast_regs)));
                }
                if ((num_bcast_regs + i_ur) < ur) {
                    prev_bcast_ofs = bcast_load(i_reduce, num_bcast_regs + i_ur,
                            prev_bcast_ofs,
                            bcast_reg_ofs + (bcast_reg_idx % num_bcast_regs));
                    bcast_reg_idx++;
                }
            }
        }
    };
    Label reduce_loop;
    Label reduce_loop_tail;

    mov(aux_reg_load_data, reg_load_data);

    mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    init();

    mov(reduce_loop_iter, reg_reduce_loop_work);
    subs_imm(reduce_loop_iter, reduce_loop_iter, jcp.reduce_loop_unroll,
            reg_tmp_imm);
    b(LE, reduce_loop_tail);

    align(32);
    L(reduce_loop);
    {
        fma_block(false);
        add_imm(aux_reg_bcast_data, aux_reg_bcast_data,
                jcp.reduce_loop_bcast_step, reg_tmp_imm);
        add_imm(aux_reg_load_data, aux_reg_load_data, jcp.reduce_loop_load_step,
                reg_tmp_imm);
        subs_imm(reduce_loop_iter, reduce_loop_iter, jcp.reduce_loop_unroll,
                reg_tmp_imm);
        b(GT, reduce_loop);
    }

    L(reduce_loop_tail);
    fma_block(true);

    store();
}

void jit_sve_512_1x1_conv_kernel::generate() {
    preamble();

    /* All 1 predicate register */
    ptrue(reg_p_all_ones.b);

    /* Pointers indicate weight, input, and output data */
    ldr(reg_bcast_data, ptr(abi_param1, GET_OFF(bcast_data))); // Input
    ldr(reg_load_data, ptr(abi_param1, GET_OFF(load_data))); // Weight
    ldr(reg_output_data, ptr(abi_param1, GET_OFF(output_data))); // Output

    /* Pointer indicates bias data if the layer has bias option */
    if (jcp.with_bias) ldr(reg_bias_data, ptr(abi_param1, GET_OFF(bias_data)));

    /* Get workloads of each loop */
    ldr(reg_load_loop_work, ptr(abi_param1, GET_OFF(load_dim)));
    ldr(reg_bcast_loop_work, ptr(abi_param1, GET_OFF(bcast_dim)));
    ldr(reg_reduce_loop_work, ptr(abi_param1, GET_OFF(reduce_dim)));

    /* A flag for controlling reduce loop */
    ldr(reg_reduce_pos_flag, ptr(abi_param1, GET_OFF(first_last_flag)));

    if (one_of(jcp.prop_kind, forward_training, forward_inference))
        mov(reg_relu_ns, reinterpret_cast<size_t>(&jcp.eltwise.alpha));

    if (jcp.prop_kind == backward_weights)
        ldr(reg_output_stride, ptr(abi_param1, GET_OFF(output_stride)));

    auto load_loop_body = [=](int load_loop_blk) {
        subs_imm(reg_load_loop_work, reg_load_loop_work,
                load_loop_blk * jcp.load_loop_iter_step, reg_tmp_imm);

        bcast_loop(load_loop_blk);
        add_imm(reg_load_data, reg_load_data,
                load_loop_blk * jcp.load_loop_load_step, reg_tmp_imm);
        switch (jcp.prop_kind) {
            case forward_training:
            case forward_inference:
                add_imm(reg_bias_data, reg_bias_data,
                        load_loop_blk * jcp.load_block * jcp.typesize_out,
                        reg_tmp_imm);
                add_imm(reg_output_data, reg_output_data,
                        load_loop_blk * jcp.load_block * jcp.typesize_out
                                * (is_out_layout_nxc(jcp)
                                                ? 1
                                                : (jcp.with_dw_conv
                                                                ? jcp.ow
                                                                : jcp.bcast_dim)),
                        reg_tmp_imm);
                break;
            case backward_data:
                add_imm(reg_output_data, reg_output_data,
                        load_loop_blk * jcp.load_block * jcp.typesize_out
                                * (is_out_layout_nxc(jcp) ? 1 : jcp.bcast_dim),
                        reg_tmp_imm);
                break;
            case backward_weights:
                for (int i_load = 0; i_load < load_loop_blk; i_load++)
                    add(reg_output_data, reg_output_data, reg_output_stride);
                break;
            default: assert(!"invalid prop_kind");
        }
    };

    const int simd_w = cpu_isa_traits<sve_512>::vlen / sizeof(float);

    Label load_loop_blk[7];

    // with an implicit load_loop_block {6, 5, 4, 3, 2,  1}
    static const int ur_cases_bcast[] = {2, 5, 6, 9, 14, 32};

    const int size_ur_cases = sizeof(ur_cases_bcast);

    const int *ur_cases = ur_cases_bcast;
    const int num_ur_cases = size_ur_cases / sizeof(*ur_cases);

    for (int ur_idx = num_ur_cases - 1; ur_idx > 0; ur_idx--) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.nb_load > label_idx && jcp.ur <= ur_cases[ur_idx]) {
            cmp_imm(reg_load_loop_work, simd_w * (label_idx + 1), reg_tmp_imm);
            b(LE, load_loop_blk[label_idx]);
        }
    }

    for (int ur_idx = 0; ur_idx < num_ur_cases; ur_idx++) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.nb_load > label_idx && jcp.ur <= ur_cases[ur_idx]) {
            L(load_loop_blk[label_idx]);
            {
                if (label_idx == 0) {
                    cmp(reg_load_loop_work, 0);
                    b(LE, load_loop_blk[num_ur_cases]);
                }
                load_loop_body(label_idx + 1);
                if (label_idx - 1 > 0) {
                    cmp_imm(reg_load_loop_work, 2 * label_idx * simd_w,
                            reg_tmp_imm);
                    b(EQ, load_loop_blk[label_idx - 1]);
                }
                cmp_imm(reg_load_loop_work, label_idx * simd_w, reg_tmp_imm);
                b(GT, load_loop_blk[label_idx]);
            }
            for (int idx = label_idx - 1; idx > 0; --idx) {
                cmp_imm(reg_load_loop_work, simd_w * (idx + 1), reg_tmp_imm);
                b(EQ, load_loop_blk[idx]);
            }
            if (ur_idx < num_ur_cases - 2) {
                cmp_imm(reg_load_loop_work, simd_w, reg_tmp_imm);
                b(LE, load_loop_blk[0]);
            }
        }
    }
    L(load_loop_blk[num_ur_cases]);

    postamble();
    if (jcp.with_eltwise) {
#ifndef DISABLE_ELTWISE
        eltwise_injector_->prepare_table();
        binCommit();
#else
        assert(!"fused eltwise error");
#endif
    }
}

bool jit_sve_512_1x1_conv_kernel::post_ops_ok(
        jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {

    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };
    auto is_convolution
            = [&](int idx) { return p.entry_[idx].is_convolution(); };

    int dw_idx = p.find(primitive_kind::convolution);
    int len = dw_idx != -1 ? dw_idx + 1 : p.len();

    switch (len) {
        case 0: return true; // no post_ops
        case 1: // eltwise OR sum OR Convolution
            return is_eltwise(0) || is_sum(0) || is_convolution(0);
        case 2: // sum -> eltwise OR eltwise -> convolution
            return (is_sum(0) && is_eltwise(1))
                    || (is_eltwise(0) && is_convolution(1));
        default: return false;
    }

    return false;
}

status_t jit_sve_512_1x1_conv_kernel::init_conf(jit_1x1_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr, int nthreads, bool reduce_src) {

    /* arch check */
    if (!mayiuse(sve_512)) return status::unimplemented;

    jcp.nthr = nthreads;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int simd_w = cpu_isa_traits<sve_512>::vlen / sizeof(float);
    const int ndims = src_d.ndims();
    /* Forward_[training, inference], backward_[data, weight] */
    jcp.prop_kind = cd.prop_kind;

    /* Check group option */
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    /* Batchsize */
    jcp.mb = src_d.dims()[0];
    /* Channel */
    jcp.oc_without_padding = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc = jcp.oc_without_padding;
    jcp.ic_without_padding = src_d.dims()[1] / jcp.ngroups;
    jcp.ic = jcp.ic_without_padding;
    /* D, H, W */
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    /* Kernel size */
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    /* padding params */
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    /* stride params */
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];
    /* bias info */
    jcp.with_bias = pick_by_prop_kind(jcp.prop_kind, cd.bias_desc.format_kind,
                            format_kind::undef, cd.diff_bias_desc.format_kind)
            != format_kind::undef;

    /* Spatials */
    jcp.os = jcp.od * jcp.oh * jcp.ow;
    jcp.is = jcp.id * jcp.ih * jcp.iw;
    jcp.tr_is = rnd_up(jcp.is, 4);

    if (!post_ops_ok(jcp, attr)) return status::unimplemented;

    /* Depthwise conv check */
    const auto &p = attr.post_ops_;
    const int dw_conv_ind = p.find(primitive_kind::convolution);
    jcp.with_dw_conv = dw_conv_ind != -1;

    /* Post operation check */
    // Using dw_conv_ind as upper-bound below, as post-ops after it will be
    // handled in depthwise convolution.
    jcp.with_sum = p.find(primitive_kind::sum, 0, dw_conv_ind) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise, 0, dw_conv_ind);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
#ifndef DISABLE_ELTWISE
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;
        if (jcp.eltwise.alg == alg_kind::eltwise_pow)
            return status::unimplemented;
        if (dst_d.data_type() == data_type::s32) return status::unimplemented;
#else
        return status::unimplemented;
#endif
    }

    /* Data format check */
    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    jcp.src_tag = src_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx16c);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx16c);
    bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, jcp.src_tag, jcp.dst_tag);
    auto required_dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;

    if (is_data_layout_nxc) return status::unimplemented;

    /* Channel padding check */
    bool ok_to_pad_channels
            = true && jcp.ngroups == 1 && src_d.data_type() == data_type::f32;

    /* Input and output must be multiple of simd_w */
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    bool args_ok = true && jcp.ngroups == 1 && jcp.src_tag == required_dat_tag
            && jcp.dst_tag == required_dat_tag
            && (jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0) && jcp.f_pad == 0
            && jcp.t_pad == 0 && jcp.l_pad == 0 && jcp.stride_w == 1
            && jcp.stride_h == 1 && jcp.stride_d == 1 && jcp.kd == 1
            && jcp.kh == 1 && jcp.kw == 1 && jcp.ow == jcp.iw
            && jcp.oh == jcp.ih && jcp.od == jcp.id; // enforce rpad=0
    if (!args_ok) return status::unimplemented;

    /* Channel blocking size is simd_w */
    jcp.ic_block = jcp.oc_block = simd_w;

    jcp.ver = ver_sve_512;
    if (everyone_is(data_type::f32, src_d.data_type(), weights_d.data_type(),
                dst_d.data_type())) {
        const int is_bwd_d = jcp.prop_kind == backward_data;
        /* Set weight data layout tag */
        format_tag_t wei_tag = with_groups
                ? pick(2 * ndims - 6 + is_bwd_d, gOIw16i16o, gIOw16o16i,
                        gOIhw16i16o, gIOhw16o16i, gOIdhw16i16o, gIOdhw16o16i)
                : pick(2 * ndims - 6 + is_bwd_d, OIw16i16o, IOw16o16i,
                        OIhw16i16o, IOhw16o16i, OIdhw16i16o, IOdhw16o16i);

        jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
        if (jcp.wei_tag != wei_tag) return status::unimplemented;

        jcp.fma_step = 1;
        jcp.typesize_in = sizeof(prec_traits<data_type::f32>::type);
        jcp.typesize_out = sizeof(prec_traits<data_type::f32>::type);
    } else {
        // TODO: currently, only support fp32
        return status::unimplemented;
    }

    /* once all the formats are set, check the padding consistency */
    args_ok = true && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= dst_d.padded_dims()[1]
            && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    // TODO: Optimize bellow params
    const int SMALL_SPATIAL = 10;
    const int BIG_SPATIAL = 65;
    const int BIG_LOAD_DIM = (jcp.reduce_dim >= 512) ? 256 : 512;

    int load_blocking {0};
    int load_blocking_max {0};
    int bcast_blocking {0};
    int bcast_blocking_max {0};
    int reduce_blocking {0};
    int reduce_blocking_max {0};

    jcp.load_grp_count = 1;

    // TODO: mov check funcs into platform files
    const int L1_capacity
            = platform::get_per_core_cache_size(1) / sizeof(float);
    const int L2_size = platform::get_per_core_cache_size(2) / sizeof(float);
    const int L2_capacity = (L2_size * 3) / 4;

    /* FWD, BWD data */
    if (one_of(jcp.prop_kind, forward_training, forward_inference,
                backward_data)) {

        if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
            /* Forward */
            if (jcp.with_dw_conv) jcp.ur = nstl::min(jcp.ow, jcp.ur);
            jcp.reduce_dim = jcp.ic; // src channel
            jcp.reduce_block = jcp.ic_block; // src simd_w

            jcp.load_dim = jcp.oc; // dst channel
            jcp.load_block = jcp.oc_block; // dst simd_W

            jcp.bcast_dim = jcp.is; // src H*W
        } else {
            /* Backward data */
            jcp.reduce_dim = jcp.oc; // src channel
            jcp.reduce_block = jcp.oc_block; // src simd_w

            jcp.load_dim = jcp.ic; // dst channel
            jcp.load_block = jcp.ic_block; // dst simd_w

            jcp.bcast_dim = jcp.os; // src H*W
        }

        /* # of consecutive channel elements  */
        jcp.reduce_loop_unroll = jcp.reduce_block;

        /* Offset to move to the next 16 input channel elements with the same H*W position */
        jcp.reduce_loop_bcast_step
                = jcp.reduce_loop_unroll * jcp.bcast_dim * jcp.typesize_in;

        /* Offset: 16o*16i (filter) */
        jcp.reduce_loop_load_step
                = jcp.reduce_loop_unroll * jcp.load_block * jcp.typesize_in;

        /* Offset: I/16 * 16o */
        jcp.load_loop_load_step
                = (utils::rnd_up(jcp.reduce_dim, jcp.reduce_block))
                * jcp.load_block * jcp.typesize_in;

        /* adjusting registry blocking */
        int max_regs, min_regs, size_threshold, ur_step;

        /* spatial : H*D of dst */
        const int spatial
                = (one_of(jcp.prop_kind, forward_training, forward_inference))
                ? jcp.od * jcp.oh // forward
                : jcp.id * jcp.ih; // backward

        max_regs = 9; // max # of ur_w
        min_regs = 6; // min # of ur_w
        size_threshold = 14;
        ur_step = 1; // step size of ur_w param checking
        jcp.ur = 1;

        /*
         *  H*D of dst  > SMALL_SPATIAL
         */
        if (jcp.load_dim > 128 && jcp.load_dim < BIG_LOAD_DIM
                && spatial > SMALL_SPATIAL && spatial < BIG_SPATIAL
                && jcp.reduce_dim < 256) {
            max_regs = 6;
            min_regs = 5;
        }

        for (int ur_w = max_regs; ur_w >= min_regs; ur_w -= ur_step) {
            /*
             *  H*D of dst >= size_threshold, (H*D of dst) % ur_w == 0
             *  or
             *  H*D of dst < size_threshold, (H*W of dst) % ur_w == 0
             */
            if ((spatial >= size_threshold && spatial % ur_w == 0)
                    || (spatial < size_threshold && jcp.os % ur_w == 0)) {
                jcp.ur = ur_w;
                break;
            }
        }

        if (jcp.ur == 1) {
            // If ur = 1, then min(max_regs, H*W of dst)
            jcp.ur = nstl::min(max_regs, jcp.os);
        }
        jcp.bcast_block = jcp.ur; // block size of bcast (input data)
        /* Number of steps for the dst address to output, used in bcast_loop() */
        jcp.bcast_loop_output_step = jcp.ur * jcp.typesize_out * jcp.load_block;
        jcp.bcast_loop_output_substep = -1; // unused

        /* Number of steps for the src address to be broadcasted in bcast_loop() */
        jcp.bcast_loop_bcast_step = jcp.ur * jcp.typesize_in * jcp.reduce_block;
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_iter_step = jcp.load_block;

        if (jcp.prop_kind == backward_data)
            jcp.loop_order = loop_lbr;
        else
            jcp.loop_order = reduce_src ? loop_blr : loop_lbr;

        int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
        int nb_load = div_up(jcp.load_dim, jcp.load_block);

        reduce_blocking = jcp.reduce_dim;
        if (jcp.load_dim <= BIG_LOAD_DIM && spatial > SMALL_SPATIAL
                && spatial < BIG_SPATIAL) {
            reduce_blocking = nstl::min(jcp.reduce_dim, 80);
        } else if (spatial > SMALL_SPATIAL)
            reduce_blocking = nstl::min(jcp.reduce_dim, 512);
        else
            reduce_blocking = nstl::min(jcp.reduce_dim, 256);

        // Check input data cache aliasing.
        // For other ISA constants may be updated.
        // 64 * 1024 is chosen due to 1MB L2 16-way cache.
        // 7 is empirical value. It is about half of 16.
        // So we leave about half of the set for other data - weights, dst
        int way_size = (16 * 1024) / jcp.typesize_in;
        int max_hits = 7;
        if (jcp.bcast_dim * reduce_blocking > way_size * max_hits) {
            int nrb = reduce_blocking / simd_w;
            int sp = jcp.bcast_dim;
            int wl = way_size / simd_w;
            for (int start_off = 0; start_off < jcp.ur; start_off++) {
                for (int off = start_off, hits = 0; off < sp * nrb; off += wl) {
                    if (off % sp >= jcp.ur || ++hits < max_hits) continue;
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

        /* Number of weight elements to be loaded for dest */
        int load_size = jcp.load_dim * jcp.reduce_dim;
        /* Number of elements to be broadcasted from src */
        auto bcast_size
                = (dim_t)jcp.mb * jcp.ngroups * jcp.bcast_dim * jcp.reduce_dim;

        /* 12 cores per CMG */
        if (jcp.nthr <= 12 && jcp.mb < jcp.nthr
                && nb_load * nb_bcast > jcp.nthr) {
            // Some heuristic here
            float calc_koef = 0.01, best_cost = FLT_MAX;
            int n_lgc = jcp.nthr;
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
                int min_tg = jcp.nthr / lgc;
                int max_tg = div_up(jcp.nthr, lgc);
                // Some heuristic here
                float mem_koef = (max_tg == 1) ? 1.f : 1.3f;
                float job_cost = 0.;
                if (jcp.nthr % lgc < nb_load % lgc) {
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
            load_blocking
                    = div_up(nb_load, jcp.load_grp_count) * jcp.load_block;
        } else {
            jcp.load_grp_count
                    = div_up(jcp.nthr, jcp.mb * jcp.ngroups * nb_bcast);
            jcp.load_grp_count = best_divider(jcp.nthr, jcp.load_grp_count,
                    2 * jcp.load_grp_count, false);
        }
        if (jcp.bcast_dim <= 49 && jcp.mb <= jcp.nthr && jcp.load_dim > 512
                && jcp.load_dim / jcp.reduce_dim >= 4) {
            jcp.load_grp_count = nstl::max(jcp.load_grp_count, 2);
            load_blocking = jcp.load_block;
        }

        bcast_blocking = div_up(jcp.mb * jcp.ngroups * nb_bcast,
                                 div_up(jcp.nthr, jcp.load_grp_count))
                * jcp.bcast_block;
        bcast_blocking = nstl::min(jcp.bcast_dim, bcast_blocking);
        bcast_blocking = rnd_up(bcast_blocking, jcp.bcast_block);

        int space_for_bcast = (L2_capacity - /* kernel_size - */
                2 * jcp.load_block * reduce_blocking - jcp.ur * reduce_blocking
                - 3 * 1024);
        if (jcp.reduce_dim * jcp.bcast_dim > L2_capacity) space_for_bcast /= 2;

        int bcast_in_cache
                = nstl::max(jcp.bcast_block, space_for_bcast / reduce_blocking);
        bcast_blocking = nstl::min(
                bcast_blocking, rnd_dn(bcast_in_cache, jcp.bcast_block));

        load_blocking_max = load_blocking;
        bcast_blocking_max = bcast_blocking * 3 / 2;
        reduce_blocking_max = reduce_blocking;

        jcp.ur_tail = (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) % jcp.ur;

    } else if (jcp.prop_kind == backward_weights) { /* BWD weight */

        jcp.reduce_dim = jcp.is;

        jcp.reduce_block = best_divider(jcp.reduce_dim, 7, 16, true);
        if (jcp.reduce_dim % jcp.reduce_block != 0)
            jcp.reduce_block = best_divider(jcp.iw, 4, jcp.iw, false);
        if (jcp.reduce_block > 256) { jcp.reduce_block = 1; }

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.ic;
        jcp.bcast_block = jcp.ic_block;

        if (jcp.reduce_block <= 19) {
            // if reduce_block is big then generated JIT code may be big
            // for small values of ur because reduce_loop_unroll = reduce_block
            jcp.ur = jcp.bcast_block / 2;
        } else {
            jcp.ur = jcp.bcast_block;
        }

        jcp.ur_tail = jcp.bcast_dim % jcp.bcast_block;
        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
                = jcp.typesize_in * jcp.reduce_loop_unroll * jcp.ic_block;
        jcp.reduce_loop_load_step
                = jcp.typesize_in * jcp.reduce_loop_unroll * jcp.oc_block;

        jcp.bcast_loop_output_step
                = jcp.oc_block * jcp.ic_block * jcp.typesize_out;
        jcp.bcast_loop_output_substep
                = jcp.oc_block * jcp.ur * jcp.typesize_out;
        jcp.bcast_loop_bcast_step = jcp.ic_block
                * utils::rnd_up(jcp.reduce_dim, jcp.reduce_block)
                * jcp.typesize_in;
        jcp.bcast_loop_bcast_substep = jcp.ur * jcp.typesize_in;

        jcp.load_loop_load_step = jcp.typesize_in * jcp.oc_block * jcp.os;
        jcp.load_loop_iter_step = jcp.oc_block;

        /* --- */
        balance(jcp);

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
        int min_reduce_blocking
                = nstl::min(L1_capacity / jcp.ur, nstl::max(jcp.iw, jcp.ih));
        reduce_blocking = best_divider(
                jcp.reduce_dim, min_reduce_blocking, max_reduce_blocking, true);
        reduce_blocking = nstl::max(
                rnd_dn(reduce_blocking, jcp.reduce_block), jcp.reduce_block);

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
    assert(reduce_blocking % jcp.reduce_block == 0);
    assert(load_blocking_max % jcp.load_block == 0);
    assert(reduce_blocking_max % jcp.reduce_block == 0);
    assert(jcp.reduce_dim % jcp.reduce_block == 0);

    assert(jcp.bcast_block % jcp.ur == 0);

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = utils::div_up(load_blocking, jcp.load_block);
    jcp.nb_load_blocking_max = utils::div_up(load_blocking_max, jcp.load_block);
    jcp.nb_reduce_blocking = utils::div_up(reduce_blocking, jcp.reduce_block);
    jcp.nb_reduce_blocking_max
            = utils::div_up(reduce_blocking_max, jcp.reduce_block);

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    return status::success;
}

void jit_sve_512_1x1_conv_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp) {

    using namespace dnnl::impl::memory_tracking::names;

    // Fox nxc layout bias is padded only for bwd_wb direction, as  bias
    // reduction kernels can't handle tails yet.
    if (jcp.with_bias && jcp.prop_kind != backward_data
            && (jcp.oc != jcp.oc_without_padding // blocked layout
                    || (jcp.prop_kind == backward_weights // nxc layout
                            && jcp.oc % jcp.oc_block != 0))) {

        const size_t nelems_padded_bias
                = jcp.ngroups * utils::rnd_up(jcp.oc, jcp.oc_block);
        scratchpad.book(
                key_conv_padded_bias, nelems_padded_bias, jcp.typesize_out);
    }

    if (jcp.prop_kind == backward_weights) {
        const size_t wei_size = (size_t)jcp.ngroups
                * rnd_up(jcp.oc, jcp.oc_block) * rnd_up(jcp.ic, jcp.ic_block);
        scratchpad.book(key_conv_wei_reduction, wei_size * (jcp.nthr_mb - 1),
                jcp.typesize_out);
    }
}

/* BWD W*/
void jit_sve_512_1x1_conv_kernel::balance(jit_1x1_conv_conf_t &jcp) {
    int nthreads = jcp.nthr;
    // initialize jcp reduction threading properties
    jcp.nthr = jcp.nthr_mb = jcp.nthr_g = jcp.nthr_oc_b = jcp.nthr_ic_b = 1;
    if (nthreads < jcp.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        return;
    }
    // bcast_dim: src H*W, bcast_block: ur (fwd, bwd_d)
    const int nb_bcast
            = div_up(jcp.bcast_dim, jcp.bcast_block); // # of H*W loop
    // load_dim: dst channel, load_block: simd_w
    const int nb_load
            = div_up(jcp.load_dim, jcp.load_block); // # of dst channel loop
    // reduce_dim: src channel, reduce_block: simd_w
    const int nb_reduce
            = div_up(jcp.reduce_dim, jcp.reduce_block); // # of src channel loop

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
                * div_up(jcp.ngroups, jcp.nthr_g) * div_up(nb_bcast, nthr_ic_b)
                * jcp.ic_block * jcp.reduce_block / jcp.stride_h
                / jcp.stride_w /* (n1) */
                + (size_t)load_koeff * div_up(jcp.mb * nb_reduce, nthr_mb)
                * div_up(jcp.ngroups, jcp.nthr_g) * div_up(nb_load, nthr_oc_b)
                * jcp.oc_block * jcp.reduce_block
                + (size_t)output_koeff /* (n2) */
                * div_up(jcp.ngroups, jcp.nthr_g) * div_up(nb_load, nthr_oc_b)
                * div_up(nb_bcast, nthr_ic_b) * jcp.ic_block * jcp.oc_block;
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

        const bool ready_for_async = utils::one_of(jcp.ver, ver_fma);
        if (!ready_for_async && !dnnl_thr_syncable()) {
            assert(nthr_mb == 1);
            break;
        }
    }
    if (jcp.nthr_mb > nthreads / 2 && jcp.nthr_mb < nthreads)
        jcp.nthr_mb = nstl::min(jcp.mb, nthreads);

    jcp.nthr = jcp.nthr_mb * jcp.nthr_g * jcp.nthr_oc_b * jcp.nthr_ic_b;
    assert(jcp.nthr <= nthreads);
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
