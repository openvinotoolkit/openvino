/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"

#define IDX(a) static_cast<uint32_t>(a.getIdx())

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    using namespace alg_kind;
    using namespace Xbyak_aarch64::util;
    preserved_vecs_count = 0;
    vecs_to_preserve = aux_vecs_count();
    const auto start_idx = *(vmm_idxs.begin());
    const auto end_idx = *(vmm_idxs.rbegin()) + 1;
    start_idx_tail = vmm_idxs.begin();

    for (size_t idx = preserved_vecs_count; idx < vecs_count; idx++) {
        if (preserved_vecs_count >= vecs_to_preserve) break;
        if (start_idx <= idx && idx < end_idx) continue;

        preserved_vec_idxs[preserved_vecs_count++] = idx;
    }

    size_t preserved_vecs_count_tail = vecs_to_preserve - preserved_vecs_count;
    for (size_t i = 0; i < preserved_vecs_count_tail; i++) {
        preserved_vec_idxs[preserved_vecs_count++] = *start_idx_tail;
        ++start_idx_tail;
    }

    assert(preserved_vecs_count == vecs_to_preserve);

    // Same logic but to allocate gprs
    size_t preserved_gprs_count = 0;
    for (size_t gpr_idx = 0; gpr_idx <= 30; ++gpr_idx) {
        int _idx = 30 - gpr_idx; // we allocate from the end
        if (preserved_gprs_count < aux_gprs_count()
                && (((unsigned)_idx) != x_table.getIdx()))
            preserved_gpr_idxs[preserved_gprs_count++] = _idx;
    }
    assert(preserved_gprs_count == aux_gprs_count());

    h->ptrue(p_all.b);

    if (save_state_) {
        h->str(x_table, pre_ptr(h->X_SP, -8));

        for (size_t i = 0; i < preserved_gprs_count; ++i) {
            /* This route has not been tested */
            h->str(XReg(preserved_gpr_idxs[i]), pre_ptr(h->X_SP, -8));
        }

        if (preserved_vecs_count)
            h->sub_imm(
                    h->X_SP, h->X_SP, preserved_vecs_count * vlen, h->X_TMP_0);

        size_t i = 0;

        while (i < preserved_vecs_count) {
            int count = 0;
            int ii = i;
            do {
                h->add_imm(h->x_tmp_vec[count++], h->X_SP, i * vlen,
                        h->X_DEFAULT_ADDR);
                i++;
            } while (i < preserved_vecs_count && count < h->x_tmp_vec_size);

            for (int j = 0; j < count; j++)
                h->st1w(ZRegS(preserved_vec_idxs[ii++]), p_all,
                        ptr(h->x_tmp_vec[j]));
        }
        load_table_addr();
    }

    assign_regs();
    set_coef_to_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble_tail(
        const injector_utils::vmm_index_set_iterator_t start_idx_it) {
    size_t tail_vecs_to_preserve = std::distance(start_idx_it, start_idx_tail);
    if (tail_vecs_to_preserve == 0) return;

    const int idx_off = vecs_to_preserve - tail_vecs_to_preserve;

    if (save_state_) {
        /* This route has not been tested */
        if (idx_off) h->add_imm(h->X_SP, h->X_SP, idx_off * vlen, h->X_TMP_0);

        size_t i = 0;

        while (i < tail_vecs_to_preserve) {
            int count = 0;
            int ii = i;
            do {
                h->add_imm(h->x_tmp_vec[count++], h->X_SP, i * vlen,
                        h->X_DEFAULT_ADDR);
                i++;
            } while (i < tail_vecs_to_preserve && count < h->x_tmp_vec_size);

            for (int j = 0; j < count; j++)
                h->ld1w(ZRegS(preserved_vec_idxs[idx_off + ii++]), p_all / T_z,
                        ptr(h->x_tmp_vec[j]));
        }
    }

    for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
        preserved_vec_idxs[idx_off + i] += tail_vecs_to_preserve;

    if (save_state_) {
        size_t i = 0;

        while (i < tail_vecs_to_preserve) {
            int count = 0;
            int ii = i;
            do {
                h->add_imm(h->x_tmp_vec[count++], h->X_SP, i * vlen,
                        h->X_DEFAULT_ADDR);
                i++;
            } while (i < tail_vecs_to_preserve && count < h->x_tmp_vec_size);

            for (int j = 0; j < count; j++)
                h->st1w(ZRegS(preserved_vec_idxs[idx_off + ii++]), p_all / T_z,
                        ptr(h->x_tmp_vec[j]));
        }

        if (idx_off) {
            h->sub_imm(h->X_SP, h->X_SP, idx_off * vlen, h->X_TMP_0);
        }
    }

    assign_regs();
    set_coef_to_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_postamble() {
    using namespace Xbyak_aarch64::util;
    if (!save_state_) return;

    size_t i = 0;

    while (i < preserved_vecs_count) {
        int count = 0;
        int ii = i;
        do {
            h->add_imm(h->x_tmp_vec[count++], h->X_SP, i * vlen,
                    h->X_DEFAULT_ADDR);
            i++;
        } while (i < preserved_vecs_count && count < h->x_tmp_vec_size);

        for (int j = 0; j < count; j++)
            h->ld1w(ZRegS(preserved_vec_idxs[ii++]), p_all / T_z,
                    ptr(h->x_tmp_vec[j]));
    }

    if (preserved_vecs_count)
        h->add_imm(h->X_SP, h->X_SP, preserved_vecs_count * vlen, h->X_TMP_0);

    for (int i = aux_gprs_count() - 1; i >= 0; --i)
        h->ldr(XReg(preserved_gpr_idxs[i]), post_ptr(h->X_SP, 8));
    h->ldr(x_table, post_ptr(h->X_SP, 8));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::assign_regs() {
    /* For translation of x64's memory operand instructions */
    z_tmp = TRegS(static_cast<uint32_t>(preserved_vec_idxs[0]));

    vmm_mask = TRegS(preserved_vec_idxs[1]);
    vmm_aux0 = TRegS(preserved_vec_idxs[1]);
    vmm_aux1 = TRegS(preserved_vec_idxs[2]);
    vmm_aux2 = TRegS(preserved_vec_idxs[3]);
    vmm_aux3 = TRegS(preserved_vec_idxs[4]);
    vmm_aux4 = TRegS(preserved_vec_idxs[5]);
    vmm_aux5 = TRegS(preserved_vec_idxs[6]);
    vmm_aux6 = TRegS(preserved_vec_idxs[7]);
    vmm_aux7 = TRegS(preserved_vec_idxs[8]);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::set_coef_to_regs() {
    using namespace alg_kind;

    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu:
                if (alpha_ != 0.f) table_val(alpha, z_tmp);
                break;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: table_val(alpha, vmm_aux4); break;
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh:
            case eltwise_square:
            case eltwise_abs:
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt:
            case eltwise_swish: break;
            case eltwise_linear:
                table_val(alpha, z_tmp);
                table_val(beta, vmm_aux0);
                break;
            case eltwise_bounded_relu: table_val(alpha, z_tmp); break;
            case eltwise_soft_relu:
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic:
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp:
            case eltwise_gelu_tanh:
            case eltwise_log: break;
            case eltwise_clip:
                table_val(alpha, z_tmp);
                table_val(beta, vmm_aux0);
                break;
            case eltwise_pow:
            case eltwise_gelu_erf:
            case eltwise_round: break;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: table_val(alpha, z_tmp); break;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu:
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh:
            case eltwise_square:
            case eltwise_abs:
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt:
            case eltwise_linear:
            case eltwise_bounded_relu:
            case eltwise_soft_relu:
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic:
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp:
            case eltwise_gelu_tanh:
            case eltwise_swish:
            case eltwise_log: break;
            case eltwise_clip:
                table_val(beta, z_tmp);
                table_val(alpha, vmm_aux0);
                break;
            case eltwise_pow:
            case eltwise_gelu_erf: break;
            default: assert(!"unsupported eltwise algorithm");
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_cmp_mask(
        const TRegS &vmm_src, const TRegS &compare_operand, int cmp_predicate) {

    enum {
        EQ_OQ = 0,
        LT_OS = 1,
        LE_OS = 2,
        UNORD_Q = 3,
        NEQ_UQ = 4,
        NLT_US = 5,
        NLE_US = 6,
        ORD_Q = 7,
        EQ_UQ = 8,
        NGE_US = 9,
        NGT_US = 10,
        FALSE_OQ = 11,
        NEQ_OQ = 12,
        GE_OS = 13,
        GT_OS = 14,
        TRUE_UQ = 15,
        EQ_OS = 16,
        LT_OQ = 17,
        LE_OQ = 18,
        UNORD_S = 19,
        NEQ_US = 20,
        NLT_UQ = 21,
        NLE_UQ = 22,
        ORD_S = 23,
        EQ_US = 24,
        NGE_UQ = 25,
        NGT_UQ = 26,
        FALSE_OS = 27,
        NEQ_OS = 28,
        GE_OQ = 29,
        GT_OQ = 30,
        TRUE_US = 31,
    };

    h->mov(PRegB(IDX(p_tmp0)), h->P_ALL_ONE / T_z, h->P_ALL_ONE.b);
    switch (cmp_predicate) {
        case EQ_OQ:
            h->fcmeq(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case LT_OS:
            h->fcmlt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case LE_OS:
            h->fcmle(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NEQ_UQ:
            h->fcmne(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NLT_US:
            h->fcmge(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NLE_US:
            h->fcmgt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case EQ_UQ:
            h->fcmeq(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NGE_US:
            h->fcmlt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NGT_US:
            h->fcmle(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NEQ_OQ:
            h->fcmne(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case GE_OS:
            h->fcmge(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case GT_OS:
            h->fcmgt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case EQ_OS:
            h->fcmeq(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case LT_OQ:
            h->fcmlt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case LE_OQ:
            h->fcmle(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NEQ_US:
            h->fcmne(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NLT_UQ:
            h->fcmge(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NLE_UQ:
            h->fcmgt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case EQ_US:
            h->fcmeq(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NGE_UQ:
            h->fcmlt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NGT_UQ:
            h->fcmle(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case NEQ_OS:
            h->fcmne(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case GE_OQ:
            h->fcmge(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case GT_OQ:
            h->fcmgt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;

        case UNORD_Q:
        case ORD_Q:
        case FALSE_OQ:
        case TRUE_UQ:
        case UNORD_S:
        case ORD_S:
        case FALSE_OS:
        case TRUE_US:
        default: assert(!"Unsupported compare mode"); break;
    }
}

// Uses injector masks objects: p_mask
// Blends a result of second input into a first input w/ a stored mask.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::blend_with_mask(
        const TRegS &vmm_dst, const TRegS &src) {
    h->sel(vmm_dst, p_mask / T_m, src, vmm_dst);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_fwd(
        const TRegS &vmm_src) {

    const auto &t0 = ZRegS(IDX(vmm_src));
    const auto &t1 = ZRegS(IDX(vmm_aux1));
    const auto &t2 = ZRegS(IDX(vmm_aux2));
    h->fmin(t0, p_all, ZRegS(IDX(table_val(exp_ln_flt_max_f, z_tmp))));
    h->fmax(t0, p_all, ZRegS(IDX(table_val(exp_ln_flt_min_f, z_tmp))));
    h->fmul(t0, t0, ZRegS(IDX(table_val(exp_log2ef, z_tmp))));
    h->movprfx(t1, p_all, t0);
    h->frintm(t1, p_all, t0);
    h->fcvtzs(t2, p_all, t1);
    h->fsub(t1, t0, t1);
    h->fadd(t0, t1, ZRegS(IDX(table_val(one, z_tmp))));
    h->lsr(t1, t0, 17);
    h->fexpa(t1, t1);
    h->fscale(t1, p_all, t2);
    h->and_(ZRegD(t2.getIdx()), ZRegD(t0.getIdx()),
            ZRegD(IDX(table_val(exp_not_mask17, z_tmp))));
    h->fsub(t2, t0, t2);
    h->movprfx(t0, p_all, ZRegS(IDX(table_val(exp_coeff2, z_tmp))));
    h->fmad(t0, p_all, t2, ZRegS(IDX(table_val(exp_coeff1, z_tmp))));
    h->fmad(t0, p_all, t2, ZRegS(IDX(table_val(one, z_tmp))));
    h->fmul(t0, t1, t0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector_fwd(
        const TRegS &vmm_src) {
    /* Negative values are multiplied by alpha.
     Positive values are not modified. */
    h->mov(ZRegD(vmm_aux0.getIdx()), ZRegD(vmm_src.getIdx()));
    h->fminnm(vmm_src, p_all, 0.f);
    h->fmaxnm(vmm_aux0, p_all, 0.f);
    /* alpha is set to z_tmp in set_coef_to_regs(). */
    h->fmul(vmm_src, vmm_src, z_tmp);
    h->fadd(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_zero_ns_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->fmaxnm(vmm_src, p_all, 0.f);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector_fwd(
        const TRegS &vmm_src) {
    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    h->mov(ZRegD(vmm_aux3.getIdx()), ZRegD(vmm_src.getIdx()));

    // compute exponent
    exp_compute_vector_fwd(vmm_src);

    // alpha * (exp(x) - 1)
    h->fsub(vmm_src, p_all / T_m, 1.f);
    h->fmul(vmm_src, vmm_src, vmm_aux4);

    // combine with mask
    h->fcmgt(p_mask.s, p_all / T_z, vmm_aux3, 0.f);
    h->mov(vmm_src, p_mask / T_m, vmm_aux3);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_fwd(
        const TRegS &vmm_src) {
    // tanh(x) = x(1 + (-1/3)x^2) for |x| < tanh_range
    // tanh(x) = 1 - 2/(1 + exp(2 x)) for otherwise

    const auto &t0 = ZRegS(IDX(vmm_src));
    const auto &t1 = ZRegS(IDX(vmm_aux1));
    const auto &t2 = ZRegS(IDX(vmm_aux2));
    const auto &t3 = ZRegS(IDX(vmm_aux3));
    const auto &oneS = ZRegS(IDX(vmm_aux4));
    const auto &mask = PReg(6); // avoid pred regs used in *conv_kernel*

    h->fcpy(oneS, p_all, 1);
    // make mask for small x
    h->mov(t3, p_all, t0);
    h->fabs(t1, p_all, t0);
    h->cmplt(mask.s, p_all, t1, ZRegS(IDX(table_val(tanh_range, z_tmp))));

    // 2x
    h->fadd(t0, t0, t0);
    // exp(2x)
    exp_compute_vector_fwd(t0);
    // 1+exp(2x)
    h->fadd(t0, t0, oneS);
    // 1/(1+exp(2x))
    // 1st aprox ; a = 1/x + e
    h->frecpe(t1, t0);
    // 2nd aprox ; a' = (2 - ax)a = 1/x - e^2 x
    h->frecps(t2, t0, t1);
    h->fmul(t2, t2, t1);
    // 3rd aprox ; a'' = (2 - a'x)a'
    h->frecps(t0, t0, t2);
    h->fmul(t0, t0, t2);

    // 2/(1+exp(2x))
    h->fadd(t0, t0, t0);
    // 1-2/(1+exp(2x))
    h->fsub(t0, oneS, t0);

    // tanh(x) = x(1 - x^2/3) for |x| < tanh_range
    h->fmul(t1, t3, t3);
    h->fmad(t1, p_all, ZRegS(IDX(table_val(tanh_m1d3, z_tmp))), oneS);
    h->fmul(t1, p_all, t3);
    // select the correct value according to mask
    h->mov(t0, mask, t1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_tanh_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->mov(ZRegD(IDX(vmm_aux0)), ZRegD(IDX(vmm_src)));

    // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
    h->fmul(vmm_src, vmm_src, vmm_src);
    h->mov(ZRegD(IDX(vmm_aux1)),
            ZRegD(IDX(table_val(gelu_tanh_fitting_const, z_tmp))));
    h->fmad(vmm_src, p_all / T_m, vmm_aux1, ZRegS(IDX(table_val(one, z_tmp))));
    h->fmul(vmm_src, vmm_src, vmm_aux0);
    h->fmul(vmm_src, vmm_src,
            ZRegS(IDX(table_val(gelu_tanh_sqrt_two_over_pi, z_tmp))));

    // save x on stack as tanh uses vmm_aux0
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->str(ZReg(IDX(vmm_aux0)), ptr(h->X_TMP_0));

    // compute tanh(G(x))
    tanh_compute_vector_fwd(vmm_src);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->ldr(ZReg(IDX(vmm_aux0)), ptr(h->X_TMP_0));
    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    // compute 0.5 * x * (1 + tanh(G(x)))
    h->fadd(vmm_src, p_all / T_m, 1.f);
    h->fmul(vmm_src, p_all / T_m, 0.5f);
    h->fmul(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->fmul(vmm_src, vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->fabs(vmm_src, p_all / T_m, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->fsqrt(vmm_src, p_all / T_m, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector_fwd(
        const TRegS &vmm_src) {
    // compute x = alpha * x + beta;
    h->fmad(vmm_src, p_all / T_m, z_tmp, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->fmaxnm(vmm_src, p_all, 0.f);
    h->fminnm(vmm_src, p_all, z_tmp);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->fmaxnm(vmm_src, p_all, z_tmp);
    h->fminnm(vmm_src, p_all, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector_fwd(
        const TRegS &vmm_src) {
    // ln(1 + exp(x)) =
    // = ln(1 + exp(n * ln(2) + r)) // divide x by ln(2) and get quot and rem
    // = ln(1 + 2^n * exp(r)) // simplify the exp(n*ln(2)) expression
    // = ln(2 ^ 0 + 2^n * exp(r)) // note 1 = 2^0
    // = ln(2 ^ (n - n) + 2^n * exp(r)) // 2^0 = 2^(n-n)
    // = ln(2 ^ n * (2^-n + exp(r))) // factorize with 2^n
    // = n * ln(2) + ln(2^-n + exp(r)) // take the 2^n factor out of the ln

    // keep src for further computations
    h->mov(ZRegD(IDX(vmm_aux2)), ZRegD(IDX(vmm_src)));

    h->fminnm(ZRegS(IDX(table_val(exp_ln_flt_max_f, z_tmp))), p_all, vmm_src);
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(z_tmp)));
    h->fmaxnm(ZRegS(IDX(table_val(exp_ln_flt_min_f, z_tmp))), p_all, vmm_src);

    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(z_tmp)));
    h->mov(ZRegD(IDX(vmm_aux1)), ZRegD(IDX(vmm_src)));

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->fmul(vmm_src, vmm_src, ZRegS(IDX(table_val(exp_log2ef, z_tmp))));
    h->fadd(vmm_src, p_all / T_m, 0.5f);

    // tmp = floorf(fx)
    h->frintm(vmm_aux0, p_all / T_m, vmm_src);

    // keep vmm_src = fx for further computations
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_aux0)));

    // x = x - fx * ln2
    h->fmul(vmm_aux0, vmm_aux0, ZRegS(IDX(table_val(ln2f, z_tmp))));
    h->fsub(vmm_aux1, vmm_aux1, vmm_aux0);
    // compute exponent polynomial
    h->mov(ZRegD(IDX(vmm_aux3)), ZRegD(IDX(table_val(exp_pol, z_tmp, 4))));
    h->fmad(vmm_aux3, p_all / T_m, vmm_aux1,
            ZRegS(IDX(table_val(exp_pol, z_tmp, 3))));
    h->fmad(vmm_aux3, p_all / T_m, vmm_aux1,
            ZRegS(IDX(table_val(exp_pol, z_tmp, 2))));
    h->fmad(vmm_aux3, p_all / T_m, vmm_aux1,
            ZRegS(IDX(table_val(exp_pol, z_tmp, 1))));
    h->fmad(vmm_aux3, p_all / T_m, vmm_aux1,
            ZRegS(IDX(table_val(exp_pol, z_tmp, 0))));
    h->fmad(vmm_aux3, p_all / T_m, vmm_aux1, ZRegS(IDX(table_val(one, z_tmp))));

    // We do not count 2^-n here, because n can reach 128 and 2^(-128) is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^-n + exp(r) will be counted (2^-(n-1) + 2*exp(r))/2, because 2^(-127)
    // and 2 are numbers representable in fp32.

    // compute 2^-(n-1)
    // vmm_src now represents n-1
    h->fsub(vmm_src, p_all / T_m, 1.f);
    h->fneg(vmm_aux1, p_all / T_m, vmm_src);

    h->frinti(vmm_aux1, p_all / T_m, vmm_aux1);
    h->fcvtzs(vmm_aux1, p_all / T_m, vmm_aux1);
    // restore vmm_src to n
    h->fadd(vmm_src, p_all / T_m, 1.f);

    h->add(vmm_aux1, vmm_aux1, ZRegS(IDX(table_val(exponent_bias, z_tmp))));
    h->lsl(vmm_aux1, vmm_aux1, n_mantissa_bits);
    // calculate ln(1 + y)
    h->fmul(vmm_aux3, p_all / T_m, 2.f); // 2*exp(r)
    h->fadd(vmm_aux3, vmm_aux3,
            vmm_aux1); // 2^-(n-1) + 2*exp(r)
    h->fmul(vmm_aux3, p_all / T_m,
            ZRegS(IDX(table_val(half, z_tmp)))); // (2^-(n-1) + 2*exp(r))/2

    // frexp()
    h->lsr(vmm_src, vmm_aux3, n_mantissa_bits);
    h->scvtf(vmm_src, p_all / T_m, vmm_src);
    // got n. where n is x = 2^n * y. y = 0.5 .. 1
    h->fsub(vmm_src, vmm_src,
            ZRegS(IDX(table_val(soft_relu_one_twenty_six, z_tmp))));

    // and with mask (to get 0.5 * mantissa)
    h->and_(ZRegD(IDX(vmm_aux3)), ZRegD(IDX(vmm_aux3)),
            ZRegD(IDX(table_val(soft_relu_mantissa_sign_mask, z_tmp))));
    // got y. (mantisa)  0.5 < y < 1 (or with (to get 0.5 * mantissa))
    h->orr(ZRegD(IDX(vmm_aux3)), ZRegD(IDX(vmm_aux3)),
            ZRegD(IDX(table_val(half, z_tmp))));
    // y  = y - 1
    h->fsub(vmm_aux3, p_all / T_m, 1.f);

    // compute log1p polynomial
    h->mov(ZRegD(IDX(vmm_aux1)),
            ZRegD(IDX(table_val(soft_relu_pol, z_tmp, 8))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 7))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 6))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 5))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 4))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 3))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 2))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 1))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux3,
            ZRegS(IDX(table_val(soft_relu_pol, z_tmp, 0))));
    //calculate ln(2) * n
    h->fmul(vmm_src, vmm_src, ZRegS(IDX(table_val(ln2f, z_tmp))));
    h->fadd(vmm_src, vmm_src, vmm_aux1);
    h->fadd(vmm_src, vmm_src, vmm_aux0);

    // get vmm_mask = src > max logf
    // y = (x < max log f) ? soft_relu(x) : x
    compute_cmp_mask(vmm_aux2, table_val(exp_ln_flt_max_f, z_tmp), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector_fwd(
        const TRegS &vmm_src) {
    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic is symmetric function.
    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    h->mov(ZRegD(IDX(vmm_aux3)), ZRegD(IDX(vmm_src)));
    // we store the original sign and make x negative
    h->and_(ZRegD(IDX(vmm_aux3)), ZRegD(IDX(vmm_aux3)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));
    h->orr(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));

    exp_compute_vector_fwd(vmm_src);

    // dup exp(x)
    h->mov(ZRegD(IDX(vmm_aux1)), ZRegD(IDX(vmm_src)));
    // (exp(x) + 1)
    h->fadd(vmm_aux1, vmm_aux1, ZRegS(IDX(table_val(one, z_tmp))));
    // y = exp(x) / (exp(x) + 1)
    h->fdiv(vmm_src, p_all, vmm_aux1);

    // Now we have to apply the "symmetry" based on original sign
    h->mov(ZRegD(IDX(vmm_aux2)), ZRegD(IDX(table_val(one, z_tmp))));
    h->fsub(vmm_aux2, vmm_aux2, vmm_src);

    h->and_(ZRegD(IDX(z_tmp)), ZRegD(IDX(vmm_aux3)), ZRegD(IDX(vmm_aux3)));
    h->cmpne(PRegS(IDX(p_mask)), p_all / T_z, z_tmp, 0);

    blend_with_mask(vmm_aux2, vmm_src);

    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_aux2)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector_fwd(
        const TRegS &vmm_src) {
    // Save src data on stack for later usage
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->str(ZReg(IDX(vmm_src)), ptr(h->X_TMP_0));
    // x*alpha
    h->fmul(vmm_src, vmm_src, ZRegS(IDX(table_val(alpha, z_tmp))));
    // sigmoid(x*alpha)
    logistic_compute_vector_fwd(vmm_src);
    // x*sigmoid(alpha*x)
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->ldr(ZReg(IDX(vmm_aux0)), ptr(h->X_TMP_0));
    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    h->fmul(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector_fwd(
        const TRegS &vmm_src) {

    const auto &t0 = ZRegS(IDX(vmm_src));
    const auto &t1 = ZRegS(IDX(vmm_aux1));
    const auto &t2 = ZRegS(IDX(vmm_aux2));
    const auto &t3 = ZRegS(IDX(vmm_aux3));
    const auto &t4 = ZRegS(IDX(vmm_aux4));
    const auto &mask = p_tmp0.s;
    const auto &wt0 = h->W_TMP_0;
    const auto &xt0 = h->X_TMP_0;
    auto set_imm = [&](const ZRegS &dst, uint32_t imm) {
        h->mov_imm(wt0, imm);
        h->cpy(dst, p_all, wt0);
        return dst;
    };
    Label tbl1L, tbl2L, exitL;
    const size_t tblL = 5;
    const size_t tblN = 1 << tblL;
    union fi {
        float f;
        uint32_t i;
    };
    //h->brk(0);
    h->mov(t4, p_all, t0);
    h->fmul(t0, t0, set_imm(z_tmp, float2int(std::sqrt(2))));
    set_imm(t3, 127 << 23);
    h->sub(t1, t0, t3);
    h->asr(t1, t1, 23); // n
    h->scvtf(t1, p_all, t1); // int -> float
    h->and_(t0, p_all, set_imm(z_tmp, 0x7fffff));
    h->asr(t2, t0, 23 - tblL); // d
    h->lsl(t2, t2, 2); // d *= 4
    h->orr(t0, p_all, t3); // y
    h->fmul(t0, t0, set_imm(z_tmp, float2int(1 / std::sqrt(2))));
    h->adr(xt0, tbl1L);
    h->ld1w(t3, p_all, ptr(xt0, t2, SXTW)); // f
    h->fcpy(z_tmp, p_all, 1.0f);
    h->fnmsb(t0, p_all, t3, z_tmp); // y = y * f - 1
    h->adr(xt0, tbl2L);
    h->ld1w(t2, p_all, ptr(xt0, t2, SXTW)); // h
    h->fsub(t3, t4, z_tmp); // x-1
    set_imm(z_tmp, float2int(1.0 / 32));
    h->facge(mask, p_all, z_tmp, t3); // 1/32 >= abs(x-1)
    h->mov(t0, mask, t3);
    h->eor(t2, mask, t2);
    h->fnmsb(t1, p_all, set_imm(z_tmp, float2int(std::log(2))),
            t2); // x = n * log2 - h
    h->movprfx(t2, p_all, set_imm(z_tmp, float2int(1.0f / 3)));
    h->fcpy(z_tmp, p_all, -0.5f);
    h->fmad(t2, p_all, t0, z_tmp); // f
    h->fcpy(z_tmp, p_all, 1.0f);
    h->fmad(t2, p_all, t0, z_tmp); // f * y + 1
    h->fmad(t0, p_all, t2, t1); // y * f + x
    // check nan/inf
    h->fcmlt(mask, p_all, t4, 0.0f); // neg
    h->mov(wt0, 0x7fc00000); // qnan
    h->cpy(t0, mask, wt0);
    h->fcmeq(mask, p_all, t4, 0.0f); // = 0
    h->mov(wt0, 0xff800000); // -Inf
    h->cpy(t0, mask, wt0);

    h->b(exitL);
    h->L(tbl1L);
    const float *tbl1Addr = (const float *)h->getCurr();
    for (size_t i = 0; i < tblN; i++) {
        fi fi;
        fi.i = (127 << 23) | (i << (23 - tblL));
        fi.f = std::sqrt(2) / fi.f;
        h->dd(fi.i);
    }
    h->L(tbl2L);
    for (size_t i = 0; i < tblN; i++) {
        fi fi;
        fi.f = std::log(tbl1Addr[i]);
        h->dd(fi.i);
    }
    h->L(exitL);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_erf_compute_vector_fwd(
        const TRegS &vmm_src) {
    // Here we approximate erf(x) using the expression by
    // Abramowitz and Stegun from ``Handbook of Mathematical
    // Functions''
    // NOTE: The performance of this kernel can be further improved
    // with a minimax polynomialial expansion, thereby avoiding division
    // and exp. However, so far, this has costed larger accuracy
    // differences with respect to glibc erf based GELU, in particular
    // ~1.0e-5 -- 1.0e-3 absolute error at s = -5.

    // x = s / sqrt(2)
    h->fmul(vmm_src, vmm_src,
            ZRegS(IDX(table_val(gelu_erf_one_over_sqrt_two, z_tmp))));

    // IMPORTANT: we use vmm_aux3 to save `x` as exp_compute does not use it.
    h->mov(ZRegD(IDX(vmm_aux3)), ZRegD(IDX(vmm_src)));

    // -exp(-x*x)
    h->fmul(vmm_src, vmm_src, vmm_src);
    h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));

    exp_compute_vector_fwd(vmm_src);
    h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));

    // get sign
    h->mov(ZRegD(IDX(vmm_aux0)), ZRegD(IDX(vmm_aux3)));
    h->and_(ZRegD(IDX(vmm_aux0)), ZRegD(IDX(vmm_aux0)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));

    // abs(x)
    h->mov(ZRegD(IDX(vmm_aux1)), ZRegD(IDX(vmm_aux3)));
    abs_compute_vector_fwd(vmm_aux1);

    // t = 1 / (p*x + 1)
    h->mov(ZRegD(IDX(vmm_aux2)),
            ZRegD(IDX(table_val(gelu_erf_approx_const, z_tmp))));
    h->fmad(vmm_aux2, p_all / T_m, vmm_aux1, ZRegS(IDX(table_val(one, z_tmp))));

    h->mov(ZRegD(IDX(vmm_aux4)), ZRegD(IDX(table_val(one, z_tmp))));
    h->fdiv(vmm_aux4, p_all, vmm_aux2);

    // -exp(-x*x)*t
    h->fmul(vmm_src, vmm_src, vmm_aux4);

    // compute polynomialial r
    h->mov(ZRegD(IDX(vmm_aux1)), ZRegD(IDX(table_val(gelu_erf_pol, z_tmp, 4))));

    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 3))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 2))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 1))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 0))));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->fmad(vmm_src, p_all / T_m, vmm_aux1, ZRegS(IDX(table_val(one, z_tmp))));
    h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_aux0)));

    // S = 0.5 * s = x / sqrt^2(2)
    h->fmul(vmm_aux3, vmm_aux3,
            ZRegS(IDX(table_val(gelu_erf_one_over_sqrt_two, z_tmp))));
    // GELU = 0.5 * s * (1 + erf) = S + S * erf
    h->fmad(vmm_src, p_all / T_m, vmm_aux3, vmm_aux3);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector_bwd(
        const TRegS &vmm_src) {
    h->fcmgt(p_mask.s, p_all / T_z, vmm_src, 0.f);
    h->mov(ZRegD(vmm_src.getIdx()), ZRegD(z_tmp.getIdx()));
    h->fmov(vmm_src, p_mask / T_m, 1.f);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector_bwd(
        const TRegS &vmm_src) {
    if (!use_dst_) {
        // R = exp(s)
        exp_compute_vector_fwd(vmm_src);
        // after exponentiation, get mask by comparing with exp(0)=1.f, not 0.f
        compute_cmp_mask(vmm_src, table_val(one, z_tmp), _cmp_gt_os);
        // R * alpha, then blend with 1.f
        h->fmul(vmm_src, vmm_src, ZRegS(IDX(table_val(alpha, z_tmp))));
    } else {
        // get mask of `d` > 0
        compute_cmp_mask(vmm_src, table_val(zero, z_tmp), _cmp_gt_os);
        // R = `d` + alpha, then blend with 1.f
        h->fadd(vmm_src, vmm_src, ZRegS(IDX(table_val(alpha, z_tmp))));
    }
    blend_with_mask(vmm_src, table_val(one, z_tmp));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = 1 - d^2 = 1 - tanh^2(s)
    if (!use_dst_) tanh_compute_vector_fwd(vmm_src);
    h->mov(ZRegD(IDX(vmm_aux0)), ZRegD(IDX(table_val(one, z_tmp))));

    h->fmls(vmm_aux0, p_all / T_m, vmm_src, vmm_src);

    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_aux0)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_tanh_compute_vector_bwd(
        const TRegS &vmm_src) {
    h->mov(ZRegD(IDX(vmm_aux0)), ZRegD(IDX(vmm_src)));

    // compute G1(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x^2)
    // compute G2(x) = sqrt_root_two_over_pi * x * (1 + 3 * fitting_const * x^2)
    h->fmul(vmm_src, vmm_src, vmm_src);

    // keep G2 in a separate register
    h->mov(ZRegD(IDX(vmm_aux2)),
            ZRegD(IDX(table_val(gelu_tanh_fitting_const_times_three, z_tmp))));
    h->fmad(vmm_aux2, p_all / T_m, vmm_src, ZRegS(IDX(table_val(one, z_tmp))));

    h->mov(ZRegD(IDX(vmm_aux1)),
            ZRegD(IDX(table_val(gelu_tanh_fitting_const, z_tmp))));
    h->fmad(vmm_src, p_all / T_m, vmm_aux1, ZRegS(IDX(table_val(one, z_tmp))));
    h->fmul(vmm_aux0, vmm_aux0,
            ZRegS(IDX(table_val(gelu_tanh_sqrt_two_over_pi, z_tmp))));
    h->fmul(vmm_src, vmm_src, vmm_aux0);
    h->fmul(vmm_aux2, vmm_aux2, vmm_aux0);

    // save G2 on stack as tanh uses all available registers
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->str(ZReg(IDX(vmm_aux2)), ptr(h->X_TMP_0));

    // T = tanh(G1(x))
    tanh_compute_vector_fwd(vmm_src);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->ldr(ZReg(IDX(vmm_aux2)), ptr(h->X_TMP_0));
    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    // compute 0.5 * (1 + T) * (1 + G2 * (1 - T))
    // 1) R = G2 * (1 - T) = G2 - G2 * T
    h->fmls(vmm_aux2, p_all / T_m, vmm_aux2, vmm_src);
    // 2) Q = 1 + T
    h->fadd(vmm_src, vmm_src, ZRegS(IDX(table_val(one, z_tmp))));
    // 3) res = Q * (1 + R) = Q + Q * R
    h->fmla(vmm_src, p_all / T_m, vmm_src, vmm_aux2);

    h->fmul(vmm_src, vmm_src, ZRegS(IDX(table_val(half, z_tmp))));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = 2 * s
    h->fmul(vmm_src, p_all / T_m, 2.f);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector_bwd(
        const TRegS &vmm_src) {
    // replace positive values with 1.f
    compute_cmp_mask(vmm_src, table_val(zero, z_tmp), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(one, z_tmp));
    // replace negative values with -1.f
    compute_cmp_mask(vmm_src, table_val(zero, z_tmp), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(minus_one, z_tmp));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = 0.5 / d = 0.5 / sqrt(s)
    if (!use_dst_) sqrt_compute_vector_fwd(vmm_src);
    h->mov(ZRegD(IDX(vmm_aux0)), ZRegD(IDX(table_val(half, z_tmp))));
    h->fdiv(vmm_aux0, p_all, vmm_src);
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_aux0)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector_bwd(
        const TRegS &vmm_src) {
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(table_val(alpha, z_tmp))));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector_bwd(
        const TRegS &vmm_src) {
    // get mask of values > alpha and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(alpha, z_tmp), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(zero, z_tmp));
    // make all negative values zeros
    h->fmov(z_tmp, 0.f);
    h->fmaxnm(vmm_src, p_all, z_tmp);

    // everything bigger than 0.f should be 1.f
    compute_cmp_mask(vmm_src, table_val(zero, z_tmp), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(one, z_tmp));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector_bwd(
        const TRegS &vmm_src) {
    logistic_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = d * (1 - d) = d - d * d; d = logistic(s)
    if (!use_dst_) logistic_compute_vector_fwd(vmm_src);
    // h->uni_vfnmadd231ps(vmm_src, vmm_src, vmm_src); // bless sse41
    h->mov(ZRegD(IDX(vmm_aux0)), ZRegD(IDX(table_val(one, z_tmp))));

    h->fsub(vmm_aux0, vmm_aux0, vmm_src);

    h->fmul(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_bwd(
        const TRegS &vmm_src) {
    if (!use_dst_) exp_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector_bwd(
        const TRegS &vmm_src) {
    // R = alpha * s
    h->fmul(vmm_src, vmm_src, ZRegS(IDX(table_val(alpha, z_tmp))));

    // Save R on stack for later usage
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->str(ZReg(IDX(vmm_src)), ptr(h->X_TMP_0));

    // Q = sigmoid(alpha * s)
    logistic_compute_vector_fwd(vmm_src);

    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->ldr(ZReg(IDX(vmm_aux0)), ptr(h->X_TMP_0));

    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    // compute Q * (1 + R * (1 - Q))
    // T = R * (1 - Q) = R - R * Q
    h->fmls(vmm_aux0, p_all / T_m, vmm_aux0, vmm_src);

    // Q * (1 + T) = Q + Q * T
    h->fmla(vmm_src, p_all / T_m, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = 1 / s
    /* Do not use 1.f, which is a float constant,
       but 1., which is a double constant. */
    h->fmov(z_tmp, 1.);
    h->fdiv(z_tmp, p_all, vmm_src);
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(z_tmp)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector_bwd(
        const TRegS &vmm_src) {
    // set result with 1.
    /* Do not use 1.f, which is a float constant,
       but 1., which is a double constant. */
    h->fmov(vmm_aux1, 1.);

    // get mask of values > beta and blend with 0.f
    h->fcmgt(p_mask.s, p_all / T_z, vmm_src, z_tmp);
    h->mov(vmm_aux1, p_mask / T_m, 0);
    // get mask of values <= alpha and blend with 0.f
    h->fcmle(p_tmp0.s, p_all / T_z, vmm_src, vmm_aux0);
    h->mov(vmm_aux1, p_tmp0 / T_m, 0);

    h->mov(ZRegD(vmm_src.getIdx()), ZRegD(vmm_aux1.getIdx()));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_erf_compute_vector_bwd(
        const TRegS &vmm_src) {
    // R = s / sqrt(2)
    h->fmul(vmm_src, vmm_src,
            ZRegS(IDX(table_val(gelu_erf_one_over_sqrt_two, z_tmp))));

    // Save R on stack for later usage
    h->sub_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->str(ZReg(IDX(vmm_src)), ptr(h->X_TMP_0));

    // Q = exp(-R*R)
    h->fmul(vmm_src, vmm_src, vmm_src);
    h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));
    exp_compute_vector_fwd(vmm_src);

    // T = R / sqrt(pi) * Q
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->ldr(ZReg(IDX(vmm_aux2)), ptr(h->X_TMP_0));
    h->fmul(vmm_aux2, vmm_aux2,
            ZRegS(IDX(table_val(gelu_erf_one_over_sqrt_pi, z_tmp))));
    h->fmul(vmm_aux2, vmm_aux2, vmm_src);

    // -Q
    h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));

    // get sign
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->ldr(ZReg(IDX(vmm_aux0)), ptr(h->X_TMP_0));
    h->and_(ZRegD(IDX(vmm_aux0)), ZRegD(IDX(vmm_aux0)),
            ZRegD(IDX(table_val(sign_mask, z_tmp))));

    // abs(x)
    h->add_imm(h->X_TMP_0, h->X_SP, 0, h->X_TMP_1);
    h->ldr(ZReg(IDX(vmm_aux1)), ptr(h->X_TMP_0));
    h->add_imm(h->X_SP, h->X_SP, vlen, h->X_TMP_0);

    abs_compute_vector_fwd(vmm_aux1);

    // W = 1 / (p * s + 1)
    h->mov(ZRegD(IDX(vmm_aux3)),
            ZRegD(IDX(table_val(gelu_erf_approx_const, z_tmp))));
    h->mov(ZRegD(IDX(vmm_aux4)), ZRegD(IDX(table_val(one, z_tmp))));
    h->fmad(vmm_aux3, p_all / T_m, vmm_aux1, vmm_aux4);
    h->fdiv(vmm_aux4, p_all, vmm_aux3);

    // Q * W
    h->fmul(vmm_src, vmm_src, vmm_aux4);

    // compute polynomial r
    h->mov(ZRegD(IDX(vmm_aux1)), ZRegD(IDX(table_val(gelu_erf_pol, z_tmp, 4))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 3))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 2))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 1))));
    h->fmad(vmm_aux1, p_all / T_m, vmm_aux4,
            ZRegS(IDX(table_val(gelu_erf_pol, z_tmp, 0))));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->fmad(vmm_src, p_all / T_m, vmm_aux1, ZRegS(IDX(table_val(one, z_tmp))));
    h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_aux0)));

    // P = T + 0.5
    h->fadd(vmm_aux2, vmm_aux2, ZRegS(IDX(table_val(half, z_tmp))));
    // res = P + 0.5 * erf
    h->fmla(vmm_aux2, p_all / T_m, vmm_src, ZRegS(IDX(table_val(half, z_tmp))));
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_aux2)));
}

template <cpu_isa_t isa>
size_t jit_uni_eltwise_injector_f32<isa>::aux_gprs_count() {
    using namespace alg_kind;
    switch (alg_) {
        case eltwise_tanh_use_dst_for_bwd:
        case eltwise_tanh:
        case eltwise_gelu_tanh: return 0;
        default: return 0;
    }
    return 0;
};

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::round_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->frintn(vmm_src, p_all / T_m, vmm_src);
}

template <cpu_isa_t isa>
size_t jit_uni_eltwise_injector_f32<isa>::aux_vecs_count() {
    using namespace alg_kind;

    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return (alpha_ == 0.f) ? 1 : 3;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: return 6; /* = exp + 2 */
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh: return 9;
            case eltwise_square: return 0;
            case eltwise_abs: return 1;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return 0;
            case eltwise_linear: return 2;
            case eltwise_bounded_relu: return 1;
            case eltwise_soft_relu: return 5;
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic: return 5; /* = exp + 1 */
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp: return 4;
            case eltwise_gelu_tanh: return 9; /* = tanh */
            case eltwise_swish: return 6; /* = logistic */
            case eltwise_log: return 6;
            case eltwise_clip: return 2;
            case eltwise_gelu_erf: return 6;
            case eltwise_round: return 0;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return 1;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: return 4; /* = exp */
            case eltwise_tanh_use_dst_for_bwd: return 2;
            case eltwise_tanh: return 9;
            case eltwise_square: return 1;
            case eltwise_abs: return 1;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return 2;
            case eltwise_linear: return 1;
            case eltwise_bounded_relu: return 1;
            case eltwise_soft_relu: return 5; /* = logistic */
            case eltwise_logistic_use_dst_for_bwd: return 2;
            case eltwise_logistic: return 5; /* = logistic */
            case eltwise_exp_use_dst_for_bwd: return 0;
            case eltwise_exp: return 4; /* = exp */
            case eltwise_gelu_tanh: return 9; /* = tanh */
            case eltwise_swish: return 6; /* = logistic */
            case eltwise_log: return 1;
            case eltwise_clip: return 3;
            case eltwise_gelu_erf: return 6;
            default: assert(!"unsupported eltwise algorithm");
        }
    }

    return 0;
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_body(
        const injector_utils::vmm_index_set_iterator_t &start_idx_it,
        const injector_utils::vmm_index_set_iterator_t &end_idx_it) {
    using namespace alg_kind;
    std::for_each(start_idx_it, end_idx_it, [&](size_t idx) {
        if (is_fwd_) {
            switch (alg_) {
                case eltwise_relu_use_dst_for_bwd:
                case eltwise_relu:
                    if (alpha_ == 0.f)
                        relu_zero_ns_compute_vector_fwd(TRegS(idx));
                    else
                        relu_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu: elu_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_square:
                    square_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_abs: abs_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_sqrt_use_dst_for_bwd:
                case eltwise_sqrt: sqrt_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_swish: swish_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_linear:
                    linear_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_bounded_relu:
                    bounded_relu_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                    logistic_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp: exp_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_gelu_tanh:
                    gelu_tanh_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_log: log_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_clip: clip_compute_vector_fwd(TRegS(idx)); break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_fwd(TRegS(idx));
                    break;
                case eltwise_round: round_compute_vector_fwd(TRegS(idx)); break;
                default: assert(!"unsupported eltwise algorithm");
            }
        } else {
            switch (alg_) {
                case eltwise_relu_use_dst_for_bwd:
                case eltwise_relu: relu_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu: elu_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_square:
                    square_compute_vector_bwd(TRegS(idx));
                    break;
                case eltwise_abs: abs_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_sqrt_use_dst_for_bwd:
                case eltwise_sqrt: sqrt_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_linear:
                    linear_compute_vector_bwd(TRegS(idx));
                    break;
                case eltwise_bounded_relu:
                    bounded_relu_compute_vector_bwd(TRegS(idx));
                    break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_bwd(TRegS(idx));
                    break;
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                    logistic_compute_vector_bwd(TRegS(idx));
                    break;
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp: exp_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_gelu_tanh:
                    gelu_tanh_compute_vector_bwd(TRegS(idx));
                    break;
                case eltwise_swish: swish_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_log: log_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_clip: clip_compute_vector_bwd(TRegS(idx)); break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_bwd(TRegS(idx));
                    break;
                default: assert(!"unsupported eltwise algorithm");
            }
        }
        if (scale_ != 1.f) {
            h->fmul(ZRegS(IDX(TRegS(idx))), ZRegS(IDX(TRegS(idx))),
                    ZRegS(IDX(table_val(scale, z_tmp))));
        }
    });
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    injector_utils::vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    const auto &start_idx_it = vmm_idxs.begin();
    const auto &end_idx_it = vmm_idxs.end();
    assert(*start_idx_it < *vmm_idxs.rbegin() + 1
            && *vmm_idxs.rbegin() <= vecs_count);

    injector_preamble(vmm_idxs);
    compute_body(start_idx_tail, end_idx_it);
    injector_preamble_tail(start_idx_it);
    compute_body(start_idx_it, start_idx_tail);
    injector_postamble();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::prepare_table(bool gen_table) {
    if (!gen_table) return;

    h->align(64);
    h->L(l_table);

    // Assumption: entries can be inserted with dd, so they should be 4 bytes.
    assert(sizeof(table_entry_val_t) == 4);

    // Assumption: iterating on entry_map_ here has the same order as
    // when we set the offsets. We verify that in asserts.
    // table_entry_val_t is assumed to be 32 bits
#ifndef NDEBUG
    size_t off = 0;
    key_t curr_key = undef_key;
    int key_occurences = 0;
#endif

    // Run through the map and insert values stored there
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        const auto &te = (*it).second; // get map entry for a given key
        const auto len = te.bcast ? vlen : sizeof(table_entry_val_t);
        /*        for (size_t d = 0; d < len; d += sizeof(table_entry_val_t))
            h->dd(te.val);*/
        for (size_t d = 0; d < len; d += sizeof(table_entry_val_t))
            h->dd(te.val);

#ifndef NDEBUG
        // we check that the precomputed offsets match the registered ones
        const auto &key = (*it).first; // get map entry key
        if (key != curr_key) {
            curr_key = key;
            key_occurences = 0;
        }
        key_occurences++;
        auto expected_off = table_off(key, key_occurences - 1);
        assert(off == expected_off);
        MAYBE_UNUSED(expected_off);
        off += len;
#endif
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::register_table_entries() {
    // This function is responsible to pick all necessary constants
    // for a given algorithm, compute right offset for them to be used
    // in table_val() and save the hexadecimal value of them, which
    // will be finally used in prepare_table(). We rely on fact that
    // the map iterator order is deterministic for a fixed map.

    // common values used in several algorithms
    static const table_t common_values {{zero, {0x00000000, true}},
            {half, {0x3f000000, true}}, {one, {0x3f800000, true}},
            {two, {0x40000000, true}}, {minus_one, {0xbf800000, true}},
            {minus_two, {0xc0000000, true}}, {ln2f, {0x3f317218, true}},
            {positive_mask, {0x7fffffff, true}},
            {sign_mask, {0x80000000, true}},
            {exponent_bias, {0x0000007f, true}}};

    // exp(x) constants
    static const table_t exp_consts {{exp_log2ef, {0x3fb8aa3b, true}},
            {exp_ln_flt_max_f, {0x42b17218, true}},
            {exp_ln_flt_min_f, {0xc2aeac50, true}}};

    // exp(x) polynomial approximation
    static const table_t exp_polynomial {
            {exp_pol, {0x3f7ffffb, true}}, // p1 = 0.999999701f
            {exp_pol, {0x3efffee3, true}}, // p2 = 0.499991506f
            {exp_pol, {0x3e2aad40, true}}, // p3 = 0.166676521f
            {exp_pol, {0x3d2b9d0d, true}}, // p4 = 0.0418978221f
            {exp_pol, {0x3c07cfce, true}} // p5 = 0.00828929059f
    };
    // exp(x) constants2
    static const table_t exp_consts2 {
            {exp_coeff1, {0x3f31721c, true}},
            {exp_coeff2, {0x3e772df2, true}},
            {exp_not_mask17, {~((1u << 17) - 1), true}},
    };

    // tanh(x) constants for four interval approximation
    static const table_t tanh_consts {
            {tanh_range, {0x3d4ccccd, true}},
            {tanh_m1d3, {0xbeaaaaab, true}},
    };

    // soft_relu(x) constants
    static const table_t soft_relu_consts {
            {soft_relu_one_twenty_six, {0x42fc0000, true}},
            {soft_relu_mantissa_sign_mask, {0x807fffff, true}},
    };

    // soft_relu ln(1 + x) polynomial approximation
    static const table_t soft_relu_polynomial {
            {soft_relu_pol, {0xb2b4637d, true}}, // p0 = 0.0000000244f
            {soft_relu_pol, {0x3f7fff8e, true}}, // p1 = 0.9999976971f
            {soft_relu_pol, {0xbf001759, true}}, // p2 = -0.5002478215f
            {soft_relu_pol, {0x3ea70608, true}}, // p3 = 0.3272714505f
            {soft_relu_pol, {0xbea3d7bf, true}}, // p4 = -0.3153830071f
            {soft_relu_pol, {0xbe361d04, true}}, // p5 = -0.1701777461f
            {soft_relu_pol, {0xbfa8f1e6, true}}, // p6 = -1.3254635147f
            {soft_relu_pol, {0xbfe1e812, true}}, // p7 = -1.7971917960f
            {soft_relu_pol, {0xbfc4d30e, true}}, // p8 = -1.5652673123f
    };

    // gelu_tanh(x) constants (formula defined)
    static const table_t gelu_tanh_consts {
            {gelu_tanh_fitting_const, {0x3d372713, true}},
            {gelu_tanh_fitting_const_times_three, {0x3e095d4f, true}},
            {gelu_tanh_sqrt_two_over_pi, {0x3f4c422a, true}},
    };

    // gelu_erf(x) constants (formula defined)
    static const table_t gelu_erf_consts {
            {gelu_erf_approx_const, {0x3ea7ba05, true}},
            {gelu_erf_one_over_sqrt_two, {0x3f3504f3, true}},
            {gelu_erf_one_over_sqrt_pi, {0x3f106eba, true}},
    };

    // gelu_erf(x) polynomial approximation
    static const table_t gelu_erf_polynomial {
            {gelu_erf_pol, {0x3e827906, true}}, // p1 = 0.254829592f
            {gelu_erf_pol, {0xbe91a98e, true}}, // p2 = -0.284496736f
            {gelu_erf_pol, {0x3fb5f0e3, true}}, // p3 = 1.421413741f
            {gelu_erf_pol, {0xbfba00e3, true}}, // p4 = -1.453152027f
            {gelu_erf_pol, {0x3f87dc22, true}}, // p5 = 1.061405429f
    };

    // This object takes care about which constants and polynomials to include.
    struct need_t {
        need_t(alg_kind_t alg) {
            using namespace alg_kind;
            switch (alg) {
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu:
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp:
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                case eltwise_swish: exp_ = true; break;
                case eltwise_gelu_erf: gelu_erf_ = true; break;
                case eltwise_gelu_tanh:
                    exp_ = true;
                    gelu_tanh_ = true;
                    break;
                case eltwise_log: log_ = true; break;
                case eltwise_soft_relu: soft_relu_ = true; break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh:
                    exp_ = true;
                    tanh_ = true;
                    break;
                default: break;
            }
        }

        bool exp_ = false;
        bool tanh_ = false;
        bool soft_relu_ = false;
        bool gelu_tanh_ = false;
        bool gelu_erf_ = false;
        bool log_ = false;

        bool exp() const { return exp_ || soft_relu_ || gelu_erf_; }
        bool tanh() const { return tanh_ || gelu_tanh_; }
        bool soft_relu() const { return soft_relu_; }
        bool gelu_tanh() const { return gelu_tanh_; }
        bool gelu_erf() const { return gelu_erf_; }
        bool log() const { return log_; }
    };

    need_t need(alg_);

    auto push_arg_entry_of = [&](const key_t key, const table_entry_val_t val,
                                     const bool broadcast) {
        mapped_table_entry_t te {0, val, broadcast};
        entry_map_.insert(std::make_pair(key, te));
    };

    auto push_entries_of = [&](const table_t &t) {
        for (auto it = t.begin(); it != t.end(); it++) {
            auto key = (*it).first;
            auto te = (*it).second; // copy values from table
            push_arg_entry_of(key, te.val, te.bcast);
        }
    };

    push_arg_entry_of(scale, float2int(scale_), true);
    push_arg_entry_of(alpha, float2int(alpha_), true);
    push_arg_entry_of(beta, float2int(beta_), true);
    push_entries_of(common_values);
    if (need.exp()) push_entries_of(exp_consts);
    if (need.exp()) push_entries_of(exp_polynomial);
    if (need.exp()) push_entries_of(exp_consts2);
    if (need.tanh()) push_entries_of(tanh_consts);
    if (need.soft_relu()) push_entries_of(soft_relu_consts);
    if (need.soft_relu()) push_entries_of(soft_relu_polynomial);
    if (need.gelu_tanh()) push_entries_of(gelu_tanh_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_polynomial);

    // Now that we registered the entries, we set the offsets.  No
    // entries should be registered after this point.  This allows to
    // expect the same order when injecting the table entries in
    // prepare_table.
    size_t off = 0;
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        auto &te = (*it).second;
        te.off = off;
        off += te.bcast ? vlen : sizeof(table_entry_val_t);
    }
}

template struct jit_uni_eltwise_injector_f32<sve_512>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
