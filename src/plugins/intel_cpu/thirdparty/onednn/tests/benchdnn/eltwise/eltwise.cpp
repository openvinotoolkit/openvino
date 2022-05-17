/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/compare.hpp"

#include "binary/binary.hpp"
#include "eltwise/eltwise.hpp"

namespace eltwise {

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &epd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_eltwise_desc_t ed;
    dnnl_memory_desc_t data_d;

    SAFE(init_md(&data_d, prb->ndims, prb->dims.data(), prb->dt, prb->tag),
            CRIT);

    dnnl_alg_kind_t alg = attr_t::post_ops_t::kind2dnnl_kind(prb->alg);

    if (prb->dir & FLAG_FWD) {
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;

        DNN_SAFE(dnnl_eltwise_forward_desc_init(
                         &ed, prop, alg, &data_d, prb->alpha, prb->beta),
                WARN);
    } else {
        dnnl_memory_desc_t diff_data_d;
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_data_d, prb->ndims,
                         prb->dims.data(), prb->dt, dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_eltwise_backward_desc_init(&ed, alg, &diff_data_d,
                         &data_d, prb->alpha, prb->beta),
                WARN);
    }

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr, prb->ndims, prb->dims.data());
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&epd, &ed, dnnl_attr, engine, nullptr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    res->impl_name = query_impl_info(epd);
    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        return res->state = SKIPPED, res->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(
                5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    }

    SAFE(check_pd_w_and_wo_attr(res, prb->attr, ed), WARN);

    return OK;
}

static bool check_abs_err(const prb_t *prb, const float &s, const float &trh) {
    const float approx_machine_eps = 2 * epsilon_dt(dnnl_f32);
    const float comp_err = approx_machine_eps / trh;

    switch (prb->alg) {
        case alg_t::ELU:
        case alg_t::ELU_DST:
            // catch catastrophic cancellation when (exp(s) - 1), s < 0 and
            // s is close to zero.
            return (prb->dir & FLAG_FWD) && std::signbit(s)
                    && (fabsf(expf(s) - 1.f) <= comp_err);
        case alg_t::GELU_TANH: {
            // catch catastrophic cancellation
            // (4.f is magic scale for f32)
            const float sqrt_2_over_pi = 0.797884;
            const float fitting_const = 0.044715;
            float v = tanhf(sqrt_2_over_pi * s * (1 + fitting_const * s * s));
            float dg = sqrt_2_over_pi * (1 + 3 * fitting_const * s * s);
            if (fabsf(1.f + v) <= comp_err) return true;
            return (prb->dir & FLAG_BWD) && std::signbit(s)
                    && fabsf(1.f + s * (1.f - v) * dg) <= 4.f * comp_err;
        }
        case alg_t::GELU_ERF: {
            // Catch catastrophic cancellation
            // which occurs at large negative s.
            // Factor 2 (in bwd) is to account for the fact that error is
            // accumulated for each summand (except the 1) when they
            // are of the same order of magnitude.
            const float sqrt_2_over_2 = 0.707106769084930419921875f;
            const float two_over_sqrt_pi = 1.12837922573089599609375f;
            float v = s * sqrt_2_over_2;
            if (prb->dir & FLAG_FWD)
                return fabsf(1.f + erff(v)) <= comp_err;
            else
                return fabsf(1.f + erff(v)
                               + v * two_over_sqrt_pi * expf(-v * v))
                        <= comp_err * 2;
        }
        case alg_t::TANH:
            // catch catastrophic cancellation, which occurs when err in tanh(s)
            // is high and tanh(s) is close to 1.
            return (prb->dir & FLAG_BWD) && (1.f - tanhf(fabsf(s))) <= comp_err;
        case alg_t::TANH_DST: // sse41 can't do fma
            // catch catastrophic cancellation, which occurs when err in tanh(s)
            // is high and tanh(s) is close to 1.
            return (prb->dir & FLAG_BWD) && (1.f - s * s) <= comp_err;
        case alg_t::SRELU:
            // when s is negative, expf(s) -> 0 rapidly
            // which leads to log1pf(expf(s)) -> 0
            // which leads to high relative error,
            // while abs error is still low.
            // (10.f is magic scale for bf16)
            return (prb->dir & FLAG_FWD) && std::signbit(s)
                    && log1pf(expf(s)) <= 10.f * comp_err;
        case alg_t::LOGSIGMOID:
            // same situation like in SRELU
            // in logsigmoid when s is positive
            // results -> 0
            return (prb->dir & FLAG_FWD) && !std::signbit(s)
                    && log1pf(expf(-s)) <= 10.f * comp_err;
        case alg_t::MISH:
            // same situation like in SRELU
            return (prb->dir & FLAG_FWD) && std::signbit(s)
                    && s * tanh(log1pf(expf(s))) <= 10.f * comp_err;
        case alg_t::LOGISTIC:
            // when s >= 4, logistic(s) -> 0 rapidly, which leads to high
            // relative error of logistic(s) * (1 - logistic(s)) due to
            // catastrohic cancellation.
            return (prb->dir & FLAG_BWD) && !std::signbit(s)
                    && (1.f / (1.f + expf(s))) <= comp_err;
        case alg_t::SWISH: {
            // catch cancellation happening when W(s) ~~ -1 in (1 + W(s))
            // formula part on backward.
            const float alpha_s = prb->alpha * s;
            return (prb->dir & FLAG_BWD)
                    && (alpha_s * (1.f - 1.f / (1.f + expf(-alpha_s)))
                            <= comp_err);
        }
        default: return false;
    }
}

float get_eltwise_threshold(dnnl_data_type_t dt, alg_t alg, bool is_fwd) {
    // Tolerate only rounding error (1 ulp) for other than fp32 precisions.
    float trh = dt == dnnl_f32 ? 4e-6 : epsilon_dt(dt);
    // Tolerate bigger compute errors for complex algorithms.
    const bool alg_has_higher_tolerance = alg == alg_t::GELU_TANH
            || alg == alg_t::ELU || alg == alg_t::SWISH || alg == alg_t::TANH
            || alg == alg_t::SRELU || alg == alg_t::LOGSIGMOID
            || alg == alg_t::MISH || alg == alg_t::LOG
            || ((alg == alg_t::ELU_DST || alg == alg_t::TANH_DST) && is_fwd);
    if (dt == dnnl_f32 && alg_has_higher_tolerance) trh = 4e-5;
    return trh;
}

static float get_eltwise_zero_trust_percent(const prb_t *prb) {
    float ztp = 60.f; // default for eltwise due to filling.
    switch (prb->alg) {
        case alg_t::LINEAR:
            if (prb->alpha == 0) ztp = 100.f;
            break;
        case alg_t::BRELU:
            if ((prb->alpha == 0) || (prb->dir & FLAG_BWD)) ztp = 100.f;
            break;
        case alg_t::CLIP:
        case alg_t::CLIP_V2:
        case alg_t::CLIP_V2_DST:
            if ((prb->alpha == 0 && prb->beta == 0) || (prb->dir & FLAG_BWD))
                ztp = 100.f;
            break;
        case alg_t::POW:
            if (prb->alpha == 0 || ((prb->dir & FLAG_BWD) && prb->beta == 0))
                ztp = 100.f;
            break;
        default: break;
    }
    // Integral data types with small float values will produce most zeros.
    // u8 with negative alpha will produce only zeros.
    if (is_integral_dt(prb->dt)) ztp = 100.f;
    return ztp;
}

int fill_data(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    dnnl::impl::parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note 1: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // we avoid it for two reasons:
        //   a. it has a complexity in O(idx_start).
        //   b. igen and fgen below might require more than 1 sample
        //   per idx, so the we cannot deterministically compute the
        //   number of states we need to discard
        // Note 2: We also advance the state to avoid having only
        // small values as first chunk input.  The +1 is necessary to
        // avoid generating zeros in first chunk.
        // Note 3: we multiply by kind + 1 to have different values in
        // src/dst and diff_dst. The +1 is to avoid 0 again.
        std::minstd_rand msr((idx_start + 1) * (kind + 1));
        msr.discard(1);
        std::uniform_int_distribution<> igen(0, 10);
        // TODO: 0.09 due to log impl doesn't give good accuracy in 0.99 points
        std::uniform_real_distribution<> fgen(0.f, 0.09f);

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            static constexpr int64_t num_of_generation_variants = 13;
            float value = FLT_MAX;
            switch (idx % num_of_generation_variants) {
                case 0: value = (float)igen(msr); break; // [0-10] pos
                case 1: value = -(float)igen(msr); break; // [0-10] neg
                case 2: value = fgen(msr); break; // [0.-0.1) pos
                case 3: value = -fgen(msr); break; // [0.-0.1) neg
                case 4: value = 10 * (float)igen(msr); break; // [0-100] pos
                case 5: value = -10 * (float)igen(msr); break; // [0-100] neg
                case 6: value = 10.f * fgen(msr); break; // [0.-1.) pos
                case 7: value = -10.f * fgen(msr); break; // [0.-1.) neg
                case 8:
                    value = 88.f + 10.f * fgen(msr);
                    break; // values close to logf(FLT_MAX) for exp alg testing
                case 9:
                    value = 22.f + 10.f * fgen(msr);
                    break; // values close to logf(FLT_MAX)/4.0 for bwd mish alg testing
                case 10:
                    value = 44.f + 10.f * fgen(msr);
                    break; // values close to logf(FLT_MAX)/2.0 for fwd mish alg testing
                case 11: value = prb->alpha; break; // `x = alpha` corner cases
                case 12: value = prb->beta; break; // `x = beta` corner cases
            }
            value = round_to_nearest_representable(prb->dt, value);

            // Hack: -0 may lead to different sign in the answer since input
            // passes through simple reorder which converts -0 into +0.
            if (value == -0.f) value = 0.f;

            mem_fp.set_elem(idx, value);
        }
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common({prb->dt}, prb->dir, res);
    if (res->state == SKIPPED) return;

    bool is_invalid = false;
    switch (prb->alg) {
        case alg_t::CLIP:
        case alg_t::CLIP_V2:
        case alg_t::CLIP_V2_DST: is_invalid = prb->beta < prb->alpha; break;
        case alg_t::BRELU:
        case alg_t::ELU_DST:
        case alg_t::RELU_DST: is_invalid = prb->alpha < 0; break;
        case alg_t::ROUND:
            is_invalid = prb->dt != dnnl_f32 || prb->dir & FLAG_BWD;
            break;
        default: break;
    };
    if (is_invalid) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
        return;
    }

    if (is_nvidia_gpu()) {
        if (!is_nvidia_eltwise_ok(prb->dir, prb->alg, prb->alpha)
                || !prb->attr.post_ops.is_def()) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    check_sum_post_ops(prb->attr, res);
    if (res->state == SKIPPED) return OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(prim, &const_pd), CRIT);

    if (check_mem_size(const_pd) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const bool is_fwd = prb->dir & FLAG_FWD;
    const auto &src_md = q(DNNL_ARG_SRC);
    const auto &dst_md = q(DNNL_ARG_DST);
    const auto &data_md = !is_fwd && prb->use_dst() ? dst_md : src_md;
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);
    const auto &test_engine = get_test_engine();

    dnn_mem_t src_fp(data_md, dnnl_f32, tag::abx, test_engine);
    dnn_mem_t src_dt(data_md, test_engine);

    // we need src_fp for proper comparison, => no in-place reference
    dnn_mem_t dst_fp(data_md, dnnl_f32, tag::abx, test_engine);
    dnn_mem_t placeholder_dst_dt;
    if (!prb->inplace) { placeholder_dst_dt = dnn_mem_t(data_md, test_engine); }
    dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;

    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(binary::setup_binary_po(
                 const_pd, binary_po_args, binary_po_dt, binary_po_fp),
            WARN);

    dnn_mem_t d_dst_dt, placeholder_d_src_dt;

    SAFE(fill_data(prb, SRC, src_dt, src_fp), WARN);

    args_t args;

    dnn_mem_t &arg_fp = !is_fwd && prb->use_dst() ? dst_fp : src_fp;

    // Shouldn't be defined inside since not available when `eltwise_add_check`
    // is invoked due to removed from stack.
    const float trh = get_eltwise_threshold(prb->dt, prb->alg, is_fwd);
    compare::compare_t cmp;
    if (is_bench_mode(CORR)) {
        cmp.set_threshold(trh);
        cmp.set_zero_trust_percent(get_eltwise_zero_trust_percent(prb));

        const auto eltwise_add_check =
                [&](const compare::compare_t::driver_check_func_args_t &args) {
                    // Some algorithms require absolute value comparison for inputs
                    // where catastrophic cancellation may happen.
                    const float src = arg_fp.get_elem(args.idx);
                    if (check_abs_err(prb, src, trh)) return args.diff <= trh;
                    if (prb->attr.post_ops.binary_index() != -1)
                        return args.diff <= trh;
                    return false;
                };
        cmp.set_driver_check_function(eltwise_add_check);
    }

    if (prb->dir & FLAG_FWD) {
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
        args.set(binary_po_args, binary_po_dt);

        SAFE(execute_and_wait(prim, args), WARN);

        if (is_bench_mode(CORR)) {
            compute_ref_fwd(prb, src_fp, binary_po_fp, dst_fp);
            SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
        }
    } else {
        const auto &d_data_md = q(DNNL_ARG_DIFF_DST);

        dnn_mem_t d_dst_fp
                = dnn_mem_t(d_data_md, dnnl_f32, tag::abx, test_engine);
        d_dst_dt = dnn_mem_t(d_data_md, test_engine);

        dnn_mem_t &d_src_fp = d_dst_fp; // in-place reference
        if (!prb->inplace) {
            placeholder_d_src_dt = dnn_mem_t(d_data_md, test_engine);
        }
        dnn_mem_t &d_src_dt = prb->inplace ? d_dst_dt : placeholder_d_src_dt;

        SAFE(fill_data(prb, DST, d_dst_dt, d_dst_fp), WARN);

        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        if (prb->use_dst()) {
            if (is_bench_mode(CORR))
                compute_ref_fwd(prb, src_fp, binary_po_fp, dst_fp);
            SAFE(dst_dt.reorder(dst_fp), WARN);
            // make dst_fp of same values as for bf16, otherwise there are high
            // relative and absolute errors due to initial difference in source
            // values which become worse particularly when (1 - x) is used.
            if (dst_dt.dt() != dst_fp.dt()) SAFE(dst_fp.reorder(dst_dt), WARN);
            args.set(DNNL_ARG_DST, dst_dt);
        } else {
            args.set(DNNL_ARG_SRC, src_dt);
        }
        SAFE(execute_and_wait(prim, args), WARN);

        if (is_bench_mode(CORR)) {
            compute_ref_bwd(prb, arg_fp, d_dst_fp, d_src_fp);
            SAFE(cmp.compare(d_src_fp, d_src_dt, prb->attr, res), WARN);
        }
    }

    return measure_perf(res->timer, prim, args);
}

} // namespace eltwise
