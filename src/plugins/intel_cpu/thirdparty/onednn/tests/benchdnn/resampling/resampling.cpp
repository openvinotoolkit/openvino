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

#include <stdio.h>
#include <stdlib.h>

#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/compare.hpp"

#include "binary/binary.hpp"
#include "resampling/resampling.hpp"

namespace resampling {

int fill_dat(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_fp.nelems();
    const auto dt = mem_dt.dt();
    const int range = 16;
    const int f_min = 0;

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((97 * i) - 19 * kind + 101) % (range + 1);
        const float value = dt == dnnl_f32 || is_integral_dt(dt)
                ? (f_min + gen) * (1.0f + 4.0f / range)
                : (f_min + gen) / range;

        mem_fp.set_elem(i, round_to_nearest_representable(dt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_src(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    return fill_dat(prb, SRC, mem_dt, mem_fp, res);
}

int fill_dst(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    return fill_dat(prb, DST, mem_dt, mem_fp, res);
}

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &rpd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_memory_desc_t src_d, dst_d;

    dnnl_dims_t src_1d_dims = {prb->mb, prb->ic, prb->iw};
    dnnl_dims_t src_2d_dims = {prb->mb, prb->ic, prb->ih, prb->iw};
    dnnl_dims_t src_3d_dims = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};
    dnnl_dim_t *src_dims = prb->ndims == 5
            ? src_3d_dims
            : prb->ndims == 4 ? src_2d_dims : src_1d_dims;

    dnnl_dims_t dst_1d_dims = {prb->mb, prb->ic, prb->ow};
    dnnl_dims_t dst_2d_dims = {prb->mb, prb->ic, prb->oh, prb->ow};
    dnnl_dims_t dst_3d_dims = {prb->mb, prb->ic, prb->od, prb->oh, prb->ow};
    dnnl_dim_t *dst_dims = prb->ndims == 5
            ? dst_3d_dims
            : prb->ndims == 4 ? dst_2d_dims : dst_1d_dims;

    std::string src_tag = (prb->dir & FLAG_FWD) ? prb->tag : tag::any;
    std::string dst_tag = (prb->dir & FLAG_BWD) ? prb->tag : tag::any;

    SAFE(init_md(&src_d, prb->ndims, src_dims, prb->sdt, src_tag), CRIT);

    SAFE(init_md(&dst_d, prb->ndims, dst_dims, prb->ddt, dst_tag), CRIT);

    dnnl_alg_kind_t alg = alg2alg_kind(prb->alg);
    dnnl_resampling_desc_t rd;

    if (prb->dir & FLAG_FWD) {
        auto prop_kind = prb->dir & FLAG_INF ? dnnl_forward_inference
                                             : dnnl_forward_training;
        DNN_SAFE(dnnl_resampling_forward_desc_init(
                         &rd, prop_kind, alg, nullptr, &src_d, &dst_d),
                WARN);
    } else {
        DNN_SAFE(dnnl_resampling_backward_desc_init(
                         &rd, alg, nullptr, &src_d, &dst_d),
                WARN);
    }

    dnnl_primitive_desc_t hint_fwd_pd_ {};
    dnnl_status_t status = dnnl_success;
    if (prb->dir & FLAG_BWD) {
        dnnl_memory_desc_t fwd_src_d, fwd_dst_d;
        SAFE(init_md(&fwd_src_d, prb->ndims, src_dims, prb->sdt, prb->tag),
                CRIT);
        SAFE(init_md(&fwd_dst_d, prb->ndims, dst_dims, prb->ddt, tag::any),
                CRIT);

        dnnl_resampling_desc_t rd_fwd;
        DNN_SAFE(dnnl_resampling_forward_desc_init(&rd_fwd,
                         dnnl_forward_training, alg, nullptr, &fwd_src_d,
                         &fwd_dst_d),
                WARN);

        status = dnnl_primitive_desc_create(
                &hint_fwd_pd_, &rd_fwd, nullptr, engine, nullptr);
        if (status == dnnl_unimplemented) return res->state = UNIMPLEMENTED, OK;
    }
    auto hint_fwd_pd = make_benchdnn_dnnl_wrapper(hint_fwd_pd_);
    SAFE(status, WARN);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr, prb->ndims, dst_dims);
    const auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    status = dnnl_primitive_desc_create(
            &rpd, &rd, dnnl_attr, engine, hint_fwd_pd);

    if (status == dnnl_unimplemented) return res->state = UNIMPLEMENTED, OK;
    SAFE(status, WARN);

    res->impl_name = query_impl_info(rpd);
    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        return res->state = SKIPPED, res->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(
                5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    }

    SAFE(check_pd_w_and_wo_attr(res, prb->attr, rd), WARN);

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common({prb->sdt, prb->ddt}, prb->dir, res);

    if (res->state == SKIPPED) return;

    if (is_nvidia_gpu()) {
        const bool dt_ok = prb->sdt != dnnl_s8 && prb->ddt != dnnl_s8;
        if (prb->ndims == 5 || prb->alg == nearest || !prb->attr.is_def()
                || !dt_ok) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

/* The following issue takes place for integer data types:
 * Sometimes there are differences in the order of operations between
 * the version of the algorithm implemented in the kernel and the reference
 * algorithm. Therefore, this function is especially important if the
 * destination data type is an integer, because when the floating-point
 * type is used to compute the algorithm and if the returned value is very
 * close to x.5, there may be a difference between the output value of
 * reference and the kernel, as one version may round up and the other down.
 * Therefore, we can assume that two values are equal to each other when:
 * - there is a difference in the order of operations,
 * - and the output value of the algorithm is very close to x.5,
 * - and the difference between the output value of reference and expected is 1,
 * - and the output type is an integer type */
void add_additional_check_to_compare(compare::compare_t &cmp) {
    using cmp_args_t = compare::compare_t::driver_check_func_args_t;
    cmp.set_driver_check_function([&](const cmp_args_t &args) -> bool {
        if (!is_integral_dt(args.dt)) return false;
        // Check that original value is close to x.5f
        static constexpr float small_eps = 9e-6;
        if (fabsf((floorf(args.exp_f32) + 0.5f) - args.exp_f32) >= small_eps)
            return false;
        // If it was, check that exp and got values reside on opposite sides of it.
        if (args.exp == floorf(args.exp_f32))
            return args.got == ceilf(args.exp_f32);
        else if (args.exp == ceilf(args.exp_f32))
            return args.got == floorf(args.exp_f32);
        else {
            assert(!"unexpected scenario");
            return false;
        }
    });
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

    const auto &src_md
            = prb->dir == BWD_D ? q(DNNL_ARG_DIFF_SRC) : q(DNNL_ARG_SRC);
    const auto &dst_md
            = prb->dir == BWD_D ? q(DNNL_ARG_DIFF_DST) : q(DNNL_ARG_DST);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_fp(src_md, fp, tag, test_engine);
    dnn_mem_t src_dt(src_md, test_engine);

    dnn_mem_t dst_fp(dst_md, fp, tag, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) >= 0)
        SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    // When post-ops occur, the relative difference can change
    // between the output from reference and the kernel. The compare
    // function usually uses to compare a relative difference.
    // Therefore, we should not lead to a situation where the
    // relative difference is very small after executing a
    // post-ops operation. Therefore, all values for binary post_ops
    // are positive when the linear algorithm is present. This is
    // important because there may be small differences in the result
    // between the expected value and the gotten value with this algorithm.
    const bool only_positive_values = prb->alg == linear;
    SAFE(binary::setup_binary_po(const_pd, binary_po_args, binary_po_dt,
                 binary_po_fp, test_engine, only_positive_values),
            WARN);

    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    args_t args;

    compare::compare_t cmp;
    const bool operations_order_can_be_different = prb->alg == linear;
    if (operations_order_can_be_different) add_additional_check_to_compare(cmp);

    if (prb->dir & FLAG_FWD) {
        SAFE(fill_src(prb, src_dt, src_fp, res), WARN);
        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(binary_po_args, binary_po_dt);

        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(prim, args), WARN);

        if (is_bench_mode(CORR)) {
            compute_ref_fwd(prb, src_fp, dst_fp, binary_po_fp);
            const float linear_trh = epsilon_dt(prb->sdt) > epsilon_dt(prb->ddt)
                    ? epsilon_dt(prb->sdt) // conversion error sdt->ddt
                    : 7 * epsilon_dt(prb->ddt); // algorithm calculation error
            float trh = prb->alg == nearest ? 0.f : linear_trh;

            if (is_nvidia_gpu()) {
                // cuDNN precision is different from ref one due to different
                // computation algorithm used for resampling.
                trh = prb->ddt == dnnl_f16 ? 4e-2 : 2e-5;
            }

            cmp.set_threshold(trh);
            // No sense to test zero trust for upsampling since it produces
            // valid zeros.
            // TODO: validate this once again.
            cmp.set_zero_trust_percent(100.f);
            SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
        }
    } else {
        SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);
        args.set(DNNL_ARG_DIFF_DST, dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, src_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE(execute_and_wait(prim, args), WARN);

        if (is_bench_mode(CORR)) {
            compute_ref_bwd(prb, src_fp, dst_fp);
            const float linear_trh = epsilon_dt(prb->ddt) > epsilon_dt(prb->sdt)
                    ? epsilon_dt(prb->ddt)
                    : 7 * epsilon_dt(prb->sdt);
            float trh = prb->alg == nearest ? 0.f : linear_trh;

            // cuDNN precision is different from ref one due to different
            // computation algorithm used for resampling.
            if (is_nvidia_gpu()) trh = 2e-5;

            cmp.set_threshold(trh);
            // No sense to test zero trust for upsampling since it produces
            // valid zeros.
            // TODO: validate this once again.
            cmp.set_zero_trust_percent(100.f);
            SAFE(cmp.compare(src_fp, src_dt, prb->attr, res), WARN);
        }
    }

    return measure_perf(res->timer, prim, args);
}

} // namespace resampling
