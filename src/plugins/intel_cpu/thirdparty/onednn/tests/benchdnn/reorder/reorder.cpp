/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
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

#include <cstring>

#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/compare.hpp"

#include "reorder.hpp"

namespace reorder {

int get_n_scales(const prb_t *prb) {
    const int mask = attr_t::get_default_mask(prb->attr.oscale.policy);
    assert(IMPLICATION(mask >= (1 << 1), prb->ndims > 1));
    switch (mask) {
        case 0: return 1;
        case (1 << 0): return prb->reorder.dims[0];
        case (1 << 1): return prb->reorder.dims[1];
        case (1 << 1) + (1 << 0):
            return prb->reorder.dims[1] * prb->reorder.dims[0];
        default: assert(!"unsupported mask"); return 1;
    }
}

// Filling for integers is different due to problematic int -> float conversion.
// And it doesn't require many different points to be tested.
int fill_memory_int(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem) {
    const dt_conf_t c_src = prb->conf_in;
    const auto dt = mem.dt();
    const int max = c_src->max;
    const auto nelems = mem.nelems();

    for (int64_t idx = 0; idx < nelems; ++idx) {
        const int gen[4] = {
                (int)max, /* saturate to max of output data type */
                (int)c_src->min, /* saturate to min of output data type */
                (int)0,
                (int)16,
        };

        const int rng = kind == SRC ? (idx % 4) : ((idx * 5 / 4) % 4);
        const int value = gen[rng];
        switch (dt) {
            case dnnl_s32: ((int *)mem)[idx] = value; break;
            case dnnl_s8: ((int8_t *)mem)[idx] = (int8_t)value; break;
            case dnnl_u8: ((uint8_t *)mem)[idx] = (uint8_t)value; break;
            default: assert(!"not expected"); break;
        }
    }

    return OK;
}

int fill_memory_fp(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem) {
    const dt_conf_t c_src = prb->conf_in;
    const int max = c_src->max;
    const int scale_mask = attr_t::get_default_mask(prb->attr.oscale.policy);

    const auto nelems = mem.nelems();

    for (int64_t idx = 0; idx < nelems; ++idx) {
        const int64_t mask_idx = mem.get_scale_idx(idx, scale_mask);
        const float scale = prb->scales[mask_idx];

        const float gen[7] = {
                (float)max, /* saturate to max of output data type */
                (float)c_src->min, /* saturate to min of output data type */
                (float)1.6 / scale, /* rounding check */
                (float)0.2 / scale, /* saturate to 0 */
                (float)1.0 / scale, /* exact multiplication check */
                (float)2.0,
                (float)scale,
        };

        const int rng = kind == SRC ? (idx % 7) : ((idx * 8 / 7) % 7);
        mem.set_elem(idx, gen[rng]);
    }

    return OK;
}

int fill_memory(const prb_t *prb, data_kind_t kind, dnn_mem_t &mem) {
    if (is_integral_dt(mem.dt())) return fill_memory_int(prb, kind, mem);
    return fill_memory_fp(prb, kind, mem);
}

int fill_memory_extra(const prb_t *prb, dnnl_memory_extra_desc_t &extra) {
    extra.flags = dnnl_memory_extra_flag_none;

    if (prb->is_reorder_with_compensation()) {
        for (const auto &i_oflag : prb->oflag) {
            if (i_oflag.first & FLAG_S8S8_COMP) {
                extra.flags |= dnnl_memory_extra_flag_compensation_conv_s8s8;
                extra.compensation_mask = i_oflag.second;

                const float s8_scale_factor = reorder_rescale_factor();
                const bool need_rescale = s8_scale_factor != 1.f;
                if (need_rescale) {
                    extra.flags |= dnnl_memory_extra_flag_scale_adjust;
                    extra.scale_adjust = s8_scale_factor;
                }
            }
            if (i_oflag.first & FLAG_ZP_COMP) {
                extra.flags
                        |= dnnl_memory_extra_flag_compensation_conv_asymmetric_src;
                extra.asymm_compensation_mask = i_oflag.second;
            }
        }
    }

    return OK;
}

int ref_reorder(const prb_t *prb, dnn_mem_t &dst, const dnn_mem_t &src) {
    auto dst_dt = dst.dt();

    const auto nelems = src.nelems();
    const int scale_mask = attr_t::get_default_mask(prb->attr.oscale.policy);
    const int src_zero_point = prb->src_zp ? prb->src_zp[0] : 0;
    const int dst_zero_point = prb->dst_zp ? prb->dst_zp[0] : 0;

    float beta = 0;
    const auto &po = prb->attr.post_ops;
    const int beta_idx = po.find(attr_t::post_ops_t::kind_t::SUM);
    if (beta_idx >= 0) beta = po.entry[beta_idx].sum.scale;

    for (int64_t idx = 0; idx < nelems; ++idx) {
        float s = src.get_elem(idx) - src_zero_point;
        float d = 0;
        if (beta_idx >= 0) d = dst.get_elem(idx) - dst_zero_point;

        const int64_t scale_idx = dst.get_scale_idx(idx, scale_mask);
        const float alpha = prb->scales[scale_idx];
        float value = alpha * s + beta * d + dst_zero_point;
        value = maybe_saturate(dst_dt, value);
        if (dst_dt == dnnl_s32 && value >= (float)INT_MAX)
            value = BENCHDNN_S32_TO_F32_SAT_CONST;

        dst.set_elem(idx, round_to_nearest_representable(dst_dt, value));
    }

    return OK;
}

int compare_bootstrap(dnn_mem_t &mem_ref, dnn_mem_t &mem_got, res_t *res) {
    bool ok = false;
    // demand bit-wise identical results
    const auto size_ref = mem_ref.size();
    if (size_ref == 0) return res->state = PASSED, OK;

    if (size_ref == mem_got.size())
        ok = !std::memcmp((void *)mem_ref, (void *)mem_got, size_ref);

    res->errors = !ok;
    res->state = ok ? PASSED : FAILED;
    res->total = 1;

    return res->state == FAILED ? FAIL : OK;
}

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &rpd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    const auto &rc = prb->reorder;
    auto dims = rc.dims;
    for (int d = 0; d < prb->ndims; ++d)
        if (prb->runtime_dim_mask & (1 << d)) dims[d] = DNNL_RUNTIME_DIM_VAL;

    dnnl_memory_desc_t src_d, dst_d;
    SAFE(init_md(&src_d, prb->ndims, dims.data(), prb->conf_in->dt, rc.tag_in),
            CRIT);
    SAFE(init_md(&dst_d, prb->ndims, dims.data(), prb->conf_out->dt,
                 rc.tag_out),
            CRIT);

    // Prepare and assign extra for dst_md.
    dnnl_memory_extra_desc_t dst_md_extra {};
    fill_memory_extra(prb, dst_md_extra);
    dst_d.extra = dst_md_extra;

    dnnl_engine_t src_engine = engine, dst_engine = engine;
    if (is_gpu()) {
        switch (prb->cross_engine) {
            case CPU2GPU: src_engine = get_cpu_engine(); break;
            case GPU2CPU: dst_engine = get_cpu_engine(); break;
            default: break;
        }
    }

    attr_args_t attr_args;
    attr_args.prepare_output_scales(prb->attr, prb->scales, get_n_scales(prb));
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    dnnl_status_t init_status = dnnl_reorder_primitive_desc_create(
            &rpd, &src_d, src_engine, &dst_d, dst_engine, dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    SAFE(init_status, WARN);

    res->impl_name = query_impl_info(rpd);
    if (maybe_skip(res->impl_name)) {
        BENCHDNN_PRINT(2, "SKIPPED: oneDNN implementation: %s\n",
                res->impl_name.c_str());
        return res->state = SKIPPED, res->reason = SKIP_IMPL_HIT, OK;
    } else {
        BENCHDNN_PRINT(
                5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    }

    if (attr_same_pd_check && !prb->attr.is_def()) {
        dnnl_primitive_desc_t pd_no_attr {};
        dnnl_primitive_attr_t dnnl_empty_attrs {};
        DNN_SAFE(dnnl_reorder_primitive_desc_create(&pd_no_attr, &src_d,
                         src_engine, &dst_d, dst_engine, dnnl_empty_attrs),
                WARN);
        auto pd_no_attr_wrapper = make_benchdnn_dnnl_wrapper(pd_no_attr);
        SAFE(check_same_pd(res, pd_no_attr_wrapper), WARN);
    }

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_NONE
    auto cross_engine = prb->cross_engine;
    if (cross_engine == CPU2GPU || cross_engine == GPU2CPU)
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
#endif

    const auto sdt = prb->conf_in->dt;
    const auto ddt = prb->conf_out->dt;
    check_known_skipped_case_common({sdt, ddt}, FWD_D, res);
    if (res->state == SKIPPED) return;

    // zero points for dst do not support sum by design
    if (!prb->attr.zero_points.is_def(DNNL_ARG_DST)
            && prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) != -1) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    if (prb->is_reorder_with_compensation()) {
        // compensation is supported for dst_dt = s8 so far
        const bool dt_ok = ddt == dnnl_s8;
        // compensation does not support any attributes but oscale
        const bool attr_ok = prb->attr.scales.is_def()
                && prb->attr.zero_points.is_def() && prb->attr.post_ops.is_def()
                && prb->attr.oscale.runtime == false;
        // compensation does not support runtime dims
        const bool rt_ok = prb->runtime_dim_mask == 0;

        if (!dt_ok || !attr_ok || !rt_ok) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }

    // bf16 reorder on cpu supports only bf16/f32 src_dt/dst_dt
    if (is_cpu()
            && (!IMPLICATION(sdt == dnnl_bf16,
                        ddt == dnnl_f32 || ddt == dnnl_bf16 || ddt == dnnl_s8
                                || ddt == dnnl_u8)
                    || !IMPLICATION(ddt == dnnl_bf16,
                            sdt == dnnl_f32 || sdt == dnnl_bf16
                                    || sdt == dnnl_s8 || sdt == dnnl_u8))) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    if (is_nvidia_gpu()) {
        const bool oscale_ok = prb->attr.oscale.policy == policy_t::COMMON;
        if (!oscale_ok) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }

    if (is_gpu()) {
        // GPU does not support run-time dims and zero-points
        if (prb->runtime_dim_mask != 0 || !prb->attr.zero_points.is_def()
                || prb->attr.oscale.runtime) {
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

    //                                       ___________________
    //                                      |                   |
    //                                      | performance timer |
    //                                      |___________________|
    //                                                |
    //   _______________           ______________     V     ________________
    //  |               | oneDNN  |              | oneDNN  |                |
    //  | dt_in fmt_ref |-------->| dt_in fmt_in |-------->| dt_out fmt_out |
    //  |_______________|         |______________|    ^    |________________|
    //           |                                    |            |
    //  benchdnn |<-------------------------------- scales         | oneDNN
    //   ________V_______                                   _______V________
    //  |                |                                 |                |
    //  | dt_out fmt_ref |         <= compare =>           | dt_out fmt_ref |
    //  |________________|                                 |________________|
    //
    // Steps:
    // 1. fill scales
    // 2. create target reorder primitive
    // 3. create memories
    // 4. fill input memory
    // 5. execute oneDNN and benchdnn reorders / q10n
    // 6. compare results
    // 7. performance measurement

    /* Step 2: create target reorder primitive */
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

    /* Step 3: create memories */
    dnnl_memory_desc_t src_md, dst_md;
    if (prb->runtime_dim_mask != 0) {
        // re-create memory descriptors with defined dims
        const auto &rc = prb->reorder;
        SAFE(init_md(&src_md, prb->ndims, rc.dims.data(), prb->conf_in->dt,
                     rc.tag_in),
                CRIT);
        SAFE(init_md(&dst_md, prb->ndims, rc.dims.data(), prb->conf_out->dt,
                     rc.tag_out),
                CRIT);
    } else {
        src_md = q(DNNL_ARG_SRC);
        dst_md = q(DNNL_ARG_DST);
    }
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto tag = tag::abx;
    const auto src_dt = src_md.data_type;
    const auto dst_dt = dst_md.data_type;

    dnnl_engine_t src_engine, dst_engine;
    DNN_SAFE(dnnl_primitive_desc_query(
                     const_pd, dnnl_query_reorder_src_engine, 0, &src_engine),
            WARN);
    DNN_SAFE(dnnl_primitive_desc_query(
                     const_pd, dnnl_query_reorder_dst_engine, 0, &dst_engine),
            WARN);

    dnn_mem_t src_dt_in_fmt_ref(src_md, src_dt, tag, src_engine);
    dnn_mem_t src_dt_in_fmt_in(src_md, src_engine);

    dnn_mem_t scratchpad_dt(scratchpad_md, src_engine);

    dnn_mem_t dst_dt_out_fmt_ref(dst_md, dst_dt, tag, dst_engine);
    dnn_mem_t dst_dt_out_fmt_out(dst_md, dst_engine);

    /* Step 4: fill input memory */
    SAFE(fill_memory(prb, SRC, src_dt_in_fmt_ref), WARN);

    /* Step 5: execute necessary reorders */
    SAFE(src_dt_in_fmt_in.reorder(src_dt_in_fmt_ref), WARN);

    const bool has_sum
            = prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) >= 0;
    if (has_sum) {
        SAFE(fill_memory(prb, DST, dst_dt_out_fmt_ref), WARN);
        SAFE(dst_dt_out_fmt_out.reorder(dst_dt_out_fmt_ref), WARN);
    }

    dnn_mem_t scales, src_zero_points_m, dst_zero_points_m;
    maybe_prepare_runtime_scales(
            scales, prb->attr.oscale, get_n_scales(prb), prb->scales);
    maybe_prepare_runtime_zero_points(
            src_zero_points_m, prb->attr, DNNL_ARG_SRC, 1, prb->src_zp);
    maybe_prepare_runtime_zero_points(
            dst_zero_points_m, prb->attr, DNNL_ARG_DST, 1, prb->dst_zp);

    args_t args;
    args.set(DNNL_ARG_FROM, src_dt_in_fmt_in);
    args.set(DNNL_ARG_TO, dst_dt_out_fmt_out);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
    args.set(DNNL_ARG_ATTR_OUTPUT_SCALES, scales);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_points_m);
    args.set(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_points_m);

    SAFE(execute_and_wait(prim, args), WARN);

    /* Step 6: check correctness */
    if (is_bench_mode(CORR)) {
        if (prb->is_reorder_with_compensation()) {
            /* "bootstrap" approach: compare to another oneDNN reorder. use
             * this when benchdnn does not know about all details of the data
             * layout, as is the case for compensated weights formats. */

            /* Step 5a: oneDNN reorder from ref format to output format */
            dnnl_memory_extra_desc_t dst_extra {};
            fill_memory_extra(prb, dst_extra);
            dnn_mem_t ref_dst_dt_out_fmt_out(dst_md, dst_engine);
            ref_dst_dt_out_fmt_out.md_.extra = dst_extra;

            const_dnnl_primitive_attr_t const_attr;
            DNN_SAFE(dnnl_primitive_desc_get_attr(const_pd, &const_attr), WARN);

            SAFE(ref_dst_dt_out_fmt_out.reorder(src_dt_in_fmt_ref, const_attr),
                    WARN);

            /* Step 5b: compare results (expect bit-wise exactness) */
            SAFE(compare_bootstrap(
                         ref_dst_dt_out_fmt_out, dst_dt_out_fmt_out, res),
                    WARN);
        } else {
            /* (default) "reference" approach: compare to benchdnn reorder */

            /* Step 5b: execute benchdnn reorder */
            SAFE(ref_reorder(prb, dst_dt_out_fmt_ref, src_dt_in_fmt_ref), WARN);

            /* Step 5c: compare benchdnn and oneDNN output */
            using cmp_t = compare::compare_t;
            cmp_t cmp;
            const bool has_s32 = src_md.data_type == dnnl_s32
                    || dst_md.data_type == dnnl_s32;
            const bool has_s8 = src_md.data_type == dnnl_s8
                    || dst_md.data_type == dnnl_s8;
            const bool has_u8 = src_md.data_type == dnnl_u8
                    || dst_md.data_type == dnnl_u8;
            if (has_u8)
                cmp.set_zero_trust_percent(58.f); // 4/7 inputs becomes 0
            else if (has_s32 || has_s8)
                cmp.set_zero_trust_percent(43.f); // 3/7 inputs becomes 0

            // A hack to avoid false-positive result from f32->s32 conversion
            // in case of sum post-op on GPU happening when two max_dt values
            // are summed together.
            using cmp_args_t = compare::compare_t::driver_check_func_args_t;
            const auto reorder_add_check = [&](const cmp_args_t &args) {
                if (args.dt == dnnl_s32 && args.got == max_dt(args.dt)
                        && is_gpu()) {
                    // 128.f = float(INT_MAX)
                    //                - BENCHDNN_S32_TO_F32_SAT_CONST;
                    return args.diff == 128.f;
                }
                return false;
            };
            cmp.set_driver_check_function(reorder_add_check);

            // TODO: enable additional checks for border values validity.
            SAFE(cmp.compare(dst_dt_out_fmt_ref, dst_dt_out_fmt_out, prb->attr,
                         res, dst_engine),
                    WARN);
        }
    }

    /* Step 7: performance measurement */
    return measure_perf(res->timer, prim, args);
}

} // namespace reorder
