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

#include "oneapi/dnnl/dnnl.h"

#include "tests/test_thread.hpp"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/compare.hpp"

#include "binary/binary.hpp"
#include "eltwise/eltwise.hpp"

namespace binary {

//TODO: Consider filling with powers of 2 for division to avoid rounding errors
int fill_mem(int input_idx, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        bool only_positive_values = false, bool only_integer_values = false) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    const auto dt = mem_dt.dt();
    const int range = 16;
    const int f_min = dt == dnnl_u8 ? 0 : -range / 2;

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const int64_t gen = (12 * i + 5 * input_idx + 16) % (range + 1);
        const float scale = only_integer_values ? 1.f : 1.25f;
        float value = (f_min + gen) * scale;
        if (only_positive_values) value = fabs(value);
        // Remove zeroes in src1 to avoid division by zero
        if (input_idx == 1 && value == 0.0f) value = 1.0f;
        mem_fp.set_elem(i, round_to_nearest_representable(dt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int setup_binary_po(const_dnnl_primitive_desc_t pd, std::vector<int> &args,
        std::vector<dnn_mem_t> &mem_dt, std::vector<dnn_mem_t> &mem_fp,
        const dnnl_engine_t &ref_engine, bool only_positive_values,
        bool only_integer_values) {
    // TODO: currently run-time dimensions are not supported in binary post-op.
    // To add a support two ways are possible: 1) add query support to the
    // library and extract expected md from pd; 2) pass a vector of pre-defined
    // (no run-time values) of `po_md`s and create memories from them in case
    // the library will lack of query mechanism.
    const_dnnl_primitive_attr_t const_attr;
    DNN_SAFE(dnnl_primitive_desc_get_attr(pd, &const_attr), WARN);

    const_dnnl_post_ops_t const_attr_po;
    DNN_SAFE(
            dnnl_primitive_attr_get_post_ops(const_attr, &const_attr_po), WARN);

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(pd, dnnl_query_exec_arg_md, index);
    };

    auto po_len = dnnl_post_ops_len(const_attr_po);
    for (int idx = 0; idx < po_len; ++idx) {
        auto kind = dnnl_post_ops_get_kind(const_attr_po, idx);
        if (kind != dnnl_binary) continue;

        int po_idx = DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1;
        const auto &po_md = q(po_idx);

        // Following call can not be executed if po_md has runtime dimension due
        // to undefined size.
        mem_fp.emplace_back(po_md, dnnl_f32, tag::abx, ref_engine);
        mem_dt.emplace_back(po_md, get_test_engine());
        args.push_back(po_idx);
        fill_mem(po_idx, mem_dt.back(), mem_fp.back(), only_positive_values,
                only_integer_values);
    }
    return OK;
}

static int init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &bpd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    dnnl_binary_desc_t bd;
    std::vector<dnnl_memory_desc_t> src_d;
    src_d.resize(prb->n_inputs());

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input) {
        const dims_t &i_sdims = prb->sdims[i_input];
        SAFE(init_md(&src_d[i_input], prb->ndims[i_input], i_sdims.data(),
                     prb->sdt[i_input], prb->stag[i_input]),
                CRIT);
    }

    dnnl_dims_t dst_dims;
    for (int d = 0; d < prb->ndims[0]; ++d)
        dst_dims[d] = std::max(prb->sdims[0][d], prb->sdims[1][d]);

    dnnl_memory_desc_t dst_d;
    SAFE(init_md(&dst_d, prb->ndims[0], dst_dims, prb->ddt, prb->dtag), WARN);

    dnnl_alg_kind_t alg = attr_t::post_ops_t::kind2dnnl_kind(prb->alg);

    DNN_SAFE(dnnl_binary_desc_init(&bd, alg, &src_d[0], &src_d[1], &dst_d),
            WARN);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr, prb->ndims[0], dst_dims);
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&bpd, &bd, dnnl_attr, engine, nullptr);

    if (init_status == dnnl_unimplemented)
        return res->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    res->impl_name = query_impl_info(bpd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", res->impl_name.c_str());

    SAFE(check_pd_w_and_wo_attr(res, prb->attr, bd), WARN);

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    std::vector<dnnl_data_type_t> dts = prb->sdt;
    dts.push_back(prb->ddt);
    check_known_skipped_case_common(dts, FWD_D, res);
    check_sum_post_ops(prb->attr, res);
    if (res->state == SKIPPED) return;

    if (prb->alg == alg_t::DIV) {
        check_binary_post_ops(prb->attr, res);
        if (res->state == SKIPPED) return;
    }

    const bool is_sum = prb->attr.post_ops.find(alg_t::SUM) >= 0;
    bool bcast_src0 = false;
    for (int d = 0; d < prb->ndims[0]; ++d)
        if (prb->sdims[0][d] != prb->sdims[1][d] && prb->sdims[0][d] == 1) {
            bcast_src0 = true;
            break;
        }

    if ((bcast_src0 && (prb->inplace || is_sum || engine_tgt_kind != dnnl_cpu))
            || (prb->inplace && prb->sdt[0] != prb->ddt)) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
        return;
    }

    if (prb->inplace && prb->sdt[0] != prb->ddt) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    if (is_gpu()) { // this is valid for Nvidia GPU as well
        for (const auto &s : prb->attr.scales.scales) {
            if (s.second.runtime) {
                res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                return;
            }
        }
    }

    if (is_nvidia_gpu()) {
        const std::vector<alg_t> supported_algs
                = {alg_t::ADD, alg_t::MUL, alg_t::MIN, alg_t::MAX};
        const bool alg_ok
                = std::any_of(supported_algs.cbegin(), supported_algs.cend(),
                        [&](const alg_t alg) { return prb->alg == alg; });
        const bool dt_ok = prb->sdt[0] == prb->sdt[1];
        const bool diff_dt_ok = dt_ok
                && IMPLICATION(
                        prb->sdt[0] != prb->ddt, prb->attr.scales.is_def());
        if (!alg_ok || !dt_ok || !diff_dt_ok || !prb->attr.post_ops.is_def()) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
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

    const auto &src0_md = q(DNNL_ARG_SRC_0);
    const auto &src1_md = q(DNNL_ARG_SRC_1);
    const auto &dst_md = q(DNNL_ARG_DST);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = tag::abx;

    const auto &test_engine = get_test_engine();

    dnn_mem_t src0_fp(src0_md, fp, tag, test_engine);
    dnn_mem_t src0_dt(src0_md, test_engine);
    SAFE(fill_mem(0, src0_dt, src0_fp), WARN);

    dnn_mem_t src1_fp(src1_md, fp, tag, test_engine);
    dnn_mem_t src1_dt(src1_md, test_engine);
    SAFE(fill_mem(1, src1_dt, src1_fp), WARN);

    dnn_mem_t dst_fp(dst_md, fp, tag, test_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    if (prb->attr.post_ops.find(alg_t::SUM) >= 0)
        SAFE(fill_mem(2, dst_dt, dst_fp), WARN);

    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(setup_binary_po(const_pd, binary_po_args, binary_po_dt, binary_po_fp),
            WARN);

    args_t args;
    args.set(DNNL_ARG_SRC_0, src0_dt);
    args.set(DNNL_ARG_SRC_1, src1_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
    args.set(binary_po_args, binary_po_dt);

    dnn_mem_t input_scales_m0;
    float scale0 = prb->attr.scales.get(DNNL_ARG_SRC_0).scale;
    maybe_prepare_runtime_scales(
            input_scales_m0, prb->attr.scales.get(DNNL_ARG_SRC_0), 1, &scale0);
    args.set(DNNL_ARG_ATTR_INPUT_SCALES | DNNL_ARG_SRC_0, input_scales_m0);
    dnn_mem_t input_scales_m1;
    float scale1 = prb->attr.scales.get(DNNL_ARG_SRC_1).scale;
    maybe_prepare_runtime_scales(
            input_scales_m1, prb->attr.scales.get(DNNL_ARG_SRC_1), 1, &scale1);
    args.set(DNNL_ARG_ATTR_INPUT_SCALES | DNNL_ARG_SRC_1, input_scales_m1);

    SAFE(execute_and_wait(prim, args), WARN);

    if (is_bench_mode(CORR)) {
        compute_ref(prb, src0_fp, src1_fp, binary_po_fp, dst_fp);

        compare::compare_t cmp;
        cmp.set_threshold(epsilon_dt(dst_dt.dt()));
        const auto binary_add_check =
                [&](const compare::compare_t::driver_check_func_args_t &args) {
                    // fp16 result can slightly mismatch for division due to difference
                    // in backends implementations.
                    return prb->alg == alg_t::DIV
                            ? args.diff < epsilon_dt(args.dt)
                            : false;
                };
        cmp.set_driver_check_function(binary_add_check);

        const std::vector<alg_t> cmp_alg = {alg_t::GE, alg_t::GT, alg_t::LE,
                alg_t::LT, alg_t::EQ, alg_t::NE};
        const bool is_cmp = std::any_of(
                cmp_alg.cbegin(), cmp_alg.cend(), [&](const alg_t alg) {
                    return (prb->alg == alg)
                            || prb->attr.post_ops.find(alg) >= 0;
                });

        if (is_cmp) cmp.set_zero_trust_percent(100.f);
        SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
    }

    return measure_perf(res->timer, prim, args);
}

} // namespace binary
