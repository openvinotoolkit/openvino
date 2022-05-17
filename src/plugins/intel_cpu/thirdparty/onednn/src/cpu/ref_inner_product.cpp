/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
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
#include "common/type_helpers.hpp"

#include "cpu/ref_io_helper.hpp"

#include "cpu/ref_inner_product.hpp"
#include "cpu/ref_inner_product_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

status_t ref_inner_product_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    auto MB = CTX_IN_BATCH(DNNL_ARG_SRC);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const auto ndims = pd()->ndims();
    const auto OC = pd()->OC();
    const auto IC = pd()->IC();

    auto ker = [=](dim_t mb, dim_t oc) {
        float d = 0;
        const dim_t KD = pd()->KD();
        const dim_t KH = pd()->KH();
        const dim_t KW = pd()->KW();
        for_(dim_t ic = 0; ic < IC; ++ic)
        for_(dim_t kd = 0; kd < KD; ++kd)
        for_(dim_t kh = 0; kh < KH; ++kh)
        for (dim_t kw = 0; kw < KW; ++kw) {
            const auto src_off = ref_ip_utils::get_data_off(
                    src_d, ndims, mb, ic, kd, kh, kw);
            const auto wei_off = ref_ip_utils::get_weights_off(
                    weights_d, ndims, oc, ic, kd, kh, kw);
            const float s
                    = io::load_float_value(src_d.data_type(), src, src_off);
            const float w = io::load_float_value(
                    weights_d.data_type(), weights, wei_off);
            d += s * w;
        }
        return d;
    };

    parallel_nd(MB, OC, [&](dim_t mb, dim_t oc) {
        float acc = ker(mb, oc);

        float d = acc;
        if (bias) {
            const auto bias_off = bias_d.off(oc);
            const float b
                    = io::load_float_value(bias_d.data_type(), bias, bias_off);
            d += b;
        }

        dim_t dst_off = dst_d.off(mb, oc);
        dim_t dst_l_off = (mb * OC + oc);

        ref_post_ops_t::args_t args;
        args.dst_val = io::load_float_value(dst_d.data_type(), dst, dst_off);
        args.ctx = &ctx;
        args.l_offset = dst_l_off;
        args.dst_md = pd()->dst_md();
        ref_post_ops->execute(d, args);

        io::store_float_value(dst_d.data_type(), d, dst, dst_off);
    });

    return status::success;
}

status_t ref_inner_product_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DIFF_SRC, status);
    CHECK(status);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());

    const auto ndims = pd()->ndims();
    const auto MB = pd()->MB();
    const auto OC = pd()->OC();
    const auto IC = pd()->IC();

    parallel_nd(MB, IC, [&](dim_t mb, dim_t ic) {
        const dim_t KD = pd()->KD();
        const dim_t KH = pd()->KH();
        const dim_t KW = pd()->KW();
        for_(dim_t kd = 0; kd < KD; ++kd)
        for_(dim_t kh = 0; kh < KH; ++kh)
        for (dim_t kw = 0; kw < KW; ++kw) {
            float ds = 0;
            for (dim_t oc = 0; oc < OC; ++oc) {
                const auto diff_dst_off = ref_ip_utils::get_data_off(
                        diff_dst_d, 2, mb, oc, 0, 0, 0);
                const auto wei_off = ref_ip_utils::get_weights_off(
                        weights_d, ndims, oc, ic, kd, kh, kw);
                const float dd = io::load_float_value(
                        diff_dst_d.data_type(), diff_dst, diff_dst_off);
                const float w = io::load_float_value(
                        weights_d.data_type(), weights, wei_off);
                ds += dd * w;
            }
            const auto diff_src_off = ref_ip_utils::get_data_off(
                    diff_src_d, ndims, mb, ic, kd, kh, kw);
            io::store_float_value(
                    diff_src_d.data_type(), ds, diff_src, diff_src_off);
        }
    });

    return status::success;
}

status_t ref_inner_product_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto diff_weights
            = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DIFF_WEIGHTS, status);
    CHECK(status);
    auto diff_bias = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DIFF_BIAS, status);
    CHECK(status);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_md(1));

    const auto ndims = src_d.ndims();
    const auto MB = pd()->MB();
    const auto OC = pd()->OC();
    const auto IC = pd()->IC();

    parallel_nd(OC, IC, [&](dim_t oc, dim_t ic) {
        const dim_t KD = pd()->KD();
        const dim_t KH = pd()->KH();
        const dim_t KW = pd()->KW();
        for_(dim_t kd = 0; kd < KD; ++kd)
        for_(dim_t kh = 0; kh < KH; ++kh)
        for (dim_t kw = 0; kw < KW; ++kw) {
            float dw = 0;
            for (dim_t mb = 0; mb < MB; ++mb) {
                const auto diff_dst_off = ref_ip_utils::get_data_off(
                        diff_dst_d, 2, mb, oc, 0, 0, 0);
                const auto src_off = ref_ip_utils::get_data_off(
                        src_d, ndims, mb, ic, kd, kh, kw);
                const float dd = io::load_float_value(
                        diff_dst_d.data_type(), diff_dst, diff_dst_off);
                const float s
                        = io::load_float_value(src_d.data_type(), src, src_off);
                dw += dd * s;
            }
            const auto diff_wei_off = ref_ip_utils::get_weights_off(
                    diff_weights_d, ndims, oc, ic, kd, kh, kw);
            io::store_float_value(
                    diff_weights_d.data_type(), dw, diff_weights, diff_wei_off);
        }
    });

    if (diff_bias) {
        parallel_nd(OC, [&](dim_t oc) {
            float db = 0;
            for (dim_t mb = 0; mb < MB; ++mb) {
                const auto diff_dst_off = ref_ip_utils::get_data_off(
                        diff_dst_d, 2, mb, oc, 0, 0, 0);
                const float dd = io::load_float_value(
                        diff_dst_d.data_type(), diff_dst, diff_dst_off);
                db += dd;
            }

            const auto diff_bia_off = diff_bias_d.off(oc);
            io::store_float_value(
                    diff_bias_d.data_type(), db, diff_bias, diff_bia_off);
        });
    }

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
