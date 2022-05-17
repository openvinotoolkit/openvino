/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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

#include <functional>
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/ref_io_helper.hpp"

#include "cpu/ref_convolution_utils.hpp"
#include "cpu/ref_deconvolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

void ref_deconvolution_fwd_t::compute_fwd_bias_common(const exec_ctx_t &ctx,
        void *dst, const float *conv_output, bool non_default_attr) const {
    const auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    auto MB = CTX_IN_BATCH(DNNL_ARG_SRC);

    const auto G = pd()->G();
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();
    const auto OD = pd()->OD();
    const auto OC = pd()->OC() / G;
    const auto ndims = pd()->desc()->src_desc.ndims;

    parallel_nd(MB, G, OC, OD, OH, OW,
            [&](dim_t mb, dim_t g, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
                const dim_t c = g * OC + oc;
                const dim_t off = ref_conv_utils::get_data_off(
                        dst_d, ndims, mb, c, od, oh, ow);
                float b = io::load_float_value(bias_d.data_type(), bias, c);
                float d = conv_output[off];
                // Use f32 if attributes happen after bias to get precise answer
                auto dt = non_default_attr ? data_type::f32 : dst_d.data_type();
                io::store_float_value(dt, d + b, dst, off);
            });
}

void ref_deconvolution_fwd_t::compute_fwd_bias_ncdhw(const exec_ctx_t &ctx,
        void *dst, const float *conv_output, bool non_default_attr) const {
    const auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    auto MB = CTX_IN_BATCH(DNNL_ARG_SRC);

    const auto OC = pd()->OC();
    const auto SP = pd()->OW() * pd()->OH() * pd()->OD();

    parallel_nd(MB, OC, [&](dim_t mb, dim_t oc) {
        const dim_t off = (mb * OC + oc) * SP;
        float b = io::load_float_value(bias_d.data_type(), bias, oc);
        PRAGMA_OMP_SIMD()
        for (dim_t sp = 0; sp < SP; ++sp) {
            float d = conv_output[off + sp];
            // Use f32 if attributes happen after bias to get precise answer.
            auto dt = non_default_attr ? data_type::f32 : dst_d.data_type();
            io::store_float_value(dt, d + b, dst, off + sp);
        }
    });
}

void ref_deconvolution_fwd_t::compute_fwd_bias_ndhwc(const exec_ctx_t &ctx,
        void *dst, const float *conv_output, bool non_default_attr) const {
    const auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    auto MB = CTX_IN_BATCH(DNNL_ARG_SRC);

    const auto OC = pd()->OC();
    const auto SP = pd()->OW() * pd()->OH() * pd()->OD();

    parallel_nd(MB, SP, [&](dim_t mb, dim_t sp) {
        const dim_t off = (mb * SP + sp) * OC;
        PRAGMA_OMP_SIMD()
        for (dim_t oc = 0; oc < OC; ++oc) {
            float b = io::load_float_value(bias_d.data_type(), bias, oc);
            float d = conv_output[off + oc];
            // Use f32 if attributes happen after bias to get precise answer.
            auto dt = non_default_attr ? data_type::f32 : dst_d.data_type();
            io::store_float_value(dt, d + b, dst, off + oc);
        }
    });
}

template <dim_t blk_size>
void ref_deconvolution_fwd_t::compute_fwd_bias_nCdhwXc(const exec_ctx_t &ctx,
        void *dst, const float *conv_output, bool non_default_attr) const {
    const auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    auto MB = CTX_IN_BATCH(DNNL_ARG_SRC);

    const auto OC = pd()->OC();
    const auto SP = pd()->OW() * pd()->OH() * pd()->OD();
    const auto stride_mb = dst_d.blocking_desc().strides[0];

    parallel_nd(MB, utils::div_up(OC, blk_size), SP,
            [&](dim_t mb, dim_t oc_blk, dim_t sp) {
                const dim_t oc = oc_blk * blk_size;
                const dim_t off = mb * stride_mb + oc * SP + sp * blk_size;
                const dim_t blk = nstl::min(blk_size, OC - oc);

                PRAGMA_OMP_SIMD()
                for (dim_t i = 0; i < blk_size; ++i) {
                    float b = i < blk ? io::load_float_value(
                                      bias_d.data_type(), bias, oc + i)
                                      : 0;
                    float d = conv_output[off + i];
                    // Use f32 if attributes happen after bias to get precise
                    // answer.
                    auto dt = non_default_attr ? data_type::f32
                                               : dst_d.data_type();
                    io::store_float_value(dt, d + b, dst, off + i);
                }
            });
}

void ref_deconvolution_fwd_t::compute_fwd_bias(const exec_ctx_t &ctx, void *dst,
        const float *conv_output, bool non_default_attr) const {
    using namespace format_tag;
    switch (pd()->dst_tag_) {
        case ncdhw:
        case nchw:
        case ncw:
            compute_fwd_bias_ncdhw(ctx, dst, conv_output, non_default_attr);
            break;
        case ndhwc:
        case nhwc:
        case nwc:
            compute_fwd_bias_ndhwc(ctx, dst, conv_output, non_default_attr);
            break;
        case nCdhw8c:
        case nChw8c:
        case nCw8c:
            compute_fwd_bias_nCdhwXc<8>(
                    ctx, dst, conv_output, non_default_attr);
            break;
        case nCdhw16c:
        case nChw16c:
        case nCw16c:
            compute_fwd_bias_nCdhwXc<16>(
                    ctx, dst, conv_output, non_default_attr);
            break;
        default:
            compute_fwd_bias_common(ctx, dst, conv_output, non_default_attr);
            break;
    }
}

status_t ref_deconvolution_fwd_t::compute_ref_attrs(const exec_ctx_t &ctx,
        const float *conv_output, void *original_dst) const {
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    DEFINE_ZERO_POINTS_BUFFER(dst_zero_point, DNNL_ARG_DST);
    const bool is_dst_zp_common
            = pd()->attr()->zero_points_.common(DNNL_ARG_DST);

    const memory_desc_wrapper dst_d(pd()->dst_md());

    auto MB = CTX_IN_BATCH(DNNL_ARG_SRC);
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();
    const auto OD = pd()->OD();
    const auto OC = pd()->OC();
    const auto OCP = dst_d.padded_dims()[1];
    const auto ndims = pd()->desc()->src_desc.ndims;

    const auto maybe_oscale = [=](float &d, dim_t oc) {
        // scale_idx_mult = 1 for per_oc scales and 0, otherwise
        const int scale_idx_mult
                = pd()->attr()->output_scales_.mask_ == (1 << 1);
        const float *scales = pd()->attr()->output_scales_.scales_;
        d *= scales[oc * scale_idx_mult];
    };

    const auto maybe_dst_zero_point = [=](float &result, dim_t oc) {
        if (is_dst_zp_common)
            result += dst_zero_point[0];
        else
            result += dst_zero_point[oc];
    };

    parallel_nd(MB, OCP, OD, OH, OW,
            [&](dim_t mb, int ocp, dim_t od, dim_t oh, dim_t ow) {
                auto dst_off = ref_conv_utils::get_data_off(
                        dst_d, ndims, mb, ocp, od, oh, ow);
                float tmp_result = 0;

                if (ocp < OC) {
                    dim_t dst_l_off = (mb * OC + ocp) * OD * OH * OW
                            + od * OH * OW + oh * OW + ow;
                    tmp_result = conv_output[dst_off];
                    maybe_oscale(tmp_result, ocp);

                    ref_post_ops_t::args_t args;
                    if (pd()->attr()->post_ops_.find(primitive_kind::sum) != -1)
                        args.dst_val = io::load_float_value(
                                dst_d.data_type(), original_dst, dst_off);
                    args.ctx = &ctx;
                    args.l_offset = dst_l_off;
                    args.dst_md = pd()->dst_md();
                    ref_post_ops->execute(tmp_result, args);
                    maybe_dst_zero_point(tmp_result, ocp);
                }
                io::store_float_value(
                        dst_d.data_type(), tmp_result, dst, dst_off);
            });

    return status_t::dnnl_success;
}

dim_t get_weights_off(const memory_desc_wrapper &wei_d, bool with_groups,
        int ndims, dim_t g, dim_t oc, dim_t ic, dim_t kd, dim_t kh, dim_t kw) {
    switch (ndims) {
        case 5:
            return with_groups ? wei_d.off(g, oc, ic, kd, kh, kw)
                               : wei_d.off(oc, ic, kd, kh, kw);
        case 4:
            return with_groups ? wei_d.off(g, oc, ic, kh, kw)
                               : wei_d.off(oc, ic, kh, kw);
        case 3:
            return with_groups ? wei_d.off(g, oc, ic, kw)
                               : wei_d.off(oc, ic, kw);
        default: assert(!"unsupported ndims"); return dim_t(0);
    }

    return 0;
};

template <data_type_t wei_type>
static void compute_src_zp_compensation(const exec_ctx_t &ctx,
        const int32_t *src_zero_point, const bool is_src_zp_common,
        typename prec_traits<wei_type>::type *wei,
        const cpu_deconvolution_fwd_pd_t *pd) {
    using namespace memory_tracking::names;

    const auto scratchpad = ctx.get_scratchpad_grantor();
    int32_t *zp_compensation = scratchpad.get<int32_t>(key_deconv_zp);
    const auto G = pd->G();
    const auto KH = pd->KH();
    const auto KW = pd->KW();
    const auto KD = pd->KD();
    const auto OC = pd->OC() / G;
    const auto IC = pd->IC() / G;
    const memory_desc_wrapper wei_d(pd->weights_md());
    const bool with_groups = pd->with_groups();
    const auto ndims = wei_d.ndims() - (with_groups ? 1 : 0);
    const auto get_wei_off
            = [=](dim_t g, dim_t oc, dim_t ic, dim_t kd, dim_t kh, dim_t kw) {
                  return get_weights_off(
                          wei_d, with_groups, ndims, g, oc, ic, kd, kh, kw);
              };

    parallel_nd(G, OC, [&](const dim_t g, const dim_t oc) {
        const auto out_offset = g * OC + oc;
        int32_t acc = 0;

        for_(dim_t kd = 0; kd < KD; ++kd)
        for_(dim_t kh = 0; kh < KH; ++kh)
        for (dim_t kw = 0; kw < KW; ++kw) {
            for (dim_t ic = 0; ic < IC; ++ic) {
                const auto weights_offset = get_wei_off(g, oc, ic, kd, kh, kw);
                const int32_t wei32 = static_cast<int32_t>(wei[weights_offset]);

                if (is_src_zp_common)
                    acc += wei32;
                else
                    acc += wei32 * src_zero_point[g * IC + ic];
            }
        }

        zp_compensation[out_offset] = acc * src_zero_point[0];
    });
}

template <data_type_t wei_type>
static std::function<int32_t(
        const dim_t, const dim_t, const dim_t, const dim_t, const dim_t)>
prepare_zp_pad_comp_ker(const dim_t ndims, const int32_t *src_zero_point,
        const bool is_src_zp_common, typename prec_traits<wei_type>::type *wei,
        const cpu_deconvolution_fwd_pd_t *deconv_pd) {

    const auto KH = deconv_pd->KH();
    const auto KW = deconv_pd->KW();
    const auto KD = deconv_pd->KD();
    const auto KSD = deconv_pd->KSD();
    const auto KSH = deconv_pd->KSH();
    const auto KSW = deconv_pd->KSW();
    const auto KDD = deconv_pd->KDD() + 1;
    const auto KDH = deconv_pd->KDH() + 1;
    const auto KDW = deconv_pd->KDW() + 1;
    const auto IC = deconv_pd->IC() / deconv_pd->G();
    const auto IH = deconv_pd->IH();
    const auto IW = deconv_pd->IW();
    const auto ID = deconv_pd->ID();
    const auto pad_front = deconv_pd->padFront();
    const auto pad_top = deconv_pd->padT();
    const auto pad_left = deconv_pd->padL();
    const bool with_groups = deconv_pd->with_groups();
    const memory_desc_wrapper wei_d(deconv_pd->weights_md());
    const auto get_wei_off
            = [=](dim_t g, dim_t oc, dim_t ic, dim_t kd, dim_t kh, dim_t kw) {
                  return get_weights_off(
                          wei_d, with_groups, ndims, g, oc, ic, kd, kh, kw);
              };

    return [=](const dim_t g, const dim_t oc, const dim_t od, const dim_t oh,
                   const dim_t ow) {
        int32_t zp_pad_compensation = 0;

        for (dim_t kd = 0; kd < KD; ++kd) {
            const dim_t id = od - kd * KDD + pad_front;
            const bool should_apply_pad_comp_d
                    = id < 0 || id % KSD != 0 || (id / KSD) >= ID;

            for (dim_t kh = 0; kh < KH; ++kh) {
                const dim_t ih = oh - kh * KDH + pad_top;
                const bool should_apply_pad_comp_h
                        = ih < 0 || ih % KSH != 0 || (ih / KSH) >= IH;

                for (dim_t kw = 0; kw < KW; ++kw) {
                    const dim_t iw = ow - kw * KDW + pad_left;
                    const bool should_apply_pad_comp_w
                            = iw < 0 || iw % KSW != 0 || (iw / KSW) >= IW;

                    if (should_apply_pad_comp_d || should_apply_pad_comp_h
                            || should_apply_pad_comp_w) {

                        for (dim_t ic = 0; ic < IC; ic++) {
                            const auto wei_off
                                    = get_wei_off(g, oc, ic, kd, kh, kw);
                            const int32_t wei32
                                    = static_cast<int32_t>(wei[wei_off]);

                            if (is_src_zp_common)
                                zp_pad_compensation += wei32;
                            else
                                zp_pad_compensation
                                        += wei32 * src_zero_point[g * IC + ic];
                        }
                    }
                }
            }
        }

        if (is_src_zp_common && zp_pad_compensation)
            zp_pad_compensation *= src_zero_point[0];

        return zp_pad_compensation;
    };
}

template <data_type_t wei_type>
static status_t apply_src_zero_point(const exec_ctx_t &ctx,
        const cpu_deconvolution_fwd_pd_t *deconv_pd, float *conv_output) {
    using wei_data_t = typename prec_traits<wei_type>::type;
    using namespace memory_tracking::names;
    using namespace data_type;

    // required by DEFINE_ZERO_POINTS_BUFFER macro
    const auto pd = [&]() { return deconv_pd; };
    const auto wei = CTX_OUT_MEM(wei_data_t *, DNNL_ARG_WEIGHTS);
    DEFINE_ZERO_POINTS_BUFFER(src_zero_point, DNNL_ARG_SRC);
    const bool is_src_zp_common
            = deconv_pd->attr()->zero_points_.common(DNNL_ARG_SRC);

    const auto scratchpad = ctx.get_scratchpad_grantor();
    const int32_t *const zp_src_compensation
            = scratchpad.get<int32_t>(key_deconv_zp);
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const auto ndims = dst_d.ndims();

    const auto G = pd()->G();
    const auto MB = pd()->MB();
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();
    const auto OD = pd()->OD();
    const auto OC = pd()->OC() / G;

    compute_src_zp_compensation<wei_type>(
            ctx, src_zero_point, is_src_zp_common, wei, deconv_pd);
    const auto zp_pad_comp_ker = prepare_zp_pad_comp_ker<wei_type>(
            ndims, src_zero_point, is_src_zp_common, wei, deconv_pd);

    parallel_nd(MB, G, OC, OD, OH, OW,
            [&](const dim_t mb, const dim_t g, const dim_t oc, const dim_t od,
                    const dim_t oh, const dim_t ow) {
                const auto oc_off = g * OC + oc;
                const auto dst_off = ref_conv_utils::get_data_off(
                        dst_d, ndims, mb, oc_off, od, oh, ow);
                int32_t conv_result
                        = conv_output[dst_off] - zp_src_compensation[oc_off];

                if (const auto zp_pad_compensation
                        = zp_pad_comp_ker(g, oc, od, oh, ow)) {
                    conv_result += zp_pad_compensation;
                }

                conv_output[dst_off] = static_cast<float>(conv_result);
            });

    return status::success;
}

status_t ref_deconvolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    const auto scratchpad = ctx.get_scratchpad_grantor();
    const bool ref_bias = pd()->with_bias() && !pd()->conv_supports_bias_;
    const bool non_default_attr = !pd()->attr()->has_default_values();

    const auto &args = ctx.args();
    exec_args_t conv_args;
    conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
    conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
    if (pd()->with_bias() && pd()->conv_supports_bias_)
        conv_args[DNNL_ARG_BIAS] = args.at(DNNL_ARG_BIAS);

    // Create intermediate memory for f32 output if needed.
    auto dst = args.at(DNNL_ARG_DST);
    memory_t tmp_memory(dst.mem->engine(), pd()->conv_pd_->diff_src_md(),
            scratchpad.get_memory_storage(key_deconv_bias));
    memory_arg_t tmp_conv_output = {&tmp_memory, false};

    conv_args[DNNL_ARG_DIFF_SRC]
            = ref_bias || non_default_attr ? tmp_conv_output : dst;

    // When sum post-op happens, we need to copy original destination memory
    // prior call to external convolution happens.
    if (pd()->attr()->post_ops_.find(primitive_kind::sum) != -1) {
        void *original_dst = scratchpad.get(key_deconv_sum);
        const memory_desc_wrapper dst_d(pd()->dst_md());
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
        const auto dt_size = dst_d.data_type_size();

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start {0}, end {0};
            balance211(dst_d.nelems(true), nthr, ithr, start, end);
            auto o_dst_start = (char *)original_dst + start * dt_size;
            auto dst_start = (char *)dst + start * dt_size;
            const auto size = (end - start) * dt_size;

            std::memcpy(o_dst_start, dst_start, size);
        });
    }

    exec_ctx_t conv_ctx(ctx, std::move(conv_args));

    nested_scratchpad_t ns(ctx, key_nested, conv_p_);
    conv_ctx.set_scratchpad_grantor(ns.grantor());
    auto status = conv_p_->execute(conv_ctx);
    if (status != status::success) return status;

    using namespace data_type;

    if (!pd()->attr()->zero_points_.has_default_values(DNNL_ARG_SRC)) {
        float *conv_output = scratchpad.get<float>(key_deconv_bias);
        const auto wei_dt = pd()->weights_md()->data_type;
        switch (wei_dt) {
            case s8: apply_src_zero_point<s8>(ctx, pd(), conv_output); break;
            case u8: apply_src_zero_point<u8>(ctx, pd(), conv_output); break;
            default: assert(!"unsupported data type");
        }
    }

    float *conv_output = scratchpad.get<float>(key_deconv_bias);

    if (ref_bias) {
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
        void *tmp_output = non_default_attr ? conv_output : dst;
        compute_fwd_bias(ctx, tmp_output, conv_output, non_default_attr);
    }

    if (non_default_attr) {
        void *original_dst = scratchpad.get<void>(key_deconv_sum);
        compute_ref_attrs(ctx, conv_output, original_dst);
    }

    return status::success;
}

status_t ref_deconvolution_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    const auto &args = ctx.args();
    exec_args_t conv_args;
    conv_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_DIFF_DST);
    conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
    conv_args[DNNL_ARG_DST] = args.at(DNNL_ARG_DIFF_SRC);
    exec_ctx_t conv_ctx(ctx, std::move(conv_args));

    nested_scratchpad_t ns(ctx, key_nested, conv_p_);
    conv_ctx.set_scratchpad_grantor(ns.grantor());
    conv_p_->execute(conv_ctx);
    return status::success;
}

void ref_deconvolution_bwd_weights_t::compute_bwd_bias(
        float *diff_bias, const float *diff_dst) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const auto G = pd()->G();
    const auto MB = pd()->MB();
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();
    const auto OC = pd()->OC() / G;
    const auto OD = pd()->OD();
    const auto ndims = pd()->desc()->src_desc.ndims;

    parallel_nd(G, OC, [&](dim_t g, dim_t oc) {
        float db = 0;
        for_(dim_t mb = 0; mb < MB; ++mb)
        for_(dim_t od = 0; od < OD; ++od)
        for_(dim_t oh = 0; oh < OH; ++oh)
        for (dim_t ow = 0; ow < OW; ++ow) {
            const auto d_dst_off = ref_conv_utils::get_data_off(
                    diff_dst_d, ndims, mb, g * OC + oc, od, oh, ow);
            db += diff_dst[d_dst_off];
        }
        diff_bias[g * OC + oc] = db;
    });
}

template <data_type_t dbia_type, data_type_t ddst_type>
void ref_deconvolution_bwd_weights_t::compute_bwd_bias_ncdhw(
        typename prec_traits<dbia_type>::type *diff_bias,
        const typename prec_traits<ddst_type>::type *diff_dst) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const auto OC = pd()->OC();
    const auto MB = pd()->MB();
    const auto SP = pd()->OH() * pd()->OW() * pd()->OD();

    parallel_nd(OC, [&](dim_t oc) {
        float db = 0;
        for (dim_t mb = 0; mb < MB; ++mb) {
            PRAGMA_OMP_SIMD(reduction(+ : db))
            for (dim_t sp = 0; sp < SP; ++sp) {
                auto offset = (size_t)(mb * OC + oc) * SP + sp;
                db += diff_dst[offset];
            }
        }
        diff_bias[oc] = db;
    });
}

template <data_type_t dbia_type, data_type_t ddst_type>
void ref_deconvolution_bwd_weights_t::compute_bwd_bias_ndhwc(
        typename prec_traits<dbia_type>::type *diff_bias,
        const typename prec_traits<ddst_type>::type *diff_dst) const {
    const auto MB = pd()->MB();
    const auto SP = pd()->OW() * pd()->OH() * pd()->OD();
    const auto OC = pd()->OC();

    parallel_nd(OC, [&](dim_t oc) {
        float db = 0;
        for (dim_t mb = 0; mb < MB; ++mb) {
            PRAGMA_OMP_SIMD(reduction(+ : db))
            for (dim_t sp = 0; sp < SP; ++sp) {
                const dim_t offset = (mb * SP + sp) * OC + oc;
                db += diff_dst[offset];
            }
        }
        diff_bias[oc] = static_cast<typename prec_traits<dbia_type>::type>(db);
    });
}

template <data_type_t dbia_type, data_type_t ddst_type, dim_t blksize>
void ref_deconvolution_bwd_weights_t::compute_bwd_bias_nCdhwXc(
        typename prec_traits<dbia_type>::type *diff_bias,
        const typename prec_traits<ddst_type>::type *diff_dst) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const auto OC = pd()->OC();
    const auto MB = pd()->MB();
    const auto SP = pd()->OH() * pd()->OW() * pd()->OD();

    const ptrdiff_t stride_mb = diff_dst_d.blocking_desc().strides[0];

    parallel_nd(utils::div_up(OC, blksize), [&](dim_t ocb) {
        float db[blksize] = {0};

        for (dim_t mb = 0; mb < MB; ++mb) {
            for (dim_t sp = 0; sp < SP; ++sp) {
                auto offset = mb * stride_mb + (ocb * SP + sp) * blksize;

                PRAGMA_OMP_SIMD()
                for (dim_t i = 0; i < blksize; ++i)
                    db[i] += diff_dst[offset + i];
            }
        }

        const dim_t blk = nstl::min(blksize, OC - ocb * blksize);

        PRAGMA_OMP_SIMD()
        for (dim_t i = 0; i < blk; ++i)
            diff_bias[ocb * blksize + i] = db[i];
    });
}

template <data_type_t dbia_type, data_type_t ddst_type>
void ref_deconvolution_bwd_weights_t::compute_bias(
        const exec_ctx_t &ctx) const {
    using dbia_data_t = typename prec_traits<dbia_type>::type;
    using ddst_data_t = typename prec_traits<ddst_type>::type;

    auto diff_bias = CTX_OUT_MEM(dbia_data_t *, DNNL_ARG_DIFF_BIAS);
    auto diff_dst = CTX_IN_MEM(const ddst_data_t *, DNNL_ARG_DIFF_DST);

    using namespace format_tag;
    switch (pd()->dst_tag_) {
        case ncdhw:
        case nchw:
        case ncw:
            compute_bwd_bias_ncdhw<dbia_type, ddst_type>(diff_bias, diff_dst);
            break;
        case ndhwc:
        case nhwc:
        case nwc:
            compute_bwd_bias_ndhwc<dbia_type, ddst_type>(diff_bias, diff_dst);
            break;
        case nCdhw8c:
        case nChw8c:
        case nCw8c:
            assert(!utils::one_of(data_type::bf16, dbia_type, ddst_type));
            compute_bwd_bias_nCdhwXc<dbia_type, ddst_type, 8>(
                    diff_bias, diff_dst);
            break;
        case nCdhw16c:
        case nChw16c:
        case nCw16c:
            compute_bwd_bias_nCdhwXc<dbia_type, ddst_type, 16>(
                    diff_bias, diff_dst);
            break;
        default:
            assert(!utils::one_of(data_type::bf16, dbia_type, ddst_type));
            compute_bwd_bias((float *)diff_bias, (const float *)diff_dst);
            break;
    }
}

status_t ref_deconvolution_bwd_weights_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    const auto &args = ctx.args();
    exec_args_t conv_args;
    conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
    conv_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_DIFF_DST);
    conv_args[DNNL_ARG_DIFF_WEIGHTS] = args.at(DNNL_ARG_DIFF_WEIGHTS);
    exec_ctx_t conv_ctx(ctx, std::move(conv_args));

    nested_scratchpad_t ns(ctx, key_nested, conv_p_);
    conv_ctx.set_scratchpad_grantor(ns.grantor());
    status_t status = conv_p_->execute(conv_ctx);
    if (status != status::success) return status;

    if (pd()->with_bias()) {
        using namespace data_type;

        auto dbia_type = pd()->diff_weights_md(1)->data_type;
        auto ddst_type = pd()->diff_dst_md()->data_type;
        if (utils::everyone_is(f32, dbia_type, ddst_type))
            compute_bias<f32, f32>(ctx);
        else if (utils::everyone_is(bf16, dbia_type, ddst_type))
            compute_bias<bf16, bf16>(ctx);
        else if (dbia_type == f32 && ddst_type == bf16)
            compute_bias<f32, bf16>(ctx);
        else {
            assert(!"unsupported data type");
            return status::runtime_error;
        }
    }
    return status::success;
}

using namespace data_type;

template void ref_deconvolution_bwd_weights_t::compute_bias<f32, f32>(
        const exec_ctx_t &ctx) const;
template void ref_deconvolution_bwd_weights_t::compute_bias<f32, bf16>(
        const exec_ctx_t &ctx) const;
template void ref_deconvolution_bwd_weights_t::compute_bias<bf16, bf16>(
        const exec_ctx_t &ctx) const;
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
