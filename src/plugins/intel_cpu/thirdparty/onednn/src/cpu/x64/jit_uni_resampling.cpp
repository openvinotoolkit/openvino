/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <bitset>
#include <cassert>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/resampling_utils.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_resampling.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace resampling_utils;

static cpu_isa_t get_supported_isa(bool is_blocked_8_format) {
    if (mayiuse(avx512_core_bf16)) return avx512_core_bf16;
    if (mayiuse(avx512_core)) return avx512_core;
    // YMM registers are used for the avx512 architecture when
    // the datastream structure is based on an 8-block format.
    // Unfortunately, registers from 16 to 31 cannot be used
    // with avx512_common when using YMM. If the kernel does this,
    // it will receive a SIGILL (Illegal instruction) error.
    // Therefore, the avx version is preferred when primitive
    // is running on avx512_common and with the 8-block format.
    if (mayiuse(avx512_common) && !is_blocked_8_format) return avx512_common;
    if (mayiuse(avx2)) return avx2;
    if (mayiuse(avx)) return avx;
    if (mayiuse(sse41)) return sse41;

    return isa_any;
}

status_t jit_uni_resampling_fwd_t::pd_t::init(engine_t *engine) {
    using namespace data_type;
    using sm = primitive_attr_t::skip_mask_t;

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper dst_d(dst_md());

    conf_.src_data_type = src_md()->data_type;
    conf_.dst_data_type = dst_md()->data_type;

    fill_format_tag_info();
    conf_.isa = get_supported_isa(conf_.is_blocked_8_format);

    const bool ok = is_fwd() && !has_zero_dim_memory()
            && conf_.src_tag != format_tag::undef
            && set_default_params(conf_.src_tag) == status::success
            && platform::has_data_type_support(conf_.src_data_type)
            && platform::has_data_type_support(conf_.dst_data_type)
            && attr()->has_default_values(sm::post_ops, conf_.dst_data_type)
            && attr_.set_default_formats(dst_md(0)) == status::success;
    if (!ok) return status::unimplemented;

    if (!memory_desc_matches_tag(*dst_md(), conf_.src_tag))
        return status::unimplemented;

    conf_.alg = desc()->alg_kind;
    conf_.c = C();
    conf_.od = OD();
    conf_.oh = OH();
    conf_.ow = OW();
    conf_.id = ID();
    conf_.ih = IH();
    conf_.iw = IW();
    conf_.ndims = ndims();

    if (conf_.alg == alg_kind::resampling_linear)
        conf_.number_of_corners = pow(2, conf_.ndims - 2);

    conf_.src_dt_size = types::data_type_size(conf_.src_data_type);
    conf_.dst_dt_size = types::data_type_size(conf_.dst_data_type);

    conf_.is_saturation_needed
            = utils::one_of(conf_.dst_data_type, s32, s8, u8);

    const size_t L3_size = static_cast<size_t>(dnnl_get_current_num_threads())
            * platform::get_per_core_cache_size(3);
    const size_t input_data_size = src_d.nelems(true) * conf_.src_dt_size;
    const size_t output_data_size = dst_d.nelems(true) * conf_.dst_dt_size;
    const size_t whole_data_size = input_data_size + output_data_size;
    conf_.output_data_size = output_data_size;
    conf_.is_data_size_bigger_than_L3
            = L3_size > 0 ? whole_data_size > L3_size : false;

    conf_.el_size_of_indices = sizeof(unsigned);

    conf_.inner_stride = src_d.blocking_desc().strides[ndims() - 1];
    conf_.stride_d = IH() * IW() * conf_.inner_stride * conf_.src_dt_size;
    conf_.stride_h = IW() * conf_.inner_stride * conf_.src_dt_size;
    conf_.stride_w = conf_.inner_stride * conf_.src_dt_size;

    const std::vector<injector::post_op_type> accepted_post_ops
            = {injector::sum, injector::eltwise, injector::binary};
    static constexpr bool sum_at_0_pos_only = false;
    static constexpr bool sum_requires_scale_one = false;
    static constexpr bool sum_requires_zp_zero = true;
    const bcast_set_t accepted_broadcasts
            = {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::per_oc_spatial};
    injector::post_ops_ok_args_t post_ops_args(conf_.isa, accepted_post_ops,
            attr()->post_ops_, &dst_d, sum_at_0_pos_only,
            sum_requires_scale_one, sum_requires_zp_zero, accepted_broadcasts);
    if (!post_ops_ok(post_ops_args)) return status::unimplemented;

    conf_.post_ops = attr()->post_ops_;

    static constexpr bool require_scale_one = false;
    conf_.with_eltwise = conf_.with_binary = conf_.with_sum = false;
    for (const auto &entry : conf_.post_ops.entry_) {
        if (entry.is_eltwise()) {
            conf_.with_eltwise = true;
        } else if (entry.is_binary()) {
            conf_.with_binary = true;
        } else if (entry.is_sum(require_scale_one) && entry.sum.scale != 0.f) {
            conf_.with_sum = true;
            conf_.sum_scales.push(entry.sum.scale);
        }
    }
    conf_.with_postops
            = conf_.with_eltwise || conf_.with_binary || conf_.with_sum;

    return status::success;
}

void jit_uni_resampling_fwd_t::pd_t::fill_format_tag_info() {
    using namespace format_tag;

    const format_tag_t blocked_16_format = memory_desc_matches_one_of_tag(
            *src_md(), nCw16c, nChw16c, nCdhw16c);
    const format_tag_t blocked_8_format
            = memory_desc_matches_one_of_tag(*src_md(), nCw8c, nChw8c, nCdhw8c);
    const format_tag_t nspc_format
            = memory_desc_matches_one_of_tag(*src_md(), nwc, nhwc, ndhwc);
    const format_tag_t ncsp_format
            = memory_desc_matches_one_of_tag(*src_md(), ncw, nchw, ncdhw);

    if (blocked_16_format != undef) {
        conf_.tag_kind = jit_memory_tag_kind_t::blocked;
        conf_.src_tag = blocked_16_format;
    } else if (blocked_8_format != undef) {
        conf_.is_blocked_8_format = true;
        conf_.tag_kind = jit_memory_tag_kind_t::blocked;
        conf_.src_tag = blocked_8_format;
    } else if (nspc_format != undef) {
        conf_.tag_kind = jit_memory_tag_kind_t::nspc;
        conf_.src_tag = nspc_format;
    } else if (ncsp_format != undef) {
        conf_.tag_kind = jit_memory_tag_kind_t::ncsp;
        conf_.src_tag = ncsp_format;
    } else {
        conf_.tag_kind = jit_memory_tag_kind_t::undef;
        conf_.src_tag = undef;
    }
}

status_t jit_uni_resampling_fwd_t::get_proper_kernel_for_avx512(
        const memory_desc_t *dst_md, const jit_resampling_conf_t &conf) {
    const format_tag_t blocked_8_tag = utils::pick(conf.ndims - 3,
            format_tag::nCw8c, format_tag::nChw8c, format_tag::nCdhw8c);
    if (memory_desc_matches_tag(*pd()->src_md(), blocked_8_tag)) {
        assert(is_superset(conf.isa, avx512_core)
                && "YMMs 16-31 are not available for avx512_common.");
        return safe_ptr_assign(kernel_,
                new jit_uni_resampling_kernel_t<avx512_common, Xbyak::Ymm>(
                        conf, dst_md));
    }

    return safe_ptr_assign(kernel_,
            new jit_uni_resampling_kernel_t<avx512_common, Xbyak::Zmm>(
                    conf, dst_md));
}

status_t jit_uni_resampling_fwd_t::get_proper_kernel_for_avx(
        const memory_desc_t *dst_md, const jit_resampling_conf_t &conf) {
    using namespace data_type;

    const bool is_src_i8 = utils::one_of(conf.src_data_type, s8, u8);
    const bool is_dst_i8 = utils::one_of(conf.dst_data_type, s8, u8);
    if (is_src_i8 || is_dst_i8)
        return safe_ptr_assign(kernel_,
                new jit_uni_resampling_kernel_t<avx, Xbyak::Xmm>(conf, dst_md));

    return safe_ptr_assign(kernel_,
            new jit_uni_resampling_kernel_t<avx, Xbyak::Ymm>(conf, dst_md));
}

status_t jit_uni_resampling_fwd_t::get_proper_kernel_for_sse(
        const memory_desc_t *dst_md, const jit_resampling_conf_t &conf) {
    return safe_ptr_assign(kernel_,
            new jit_uni_resampling_kernel_t<sse41, Xbyak::Xmm>(conf, dst_md));
}

status_t jit_uni_resampling_fwd_t::init(engine_t *engine) {
    using namespace format_tag;

    const memory_desc_t *dst_md = pd()->dst_md();
    const jit_resampling_conf_t &conf = pd()->get_conf();

    if (is_superset(conf.isa, avx512_common))
        CHECK(get_proper_kernel_for_avx512(dst_md, conf));
    else if (is_superset(conf.isa, avx))
        CHECK(get_proper_kernel_for_avx(dst_md, conf));
    else if (conf.isa == sse41) {
        CHECK(get_proper_kernel_for_sse(dst_md, conf));
    } else {
        assert(!"Unsupported isa.");
        return status::runtime_error;
    }

    CHECK(kernel_->create_kernel());

    return fill_data_for_interpolation();
}

status_t jit_uni_resampling_fwd_t::fill_data_for_interpolation() {
    switch (pd()->desc()->alg_kind) {
        case alg_kind::resampling_nearest: return fill_data_for_nearest();
        case alg_kind::resampling_linear: return fill_data_for_linear();
        default:
            assert(!"Invalid resampling algorithm.");
            return status::invalid_arguments;
    }
}

status_t jit_uni_resampling_fwd_t::fill_data_for_nearest() {
    // In kernel is used vmovdqu to get indices. This instruction don't have
    // tail processing possibilities on sse41 and avx. To avoid problems
    // with that, OW is aligned to simd width, because indices for ow
    // are read in the kernel.
    indices_.reserve(pd()->OD() + pd()->OH()
            + utils::rnd_up(pd()->OW(), kernel_->get_simd_w()));

    for (dim_t od = 0; od < pd()->OD(); od++) {
        const int offset_id = nearest_idx(od, pd()->OD(), pd()->ID())
                * pd()->get_conf().stride_d;
        indices_.emplace_back(offset_id);
    }
    for (dim_t oh = 0; oh < pd()->OH(); oh++) {
        const int offset_ih = nearest_idx(oh, pd()->OH(), pd()->IH())
                * pd()->get_conf().stride_h;
        indices_.emplace_back(offset_ih);
    }
    for (dim_t ow = 0; ow < pd()->OW(); ow++) {
        const int offset_iw = nearest_idx(ow, pd()->OW(), pd()->IW())
                * pd()->get_conf().stride_w;
        indices_.emplace_back(offset_iw);
    }

    return status::success;
}

status_t jit_uni_resampling_fwd_t::fill_data_for_linear() {
    using namespace resampling_utils;

    const unsigned number_of_corners = pd()->get_conf().number_of_corners;
    const unsigned stride_w = pd()->get_conf().stride_w;
    const unsigned stride_h = pd()->get_conf().stride_h;
    const unsigned stride_d = pd()->get_conf().stride_d;

    unsigned num_of_elements = 0;
    if (pd()->get_conf().tag_kind == jit_memory_tag_kind_t::ncsp) {
        // In kernel is used vmovdqu to get indices. This instruction don't have
        // tail processing possibilities on sse41 and avx. To avoid problems
        // with that, number of spatial points is aligned to simd width, because
        // all of them are read in the kernel.
        num_of_elements = number_of_corners
                * utils::rnd_up(pd()->OD() * pd()->OH() * pd()->OW(),
                        kernel_->get_simd_w());

        indices_.resize(num_of_elements);
        weights_.resize(num_of_elements);

        const size_t indices_stride = pd()->OW() * pd()->OH() * pd()->OD();
        const size_t weights_stride = pd()->OW() * pd()->OH() * pd()->OD();

        parallel_nd(pd()->OD(), pd()->OH(), [&](dim_t od, dim_t oh) {
            const linear_coeffs_t coeffs_id(od, pd()->OD(), pd()->ID());
            const linear_coeffs_t coeffs_ih(oh, pd()->OH(), pd()->IH());

            for (dim_t ow = 0; ow < pd()->OW(); ow++) {
                const size_t offset
                        = od * pd()->OH() * pd()->OW() + oh * pd()->OW() + ow;

                const linear_coeffs_t coeffs_iw(ow, pd()->OW(), pd()->IW());

                for (unsigned i = 0; i < number_of_corners; i++) {
                    std::bitset<3> corners(i);
                    indices_[i * indices_stride + offset]
                            = coeffs_id.idx[corners.test(2)] * stride_d
                            + coeffs_ih.idx[corners.test(1)] * stride_h
                            + coeffs_iw.idx[corners.test(0)] * stride_w;
                    weights_[i * weights_stride + offset]
                            = coeffs_id.wei[corners.test(2)]
                            * coeffs_ih.wei[corners.test(1)]
                            * coeffs_iw.wei[corners.test(0)];
                }
            }
        });
    } else if (pd()->get_conf().tag_kind == jit_memory_tag_kind_t::nspc
            || pd()->get_conf().tag_kind == jit_memory_tag_kind_t::blocked) {
        num_of_elements = 2 * (pd()->OD() + pd()->OH() + pd()->OW());

        indices_.resize(num_of_elements);
        weights_.resize(num_of_elements);

        unsigned *indices_w = &indices_[0];
        unsigned *indices_h = &indices_[2 * pd()->OW()];
        unsigned *indices_d = &indices_[2 * (pd()->OW() + pd()->OH())];
        float *weights_w = &weights_[0];
        float *weights_h = &weights_[2 * pd()->OW()];
        float *weights_d = &weights_[2 * (pd()->OW() + pd()->OH())];

        for (dim_t ow = 0; ow < pd()->OW(); ow++) {
            const linear_coeffs_t coeffs_iw(ow, pd()->OW(), pd()->IW());

            // The right and left corners are set one after
            // the other because in the kernel these values
            // are read one by one, which makes it easier
            // to read and makes the operation faster.
            weights_w[2 * ow] = coeffs_iw.wei[0];
            weights_w[2 * ow + 1] = coeffs_iw.wei[1];
            indices_w[2 * ow] = coeffs_iw.idx[0] * stride_w;
            indices_w[2 * ow + 1] = coeffs_iw.idx[1] * stride_w;
        }

        for (dim_t oh = 0; oh < pd()->OH(); oh++) {
            const linear_coeffs_t coeffs_ih(oh, pd()->OH(), pd()->IH());

            weights_h[oh] = coeffs_ih.wei[0];
            weights_h[pd()->OH() + oh] = coeffs_ih.wei[1];
            indices_h[oh] = coeffs_ih.idx[0] * stride_h;
            indices_h[pd()->OH() + oh] = coeffs_ih.idx[1] * stride_h;
        }

        for (dim_t od = 0; od < pd()->OD(); od++) {
            const linear_coeffs_t coeffs_id(od, pd()->OD(), pd()->ID());

            weights_d[od] = coeffs_id.wei[0];
            weights_d[pd()->OD() + od] = coeffs_id.wei[1];
            indices_d[od] = coeffs_id.idx[0] * stride_d;
            indices_d[pd()->OD() + od] = coeffs_id.idx[1] * stride_d;
        }
    } else {
        assert(!"Invalid memory format kind.");
        return status::invalid_arguments;
    }

    return status::success;
}

status_t jit_uni_resampling_fwd_t::execute(const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const uint8_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(uint8_t *, DNNL_ARG_DST);

    const std::vector<const void *> post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(
                    pd()->get_conf().post_ops, ctx);

    switch (pd()->desc()->alg_kind) {
        case alg_kind::resampling_nearest:
            return interpolate_nearest(src, dst, post_ops_binary_rhs_arg_vec);
        case alg_kind::resampling_linear:
            return interpolate_linear(src, dst, post_ops_binary_rhs_arg_vec);
        default:
            assert(!"Invalid resampling algorithm.");
            return status::invalid_arguments;
    }
}

status_t jit_uni_resampling_fwd_t::interpolate_nearest(const uint8_t *src,
        uint8_t *dst, const std::vector<const void *> &post_ops_args) const {
    const size_t src_dt_size = pd()->get_conf().src_dt_size;
    const size_t dst_dt_size = pd()->get_conf().dst_dt_size;
    const size_t inner_stride = pd()->get_conf().inner_stride;

    const dim_t MB = pd()->MB();
    const dim_t C = pd()->C();
    const dim_t CB = utils::div_up(C, inner_stride);
    const dim_t nsp_outer = MB * CB;
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();

    const unsigned *indices_d = &indices_[0];
    const unsigned *indices_h = &indices_[OD];
    const unsigned *indices_w = &indices_[OD + OH];

    if (pd()->get_conf().tag_kind == jit_memory_tag_kind_t::ncsp) {
        parallel_nd(MB, C, OD, [&](dim_t mb, dim_t c, dim_t od) {
            const dim_t src_off
                    = (mb * C + c) * ID * IH * IW * src_dt_size + indices_d[od];
            const dim_t dst_off = ((mb * C + c) * OD * OH * OW + od * OH * OW)
                    * dst_dt_size;

            jit_resampling_call_s args = jit_resampling_call_s();
            args.src = src + src_off;
            args.dst = dst + dst_off;
            args.indices = &indices_h[0];
            args.post_ops_binary_rhs_arg_vec = post_ops_args.data();
            args.c_offset = static_cast<size_t>(c);

            (*kernel_)(&args);
        });
    } else if (pd()->get_conf().tag_kind == jit_memory_tag_kind_t::nspc
            || pd()->get_conf().tag_kind == jit_memory_tag_kind_t::blocked) {
        parallel_nd(nsp_outer, OD, OH, [&](dim_t nsp, dim_t od, dim_t oh) {
            const dim_t src_off
                    = nsp * ID * IH * IW * inner_stride * src_dt_size
                    + indices_d[od] + indices_h[oh];
            const dim_t dst_off = ((nsp * OD + od) * OH + oh) * OW
                    * inner_stride * dst_dt_size;

            const size_t cb = std::div(nsp, CB).rem;

            jit_resampling_call_s args = jit_resampling_call_s();
            args.batch_of_sp_points_to_process = OW;
            args.src = src + src_off;
            args.dst = dst + dst_off;
            args.indices = &indices_w[0];
            args.post_ops_binary_rhs_arg_vec = post_ops_args.data();
            args.c_offset = static_cast<size_t>(cb * inner_stride);

            (*kernel_)(&args);
        });
    } else {
        assert(!"Invalid memory format kind.");
        return status::invalid_arguments;
    }

    return status::success;
}

status_t jit_uni_resampling_fwd_t::interpolate_linear(const uint8_t *src,
        uint8_t *dst, const std::vector<const void *> &post_ops_args) const {
    const size_t src_dt_size = pd()->get_conf().src_dt_size;
    const size_t dst_dt_size = pd()->get_conf().dst_dt_size;
    const size_t inner_stride = pd()->get_conf().inner_stride;

    const dim_t MB = pd()->MB();
    const dim_t C = pd()->C();
    const dim_t CB = utils::div_up(C, inner_stride);
    const dim_t nsp_outer = MB * CB;
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();

    if (pd()->get_conf().tag_kind == jit_memory_tag_kind_t::ncsp) {
        parallel_nd(MB, C, [&](dim_t mb, dim_t c) {
            const dim_t src_off = (mb * C + c) * ID * IH * IW * src_dt_size;
            const dim_t dst_off = (mb * C + c) * OD * OH * OW * dst_dt_size;

            jit_resampling_call_s args = jit_resampling_call_s();
            args.batch_of_sp_points_to_process = OW * OH * OD;
            args.src = src + src_off;
            args.dst = dst + dst_off;
            args.indices = &indices_[0];
            args.weights = &weights_[0];
            args.post_ops_binary_rhs_arg_vec = post_ops_args.data();
            args.c_offset = static_cast<size_t>(c);

            (*kernel_)(&args);
        });
    } else if (pd()->get_conf().tag_kind == jit_memory_tag_kind_t::nspc
            || pd()->get_conf().tag_kind == jit_memory_tag_kind_t::blocked) {
        const unsigned *indices_top = &indices_[2 * OW];
        const unsigned *indices_bottom = &indices_[2 * OW + OH];
        const unsigned *indices_front = &indices_[2 * (OW + OH)];
        const unsigned *indices_back = &indices_[2 * (OW + OH) + OD];
        const float *weights_top = &weights_[2 * OW];
        const float *weights_bottom = &weights_[2 * OW + OH];
        const float *weights_front = &weights_[2 * (OW + OH)];
        const float *weights_back = &weights_[2 * (OW + OH) + OD];

        parallel_nd(nsp_outer, OD, OH, [&](dim_t nsp, dim_t od, dim_t oh) {
            const dim_t src_off
                    = nsp * ID * IH * IW * inner_stride * src_dt_size;
            const dim_t dst_off = (((nsp * OD + od) * OH + oh) * OW)
                    * inner_stride * dst_dt_size;

            const size_t cb = std::div(nsp, CB).rem;

            jit_resampling_call_s args = jit_resampling_call_s();
            args.batch_of_sp_points_to_process = OW;
            args.src = src + src_off;
            args.dst = dst + dst_off;
            args.indices = &indices_[0];
            args.weights = &weights_[0];
            args.post_ops_binary_rhs_arg_vec = post_ops_args.data();
            args.c_offset = static_cast<size_t>(cb * inner_stride);

            args.src_offset_front = indices_front[od];
            args.src_offset_back = indices_back[od];
            args.src_offset_top = indices_top[oh];
            args.src_offset_bottom = indices_bottom[oh];
            args.weight_front = weights_front[od];
            args.weight_back = weights_back[od];
            args.weight_top = weights_top[oh];
            args.weight_bottom = weights_bottom[oh];

            (*kernel_)(&args);
        });
    } else {
        assert(!"Invalid memory format kind.");
        return status::invalid_arguments;
    }

    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
