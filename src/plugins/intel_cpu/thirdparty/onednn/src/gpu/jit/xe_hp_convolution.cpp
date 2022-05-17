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

#include "gpu/jit/xe_hp_convolution.hpp"

#include "gpu/jit/xe_hp_conv_bwd_wei_kernel.hpp"
#include "gpu/jit/xe_hp_conv_data_kernel.hpp"
#include "gpu/ocl/ocl_gpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;

status_t xe_hp_convolution_data_common_init_conf(engine_t *engine,
        conv_conf_t &conf, const memory_desc_t &src_md,
        const memory_desc_t &wei_md, const memory_desc_t &dst_md) {

    const memory_desc_wrapper src_mdw(src_md);
    const memory_desc_wrapper wei_mdw(wei_md);
    const memory_desc_wrapper dst_mdw(dst_md);

    bool is_bwd = (conf.prop_kind == prop_kind::backward_data);
    bool is_fwd = !is_bwd;

    bool is_1st = utils::one_of(conf.ic, 3, 4) && (conf.kw == 7);
    bool is_int8
            = utils::one_of(conf.src_data_type, data_type::s8, data_type::u8);

    if (conf.with_groups && conf.ngroups > 1) return status::unimplemented;
    if (conf.ic < 16 && !is_1st) return status::unimplemented;
    if (conf.mb < 16) return status::unimplemented;
    if (conf.prop_kind == prop_kind::backward_data && is_1st)
        return status::unimplemented;

    // Reduce dimensions for 1x1 kernel.
    bool is_1x1 = (conf.kd * conf.kh * conf.kw == 1);
    bool is_stride1
            = (conf.stride_d == 1 && conf.stride_h == 1 && conf.stride_w == 1);
    bool is_eq_oi
            = (conf.od == conf.id && conf.oh == conf.ih && conf.ow == conf.iw);
    if (is_1x1 && is_stride1 && is_eq_oi) {
        assert(conf.f_pad == 0 && conf.t_pad == 0 && conf.l_pad == 0);
        conf.ow = conf.od * conf.oh * conf.ow;
        conf.iw = conf.id * conf.ih * conf.iw;
        conf.od = conf.id = conf.kd = 1;
        conf.oh = conf.ih = conf.kh = 1;
    }

    conf.mb_block = 32;
    conf.oc_block = 32;
    if (is_1st) {
        conf.ic_block = is_int8 ? 4 : 2;
    } else {
        conf.ic_block = is_int8 ? 32 : 16;
    }

    conf.oc_group = (conf.oc <= 64 ? 2 : 4);
    conf.sp_group = 4;

    // Dispatch between ver_v1 and ver_v2:

    // - v1 does not support non-unit strides with BWD_D
    auto can_use_v1 = [&]() {
        if (is_fwd) return true;
        return (conf.stride_d == 1 && conf.stride_h == 1 && conf.stride_w == 1);
    };

    auto can_use_v2_with_spatial
            = [&](int o, int i, int k, int p, int s, int d) {
                  if (d >= 7) return false;

                  o = utils::rnd_up(o, conf.sp_group);
                  int bound = std::numeric_limits<int16_t>::max();
                  if (is_fwd) {
                      int i_max = (o - 1) * s - p + (k - 1) * (1 + d);
                      return i_max <= bound;
                  }
                  // Backward.
                  int os_max = (o - 1) + p;
                  return os_max <= bound;
              };

    // - v2 does not support 1st convolution
    // - v2 does not support non-multiple OC
    // - v2 does full unrolling for (KD * KH * KW), hence (KDHW <= 9) limitation
    // - v2 uses 4 bits for (1 + dilation)
    // - v2 uses 16-bit integers for spatial
    auto can_use_v2 = [&]() {
        if (is_1st) return false;

        if (conf.kd * conf.kh * conf.kw > 9) return false;

        int oc_padded = utils::rnd_up(conf.oc, conf.oc_block);
        if ((oc_padded % (conf.oc_block * conf.oc_group) != 0)) return false;

        if (!can_use_v2_with_spatial(conf.od, conf.id, conf.kd, conf.f_pad,
                    conf.stride_d, conf.dilate_d))
            return false;
        if (!can_use_v2_with_spatial(conf.oh, conf.ih, conf.kh, conf.t_pad,
                    conf.stride_h, conf.dilate_h))
            return false;
        if (!can_use_v2_with_spatial(conf.ow, conf.iw, conf.kw, conf.l_pad,
                    conf.stride_w, conf.dilate_w))
            return false;

        if (is_bwd) {
            // Powers of 2 are supported only for strides.
            if ((conf.stride_d & (conf.stride_d - 1)) != 0) return false;
            if ((conf.stride_h & (conf.stride_h - 1)) != 0) return false;
            if ((conf.stride_w & (conf.stride_w - 1)) != 0) return false;
        }

        return true;
    };

    if (can_use_v2()) {
        conf.ver = ver_v2;
    } else if (can_use_v1()) {
        conf.ver = ver_v1;
    } else {
        return status::unimplemented;
    }

    conf.sub_group_size = 8;
    conf.gws_d[0] = utils::rnd_up(conf.oc, conf.oc_group * conf.oc_block)
            / conf.oc_block * 8;
    if (conf.ver == ver_v1) {
        conf.gws_d[1]
                = conf.od * conf.oh * utils::rnd_up(conf.ow, conf.sp_group);
    } else {
        conf.gws_d[1]
                = utils::rnd_up(conf.od * conf.oh * conf.ow, conf.sp_group);
    }
    conf.gws_d[2] = utils::div_up(conf.mb, conf.mb_block);
    conf.lws_d[0] = 8 * conf.oc_group;
    conf.lws_d[1] = conf.sp_group;
    conf.lws_d[2] = 1;

    format_tag_t src_tag;
    format_tag_t wei_tag;
    format_tag_t dst_tag;

    auto tag_4n2c = utils::pick(conf.ndims - 3, ABc4a2b, ABcd4a2b, ABcde4a2b);
    auto tag_4n4c = utils::pick(conf.ndims - 3, ABc4a4b, ABcd4a4b, ABcde4a4b);
    auto tag_32n16c
            = utils::pick(conf.ndims - 3, ABc32a16b, ABcd32a16b, ABcde32a16b);
    auto tag_32n32c
            = utils::pick(conf.ndims - 3, ABc32a32b, ABcd32a32b, ABcde32a32b);
    auto tag_40n16c
            = utils::pick(conf.ndims - 3, ABc40a16b, ABcd40a16b, ABcde40a16b);
    auto tag_40n32c
            = utils::pick(conf.ndims - 3, ABc40a32b, ABcd40a32b, ABcde40a32b);

    if (conf.prop_kind == prop_kind::backward_data) {
        // only f16, bf16 support, so only 16c variants
        auto tag_g_4i8o8i2o = utils::pick(
                conf.ndims - 3, aCBd4c8b8c2b, aCBde4c8b8c2b, aCBdef4c8b8c2b);
        auto tag_4i8o8i2o = utils::pick(
                conf.ndims - 3, BAc4b8a8b2a, BAcd4b8a8b2a, BAcde4b8a8b2a);

        src_tag = conf.mb_block == 32 ? tag_32n16c : tag_40n16c;
        wei_tag = conf.with_groups ? tag_g_4i8o8i2o : tag_4i8o8i2o;
        dst_tag = conf.mb_block == 32 ? tag_32n16c : tag_40n16c;
    } else { // forward
        auto tag_g_4o8i8o2i = utils::pick(
                conf.ndims - 3, aBCd4b8c8b2c, aBCde4b8c8b2c, aBCdef4b8c8b2c);
        auto tag_g_4o8i8o4i = utils::pick(
                conf.ndims - 3, aBCd4b8c8b4c, aBCde4b8c8b4c, aBCdef4b8c8b4c);
        auto tag_g_8o2i
                = utils::pick(conf.ndims - 3, aBCd8b2c, aBCde8b2c, aBCdef8b2c);
        auto tag_g_8o4i
                = utils::pick(conf.ndims - 3, aBCd8b4c, aBCde8b4c, aBCdef8b4c);

        auto tag_4o8i8o2i = utils::pick(
                conf.ndims - 3, ABc4a8b8a2b, ABcd4a8b8a2b, ABcde4a8b8a2b);
        auto tag_4o8i8o4i = utils::pick(
                conf.ndims - 3, ABc4a8b8a4b, ABcd4a8b8a4b, ABcde4a8b8a4b);

        auto tag_8o2i
                = utils::pick(conf.ndims - 3, ABc8a2b, ABcd8a2b, ABcde8a2b);
        auto tag_8o4i
                = utils::pick(conf.ndims - 3, ABc8a4b, ABcd8a4b, ABcde8a4b);

        if (is_int8) {
            src_tag = is_1st ? tag_4n4c
                             : conf.mb_block == 32 ? tag_32n32c : tag_40n32c;
            if (is_1st) {
                wei_tag = conf.with_groups ? tag_g_8o4i : tag_8o4i;
            } else {
                wei_tag = conf.with_groups ? tag_g_4o8i8o4i : tag_4o8i8o4i;
            }
            dst_tag = conf.mb_block == 32 ? tag_32n32c : tag_40n32c;
        } else { // f16 or bf16.
            src_tag = is_1st ? tag_4n2c
                             : conf.mb_block == 32 ? tag_32n16c : tag_40n16c;
            if (is_1st) {
                wei_tag = conf.with_groups ? tag_g_8o2i : tag_8o2i;
            } else {
                wei_tag = conf.with_groups ? tag_g_4o8i8o2i : tag_4o8i8o2i;
            }
            dst_tag = conf.mb_block == 32 ? tag_32n16c : tag_40n16c;
        }
    }

    if (src_mdw.format_kind() != format_kind::any
            && src_mdw.matches_one_of_tag(src_tag) == format_tag::undef)
        return status::unimplemented;

    if (wei_mdw.format_kind() != format_kind::any
            && wei_mdw.matches_one_of_tag(wei_tag) == format_tag::undef)
        return status::unimplemented;

    if (dst_mdw.format_kind() != format_kind::any
            && dst_mdw.matches_one_of_tag(dst_tag) == format_tag::undef)
        return status::unimplemented;

    conf.src_tag = src_tag;
    conf.wei_tag = wei_tag;
    conf.dst_tag = dst_tag;

    return status::success;
}

status_t xe_hp_convolution_fwd_t::pd_t::init_conf(engine_t *engine) {
    const convolution_desc_t &cd = *desc();

    set_default_conf(conf, cd, *src_md(), *weights_md(), *dst_md(),
            *weights_md(1), *attr());
    return xe_hp_convolution_data_common_init_conf(
            engine, conf, *src_md(), *weights_md(), *dst_md());
}

status_t xe_hp_convolution_fwd_t::init(engine_t *engine) {
    CHECK(xe_hp_conv_data_create_kernel(
            pd()->conf, pd()->attr()->post_ops_, &kernel_, this, engine));
    return status::success;
}

status_t xe_hp_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &wei = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bia = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;
    const auto &attr_info = conf.attr_info;

    auto &oscales
            = (attr_info.with_per_oc_oscales && !attr_info.with_runtime_oscales)
            ? CTX_GPU_RES_STORAGE(OSCALES_)
            : CTX_IN_STORAGE(DNNL_ARG_ATTR_OUTPUT_SCALES);
    auto &binaries = attr_info.with_binary
            ? CTX_IN_STORAGE(
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(attr_info.binary_idx)
                    | DNNL_ARG_SRC_1)
            : memory_storage_t::empty_storage();

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, wei);
    arg_list.set(2, bia);
    arg_list.set(3, dst);
    arg_list.set(4, oscales);
    arg_list.set(5, attr_info.common_oscales);
    arg_list.set(6, binaries);

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

status_t xe_hp_convolution_bwd_data_t::pd_t::init_conf(engine_t *engine) {
    const convolution_desc_t &cd = *desc();

    // The data kernel is expressed in terms of FWD convolution
    // So we need to swap diff_src and diff_dst mds to fill the conf properly
    set_default_conf(conf, cd, *diff_dst_md(), *weights_md(), *diff_src_md(),
            *weights_md(1), *attr());
    return xe_hp_convolution_data_common_init_conf(
            engine, conf, *diff_dst_md(), *weights_md(), *diff_src_md());
}

status_t xe_hp_convolution_bwd_data_t::init(engine_t *engine) {
    CHECK(xe_hp_conv_data_create_kernel(
            pd()->conf, pd()->attr()->post_ops_, &kernel_, this, engine));
    return status::success;
}

status_t xe_hp_convolution_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    // TODO: we can add bias here if we want to support deconvolution
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &wei = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_dst);
    arg_list.set(1, wei);
    arg_list.set(2, memory_storage_t::empty_storage());
    arg_list.set(3, diff_src);
    arg_list.set(4, memory_storage_t::empty_storage());
    arg_list.set(5, memory_storage_t::empty_storage());
    arg_list.set(6, memory_storage_t::empty_storage());

    const auto &conf = pd()->conf;
    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

status_t xe_hp_convolution_bwd_weights_t::pd_t::init_conf(engine_t *engine) {
    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper wei_mdw(diff_weights_md());
    const memory_desc_wrapper dst_mdw(diff_dst_md());
    const memory_desc_wrapper bia_mdw(diff_weights_md(1));

    set_default_conf(conf, cd, *src_md(), *diff_weights_md(), *diff_dst_md(),
            *diff_weights_md(1), *attr());

    if (conf.is_depthwise) return status::unimplemented;
    if (conf.with_groups && conf.ngroups > 1) return status::unimplemented;
    if (conf.mb < 16) return status::unimplemented;

    // TODO: only 128ic x 128oc workgroup, add other configurations later
    if (conf.ic % 128 != 0 || conf.oc % 128 != 0) return status::unimplemented;

    conf.mb_block = 32;
    conf.oc_block = 16;
    conf.ic_block = 16;

    conf.ic_blk_wg = 4;
    conf.oc_blk_wg = 4;

    conf.oc_group = 1;
    conf.ow_group = 1;

    conf.sub_group_size = 8;

    conf.icb = utils::div_up(conf.ic, conf.ic_block);
    conf.ocb = utils::div_up(conf.oc, conf.oc_block);

    const int num_mb_blocks = utils::div_up(conf.mb, conf.mb_block);
    const int kdhw_size = conf.kd * conf.kh * conf.kw;

    conf.sp_block = 64;
    conf.osp_chunk = utils::div_up(conf.od * conf.oh * conf.ow, conf.sp_block);

    conf.gws_d[0]
            = kdhw_size * (conf.icb / 2) * (conf.ocb / 2) * conf.sub_group_size;
    conf.gws_d[1] = conf.osp_chunk;
    conf.gws_d[2] = conf.ngroups * num_mb_blocks;

    conf.lws_d[0] = conf.ic_blk_wg * conf.oc_blk_wg * conf.sub_group_size;
    conf.lws_d[1] = 1;
    conf.lws_d[2] = 1;

    format_tag_t src_tag
            = utils::pick(conf.ndims - 3, NCw32n16c, NChw32n16c, NCdhw32n16c);
    format_tag_t dst_tag
            = utils::pick(conf.ndims - 3, NCw32n16c, NChw32n16c, NCdhw32n16c);

    format_tag_t wei_tag = conf.with_groups
            ? utils::pick(conf.ndims - 3, gOIw16o16i, gOIhw16o16i, gOIdhw16o16i)
            : utils::pick(conf.ndims - 3, OIw16o16i, OIhw16o16i, OIdhw16o16i);

    if (src_mdw.format_kind() != format_kind::any
            && src_mdw.matches_one_of_tag(src_tag) == format_tag::undef)
        return status::unimplemented;

    if (wei_mdw.format_kind() != format_kind::any
            && wei_mdw.matches_one_of_tag(wei_tag) == format_tag::undef)
        return status::unimplemented;

    if (dst_mdw.format_kind() != format_kind::any
            && dst_mdw.matches_one_of_tag(dst_tag) == format_tag::undef)
        return status::unimplemented;

    conf.src_tag = src_tag;
    conf.wei_tag = wei_tag;
    conf.dst_tag = dst_tag;

    return status::success;
}

void xe_hp_convolution_bwd_weights_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();

    if (conf.weights_data_type == data_type::bf16) {
        size_t size = conf.ngroups * utils::rnd_up(conf.oc, conf.oc_block)
                * utils::rnd_up(conf.ic, conf.ic_block) * conf.kd * conf.kh
                * conf.kw;
        scratchpad.book(memory_tracking::names::key_conv_wei_reduction, size,
                types::data_type_size(data_type::f32));
    }
    if (conf.bias_data_type == data_type::bf16) {
        size_t size = conf.ngroups * utils::rnd_up(conf.oc, conf.oc_block);
        scratchpad.book(memory_tracking::names::key_conv_bia_reduction, size,
                types::data_type_size(data_type::f32));
    }
}

status_t xe_hp_convolution_bwd_weights_t::init(engine_t *engine) {
    return xe_hp_conv_bwd_weights_create_kernels(
            pd()->conf, kernels_, this, engine);
}

status_t xe_hp_convolution_bwd_weights_t::execute(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_wei = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS);
    auto &diff_bia = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

    std::unique_ptr<memory_storage_t> temp_wei, temp_bia;

    const auto &conf = pd()->conf;
    const bool is_wei_bf16 = conf.weights_data_type == data_type::bf16;
    const bool is_bia_bf16 = conf.bias_data_type == data_type::bf16;

    if (is_wei_bf16)
        temp_wei = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_conv_wei_reduction);
    if (is_bia_bf16)
        temp_bia = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_conv_bia_reduction);

    auto &conv_wei = is_wei_bf16 ? *temp_wei : diff_wei;
    auto &conv_bia = conf.with_bias ? (is_bia_bf16 ? *temp_bia : diff_bia)
                                    : memory_storage_t::empty_storage();

    {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, conv_wei);
        arg_list.set(1, conv_bia);

        const size_t gws[3] = {size_t(conf.icb * conf.kd * conf.kh * conf.kw),
                size_t(conf.ngroups * conf.ocb), 1};
        const size_t lws[3] = {1, 1, 1};

        auto nd_range = compute::nd_range_t(gws, lws);
        CHECK(parallel_for(ctx, nd_range, kernels_[0], arg_list));
    }
    {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, src);
        arg_list.set(1, conv_wei);
        arg_list.set(2, conv_bia);
        arg_list.set(3, diff_dst);

        auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
        CHECK(parallel_for(ctx, nd_range, kernels_[1], arg_list));
    }
    if (is_wei_bf16 || is_bia_bf16) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, diff_wei);
        arg_list.set(1, diff_bia);
        arg_list.set(2, conv_wei);
        arg_list.set(3, conv_bia);

        const size_t gws[3] = {
                size_t(conf.ic_block * conf.icb * conf.kd * conf.kh * conf.kw),
                size_t(conf.ngroups * conf.ocb), 1};
        const size_t lws[3] = {1, 1, 1};

        auto nd_range = compute::nd_range_t(gws, lws);
        CHECK(parallel_for(ctx, nd_range, kernels_[2], arg_list));
    }

    return status::success;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
