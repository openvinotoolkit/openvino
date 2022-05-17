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

#include "gpu/ocl/gen9_wino_convolution.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;

static bool is_impl_optimal(conv_conf_t &conf, const convolution_desc_t &cd,
        const compute::gpu_arch_t arch) {
    if (cd.alg_kind == alg_kind::convolution_winograd) return true;

    int ow_blocks = conf.wino_ow / conf.ow_block;
    float ow_util = (float)conf.ow / conf.wino_ow;
    int oh_blocks = conf.wino_oh / conf.oh_block;
    float oh_util = (float)conf.oh / conf.wino_oh;
    int oc_blocks = conf.ocb;
    float oc_util = (float)conf.oc_without_padding / conf.wino_oc;
    float ic_util = (float)conf.ic_without_padding / conf.wino_ic;

    int blocks = ow_blocks * oh_blocks * oc_blocks;
    float utilization = ow_util * oh_util * oc_util * ic_util;
    float score;

    switch (arch) {
        case compute::gpu_arch_t::gen9:
            score = blocks * utilization;
            if (score >= 128 && utilization >= 0.50) return true;
            return false;
        case compute::gpu_arch_t::xe_lp:
            // Performance is poor with large oc*ic and small spatial, this is
            // likely due to overflowing cache and no blocking on ic.
            score = (float)conf.oc * conf.ic / (oh_blocks * ow_blocks);
            if (score < 32 * 1024 && utilization >= 0.50) return true;
            return false;
        default: return false;
    }
}

static void fwd_compute_block_sizes(
        conv_conf_t &conf, const compute::gpu_arch_t arch) {

    if (conf.ver == ver_16mb16c) {
        conf.mb_block = (conf.src_data_type == data_type::f16)
                ? (conf.mb % 32 == 0 ? 32 : 16)
                : 16;
    } else {
        conf.mb_block = 1;
    }

    //Using F(m, r) for r = 3 and tile_size = m + r - 1
    const int m = utils::div_up(conf.oh, 6) < utils::div_up(conf.oh, 4)
            ? 6
            : conf.oh > 2 ? 4 : 2;
    const int r = 3;
    conf.is_fused = true;

    conf.wino_m = m;
    conf.wino_r = r;
    conf.tile_size = m + r - 1;

    conf.vect_size = (arch == compute::gpu_arch_t::gen9)
            ? static_cast<int>(16 / types::data_type_size(conf.src_data_type))
            : 8;
    conf.oc_block = 16;
    conf.ic_block = nstl::min(conf.ic, 16);
    if (conf.src_data_type == data_type::f16)
        conf.wino_ic_block = 32;
    else if (arch != compute::gpu_arch_t::gen9 && conf.ow * conf.oh <= 256)
        conf.wino_ic_block = 32;
    else
        conf.wino_ic_block = 16;

    conf.ocb = utils::div_up(conf.oc, conf.oc_block);

    if (conf.is_fused) {
        conf.wino_oc_block = 16;
        conf.oh_block = conf.wino_m;
        conf.ow_block = conf.ow > 14 ? 14 : utils::rnd_up(conf.ow, 2);
    } else {
        conf.wino_oc_block = 32;
        conf.oh_block = 8;
        conf.ow_block = conf.wino_m;
    }

    // Used for the internal data transform
    conf.wino_ow = utils::rnd_up(conf.ow, conf.ow_block);
    conf.wino_iw = conf.wino_ow;
    conf.wino_oh = utils::rnd_up(conf.oh, conf.oh_block);
    conf.wino_ih = conf.wino_oh + conf.t_pad + conf.b_pad;
    conf.wino_ic = utils::rnd_up(conf.ic, conf.wino_ic_block);
    conf.wino_oc = utils::rnd_up(conf.oc, conf.wino_oc_block);
}

status_t gen9_wino_convolution_fwd_t::pd_t::init_conf(
        compute::compute_engine_t *engine) {

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    const memory_desc_wrapper bias_mdw(weights_md(1));

    set_default_conf(conf, cd, *src_md(), *weights_md(), *dst_md(),
            *weights_md(1), *attr());

    conf.ic = utils::rnd_up(conf.ic_without_padding, 16);
    conf.oc = utils::rnd_up(conf.oc_without_padding, 16);

    const bool is_wino_shape = conf.ndims == 4 && conf.kh == 3 && conf.kw == 3
            && conf.ngroups == 1 && conf.stride_h == 1 && conf.stride_w == 1
            && conf.dilate_h == 0 && conf.dilate_w == 0 && conf.l_pad < conf.kw
            && conf.r_pad < conf.kw && conf.t_pad < conf.kh
            && conf.b_pad < conf.kh;
    if (!is_wino_shape) return status::unimplemented;

    const bool is_16oc = conf.oc % 16 == 0;
    const bool is_16ic = conf.ic % 16 == 0;

    if (src_mdw.matches_one_of_tag(nhwc)
            && (dst_mdw.matches_one_of_tag(nhwc)
                    || dst_mdw.format_kind() == format_kind::any)) {
        // Technically this implementation currently requires ic is a multiple
        // of VTRANS_BLOCK = 4. This condition was not implemented yet due to no
        // known use case, and small IC is expected to have poor performance
        // because of extra work created by the current blocking.
        if (conf.ic_without_padding % 16 != 0
                || conf.oc_without_padding % 16 != 0)
            return status::unimplemented;
        conf.ver = ver_nhwc;
    } else if ((is_16oc && is_16ic)) {
        conf.ver = (conf.mb % 16 == 0) ? ver_16mb16c : ver_8ow16c;
    } else {
        return status::unimplemented;
    }

    const compute::gpu_arch_t arch = engine->device_info()->gpu_arch();
    fwd_compute_block_sizes(conf, arch);
    if (!is_impl_optimal(conf, cd, arch)) return status::unimplemented;

    size_t U_sz = conf.tile_size * conf.kh * conf.wino_ic * conf.wino_oc;
    size_t M_sz = 0, V_sz = 0;
    if (!conf.is_fused) {
        M_sz = conf.tile_size * conf.mb * conf.wino_oc * conf.wino_oh
                * conf.wino_ow;
        V_sz = conf.tile_size * conf.mb * conf.wino_ic * conf.wino_ih
                * conf.wino_iw;
    }

    // Limit max problem size since this method uses more memory
    if (U_sz + M_sz + V_sz > 300000000) return status::unimplemented;

    //Using F(m, r) for r = 3 and tile_size = m + r - 1
    if (!conf.is_fused) {
        conf.mb_block = 1;
        conf.lws_d[0] = 8;
        conf.lws_d[1] = 1;
        conf.lws_d[2] = 1;
        conf.gws_d[0] = (conf.wino_oc / conf.wino_oc_block) * conf.lws_d[0];
        conf.gws_d[1] = conf.wino_ow * (conf.wino_oh / conf.oh_block);
        conf.gws_d[2] = (conf.mb / conf.mb_block) * conf.tile_size;

        conf.U_lws_d[0] = 1;
        conf.U_lws_d[1] = 1;
        conf.U_lws_d[2] = 1;
        conf.U_gws_d[0] = 1;
        conf.U_gws_d[1] = 3; // kh or kw depending
        conf.U_gws_d[2] = conf.wino_ic * conf.wino_oc;

        conf.V_lws_d[0] = 1;
        conf.V_lws_d[1] = 1;
        conf.V_lws_d[2] = 1;
        conf.V_gws_d[0] = conf.wino_ow;
        conf.V_gws_d[1] = conf.wino_ih;
        conf.V_gws_d[2] = conf.wino_ic / conf.ic_block * conf.mb;

        conf.M_lws_d[0] = 1;
        conf.M_lws_d[1] = 1;
        conf.M_lws_d[2] = 1;
        conf.M_gws_d[0] = utils::div_up(conf.ow, conf.ow_block);
        conf.M_gws_d[1] = conf.oh;
        conf.M_gws_d[2] = conf.oc / conf.oc_block * conf.mb;
    } else {
        conf.mb_block = 1;
        conf.lws_d[0] = conf.wino_ic_block / 2;
        conf.lws_d[1] = 8;
        conf.lws_d[2] = 1;
        conf.gws_d[0]
                = utils::div_up(conf.wino_ow, conf.ow_block) * conf.lws_d[0];
        conf.gws_d[1]
                = utils::div_up(conf.wino_oh, conf.oh_block) * conf.lws_d[1];
        conf.gws_d[2] = (conf.mb / conf.mb_block)
                * (conf.wino_oc / conf.wino_oc_block);

        conf.U_lws_d[0] = conf.wino_ic_block / 2;
        conf.U_lws_d[1] = 1;
        conf.U_lws_d[2] = 1;
        conf.U_gws_d[0] = conf.wino_ic * conf.wino_oc / conf.vect_size;
        conf.U_gws_d[1] = 3;
        conf.U_gws_d[2] = 1; // kh or kw depending
    }

    format_tag_t src_tag, dst_tag, wei_tag;

    switch (conf.ver) {
        case ver_16mb16c:
            src_tag = NChw16n16c;
            dst_tag = NChw16n16c;
            wei_tag = conf.with_groups ? gOIhw16i16o : OIhw16i16o;
            break;
        case ver_8ow16c:
            src_tag = nChw16c;
            dst_tag = nChw16c;
            wei_tag = conf.with_groups ? gOIhw16i16o : OIhw16i16o;
            break;
        case ver_nhwc:
            src_tag = nhwc;
            dst_tag = nhwc;
            wei_tag = conf.with_groups ? gOIhw16i16o : OIhw16i16o;
            break;
        default: return status::unimplemented;
    }

    if (src_mdw.format_kind() == format_kind::any) {
        conf.src_tag = src_tag;
    } else {
        conf.src_tag = src_mdw.matches_one_of_tag(src_tag);
    }
    if (conf.src_tag != src_tag) return status::unimplemented;

    if (weights_mdw.format_kind() == format_kind::any) {
        conf.wei_tag = wei_tag;
    } else {
        conf.wei_tag = weights_mdw.matches_one_of_tag(wei_tag);
    }
    if (conf.wei_tag != wei_tag) return status::unimplemented;

    if (dst_mdw.format_kind() == format_kind::any) {
        conf.dst_tag = dst_tag;
    } else {
        conf.dst_tag = dst_mdw.matches_one_of_tag(dst_tag);
    }
    if (conf.dst_tag != dst_tag) return status::unimplemented;

    return status::success;
}

void gen9_wino_convolution_fwd_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();

    auto wei_data_t = this->desc()->weights_desc.data_type;
    size_t U_sz = conf.tile_size * conf.kh * conf.wino_ic * conf.wino_oc;
    scratchpad.book(key_wino_U, U_sz, types::data_type_size(wei_data_t),
            OCL_BUFFER_ALIGNMENT);

    if (!conf.is_fused) {
        auto dst_data_t = this->desc()->dst_desc.data_type;
        size_t M_sz = conf.tile_size * conf.mb * conf.wino_oc * conf.wino_oh
                * conf.wino_ow;
        scratchpad.book(key_wino_M, M_sz, types::data_type_size(dst_data_t),
                OCL_BUFFER_ALIGNMENT);

        auto src_data_t = this->desc()->src_desc.data_type;
        size_t V_sz = conf.tile_size * conf.mb * conf.wino_ic * conf.wino_ih
                * conf.wino_iw;
        scratchpad.book(key_wino_V, V_sz, types::data_type_size(src_data_t),
                OCL_BUFFER_ALIGNMENT);
    }
}

status_t gen9_wino_convolution_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("G", conf.ngroups);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OC", conf.oc);
    kernel_ctx.define_int("OD", conf.od);
    kernel_ctx.define_int("OH", conf.oh);
    kernel_ctx.define_int("OW", conf.ow);
    kernel_ctx.define_int("KD", conf.kd);
    kernel_ctx.define_int("KH", conf.kh);
    kernel_ctx.define_int("KW", conf.kw);
    kernel_ctx.define_int("PH", conf.t_pad);
    kernel_ctx.define_int("PW", conf.l_pad);
    kernel_ctx.define_int("OCB", conf.ocb);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OH_BLOCK", conf.oh_block);
    kernel_ctx.define_int("OW_BLOCK", conf.ow_block);
    kernel_ctx.define_int("OW_LAST", utils::rnd_dn(conf.ow, conf.ow_block));
    kernel_ctx.define_int("OWB", utils::div_up(conf.ow, conf.ow_block));
    kernel_ctx.define_int("OHB", utils::div_up(conf.oh, conf.oh_block));
    kernel_ctx.define_int("OC_WO_PADDING", conf.oc_without_padding);
    kernel_ctx.define_int("WINO_M", conf.wino_m);
    kernel_ctx.define_int("WINO_R", conf.wino_r);
    kernel_ctx.define_int("WINO_IC_BLOCK", conf.wino_ic_block);
    kernel_ctx.define_int("WINO_OC_BLOCK", conf.wino_oc_block);
    kernel_ctx.define_int("WINO_IC", conf.wino_ic);
    kernel_ctx.define_int("WINO_OC", conf.wino_oc);
    kernel_ctx.define_int("WINO_IH", conf.wino_ih);
    kernel_ctx.define_int("WINO_IW", conf.wino_iw);
    kernel_ctx.define_int("WINO_OH", conf.wino_oh);
    kernel_ctx.define_int("WINO_OW", conf.wino_ow);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);
    kernel_ctx.define_int("VECT_DT_N", conf.vect_size);

    kernel_ctx.set_data_type(conf.src_data_type);

    kernel_ctx.define_int("VER_8OW16C", conf.ver == ver_8ow16c);
    kernel_ctx.define_int("VER_16MB16C", conf.ver == ver_16mb16c);

    kernel_ctx.define_int("SRC_NHWC", utils::one_of(conf.src_tag, nhwc));
    kernel_ctx.define_int(
            "SRC_16N16C", utils::one_of(conf.src_tag, NChw16n16c));
    kernel_ctx.define_int("SRC_W16C", utils::one_of(conf.src_tag, nChw16c));

    kernel_ctx.define_int(
            "WEI_16I16O", utils::one_of(conf.wei_tag, gOIhw16i16o, OIhw16i16o));
    kernel_ctx.define_int("WEI_16I16O_FLIPPED",
            utils::one_of(conf.wei_tag, gIOhw16i16o, IOhw16i16o));

    kernel_ctx.define_int("DST_NHWC", utils::one_of(conf.src_tag, nhwc));
    kernel_ctx.define_int(
            "DST_16N16C", utils::one_of(conf.dst_tag, NChw16n16c));
    kernel_ctx.define_int("DST_W16C", utils::one_of(conf.dst_tag, nChw16c));

    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);

    dnnl_dims_t dst_dims;
    dst_dims[0] = conf.mb;
    dst_dims[1] = conf.oc_without_padding;
    dst_dims[2] = conf.ndims > 4 ? conf.od : conf.oh;
    dst_dims[3] = conf.ndims > 4 ? conf.oh : conf.ow;
    dst_dims[4] = conf.ow;
    def_attr_info(kernel_ctx, conf.attr_info, attr()->post_ops_, &dst_dims);

    kernel_ctx.print_options();
    return status::success;
}

status_t gen9_wino_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;
    const auto &attr_info = conf.attr_info;

    std::unique_ptr<memory_storage_t> wei_trans
            = ctx.get_scratchpad_grantor().get_memory_storage(key_wino_U);
    compute::kernel_arg_list_t wei_transform_args;
    wei_transform_args.set(0, *wei_trans);
    wei_transform_args.set(1, weights);
    auto wei_trans_nd_range = compute::nd_range_t(conf.U_gws_d, conf.U_lws_d);
    status_t status = parallel_for(
            ctx, wei_trans_nd_range, wei_trans_kernel_, wei_transform_args);

    if (conf.is_fused) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, dst);
        arg_list.set(1, src);
        arg_list.set(2, *wei_trans);
        arg_list.set(3, bias);
        append_post_ops_to_arg_list(ctx, arg_list, 4, pd()->attr()->post_ops_);
        auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
        status = parallel_for(ctx, nd_range, kernel_, arg_list);
    } else {
        std::unique_ptr<memory_storage_t> src_trans
                = ctx.get_scratchpad_grantor().get_memory_storage(key_wino_V);
        compute::kernel_arg_list_t src_transform_args;
        src_transform_args.set(0, *src_trans);
        src_transform_args.set(1, src);
        auto src_trans_nd_range
                = compute::nd_range_t(conf.V_gws_d, conf.V_lws_d);
        status = parallel_for(
                ctx, src_trans_nd_range, src_trans_kernel_, src_transform_args);

        std::unique_ptr<memory_storage_t> M_buf
                = ctx.get_scratchpad_grantor().get_memory_storage(key_wino_M);
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, *M_buf);
        arg_list.set(1, *src_trans);
        arg_list.set(2, *wei_trans);
        auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
        status = parallel_for(ctx, nd_range, kernel_, arg_list);

        compute::kernel_arg_list_t dst_transform_args;
        dst_transform_args.set(0, dst);
        dst_transform_args.set(1, *M_buf);
        dst_transform_args.set(2, bias);
        append_post_ops_to_arg_list(
                ctx, dst_transform_args, 3, pd()->attr()->post_ops_);
        auto dst_trans_nd_range
                = compute::nd_range_t(conf.M_gws_d, conf.M_lws_d);
        status = parallel_for(
                ctx, dst_trans_nd_range, dst_trans_kernel_, dst_transform_args);
    }

    if (attr_info.with_eltwise
            && !gpu_eltwise_fwd_pd_t::eltwise_preserves_zero(
                    attr_info.eltwise_alg, attr_info.eltwise_alpha,
                    attr_info.eltwise_beta)) {
        ctx.zero_pad_output(DNNL_ARG_DST);
    }
    return status;
}
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
