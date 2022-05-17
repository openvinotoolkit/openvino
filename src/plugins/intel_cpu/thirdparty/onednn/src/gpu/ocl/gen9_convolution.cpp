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

#include "gpu/ocl/gen9_convolution.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/reorder.hpp"
#include "common/type_helpers.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "gpu/primitive_conf.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;

static void fwd_compute_block_sizes(
        conv_conf_t &conf, const convolution_pd_t *pd) {

    int max_ow_block = (conf.src_data_type == data_type::f16 ? 20 : 16);
    if (conf.ver == ver_16mb16c || conf.ver == ver_32mb16c) {
        max_ow_block = 1;
    } else if (conf.is_depthwise || conf.ver == ver_1stconv) {
        max_ow_block = 8;
    }
    max_ow_block = nstl::min(conf.ow, max_ow_block);

    if (conf.ver == ver_16mb16c) {
        conf.mb_block
                = (conf.src_data_type == data_type::f16 && !conf.is_depthwise)
                ? (conf.mb % 32 == 0 ? 32 : 16)
                : 16;
    } else if (conf.ver == ver_32mb16c) {
        conf.mb_block = 32;
    } else {
        conf.mb_block = 1;
    }

    conf.ow_block = utils::max_div(conf.ow, max_ow_block);

    if (conf.ow_block < max_ow_block / 2) {
        float min_tail_ratio = 1;
        int best_ow_block = -1;
        for (int ow_block = 8; ow_block <= max_ow_block; ow_block++) {
            float tail_ratio
                    = (ow_block - (conf.ow % ow_block)) / (float)conf.ow;
            if (tail_ratio <= min_tail_ratio) {
                min_tail_ratio = tail_ratio;
                best_ow_block = ow_block;
            }
        }
        assert(best_ow_block > 0);
        conf.ow_block = best_ow_block;
    }

    if (conf.is_depthwise) {
        conf.oc_block = 16;
        conf.ic_block = 16;
        conf.omb = conf.mb_block;
        return;
    }

    if (conf.ver == ver_1stconv && conf.mb_block == 1 && conf.oc % 32 == 0) {
        conf.oc_block = 32;
    } else {
        conf.oc_block = 16;
    }
    conf.ic_block = nstl::min(conf.ic, 16);

    conf.omb = (conf.mb_block == 1 && conf.mb % 16 == 0) ? 16 : conf.mb_block;
    conf.ocb = utils::max_div(conf.oc / 16, 8) * 16;
}

status_t gen9_convolution_fwd_t::pd_t::init_conf(engine_t *engine) {

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    const memory_desc_wrapper bias_mdw(weights_md(1));

    set_default_conf(conf, cd, *src_md(), *weights_md(), *dst_md(),
            *weights_md(1), *attr());

    const bool int8_dst = conf.dst_data_type == data_type::s8;
    const bool is_src_nhwc
            = src_mdw.matches_one_of_tag(nwc, nhwc, ndhwc) != format_tag::undef;
    const bool is_dst_nhwc
            = dst_mdw.matches_one_of_tag(nwc, nhwc, ndhwc) != format_tag::undef;
    const bool is_nhwc = is_src_nhwc || is_dst_nhwc;

    const bool is_1stconv = conf.ic_without_padding == 3;
    const bool is_depthwise = conf.with_groups && (conf.ic_without_padding == 1)
            && (conf.oc_without_padding == 1);

    conf.is_nhwc = is_1stconv ? is_dst_nhwc : is_nhwc;
    conf.is_depthwise = is_depthwise;

    const int out_block = int8_dst && !is_1stconv ? 32 : 16;
    if (is_1stconv || (conf.with_groups && conf.ngroups > 1)) {
        conf.ic = conf.ic_without_padding;
        conf.oc = is_1stconv ? utils::rnd_up(conf.oc_without_padding, out_block)
                             : conf.oc_without_padding;
    } else {
        conf.ic = utils::rnd_up(conf.ic_without_padding, 16);
        conf.oc = utils::rnd_up(conf.oc_without_padding, out_block);
    }

    conf.ngroups_without_padding = conf.ngroups;
    if (is_depthwise)
        conf.ngroups = utils::rnd_up(conf.ngroups, int8_dst ? 32 : 16);

    const bool is_dw_16g = (conf.is_depthwise && conf.ngroups % 16 == 0);
    const bool is_16oc = conf.oc % out_block == 0;
    const bool is_16ic = conf.ic % 16 == 0;

    conf.mb_block = 1;
    conf.oc_block = 1;
    conf.ic_block = 1;
    conf.od_block = 1;
    conf.oh_block = 1;
    conf.ow_block = 1;
    conf.omb = 1;
    conf.ocb = 1;
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    const bool is_xe_hp_plus
            = compute_engine->is_xe_hp() || compute_engine->is_xe_hpg();
    const bool has_non_uniform_wg
            = compute_engine->mayiuse_non_uniform_work_groups();

    if (conf.is_nhwc) {
        if (!utils::one_of(src_mdw.data_type(), f32, f16))
            return status::unimplemented;
        if (conf.is_depthwise && conf.ngroups_without_padding % 16)
            return status::unimplemented;
        // TODO: Add group convolution support in NHWC kernel.
        if (!conf.is_depthwise && conf.ngroups > 1 && !(is_16oc && is_16ic)) {
            return status::unimplemented;
        }
        if (int8_dst) { return status::unimplemented; }
        conf.ver = ver_nhwc;
    } else if (is_1stconv) {
        if (!is_16oc) return status::unimplemented;
        conf.ver = ver_1stconv;
    } else if ((is_16oc && is_16ic) || is_dw_16g) {
        if (conf.mb % 32 == 0 && conf.is_depthwise
                && utils::one_of(src_mdw.data_type(), bf16, f16)
                && is_xe_hp_plus) {
            conf.ver = ver_32mb16c;
        } else {
            conf.ver = (conf.mb % out_block == 0) ? ver_16mb16c : ver_8ow16c;
        }
    } else {
        return status::unimplemented;
    }

    const bool is_fp16 = src_mdw.data_type() == data_type::f16;

    switch (conf.ver) {
        case ver_nhwc: {
            conf.mb_block = 1;
            conf.oc_block = 16;
            conf.ic_block = is_1stconv ? 1 : 16;

            int max_ow_block = (conf.kw > 1) ? 8 : 16;
            if (conf.oc <= 64 && conf.ic <= 64) max_ow_block = 8;

            conf.ow_block = utils::max_div(conf.ow, max_ow_block);

            if (conf.ow_block <= 8) {
                int max_tail = 0;
                for (int j = 8; j < max_ow_block; j++) {
                    if (conf.ow % j > max_tail) {
                        max_tail = conf.ow % j;
                        conf.ow_block = j;
                    }
                }
            }
            if (conf.ow_block <= 8) conf.ow_block = 8;
            if (conf.ow <= 8 || conf.oc <= 32) conf.ow_block = 8;

            conf.oh_block = 1;
            conf.sub_group_size = 16;
            conf.lws_d[0] = 16;
            conf.lws_d[1] = 1;
            conf.lws_d[2] = 1;

            int max_oc_block = 8;
            if (conf.is_depthwise) {
                conf.ocb = conf.ngroups;
            } else {
                conf.ocb = conf.oc_block
                        * utils::max_div(utils::div_up(conf.oc, conf.oc_block),
                                max_oc_block);
            }

            conf.gws_d[0] = conf.ocb;
            conf.gws_d[1] = utils::div_up(conf.oh, conf.oh_block)
                    * utils::div_up(conf.ow, conf.ow_block) * conf.od;
            if (conf.is_depthwise) {
                conf.gws_d[2] = conf.mb;
            } else {
                conf.gws_d[2] = conf.mb * utils::div_up(conf.oc, conf.ocb)
                        * conf.ngroups;
            }
        } break;
        case ver_1stconv:
        case ver_8ow16c:
        case ver_16mb16c:
        case ver_32mb16c: {
            fwd_compute_block_sizes(conf, this);
            conf.sub_group_size = 16;
            conf.gws_d[0] = conf.ngroups * conf.ocb / (conf.oc_block / 16);
            conf.gws_d[1]
                    = (conf.od * conf.oh * utils::div_up(conf.ow, conf.ow_block)
                            * (conf.omb / conf.mb_block));
            conf.gws_d[2] = (conf.oc / conf.ocb) * (conf.mb / conf.omb);
            conf.lws_d[0] = is_xe_hp_plus ? 32 : 16;
            conf.lws_d[1] = 1;
            conf.lws_d[2] = 1;
            break;
        }
        default: return status::unimplemented;
    }

    maybe_fix_non_uniform_work_sizes(has_non_uniform_wg, conf);

    format_tag_t src_tag, dst_tag, wei_tag;

    switch (conf.ver) {
        case ver_nhwc:
            src_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            dst_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            if (is_1stconv) {
                wei_tag = conf.with_groups ? utils::pick(
                                  conf.ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                                           : utils::pick(conf.ndims - 3, Owi16o,
                                                   Ohwi16o, Odhwi16o);
            } else if (conf.is_depthwise) {
                wei_tag = utils::pick(
                        conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g);
            } else {
                wei_tag = conf.with_groups
                        ? utils::pick(conf.ndims - 3, gOIw16i16o, gOIhw16i16o,
                                gOIdhw16i16o)
                        : utils::pick(conf.ndims - 3, OIw16i16o, OIhw16i16o,
                                OIdhw16i16o);
            }
            break;
        case ver_1stconv:
            if (is_src_nhwc)
                src_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            else
                src_tag = utils::pick(conf.ndims - 3, ncw, nchw, ncdhw);

            if (is_xe_hp_plus && is_fp16) {
                dst_tag = conf.mb % 32 == 0 ? utils::pick(conf.ndims - 3,
                                  NCw32n16c, NChw32n16c, NCdhw32n16c)
                                            : utils::pick(conf.ndims - 3,
                                                    nCw16c, nChw16c, nCdhw16c);
            } else {
                dst_tag = conf.mb % 16 == 0 ? utils::pick(conf.ndims - 3,
                                  NCw16n16c, NChw16n16c, NCdhw16n16c)
                                            : utils::pick(conf.ndims - 3,
                                                    nCw16c, nChw16c, nCdhw16c);
            }
            wei_tag = conf.with_groups
                    ? utils::pick(conf.ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                    : utils::pick(conf.ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
            break;
        case ver_16mb16c:
            src_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            dst_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                                        : utils::pick(conf.ndims - 3, IOw16i16o,
                                                IOhw16i16o, IOdhw16i16o));
            break;
        case ver_32mb16c:
            src_tag = utils::pick(
                    conf.ndims - 3, NCw32n16c, NChw32n16c, NCdhw32n16c);
            dst_tag = utils::pick(
                    conf.ndims - 3, NCw32n16c, NChw32n16c, NCdhw32n16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                                        : utils::pick(conf.ndims - 3, IOw16i16o,
                                                IOhw16i16o, IOdhw16i16o));
            break;
        case ver_8ow16c:
            src_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gOIw16i16o, gOIhw16i16o, gOIdhw16i16o)
                                        : utils::pick(conf.ndims - 3, OIw16i16o,
                                                OIhw16i16o, OIdhw16i16o));
            break;
        default: return status::unimplemented;
    }
    if (int8_dst) {
        if (is_1stconv && conf.ic_without_padding < 4) {
            dst_tag = utils::pick(conf.ndims - 3, ncw, nchw, ncdhw);
        } else if (conf.ver == ver_16mb16c || conf.ver == ver_32mb16c) {
            dst_tag = utils::pick(
                    conf.ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
        } else {
            dst_tag = utils::pick(conf.ndims - 3, nCw32c, nChw32c, nCdhw32c);
        }
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

    conf.is_src_nchw = utils::one_of(src_tag, ncw, nchw, ncdhw);
    conf.is_src_nhwc = utils::one_of(src_tag, nwc, nhwc, ndhwc);

    return status::success;
}

status_t gen9_convolution_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("IS_DW", conf.is_depthwise);
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
    kernel_ctx.define_int("SD", conf.stride_d);
    kernel_ctx.define_int("SH", conf.stride_h);
    kernel_ctx.define_int("SW", conf.stride_w);
    kernel_ctx.define_int("PD", conf.f_pad);
    kernel_ctx.define_int("PH", conf.t_pad);
    kernel_ctx.define_int("PW", conf.l_pad);
    kernel_ctx.define_int("PD_R", conf.back_pad);
    kernel_ctx.define_int("PH_R", conf.b_pad);
    kernel_ctx.define_int("PW_R", conf.r_pad);
    kernel_ctx.define_int("DD", conf.dilate_d);
    kernel_ctx.define_int("DH", conf.dilate_h);
    kernel_ctx.define_int("DW", conf.dilate_w);
    kernel_ctx.define_int("OW_PADDED", utils::rnd_up(conf.ow, 4));
    kernel_ctx.define_int("OC_PADDED", conf.oc);
    kernel_ctx.define_int("OMB", conf.omb);
    kernel_ctx.define_int("OCB", conf.ocb);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OH_BLOCK", conf.oh_block);
    kernel_ctx.define_int("OW_BLOCK", conf.ow_block);
    kernel_ctx.define_int("OW_LAST", utils::rnd_dn(conf.ow, conf.ow_block));
    kernel_ctx.define_int("OWB", utils::div_up(conf.ow, conf.ow_block));
    kernel_ctx.define_int("OHB", utils::div_up(conf.oh, conf.oh_block));
    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);
    kernel_ctx.define_int("G_WO_PADDING", conf.ngroups_without_padding);
    kernel_ctx.define_int("IC_WO_PADDING", conf.ic_without_padding);
    kernel_ctx.define_int("OC_WO_PADDING", conf.oc_without_padding);
    kernel_ctx.define_int("OC_GROUP", conf.lws_d[0] / 8);
    kernel_ctx.define_int("MB_GROUP", 1);
    kernel_ctx.define_int("SP_GROUP", conf.lws_d[1]);
    if (conf.kw == 1)
        kernel_ctx.define_int("SRC_SP_GROUP", conf.lws_d[1] + conf.kw - 1);
    else
        kernel_ctx.define_int(
                "SRC_SP_GROUP", conf.stride_w * (conf.lws_d[1] - 1) + conf.kw);

    kernel_ctx.set_data_type(conf.src_data_type);
    def_data_type(kernel_ctx, conf.dst_data_type, "DST");

    kernel_ctx.define_int("VER_1STCONV", conf.ver == ver_1stconv);
    kernel_ctx.define_int("VER_8OW16C", conf.ver == ver_8ow16c);
    kernel_ctx.define_int("VER_16MB16C", conf.ver == ver_16mb16c);
    kernel_ctx.define_int("VER_32MB16C", conf.ver == ver_32mb16c);

    kernel_ctx.define_int("SRC_NCHW", conf.is_src_nchw);
    kernel_ctx.define_int("SRC_NHWC", conf.is_src_nhwc);
    kernel_ctx.define_int("SRC_16N16C",
            utils::one_of(conf.src_tag, NCw16n16c, NChw16n16c, NCdhw16n16c));
    kernel_ctx.define_int(
            "SRC_W16C", utils::one_of(conf.src_tag, nCw16c, nChw16c, nCdhw16c));

    kernel_ctx.define_int("WEI_I16O",
            utils::one_of(conf.wei_tag, gOwi16o, gOhwi16o, gOdhwi16o, Owi16o,
                    Ohwi16o, Odhwi16o));
    kernel_ctx.define_int("WEI_16I16O",
            utils::one_of(conf.wei_tag, gOIw16i16o, gOIhw16i16o, gOIdhw16i16o,
                    OIw16i16o, OIhw16i16o, OIdhw16i16o));
    kernel_ctx.define_int("WEI_16I16O_FLIPPED",
            utils::one_of(conf.wei_tag, gIOw16i16o, gIOhw16i16o, gIOdhw16i16o,
                    IOw16i16o, IOhw16i16o, IOdhw16i16o));

    kernel_ctx.define_int(
            "DST_W16C", utils::one_of(conf.dst_tag, nCw16c, nChw16c, nCdhw16c));
    kernel_ctx.define_int("DST_16N16C",
            utils::one_of(conf.dst_tag, NCw16n16c, NChw16n16c, NCdhw16n16c));
    kernel_ctx.define_int("DST_32N16C",
            utils::one_of(conf.dst_tag, NCw32n16c, NChw32n16c, NCdhw32n16c));
    kernel_ctx.define_int("DST_32N32C",
            utils::one_of(conf.dst_tag, NCw32n32c, NChw32n32c, NCdhw32n32c));
    kernel_ctx.define_int(
            "DST_W32C", utils::one_of(conf.dst_tag, nCw32c, nChw32c, nCdhw32c));
    kernel_ctx.define_int(
            "DST_NCHW", utils::one_of(conf.dst_tag, ncw, nchw, ncdhw));

    kernel_ctx.define_int("GWS_0", conf.gws_d[0]);
    kernel_ctx.define_int("GWS_1", conf.gws_d[1]);
    kernel_ctx.define_int("GWS_2", conf.gws_d[2]);

    kernel_ctx.define_int("GWS_ORIG_0", conf.gws_orig_d[0]);
    kernel_ctx.define_int("GWS_ORIG_1", conf.gws_orig_d[1]);
    kernel_ctx.define_int("GWS_ORIG_2", conf.gws_orig_d[2]);

    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    dnnl_dims_t dst_dims;
    dst_dims[0] = conf.mb;
    dst_dims[1] = conf.ngroups_without_padding * conf.oc_without_padding;
    dst_dims[2] = conf.ndims > 4 ? conf.od : conf.oh;
    dst_dims[3] = conf.ndims > 4 ? conf.oh : conf.ow;
    dst_dims[4] = conf.ow;
    kernel_ctx.add_option("-cl-std=CL2.0");
    def_attr_info(kernel_ctx, conf.attr_info, attr()->post_ops_, &dst_dims);

    kernel_ctx.print_options();
    return status::success;
}

status_t gen9_convolution_bwd_data_t::pd_t::init_conf(engine_t *engine) {
    using namespace dnnl::impl::format_tag;
    using namespace data_type;

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(diff_src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(diff_dst_md());
    const memory_desc_wrapper bias_mdw(weights_md(1));

    set_default_conf(conf, cd, *diff_src_md(), *weights_md(), *diff_dst_md(),
            *weights_md(1), *attr());
    const bool is_nhwc
            = src_mdw.matches_one_of_tag(nwc, nhwc, ndhwc) != format_tag::undef
            || dst_mdw.matches_one_of_tag(nwc, nhwc, ndhwc)
                    != format_tag::undef;
    const bool is_1stconv = conf.ic_without_padding == 3;
    const bool is_depthwise = conf.with_groups && (conf.ic_without_padding == 1)
            && (conf.oc_without_padding == 1);
    conf.is_nhwc = is_nhwc;
    conf.is_depthwise = is_depthwise;

    if (is_nhwc && (is_depthwise || is_1stconv)) return status::unimplemented;

    if (is_1stconv || (conf.with_groups && conf.ngroups > 1) || is_depthwise) {
        conf.ic = conf.ic_without_padding;
        conf.oc = is_1stconv ? utils::rnd_up(conf.oc_without_padding, 16)
                             : conf.oc_without_padding;
    } else {
        conf.ic = utils::rnd_up(conf.ic_without_padding, 16);
        conf.oc = utils::rnd_up(conf.oc_without_padding, 16);
    }
    conf.ngroups_without_padding = conf.ngroups;
    if (is_depthwise) conf.ngroups = utils::rnd_up(conf.ngroups, 16);
    const bool is_dw_16g = (conf.is_depthwise && conf.ngroups % 16 == 0);

    const bool is_16ic = conf.ic % 16 == 0;
    const bool is_16oc = conf.oc % 16 == 0;
    const bool use_16mb_unroll = !is_nhwc
            && !(conf.mb == 1 || conf.mb % 16 != 0) && !is_1stconv
            && ((is_16ic && is_16oc) || is_dw_16g);
    conf.mb_block = 1;
    conf.oc_block = 1;
    conf.ic_block = 1;
    conf.od_block = 1;
    conf.oh_block = 1;
    conf.ow_block = 1;
    conf.icb = 1;
    if (is_nhwc)
        conf.ver = ver_nhwc;
    else if (use_16mb_unroll)
        conf.ver = ver_16mb16c;
    else if (conf.mb % 16 != 0 && ((is_16oc && is_16ic) || is_dw_16g))
        conf.ver = ver_8ow16c;
    else
        return status::unimplemented;

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    //TODO: Fix Gtests and reenable
    const bool is_xe_hp_plus
            = compute_engine->is_xe_hp() || compute_engine->is_xe_hpg();
    const bool has_non_uniform_wg
            = compute_engine->mayiuse_non_uniform_work_groups();

    status_t status = status::success;
    switch (conf.ver) {
        case ver_16mb16c:
            conf.mb_block = 16;
            conf.oc_block = 16;
            conf.ic_block = 16;
            conf.od_block = 1;
            conf.ih_block = 1;
            conf.iw_block = 1;
            conf.sub_group_size = 16;
            if (conf.is_depthwise) {
                conf.icb = conf.ngroups;
                conf.lws_d[0] = 1;
                conf.lws_d[1] = is_xe_hp_plus ? 32 : 16;
                conf.lws_d[2] = 1;
                conf.gws_d[0] = conf.ih * conf.iw * conf.id;
                conf.gws_d[1] = conf.ic * conf.ngroups;
                conf.gws_d[2] = conf.mb / 16;
            } else {
                conf.icb = 64;
                while (conf.icb > 16) {
                    if (conf.ic % conf.icb == 0) break;
                    conf.icb /= 2;
                }
                conf.lws_d[0] = is_xe_hp_plus ? 32 : 16;
                conf.lws_d[1] = 1;
                conf.lws_d[2] = 1;
                conf.gws_d[0] = conf.icb;
                conf.gws_d[1] = conf.ih * conf.iw * conf.id;
                conf.gws_d[2]
                        = conf.mb / 16 * (conf.ic / conf.icb) * conf.ngroups;
            }
            break;
        case ver_8ow16c:
        case ver_nhwc: {
            conf.mb_block = 1;
            conf.oc_block = 16;
            conf.ic_block = 16;
            conf.od_block = 1;
            conf.ih_block = 1;
            int max_iw_block = 16;
            if (conf.ver == ver_nhwc) { max_iw_block = (conf.kw > 1) ? 8 : 16; }
            conf.iw_block = nstl::max(8, utils::max_div(conf.iw, max_iw_block));
            conf.sub_group_size = 16;
            if (conf.is_depthwise) {
                conf.icb = conf.ngroups;
                conf.lws_d[0] = 1;
                conf.lws_d[1] = conf.ic_block;
                conf.lws_d[2] = 1;
                conf.gws_d[0] = conf.ih * utils::div_up(conf.iw, conf.iw_block)
                        * conf.id;
                conf.gws_d[1] = conf.ic * conf.ngroups;
                conf.gws_d[2] = conf.mb;
            } else {
                conf.icb = 64;
                while (conf.icb > conf.ic_block) {
                    if (utils::rnd_up(conf.ic, conf.ic_block) % conf.icb == 0)
                        break;
                    conf.icb /= 2;
                }
                conf.lws_d[0] = conf.ic_block;
                conf.lws_d[1] = 1;
                conf.lws_d[2] = 1;
                conf.gws_d[0] = conf.icb;
                conf.gws_d[1] = conf.ih * utils::div_up(conf.iw, conf.iw_block)
                        * conf.id;
                conf.gws_d[2] = conf.mb
                        * (utils::rnd_up(conf.ic, conf.ic_block) / conf.icb)
                        * conf.ngroups;
            }
            break;
        }
        default: status = status::unimplemented;
    }

    maybe_fix_non_uniform_work_sizes(has_non_uniform_wg, conf);

    format_tag_t src_tag, dst_tag, wei_tag;

    switch (conf.ver) {
        case ver_nhwc:
            src_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            dst_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            wei_tag = conf.with_groups ? utils::pick(conf.ndims - 3, gOIw16o16i,
                              gOIhw16o16i, gOIdhw16o16i)
                                       : utils::pick(conf.ndims - 3, OIw16o16i,
                                               OIhw16o16i, OIdhw16o16i);
            break;
        case ver_16mb16c:
            src_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            dst_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gOIw16o16i, gOIhw16o16i, gOIdhw16o16i)
                                        : utils::pick(conf.ndims - 3, OIw16o16i,
                                                OIhw16o16i, OIdhw16o16i));
            break;
        case ver_8ow16c:
            src_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gOIw16o16i, gOIhw16o16i, gOIdhw16o16i)
                                        : utils::pick(conf.ndims - 3, OIw16o16i,
                                                OIhw16o16i, OIdhw16o16i));
            break;
        default: status = status::unimplemented;
    }
    if (status != status::success) return status;

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

    conf.is_src_nchw = utils::one_of(src_tag, ncw, nchw, ncdhw);
    conf.is_src_nhwc = utils::one_of(src_tag, nwc, nhwc, ndhwc);

    return status::success;
}

status_t gen9_convolution_bwd_data_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("IS_DW", conf.is_depthwise);
    kernel_ctx.define_int("BWD_DATA", 1);
    kernel_ctx.define_int("G", conf.ngroups);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ICB", conf.icb);
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
    kernel_ctx.define_int("SD", conf.stride_d);
    kernel_ctx.define_int("SH", conf.stride_h);
    kernel_ctx.define_int("SW", conf.stride_w);
    kernel_ctx.define_int("PD", conf.f_pad);
    kernel_ctx.define_int("PH", conf.t_pad);
    kernel_ctx.define_int("PW", conf.l_pad);
    kernel_ctx.define_int("PD_R", conf.back_pad);
    kernel_ctx.define_int("PH_R", conf.b_pad);
    kernel_ctx.define_int("PW_R", conf.r_pad);
    kernel_ctx.define_int("DD", conf.dilate_d);
    kernel_ctx.define_int("DH", conf.dilate_h);
    kernel_ctx.define_int("DW", conf.dilate_w);
    kernel_ctx.define_int("OC_PADDED", utils::rnd_up(conf.oc, conf.oc_block));
    kernel_ctx.define_int("IC_PADDED", utils::rnd_up(conf.ic, conf.ic_block));
    kernel_ctx.define_int("G_WO_PADDING", conf.ngroups_without_padding);
    kernel_ctx.define_int("OC_WO_PADDING", conf.oc_without_padding);
    kernel_ctx.define_int("IC_WO_PADDING", conf.ic_without_padding);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("IH_BLOCK", conf.ih_block);
    kernel_ctx.define_int("IW_BLOCK", conf.iw_block);
    kernel_ctx.define_int("IWB", utils::div_up(conf.iw, conf.iw_block));
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);
    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);

    kernel_ctx.define_int("GWS_0", conf.gws_d[0]);
    kernel_ctx.define_int("GWS_1", conf.gws_d[1]);
    kernel_ctx.define_int("GWS_2", conf.gws_d[2]);

    kernel_ctx.define_int("GWS_ORIG_0", conf.gws_orig_d[0]);
    kernel_ctx.define_int("GWS_ORIG_1", conf.gws_orig_d[1]);
    kernel_ctx.define_int("GWS_ORIG_2", conf.gws_orig_d[2]);

    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    kernel_ctx.set_data_type(conf.src_data_type);

    switch (conf.ver) {
        case ver_16mb16c: kernel_ctx.define_int("VER_16MB16C", 1); break;
        case ver_8ow16c: kernel_ctx.define_int("VER_8OW16C", 1); break;
        default: break;
    }

    kernel_ctx.add_option("-cl-std=CL2.0");

    return status::success;
}

status_t gen9_convolution_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {

    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, weights);
    arg_list.set(2, diff_dst);
    arg_list.set(3, bias);

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

status_t gen9_convolution_bwd_weights_t::pd_t::init_conf(engine_t *engine) {
    using namespace dnnl::impl::format_tag;
    using namespace data_type;

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper weights_mdw(diff_weights_md());
    const memory_desc_wrapper dst_mdw(diff_dst_md());
    const memory_desc_wrapper bias_mdw(diff_weights_md(1));

    set_default_conf(conf, cd, *src_md(), *diff_weights_md(), *diff_dst_md(),
            *diff_weights_md(1), *attr());

    const bool is_nhwc
            = src_mdw.matches_one_of_tag(nwc, nhwc, ndhwc) != format_tag::undef
            || dst_mdw.matches_one_of_tag(nwc, nhwc, ndhwc)
                    != format_tag::undef;

    const bool is_1stconv = conf.ic_without_padding == 3;
    const bool is_depthwise = conf.with_groups && (conf.ic_without_padding == 1)
            && (conf.oc_without_padding == 1);

    conf.is_nhwc = is_nhwc;
    conf.is_depthwise = is_depthwise;

    if (is_1stconv || (conf.with_groups && conf.ngroups > 1) || is_depthwise
            || is_nhwc) {
        conf.ic = conf.ic_without_padding;
        conf.oc = is_1stconv ? utils::rnd_up(conf.oc_without_padding, 16)
                             : conf.oc_without_padding;
    } else {
        conf.ic = utils::rnd_up(conf.ic_without_padding, 16);
        conf.oc = utils::rnd_up(conf.oc_without_padding, 16);
    }

    conf.ngroups_without_padding = conf.ngroups;
    if (is_depthwise && !is_nhwc)
        conf.ngroups = utils::rnd_up(conf.ngroups, 16);
    const bool is_dw_16g = (conf.is_depthwise && conf.ngroups % 16 == 0);

    const bool is_16ic = conf.ic % 16 == 0;
    const bool is_16oc = conf.oc % 16 == 0;
    const bool use_16mb_unroll = !is_nhwc
            && !(conf.mb == 1 || conf.mb % 16 != 0) && !is_1stconv
            && ((is_16ic && is_16oc) || is_dw_16g);

    conf.mb_block = 1;
    conf.oc_block = 1;
    conf.ic_block = 1;
    conf.od_block = 1;
    conf.oh_block = 1;
    conf.ow_block = 1;
    conf.osp_chunk = 1;
    conf.mb_chunk = 1;
    if (is_nhwc)
        conf.ver = ver_nhwc;
    else if (use_16mb_unroll)
        conf.ver = ver_16mb16c;
    else if (conf.mb % 16 != 0 && ((is_16oc && is_16ic) || is_dw_16g))
        conf.ver = ver_8ow16c;
    else if (is_1stconv && is_16oc)
        conf.ver = ver_1stconv;
    else
        return status::unimplemented;

    switch (conf.ver) {
        case ver_1stconv:
        case ver_8ow16c:
        case ver_nhwc:
            conf.mb_block = 1;
            conf.oc_block = 16;
            conf.ic_block = is_1stconv ? 1 : 16;
            conf.ow_block = 8;
            break;
        case ver_16mb16c:
            conf.mb_block = 16;
            conf.oc_block = 16;
            conf.ic_block = 16;
            conf.ow_block = 1;
            break;
    }

    bwd_w_compute_block_sizes(conf, engine);

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);

    //TODO: Fix Gtests and reenable
    const bool is_xe_hp_plus
            = compute_engine->is_xe_hp() || compute_engine->is_xe_hpg();
    const bool has_non_uniform_wg
            = compute_engine->mayiuse_non_uniform_work_groups();

    conf.sub_group_size = 16;
    conf.lws_d[0] = is_xe_hp_plus ? 32 : 16;
    conf.lws_d[1] = 1;
    conf.lws_d[2] = 1;

    if (conf.is_depthwise) {
        conf.gws_d[0] = utils::rnd_up(conf.ngroups, 16);
    } else {
        conf.gws_d[0] = is_1stconv ? conf.ocb * conf.ngroups
                                   : conf.ocb * (conf.icb / 16) * conf.ngroups;
    }
    conf.gws_d[1] = is_1stconv && !is_nhwc
            ? utils::div_up(conf.kh * conf.kw * conf.kd * conf.ic, 16)
            : conf.kh * conf.kw * conf.kd;
    conf.gws_d[2] = conf.nchunk * utils::div_up(conf.ic, conf.icb)
            * utils::div_up(conf.oc, conf.ocb);

    maybe_fix_non_uniform_work_sizes(has_non_uniform_wg, conf);

    format_tag_t src_tag, dst_tag, wei_tag;

    switch (conf.ver) {
        case ver_nhwc:
            src_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            dst_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            if (is_1stconv) {
                wei_tag = conf.with_groups ? utils::pick(
                                  conf.ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                                           : utils::pick(conf.ndims - 3, Owi16o,
                                                   Ohwi16o, Odhwi16o);
            } else if (conf.is_depthwise) {
                wei_tag = utils::pick(
                        conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g);
            } else {
                wei_tag = conf.with_groups
                        ? utils::pick(conf.ndims - 3, gIOw16i16o, gIOhw16i16o,
                                gIOdhw16i16o)
                        : utils::pick(conf.ndims - 3, IOw16i16o, IOhw16i16o,
                                IOdhw16i16o);
            }
            break;
        case ver_1stconv:
            assert(!conf.is_depthwise);
            src_tag = utils::pick(conf.ndims - 3, ncw, nchw, ncdhw);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.with_groups
                    ? utils::pick(conf.ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                    : utils::pick(conf.ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
            break;
        case ver_16mb16c:
            src_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            dst_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                                        : utils::pick(conf.ndims - 3, IOw16i16o,
                                                IOhw16i16o, IOdhw16i16o));
            break;
        case ver_8ow16c:
            src_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                                        : utils::pick(conf.ndims - 3, IOw16i16o,
                                                IOhw16i16o, IOdhw16i16o));
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

    conf.is_src_nchw = utils::one_of(src_tag, ncw, nchw, ncdhw);
    conf.is_src_nhwc = utils::one_of(src_tag, nwc, nhwc, ndhwc);

    bool ok = set_default_formats_common(
            conf.src_tag, conf.wei_tag, conf.dst_tag);
    if (!ok) return status::unimplemented;
    if (is_1stconv && !is_nhwc) {
        if (data_type::bf16 == conf.weights_data_type) {
            conf.reorder_wei = true;
            auto temp_wei_md = *diff_weights_md();
            temp_wei_md.data_type = data_type::f32;

            primitive_attr_t r_attr(default_attr());
            if (!r_attr.is_initialized()) return status::out_of_memory;

            CHECK(reorder_primitive_desc_create(rpd_wei_, engine, &temp_wei_md,
                    diff_weights_md(), &r_attr));
        }

        if (conf.with_bias && data_type::bf16 == conf.bias_data_type) {
            conf.reorder_bias = true;
            auto temp_bias_md = *diff_weights_md(1);
            temp_bias_md.data_type = data_type::f32;
            primitive_attr_t r_attr(default_attr());
            if (!r_attr.is_initialized()) return status::out_of_memory;

            CHECK(reorder_primitive_desc_create(rpd_bia_, engine, &temp_bias_md,
                    diff_weights_md(1), &r_attr));
        }
    }

    return status::success;
}

status_t gen9_convolution_bwd_weights_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("IS_DW", conf.is_depthwise);
    kernel_ctx.define_int("BWD_WEIGHTS", 1);
    kernel_ctx.define_int("G", conf.ngroups);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ICB", conf.icb);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OC", conf.oc);
    kernel_ctx.define_int("OCB", conf.ocb);
    kernel_ctx.define_int("OD", conf.od);
    kernel_ctx.define_int("OH", conf.oh);
    kernel_ctx.define_int("OW", conf.ow);
    kernel_ctx.define_int("KD", conf.kd);
    kernel_ctx.define_int("KH", conf.kh);
    kernel_ctx.define_int("KW", conf.kw);
    kernel_ctx.define_int("SD", conf.stride_d);
    kernel_ctx.define_int("SH", conf.stride_h);
    kernel_ctx.define_int("SW", conf.stride_w);
    kernel_ctx.define_int("PD", conf.f_pad);
    kernel_ctx.define_int("PH", conf.t_pad);
    kernel_ctx.define_int("PW", conf.l_pad);
    kernel_ctx.define_int("PD_R", conf.back_pad);
    kernel_ctx.define_int("PH_R", conf.b_pad);
    kernel_ctx.define_int("PW_R", conf.r_pad);
    kernel_ctx.define_int("DD", conf.dilate_d);
    kernel_ctx.define_int("DH", conf.dilate_h);
    kernel_ctx.define_int("DW", conf.dilate_w);
    kernel_ctx.define_int("OC_PADDED", conf.oc);
    kernel_ctx.define_int("OC_WO_PADDING", conf.oc_without_padding);
    kernel_ctx.define_int("G_WO_PADDING", conf.ngroups_without_padding);

    kernel_ctx.define_int("OW_BLOCK", conf.ow_block);
    kernel_ctx.define_int("ODB", conf.odb);
    kernel_ctx.define_int("OHB", conf.ohb);
    kernel_ctx.define_int("OWB", conf.owb);

    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);
    kernel_ctx.define_int("NCHUNK", conf.nchunk);
    kernel_ctx.define_int("OSP_CHUNK", conf.osp_chunk);
    kernel_ctx.define_int("MB_CHUNK", conf.mb_chunk);
    kernel_ctx.define_int(
            "MB_CHUNK_SIZE", utils::div_up(conf.mb, conf.mb_chunk));
    kernel_ctx.define_int("OW_BLOCK", conf.ow_block);

    kernel_ctx.define_int("GWS_0", conf.gws_d[0]);
    kernel_ctx.define_int("GWS_1", conf.gws_d[1]);
    kernel_ctx.define_int("GWS_2", conf.gws_d[2]);

    kernel_ctx.define_int("GWS_ORIG_0", conf.gws_orig_d[0]);
    kernel_ctx.define_int("GWS_ORIG_1", conf.gws_orig_d[1]);
    kernel_ctx.define_int("GWS_ORIG_2", conf.gws_orig_d[2]);

    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    kernel_ctx.add_option("-cl-std=CL2.0");

    kernel_ctx.set_data_type(data_type::f32);
    def_data_type(kernel_ctx, src_md()->data_type, "SRC");

    def_data_type(kernel_ctx, diff_dst_md()->data_type, "DST");

    def_data_type(kernel_ctx,
            diff_weights_md(conf.with_bias ? 1 : 0)->data_type, "BIA");

    def_data_type(kernel_ctx, data_type::f32, "WEI");

    switch (conf.ver) {
        case ver_16mb16c: kernel_ctx.define_int("VER_16MB16C", 1); break;
        case ver_1stconv:
        case ver_8ow16c: kernel_ctx.define_int("VER_8OW16C", 1); break;
        default: break;
    }

    return status::success;
}

status_t gen9_convolution_bwd_weights_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();
    if (!conf.reorder_wei && !conf.reorder_bias) return status::success;
    if (conf.reorder_wei) {
        auto temp_wei_md = *diff_weights_md();
        temp_wei_md.data_type = data_type::f32;
        memory_desc_wrapper wei_md_d(temp_wei_md);
        scratchpad.book(memory_tracking::names::key_conv_bwd_w_1st_wei_reorder,
                wei_md_d.size(), 1, OCL_BUFFER_ALIGNMENT);
        scratchpad.book(memory_tracking::names::key_nested_multiple,
                rpd_wei_->scratchpad_registry());
    }
    if (!conf.reorder_bias) return status::success;
    auto temp_bias_md = *diff_weights_md(1);
    temp_bias_md.data_type = data_type::f32;
    memory_desc_wrapper bia_md_d(temp_bias_md);
    scratchpad.book(memory_tracking::names::key_conv_bwd_w_1st_bia_reorder,
            bia_md_d.size(), 1, OCL_BUFFER_ALIGNMENT);
    scratchpad.book(memory_tracking::names::key_nested_multiple + 1,
            rpd_bia_->scratchpad_registry());

    return status::success;
}

status_t gen9_convolution_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, dst);
    append_post_ops_to_arg_list(ctx, arg_list, 4, pd()->attr()->post_ops_);

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    if (!post_ops_preserves_zeroes(ctx, pd()->attr()->post_ops_)) {
        ctx.zero_pad_output(DNNL_ARG_DST);
    }
    return status;
}

status_t gen9_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_weights = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS);
    auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

    const auto &conf = pd()->conf;

    const uint8_t zero = 0;
    std::unique_ptr<memory_t> wspace_wei;
    std::unique_ptr<memory_t> wspace_bia;
    auto temp_wei_md = *pd()->diff_weights_md();
    auto temp_bia_md = *pd()->diff_weights_md(1);
    std::unique_ptr<memory_storage_t> wspace_ptr_wei;
    std::unique_ptr<memory_storage_t> wspace_ptr_bia;
    if (conf.reorder_wei) {
        wspace_ptr_wei = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_conv_bwd_w_1st_wei_reorder);

        temp_wei_md.data_type = data_type::f32;
    }
    if (conf.reorder_bias) {
        wspace_ptr_bia = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_conv_bwd_w_1st_bia_reorder);

        temp_bia_md.data_type = data_type::f32;
    }

    memory_desc_wrapper wei_mdw(temp_wei_md);
    CHECK(compute_stream->fill(
            conf.reorder_wei ? *wspace_ptr_wei : diff_weights, zero,
            wei_mdw.size()));
    if (conf.with_bias) {
        memory_desc_wrapper bia_mdw(temp_bia_md);
        CHECK(compute_stream->fill(
                conf.reorder_bias ? *wspace_ptr_bia : diff_bias, zero,
                bia_mdw.size()));
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, conf.reorder_wei ? *wspace_ptr_wei : diff_weights);
    arg_list.set(2, conf.reorder_bias ? *wspace_ptr_bia : diff_bias);
    arg_list.set(3, diff_dst);

    status_t status = parallel_for(ctx,
            compute::nd_range_t(conf.gws_d, conf.lws_d), kernel_, arg_list);
    if (status != status::success) return status;
    auto exec_reorder = [&](memory_t *in, memory_t *out,
                                const std::shared_ptr<primitive_t> &prim,
                                int r_num) -> status_t {
        exec_args_t r_args;
        r_args[DNNL_ARG_FROM] = memory_arg_t {in, true};
        r_args[DNNL_ARG_TO] = memory_arg_t {out, false};
        exec_ctx_t r_ctx(ctx, std::move(r_args));
        nested_scratchpad_t ns(
                ctx, memory_tracking::names::key_nested_multiple + r_num, prim);
        r_ctx.set_scratchpad_grantor(ns.grantor());
        return prim->execute(r_ctx);
    };

    if (conf.reorder_wei) {
        CHECK(safe_ptr_assign(wspace_wei,
                new memory_t(ctx.stream()->engine(), &temp_wei_md,
                        std::move(wspace_ptr_wei))));
        CHECK(exec_reorder(wspace_wei.get(), ctx.output(DNNL_ARG_DIFF_WEIGHTS),
                wei_reorder_, 0));
    }
    if (conf.reorder_bias) {
        CHECK(safe_ptr_assign(wspace_bia,
                new memory_t(ctx.stream()->engine(), &temp_bia_md,
                        std::move(wspace_ptr_bia))));
        CHECK(exec_reorder(wspace_bia.get(), ctx.output(DNNL_ARG_DIFF_BIAS),
                bia_reorder_, 1));
    }

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
