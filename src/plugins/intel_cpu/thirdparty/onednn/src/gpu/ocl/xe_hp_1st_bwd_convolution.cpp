/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include "gpu/ocl/xe_hp_1st_bwd_convolution.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/reorder.hpp"
#include "common/type_helpers.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;

status_t xe_hp_1st_convolution_bwd_weights_t::pd_t::init_conf(
        engine_t *engine) {
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

    if (is_1stconv) {
        conf.ic = conf.ic_without_padding;
        conf.oc = utils::rnd_up(conf.oc_without_padding, 16);
    }

    conf.ngroups_without_padding = conf.ngroups;

    const bool is_16oc = conf.oc % 16 == 0;

    conf.mb_block = 1;
    conf.oc_block = 1;
    conf.ic_block = 1;
    conf.od_block = 1;
    conf.oh_block = 1;
    conf.ow_block = 1;
    conf.osp_chunk = 1;
    conf.mb_chunk = 1;

    if (is_1stconv && is_16oc && !conf.is_nhwc) conf.ver = ver_1stconv;

    if (conf.mb % 16 != 0 || conf.oc % 32 != 0 || conf.dilate_w > 0
            || conf.kw > 8 || conf.stride_w > 2)
        return status::unimplemented;
    switch (conf.ver) {
        case ver_1stconv:
            conf.mb_block = 1;
            conf.oc_block = 16;
            conf.ic_block = 1;
            conf.ow_block = 8;
            break;
    }

    bwd_w_compute_block_sizes(conf, engine);

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    const bool is_xe_hp_plus
            = compute_engine->is_xe_hp() || compute_engine->is_xe_hpg();
    const bool has_non_uniform_wg
            = compute_engine->mayiuse_non_uniform_work_groups();

    conf.sub_group_size = 8;
    conf.lws_d[0] = is_xe_hp_plus ? 32 : 16;
    conf.lws_d[1] = 1;
    conf.lws_d[2] = 1;
    conf.kwb = utils::rnd_up(conf.kw, 8);

    conf.gws_d[0] = conf.ocb * conf.ngroups;
    conf.gws_d[1] = utils::div_up(conf.kh * conf.kwb * conf.kd * conf.ic, 32);
    conf.gws_d[2] = conf.nchunk * utils::div_up(conf.ic, conf.icb)
            * utils::div_up(conf.oc, conf.ocb);

    maybe_fix_non_uniform_work_sizes(has_non_uniform_wg, conf);

    format_tag_t src_tag, dst_tag, wei_tag;

    switch (conf.ver) {
        case ver_1stconv:
            assert(!conf.is_depthwise);
            src_tag = utils::pick(conf.ndims - 3, ncw, nchw, ncdhw);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.with_groups
                    ? utils::pick(conf.ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                    : utils::pick(conf.ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
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
            r_attr.set_scratchpad_mode(scratchpad_mode::user);
            auto st = reorder_primitive_desc_create(
                    rpd_wei_, engine, &temp_wei_md, diff_weights_md(), &r_attr);
            if (st != status::success) return status::unimplemented;
        }

        if (conf.with_bias && data_type::bf16 == conf.bias_data_type) {
            conf.reorder_bias = true;
            auto temp_bias_md = *diff_weights_md(1);
            temp_bias_md.data_type = data_type::f32;
            primitive_attr_t r_attr(default_attr());
            if (!r_attr.is_initialized()) return status::out_of_memory;
            r_attr.set_scratchpad_mode(scratchpad_mode::user);
            auto st = reorder_primitive_desc_create(rpd_bia_, engine,
                    &temp_bias_md, diff_weights_md(1), &r_attr);
            if (st != status::success) return status::unimplemented;
        }
    }

    return status::success;
}

status_t xe_hp_1st_convolution_bwd_weights_t::pd_t::init_kernel_ctx(
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
    kernel_ctx.define_int("KWB", conf.kwb);
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
        case ver_1stconv: kernel_ctx.define_int("VER_8OW16C", 1); break;
        default: break;
    }

    return status::success;
}

status_t xe_hp_1st_convolution_bwd_weights_t::pd_t::init_scratchpad() {
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

status_t xe_hp_1st_convolution_bwd_weights_t::execute_backward_weights(
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
