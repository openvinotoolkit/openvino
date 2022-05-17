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

#include "gpu/ocl/xe_hp_bf16_convolution.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::memory_tracking::names;

status_t xe_hp_bf16_convolution_bwd_weights_t::pd_t::init_conf(
        engine_t *engine) {
    const convolution_desc_t &cd = *desc();

    set_default_conf(conf, cd, *src_md(), *diff_weights_md(), *diff_dst_md(),
            *diff_weights_md(1), *attr());

    //TODO: add depthwise
    if (conf.is_depthwise) return status::unimplemented;

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.use_256grf_per_thread = compute_engine->mayiuse(
            compute::device_ext_t::intel_variable_eu_thread_count);

    conf.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;

    using namespace dnnl::impl::format_tag;

    format_tag_t src_tag
            = utils::pick(conf.ndims - 3, NCw32n16c, NChw32n16c, NCdhw32n16c);
    format_tag_t dst_tag
            = utils::pick(conf.ndims - 3, NCw32n16c, NChw32n16c, NCdhw32n16c);
    format_tag_t wei_tag = conf.with_groups
            ? utils::pick(conf.ndims - 3, gOIw16o16i, gOIhw16o16i, gOIdhw16o16i)
            : utils::pick(conf.ndims - 3, OIw16o16i, OIhw16o16i, OIdhw16o16i);

    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper weights_mdw(diff_weights_md());
    const memory_desc_wrapper dst_mdw(diff_dst_md());

    conf.src_tag = src_mdw.format_kind() == format_kind::any
            ? src_tag
            : src_mdw.matches_one_of_tag(src_tag);
    conf.wei_tag = weights_mdw.format_kind() == format_kind::any
            ? wei_tag
            : weights_mdw.matches_one_of_tag(wei_tag);
    conf.dst_tag = dst_mdw.format_kind() == format_kind::any
            ? dst_tag
            : dst_mdw.matches_one_of_tag(dst_tag);

    if (conf.src_tag != src_tag || conf.wei_tag != wei_tag
            || conf.dst_tag != dst_tag)
        return status::unimplemented;

    // TODO: try workgroup size 32
    //  1 tile with 4thr/EU can dispatch maximum 2048 subgroups,
    //  but 4096 seems to better for resnet_50 convolutions (when measured through gsim)
    const int max_workgroup_size = 16;
    const int max_subgroups = 4096;
    conf.sub_group_size = 8;
    conf.mb_block = 16;
    conf.oc_block = 16;
    conf.ic_block = 16;

    // sometimes kernel hangs for this case,
    // when run using emulation on gen9, reason unknown.
    if (conf.oc % conf.oc_block != 0) return status::unimplemented;

    // Each workgroup loads:
    // SRC: (mb_blk_wg * mb_block) * (ic_blk_wg * ic_block),
    // DIFF_DST: (mb_blk_wg * mb_block) * (oc_blk_wg * oc_block)
    // to compute and store WEI : (oc_blk_wg * oc_block) * (ic_blk_wg * ic_block).
    //conf.mb_blk_wg = nstl::min(2, utils::div_up(conf.mb, conf.mb_block));
    conf.mb_blk_wg = conf.mb > 16 ? 2 : 1; // mb is padded by 32

    conf.ic = utils::rnd_up(conf.ic, conf.ic_block);
    conf.oc = utils::rnd_up(conf.oc, conf.oc_block);
    conf.max_blk_wg = 16;
    conf.oc_blk_wg = std::min(
            utils::max_div(conf.oc / conf.oc_block, conf.max_blk_wg), 4);
    conf.ic_blk_wg = std::min(
            utils::max_div(conf.ic / conf.ic_block, conf.max_blk_wg), 4);

    // TODO: Fine-tune blocking sizes on real hardware
    if (conf.oc_blk_wg * conf.ic_blk_wg <= max_workgroup_size) {
        conf.ic_blk_sg = 1;
        conf.oc_blk_sg = 1;
    } else {
        conf.ic_blk_sg = (conf.ic_blk_wg % 2) == 0 ? 2 : 1;
        conf.oc_blk_sg = (conf.oc_blk_wg % 2) == 0 ? 2 : 1;
    }
    int num_subgroups_for_compute
            = conf.oc_blk_wg / conf.oc_blk_sg * conf.ic_blk_wg / conf.ic_blk_sg;
    if (num_subgroups_for_compute > max_workgroup_size) {
        do {
            conf.ic_blk_wg = utils::max_div(conf.ic_blk_wg, conf.ic_blk_wg / 2);
            conf.ic_blk_sg = (conf.ic_blk_wg % 2) == 0 ? conf.ic_blk_sg : 1;
            num_subgroups_for_compute = conf.oc_blk_wg / conf.oc_blk_sg
                    * conf.ic_blk_wg / conf.ic_blk_sg;
            if (num_subgroups_for_compute > max_workgroup_size) {
                conf.oc_blk_wg
                        = utils::max_div(conf.oc_blk_wg, conf.oc_blk_wg / 2);
                conf.oc_blk_sg = (conf.oc_blk_wg % 2) == 0 ? conf.oc_blk_sg : 1;
                num_subgroups_for_compute = conf.oc_blk_wg / conf.oc_blk_sg
                        * conf.ic_blk_wg / conf.ic_blk_sg;
            }
        } while (num_subgroups_for_compute > max_workgroup_size);
    }

    // Each subgroups loads
    // SRC: mb_block * ic_block,
    // DIFF_DST: mb_block * oc_block
    const int num_subgroups_for_load_global_to_slm
            = conf.mb_blk_wg * nstl::max(conf.oc_blk_wg, conf.ic_blk_wg);
    if (num_subgroups_for_load_global_to_slm > num_subgroups_for_compute)
        conf.mb_blk_wg = 1;

    // TODO: experiment with triple buffering by simply changing this to 3
    conf.num_buffers = 2;
    if (conf.num_buffers > 2) conf.mb_blk_wg = 1;
    // For  maximum parallelization (4 workgroups/DSS)
    // total SLM size per WG shouldn't exceed (128/4 =)32 KB.
    conf.src_slm_size = conf.num_buffers * conf.mb_blk_wg * conf.mb_block
            * conf.ic_block * conf.ic_blk_wg / 2;
    conf.dst_slm_size = conf.num_buffers * conf.mb_blk_wg * conf.mb_block
            * conf.oc_block * conf.oc_blk_wg / 2;

    int max_needed_subgroups = nstl::max(
            num_subgroups_for_load_global_to_slm, num_subgroups_for_compute);
    conf.lws_d[0] = conf.sub_group_size
            * (max_needed_subgroups <= 4
                            ? 4
                            : (max_needed_subgroups <= 8 ? 8 : 16));
    conf.lws_d[1] = 1;
    conf.lws_d[2] = 1;

    const int num_workgroups_for_compute = conf.oc * conf.ic
            / (conf.ic_blk_wg * conf.ic_block * conf.oc_blk_wg * conf.oc_block);
    conf.gws_d[0] = num_workgroups_for_compute * conf.lws_d[0];

    conf.use_dpasw = false; // TODO: switch it on if better performance
    conf.use_split_barrier = false; //TODO: experiment on real hardware
    // Parallelize along k-dimension to utilize all logical threads
    const int k_dim = utils::div_up(conf.mb, (conf.mb_blk_wg * conf.mb_block))
            * conf.od * conf.oh * conf.ow;

    conf.workgroups_along_k = utils::max_div(k_dim,
            utils::div_up(max_subgroups,
                    (conf.gws_d[0] / conf.sub_group_size) * conf.ngroups
                            * conf.kd * conf.kh * conf.kw));

    conf.k_blocks = k_dim / conf.workgroups_along_k;

    conf.gws_d[1] = conf.ngroups * conf.kd * conf.kh * conf.kw
            * conf.workgroups_along_k;
    conf.gws_d[2] = 1;

    size_t wei_size = conf.weights_data_type == data_type::bf16
            ? conf.ngroups * conf.oc * conf.ic * conf.kd * conf.kh * conf.kw
                    * sizeof(float)
            : 0;

    auto scratchpad = scratchpad_registry().registrar();
    if (wei_size)
        scratchpad.book(memory_tracking::names::key_conv_wei_reduction,
                wei_size, types::data_type_size(conf.weights_data_type),
                OCL_BUFFER_ALIGNMENT);
    //scratchpad_registry());
    size_t bia_size
            = ((conf.with_bias && conf.bias_data_type == data_type::bf16)
                              ? conf.ngroups * conf.oc
                              : 0)
            * sizeof(float);
    if (bia_size)
        scratchpad.book(memory_tracking::names::key_conv_bia_reduction,
                bia_size, types::data_type_size(conf.bias_data_type),
                OCL_BUFFER_ALIGNMENT);
    //bia_size, OCL_BUFFER_ALIGNMENT);

    if (!set_default_formats_common(conf.src_tag, conf.wei_tag, conf.dst_tag)) {
        return status::unimplemented;
    }
    if (conf.ic_without_padding == 3) return status::unimplemented;
    set_offsets(src_md(), off.src_off);
    set_offsets(diff_weights_md(), off.wei_off);
    set_offsets(diff_dst_md(), off.dst_off);

    return status::success;
}

status_t xe_hp_bf16_convolution_bwd_weights_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("WITH_GROUPS", conf.with_groups);
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

    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);
    kernel_ctx.define_int("MB_BLK_WORKGROUP", conf.mb_blk_wg);
    kernel_ctx.define_int("MAX_BLK_WORKGROUP", conf.max_blk_wg);
    kernel_ctx.define_int("IC_BLK_WORKGROUP", conf.ic_blk_wg);
    kernel_ctx.define_int("OC_BLK_WORKGROUP", conf.oc_blk_wg);
    kernel_ctx.define_int("IC_BLK_SUBGROUP", conf.ic_blk_sg);
    kernel_ctx.define_int("OC_BLK_SUBGROUP", conf.oc_blk_sg);
    kernel_ctx.define_int("K_WORKGROUPS", conf.workgroups_along_k);
    kernel_ctx.define_int("K_BLOCKS", conf.k_blocks);

    kernel_ctx.define_int("NUM_BUF", conf.num_buffers);
    kernel_ctx.define_int("SRC_SLM_SIZE", conf.src_slm_size);
    kernel_ctx.define_int("DST_SLM_SIZE", conf.dst_slm_size);
    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);

    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);
    kernel_ctx.define_int("GWS_0", conf.gws_d[0]);
    kernel_ctx.define_int("GWS_1", conf.gws_d[1]);
    kernel_ctx.define_int("GWS_2", conf.gws_d[2]);

    kernel_ctx.define_int("USE_DPASW", conf.use_dpasw);

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);
    def_offsets(off.wei_off, kernel_ctx, "WEI", conf.ndims + conf.with_groups);
    def_offsets(off.dst_off, kernel_ctx, "DST", conf.ndims);

    def_data_type(kernel_ctx, conf.weights_data_type, "WEI");
    if (conf.with_bias) def_data_type(kernel_ctx, conf.bias_data_type, "BIA");
    kernel_ctx.set_data_type(
            data_type::bf16); // for enabling correct mmad8x8/dpas macro

    kernel_ctx.add_option("-cl-std=CL2.0");
    kernel_ctx.add_option("-cl-uniform-work-group-size");

    if (conf.use_256grf_per_thread)
        kernel_ctx.add_option("-cl-intel-256-GRF-per-thread");

    kernel_ctx.print_options();
    return status::success;
}

status_t xe_hp_bf16_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_weights = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS);
    auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

    const auto &conf = pd()->conf;
    compute::kernel_arg_list_t arg_list, arg_list_zero, arg_list_cvt;
    std::unique_ptr<memory_storage_t> wei_f32_reduce, bia_f32_reduce;

    if (conf.weights_data_type == data_type::bf16) {
        wei_f32_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
                key_conv_wei_reduction);
        arg_list_zero.set(0, *wei_f32_reduce);
    } else {
        arg_list_zero.set(0, diff_weights);
    }
    if (conf.with_bias) {
        if (conf.bias_data_type == data_type::bf16) {
            bia_f32_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
                    key_conv_bia_reduction);
            arg_list_zero.set(1, *bia_f32_reduce);
        } else
            arg_list_zero.set(1, diff_bias);
    } else
        arg_list_zero.set(1, memory_storage_t::empty_storage());

    auto nd_range = compute::nd_range_t(
            {conf.ic / (conf.ic_block / conf.sub_group_size) * conf.kd * conf.kh
                            * conf.kw,
                    conf.oc * conf.ngroups, 1},
            {conf.sub_group_size, 16, 1});
    CHECK(parallel_for(ctx, nd_range, zero_init_kernel_, arg_list_zero));

    arg_list.set(0, src);

    if (conf.weights_data_type == data_type::bf16) {
        arg_list.set(1, *wei_f32_reduce);
    } else {
        arg_list.set(1, diff_weights);
    }

    if (conf.with_bias) {
        if (conf.bias_data_type == data_type::bf16)
            arg_list.set(2, *bia_f32_reduce);
        else
            arg_list.set(2, diff_bias);
    } else
        arg_list.set(2, memory_storage_t::empty_storage());

    arg_list.set(3, diff_dst);

    nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
    status_t status = parallel_for(ctx, nd_range, conv_kernel_, arg_list);

    if (utils::one_of(
                data_type::bf16, conf.weights_data_type, conf.bias_data_type)) {
        if (conf.weights_data_type == data_type::bf16) {
            arg_list_cvt.set(0, *wei_f32_reduce);
            arg_list_cvt.set(2, diff_weights);
        } else {
            arg_list_cvt.set(0, memory_storage_t::empty_storage());
            arg_list_cvt.set(2, memory_storage_t::empty_storage());
        }

        if (conf.with_bias && conf.bias_data_type == data_type::bf16) {
            arg_list_cvt.set(1, *bia_f32_reduce);
            arg_list_cvt.set(3, diff_bias);
        } else {
            arg_list_cvt.set(1, memory_storage_t::empty_storage());
            arg_list_cvt.set(3, memory_storage_t::empty_storage());
        }

        nd_range = compute::nd_range_t(
                {conf.ic / (conf.ic_block / conf.sub_group_size) * conf.kd
                                * conf.kh * conf.kw,
                        conf.oc * conf.ngroups, 1},
                {conf.sub_group_size, 16, 1});
        status = parallel_for(
                ctx, nd_range, convert_f32_to_bf16_kernel_, arg_list_cvt);
    }

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
