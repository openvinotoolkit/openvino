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

#include "gpu/ocl/xe_hp_1x1_convolution.hpp"

#include "gpu/ocl/ocl_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::format_tag;

status_t xe_hp_1x1_convolution_fwd_t::pd_t::init_conf(engine_t *engine) {
    const convolution_desc_t &cd = *desc();

    set_default_conf(conf, cd, *src_md(), *weights_md(), *dst_md(),
            *weights_md(1), *attr());

    status_t status = status::success;

    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.use_256grf_per_thread = compute_engine->mayiuse(
            compute::device_ext_t::intel_variable_eu_thread_count);

    bool is_int8 = src_mdw.data_type() == data_type::u8
            || src_mdw.data_type() == data_type::s8;

    format_tag_t src_tag = dnnl_format_tag_undef;
    format_tag_t dst_tag = dnnl_format_tag_undef;
    format_tag_t wei_tag = dnnl_format_tag_undef;

    conf.calc_block = 32;
    conf.oc_block = (is_int8) ? 32 : 16;
    conf.ic_block = (is_int8) ? 32 : 16;
    conf.nchunk = utils::div_up(conf.oc * conf.ngroups, conf.calc_block);
    if (conf.is_depthwise != false || conf.kd != 1 || conf.kh != 1
            || conf.kw != 1)
        return status::unimplemented;
    if (conf.with_groups && conf.ngroups > 1
            && (conf.oc % conf.oc_block != 0 || conf.ic % conf.ic_block != 0))
        return status::unimplemented;

    if (conf.l_pad > 0 || conf.r_pad > 0 || conf.t_pad > 0 || conf.f_pad > 0)
        return status::unimplemented;
    const bool is_stride1
            = conf.stride_d == 1 && conf.stride_h == 1 && conf.stride_w == 1;

    if (is_stride1) {
        // reshape to nCxc
        conf.iw = conf.iw * conf.ih * conf.id;
        conf.ow = conf.ow * conf.oh * conf.od;
        conf.ih = conf.id = 1;
        conf.oh = conf.od = 1;
    }

    if (conf.mb == 8 || conf.mb == 16 || conf.mb % 32 == 0) {
        conf.mb_block = 32;
        conf.sp_block = 1;
    } else {
        conf.mb_block = 1;
        conf.sp_block = 8;
        // TODO: compute sp_block
        /*
        auto approx_clocks = [&](const int block) {
            int ic_chunks = utils::div_up(conf.ic, conf.ic_block);
            bool use_slm = utils::div_up(conf.ow, block) % 8 == 0;
            int mem_clocks = ic_chunks * (16 - use_slm * 6)
                    + block / 2 * (ic_chunks + 1);
            int compute_clocks = 32 * block * ic_chunks;
            int num_threads = conf.nchunk * conf.mb * conf.od * conf.oh
                    * utils::div_up(conf.ow, block);
            return utils::div_up(num_threads, dev_info->hw_threads())
                    * (compute_clocks + mem_clocks);
        };
        auto clock_compare = [&](const int &block1, const int &block2) {
            return approx_clocks(block1) < approx_clocks(block2);
        };
        std::vector<int> sorted_blocks = {4, 8, 12, 16};
        std::sort(sorted_blocks.begin(), sorted_blocks.end(), clock_compare);
        conf.sp_block = sorted_blocks[0]; */
    }

    const int ow_group
            = (conf.mb_block == 32
                      || utils::div_up(conf.ow, conf.sp_block) % 8 == 0)
            ? 8
            : 1;
    conf.sub_group_size = 8;

    // IC loop splitting is turned OFF by default
    conf.ic_split = 1;
    // Resnet-50 mb1 heuristics. TODO: compute ic_split
    if (conf.mb_block == 1 && conf.ic % 2 * conf.ic_block == 0 && conf.ic >= 256
            && conf.oc >= 256) {
        conf.ic_split = utils::max_div(utils::div_up(conf.ic, 128), 4);
    }

    conf.lws_d[0] = conf.sub_group_size * conf.ic_split;
    conf.lws_d[1] = ow_group;
    conf.lws_d[2] = 1;

    const int num_sp_threads
            = utils::div_up(conf.ow, conf.sp_block) * conf.oh * conf.od;
    conf.gws_d[0] = utils::rnd_up(
            conf.nchunk * conf.sub_group_size * conf.ic_split, conf.lws_d[0]);
    conf.gws_d[1] = utils::rnd_up(num_sp_threads, conf.lws_d[1]);
    conf.gws_d[2] = utils::div_up(conf.mb, conf.mb_block);

    conf.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    if (is_int8) {
        if (conf.mb_block == 32) {
            src_tag = utils::pick(conf.ndims - 3, NCw32n32c, NChw32n32c);
            dst_tag = utils::pick(conf.ndims - 3, NCw32n32c, NChw32n32c);
        } else {
            src_tag = utils::pick(conf.ndims - 3, nCw32c, nChw32c);
            dst_tag = utils::pick(conf.ndims - 3, nCw32c, nChw32c);
        }

        wei_tag = conf.with_groups
                ? utils::pick(conf.ndims - 3, gOIw4o8i8o4i, gOIhw4o8i8o4i)
                : utils::pick(conf.ndims - 3, OIw4o8i8o4i, OIhw4o8i8o4i);
    } else {
        if (conf.mb_block == 32) {
            src_tag = utils::pick(conf.ndims - 3, NCw32n16c, NChw32n16c);
            dst_tag = utils::pick(conf.ndims - 3, NCw32n16c, NChw32n16c);
        } else {
            src_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c);
        }

        wei_tag = conf.with_groups
                ? utils::pick(conf.ndims - 3, gOIw4o8i8o2i, gOIhw4o8i8o2i)
                : utils::pick(conf.ndims - 3, OIw4o8i8o2i, OIhw4o8i8o2i);
    }

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

    return status;
}

status_t xe_hp_1x1_convolution_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("G", conf.ngroups);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic_without_padding);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OC", conf.oc_without_padding);
    kernel_ctx.define_int("OD", conf.od);
    kernel_ctx.define_int("OH", conf.oh);
    kernel_ctx.define_int("OW", conf.ow);

    kernel_ctx.define_int("SD", conf.stride_d);
    kernel_ctx.define_int("SH", conf.stride_h);
    kernel_ctx.define_int("SW", conf.stride_w);

    kernel_ctx.define_int("SP_BLOCK", conf.sp_block);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);
    kernel_ctx.define_int("OC_CALC_BLOCK", conf.calc_block);

    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);
    def_attr_info(kernel_ctx, conf.attr_info, attr()->post_ops_);

    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);

    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    kernel_ctx.define_int("OC_NCHUNK", utils::div_up(conf.oc, conf.oc_block));
    kernel_ctx.define_int("IC_NCHUNK", utils::div_up(conf.ic, conf.ic_block));

    int ic_loop_groups = utils::max_div(
            utils::div_up(utils::div_up(conf.ic, conf.ic_split), conf.ic_block),
            4);
    kernel_ctx.define_int("IC_LOOP_GROUPS", ic_loop_groups);

    kernel_ctx.define_int("USE_DOUBLE_BUFFER", 0);
    kernel_ctx.define_int("USE_WEI_SLM",
            conf.ic_split <= 1
                    && (conf.mb_block == 32
                            || utils::div_up(conf.ow, conf.sp_block) % 8 == 0));
    kernel_ctx.define_int("SP_TAIL",
            utils::div_up(conf.ow, conf.sp_block) % conf.lws_d[1] != 0);
    kernel_ctx.define_int("OUT_SP_TAIL", conf.ow % conf.sp_block);
    kernel_ctx.define_int("IC_SPLIT", conf.ic_split);

    kernel_ctx.set_data_type(conf.dst_data_type);

    def_data_type(kernel_ctx, conf.src_data_type, "SRC");
    def_data_type(kernel_ctx, conf.dst_data_type, "DST");
    def_data_type(kernel_ctx, conf.weights_data_type, "WEI");
    def_data_type(kernel_ctx, conf.bias_data_type, "BIA");
    def_data_type(kernel_ctx,
            conf.attr_info.sum_data_type == dnnl_data_type_undef
                    ? conf.dst_data_type
                    : conf.attr_info.sum_data_type,
            "SUM");

    kernel_ctx.add_option("-Dcl_intel_subgroups_char");

    if (conf.use_256grf_per_thread)
        kernel_ctx.add_option("-cl-intel-256-GRF-per-thread");

    return status::success;
}

status_t xe_hp_1x1_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &oscales = CTX_IN_STORAGE(DNNL_ARG_ATTR_OUTPUT_SCALES);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, dst);

    unsigned arg_idx = append_post_ops_to_arg_list(
            ctx, arg_list, 4, pd()->attr()->post_ops_);
    if (conf.attr_info.common_oscales) {
        float scales = pd()->attr()->output_scales_.scales_[0];
        arg_list.set(arg_idx++, scales);
    } else {
        arg_list.set(arg_idx++, 1.0f);
    }

    if (conf.attr_info.with_per_oc_oscales) {
        if (conf.attr_info.with_runtime_oscales)
            arg_list.set(arg_idx++, oscales);
        else
            arg_list.set(arg_idx++, CTX_GPU_RES_STORAGE(SCALES_));
    } else {
        arg_list.set(arg_idx++, memory_storage_t::empty_storage());
    }

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
