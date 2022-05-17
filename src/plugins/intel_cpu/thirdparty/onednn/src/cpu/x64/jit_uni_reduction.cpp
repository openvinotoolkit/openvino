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

#include "jit_uni_reduction.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

static cpu_isa_t get_supported_isa() {
    if (mayiuse(avx512_core_bf16)) return avx512_core_bf16;
    if (mayiuse(avx512_core)) return avx512_core;
    if (mayiuse(avx512_common)) return avx512_common;
    if (mayiuse(avx2)) return avx2;
    if (mayiuse(avx)) return avx;
    if (mayiuse(sse41)) return sse41;

    return isa_any;
}

status_t jit_uni_reduction_t::pd_t::init(engine_t *engine) {
    using namespace alg_kind;
    using namespace data_type;
    using namespace format_tag;
    using sm = primitive_attr_t::skip_mask_t;

    conf_.isa = get_supported_isa();

    conf_.src_type = src_md()->data_type;
    conf_.dst_type = dst_md()->data_type;
    conf_.acc_type
            = types::default_accum_data_type(conf_.src_type, conf_.dst_type);
    conf_.src_dt_size = types::data_type_size(conf_.src_type);
    conf_.dst_dt_size = types::data_type_size(conf_.dst_type);
    conf_.acc_dt_size = types::data_type_size(conf_.acc_type);

    const bool ok = platform::has_data_type_support(conf_.src_type)
            && platform::has_data_type_support(conf_.dst_type)
            && set_default_params() == status::success
            && attr()->has_default_values(sm::post_ops);
    if (!ok) return status::unimplemented;

    if (attr()->post_ops_.len() > 0) return status::unimplemented;

    const auto src_mdw = memory_desc_wrapper(src_md());
    const auto dst_mdw = memory_desc_wrapper(dst_md());

    const format_tag_t src_md_desired_format = memory_desc_matches_one_of_tag(
            *src_md(), x, nc, ncw, nchw, ncdhw);
    const format_tag_t dst_md_desired_format = memory_desc_matches_one_of_tag(
            *dst_md(), x, nc, ncw, nchw, ncdhw);
    if (src_md_desired_format != dst_md_desired_format
            || src_md_desired_format == format_tag::undef)
        return status::unimplemented;

    const int ndims = src_mdw.ndims();
    const auto &src_dims = src_mdw.dims();
    const auto &dst_dims = dst_mdw.dims();

    conf_.is_saturation_needed = utils::one_of(conf_.dst_type, s32, s8, u8);

    int num_of_reduced_dims = 0;
    conf_.idle_size = dst_mdw.nelems();
    conf_.reduce_size = 1;
    for (int d = ndims - 1; d >= 0; --d) {
        if (src_dims[d] != dst_dims[d]) {
            num_of_reduced_dims++;
            conf_.reduce_size *= src_dims[d];
        } else
            break;
    }

    if (num_of_reduced_dims == 0) return status::unimplemented;

    for (int d = 0; d < ndims - num_of_reduced_dims; ++d)
        if (src_dims[d] != dst_dims[d]) return status::unimplemented;

    conf_.alg = desc()->alg_kind;
    if (utils::one_of(conf_.alg, reduction_norm_lp_max, reduction_norm_lp_sum,
                reduction_norm_lp_power_p_max, reduction_norm_lp_power_p_sum))
        return status::unimplemented;

    return status::success;
}

status_t jit_uni_reduction_t::init(engine_t *engine) {
    using namespace format_tag;

    const memory_desc_t *dst_md = pd()->dst_md();
    const jit_reduction_conf_t &conf = pd()->get_conf();

    CHECK(get_proper_kernel(dst_md, conf));
    CHECK(kernel_->create_kernel());

    return status::success;
}

status_t jit_uni_reduction_t::execute(const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const uint8_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(uint8_t *, DNNL_ARG_DST);

    const dim_t idle_size = pd()->get_conf().idle_size;
    const dim_t reduce_size = pd()->get_conf().reduce_size;
    const std::size_t src_dt_size = pd()->get_conf().src_dt_size;
    const std::size_t dst_dt_size = pd()->get_conf().dst_dt_size;

    parallel_nd(idle_size, [&](dim_t i) {
        const dim_t src_off = i * reduce_size * src_dt_size;
        const dim_t dst_off = i * dst_dt_size;

        jit_reduction_call_s args = jit_reduction_call_s();
        args.src = src + src_off;
        args.dst = dst + dst_off;

        (*kernel_)(&args);
    });

    return status::success;
}

status_t jit_uni_reduction_t::get_proper_kernel(
        const memory_desc_t *dst_md, const jit_reduction_conf_t &conf) {
    using namespace data_type;

    if (is_superset(conf.isa, avx512_common))
        return safe_ptr_assign(kernel_,
                new jit_uni_reduction_kernel_t<Xbyak::Zmm>(conf, dst_md));
    else if (is_superset(conf.isa, avx)) {
        const bool is_src_i8 = utils::one_of(conf.src_type, s8, u8);
        const bool is_dst_i8 = utils::one_of(conf.dst_type, s8, u8);
        if (is_src_i8 || is_dst_i8)
            return safe_ptr_assign(kernel_,
                    new jit_uni_reduction_kernel_t<Xbyak::Xmm>(conf, dst_md));

        return safe_ptr_assign(kernel_,
                new jit_uni_reduction_kernel_t<Xbyak::Ymm>(conf, dst_md));
    } else if (is_superset(conf.isa, sse41))
        return safe_ptr_assign(kernel_,
                new jit_uni_reduction_kernel_t<Xbyak::Xmm>(conf, dst_md));
    else
        return status::runtime_error;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
