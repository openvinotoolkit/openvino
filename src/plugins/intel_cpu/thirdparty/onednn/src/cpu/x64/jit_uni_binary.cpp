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

#include <functional>

#include "cpu/cpu_primitive.hpp"
#include "cpu/x64/jit_uni_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

static bcast_set_t get_supported_bcast_strategies() {
    return {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial};
}

static bool compare_layouts(const memory_desc_wrapper &src0_md,
        const memory_desc_wrapper &src1_md) {
    const strides_t &strides0 = src0_md.blocking_desc().strides;
    const strides_t &strides1 = src1_md.blocking_desc().strides;
    const dims_t &dims0 = src0_md.dims();
    const dims_t &dims1 = src1_md.dims();
    const int ndims = src0_md.ndims();

    //different layouts are not allowed for bcast (allow 1 == 1)
    bool is_bcast = false;
    for (int d = 0; d < ndims; d++)
        is_bcast = is_bcast || dims0[d] != dims1[d];
    if (is_bcast) return true;

    bool same_layouts = true;
    for (int d = 0; d < ndims; d++)
        same_layouts = same_layouts && strides0[d] == strides1[d];
    return same_layouts;
}

static dim_t get_different_layout_stride(
        const strides_t &strides0, const strides_t &strides1, const int ndims) {
    for (int d = 0; d < ndims; d++)
        if (strides0[d] == 1) return strides1[d];
    return strides1[ndims - 1];
}

static dim_t get_outer_dims_product(
        const strides_t &strides0, const dims_t &dims, const int ndims) {
    // nchw:nhwc->nchw
    if (strides0[1] == 1) return dims[1];
    // nhwc:nchw->nhwc
    else if (strides0[ndims - 1] == 1)
        return utils::array_product(dims + 2, ndims - 2);
    else
        return dims[ndims - 1];
}

using namespace data_type;

static bool data_type_supported(const data_type_t dtype) {
    return utils::one_of(dtype, f32, bf16, s8, u8);
}

status_t jit_uni_binary_t::pd_t::init(engine_t *engine) {
    using sm = primitive_attr_t::skip_mask_t;

    conf_.dst_type = dst_md()->data_type;
    conf_.src0_type = src_md(0)->data_type;
    conf_.src1_type = src_md(1)->data_type;

    memory_desc_wrapper dst_md_(dst_md());
    memory_desc_wrapper src0_md_(src_md(0));
    memory_desc_wrapper src1_md_(src_md(1));

    const auto &po = attr()->post_ops_;
    const int elt_idx = po.find(primitive_kind::eltwise);
    conf_.is_i8 = utils::one_of(conf_.dst_type, s8, u8);

    bool ok = data_type_supported(conf_.dst_type)
            && data_type_supported(conf_.src0_type)
            && data_type_supported(conf_.src1_type)
            && IMPLICATION(conf_.src0_type == bf16, mayiuse(avx512_core))
            && set_default_params() == status::success && !has_zero_dim_memory()
            && IMPLICATION(!conf_.is_i8, src0_md_ == dst_md_) && is_applicable()
            && attr()->has_default_values(sm::post_ops | sm::scales_runtime)
            && attr_.set_default_formats(dst_md(0)) == status::success;
    if (!ok) return status::unimplemented;

    // All operations over blocking descriptors should have md initialized.
    conf_.is_src_different_layouts = !compare_layouts(src0_md_, src1_md_);
    ok = post_ops_ok(
                 attr(), src_md(0), dst_md(), conf_.is_src_different_layouts)
            && (conf_.is_i8 || elt_idx == -1
                    || IMPLICATION(!dst_md_.is_dense(),
                            cpu_eltwise_fwd_pd_t::eltwise_preserves_zero(
                                    po.entry_[elt_idx].eltwise)))
            && IMPLICATION((!attr()->scales_.has_default_values()),
                    check_scales_mask())
            && (conf_.is_i8
                    || IMPLICATION(!mayiuse(avx2),
                            src0_md_.consistent_with(src1_md_)
                                    || src0_md_.is_plain()));

    if (!ok) return status::unimplemented;

    conf_.postops_per_oc_broadcast_exists
            = binary_injector::any_binary_postop_rhs_per_oc_broadcast(
                    po, src0_md_, get_supported_bcast_strategies());
    conf_.is_bf16 = conf_.dst_type == bf16;
    conf_.op_type = get_op_type(src0_md_);
    assert(conf_.op_type != op_t::none);
    conf_.do_scale_src0 = !attr()->scales_.get(DNNL_ARG_SRC_0).defined()
            || !attr()->scales_.get(DNNL_ARG_SRC_0).has_default_values();
    conf_.do_scale_src1 = !attr()->scales_.get(DNNL_ARG_SRC_1).defined()
            || !attr()->scales_.get(DNNL_ARG_SRC_1).has_default_values();
    conf_.do_sum = po.contain(primitive_kind::sum, 0)
            && po.entry_[0].sum.scale != 0.f;
    conf_.with_eltwise = po.find(primitive_kind::eltwise) != -1;
    const bool with_binary = po.find(primitive_kind::binary) != -1;
    conf_.with_postops = with_binary || conf_.with_eltwise;
    conf_.sum_scale = conf_.do_sum ? po.entry_[0].sum.scale : 0.f;
    const auto &bcast_dims = broadcast_dims();
    conf_.bcast_type = is_tensor_op() ? bcast_t::none
                                      : get_bcast_type(src1_md_, bcast_dims);
    conf_.broadcast_src1_value = (conf_.op_type == op_t::n_c_spatial
                                         && conf_.bcast_type == bcast_t::per_c)
            || (utils::one_of(conf_.op_type, op_t::n_spatial_c, op_t::c_blocked)
                    && conf_.bcast_type == bcast_t::per_w)
            || conf_.bcast_type == bcast_t::scalar;
    conf_.use_stride_src1 = !conf_.broadcast_src1_value
            && (conf_.bcast_type == bcast_t::none
                    || (conf_.op_type == op_t::n_spatial_c
                            && conf_.bcast_type == bcast_t::per_c)
                    || (conf_.op_type == op_t::n_c_spatial
                            && conf_.bcast_type == bcast_t::per_w));
    conf_.use_stride_rhs_postops = conf_.postops_per_oc_broadcast_exists
            && conf_.op_type == op_t::n_spatial_c;

    const auto ndims = src0_md_.ndims();
    if (conf_.is_src_different_layouts) {
        const auto &strides0 = src0_md_.blocking_desc().strides;
        const auto &strides1 = src1_md_.blocking_desc().strides;
        conf_.src1_stride
                = get_different_layout_stride(strides0, strides1, ndims);
        conf_.outer_dims
                = get_outer_dims_product(strides0, src0_md_.dims(), ndims);
    }
    if (conf_.bcast_type == bcast_t::per_w) {
        for (int d = 2; d < ndims; ++d)
            conf_.not_bcasted_sp_dims += !bcast_dims[d];
    }

    return status::success;
}

op_t jit_uni_binary_t::pd_t::get_op_type(const memory_desc_wrapper &src0_d) {
    const auto &strides = src0_d.blocking_desc().strides;
    const auto ndims = src0_d.ndims();

    if (!src0_d.is_plain())
        return op_t::c_blocked;
    else if (strides[1] == 1)
        return op_t::n_spatial_c;
    else if (strides[0] >= strides[1]
            && IMPLICATION(ndims >= 3, strides[1] >= strides[2]))
        return op_t::n_c_spatial;
    return op_t::none;
}

bcast_t jit_uni_binary_t::pd_t::get_bcast_type(
        const memory_desc_wrapper &src1_d, const dims_t &bcast_dims) {
    if (src1_d.nelems() == 1)
        return bcast_t::scalar;
    else if (bcast_dims[1] == 1)
        return bcast_t::per_w;
    else
        return bcast_t::per_c;
}

bool jit_uni_binary_t::pd_t::alg_preserves_zero() const {
    using namespace utils;
    using namespace alg_kind;
    return utils::one_of(desc()->alg_kind, binary_add, binary_max, binary_min,
            binary_mul, binary_sub, binary_ge, binary_gt, binary_le, binary_lt,
            binary_eq, binary_ne, binary_prelu);
}

bool jit_uni_binary_t::pd_t::check_scales_mask() const {
    for (const auto &s : attr()->scales_.scales_) {
        if (s.second.mask_ != 0) return false;
    }
    return true;
}

bool jit_uni_binary_t::pd_t::is_bcast_pattern(const dims_t &bcast_dims,
        const dim_t ndims, const dim_t N_bcast, const dim_t C_bcast,
        const dim_t W_bcast) const {
    return bcast_dims[0] == N_bcast && bcast_dims[1] == C_bcast
            && bcast_dims[ndims - 1] == W_bcast;
}

bool jit_uni_binary_t::pd_t::is_bcast_allowed(const int ndims) const {
    // supported cases: NxCxDxHxW:{NxCx1x1x1,1xCx1x1x1,Nx1xDxHxW,Nx1x1xHxW,
    //                            Nx1x1x1xW,1x1xDxHxW,1x1x1xHxW,1x1x1x1xW,
    //                            1x1x1x1x1}
    const auto &bcast_dims = broadcast_dims();
    // check if there is continuous broadcast between non-broadcast dims
    int next_bcast_expected = 1;
    bool ok = true;
    for (int d = 2; d < ndims; ++d) {
        if (bcast_dims[d] == 0) next_bcast_expected = 0;
        ok = ok && bcast_dims[d] == next_bcast_expected;
    }
    if (ndims > 2)
        ok = ok
                && (is_bcast_pattern(bcast_dims, ndims, 0, 1, 0)
                        || is_bcast_pattern(bcast_dims, ndims, 1, 1, 0)
                        || (!!next_bcast_expected // not broadcast dim not met
                                && (is_bcast_pattern(bcast_dims, ndims, 0, 0, 1)
                                        || is_bcast_pattern(
                                                bcast_dims, ndims, 1, 0, 1)
                                        || is_bcast_pattern(
                                                bcast_dims, ndims, 1, 1, 1))));
    return ok;
}

// check for different src formats with same dims
// broadcast can be accepted if src_dim == src1_dims (1 == 1)
bool jit_uni_binary_t::pd_t::is_different_layouts_allowed(
        const memory_desc_wrapper &src0_d,
        const memory_desc_wrapper &src1_d) const {
    const dims_t &src0_dims = src0_d.dims();
    const dims_t &src1_dims = src1_d.dims();
    const int ndims = src0_d.ndims();
    const auto &bd0 = src0_d.blocking_desc();
    const auto &bd1 = src1_d.blocking_desc();

    bool skippable = true;
    for (int d = 0; d < ndims; d++)
        skippable = skippable && src0_dims[d] == src1_dims[d];

    // allow nchw:nhwc and nhwc:nchw
    const bool aligned_batch = bd0.strides[0] > bd0.strides[1]
            && bd1.strides[0] > bd1.strides[1];
    //disable for blocked layouts
    return skippable && utils::everyone_is(0, bd0.inner_nblks, bd1.inner_nblks)
            && aligned_batch;
}

bool jit_uni_binary_t::pd_t::is_applicable() {
    const memory_desc_wrapper src0_d(src_md(0));
    const memory_desc_wrapper src1_d(src_md(1));
    const memory_desc_wrapper dst_d(dst_md());
    const auto ndims = src0_d.ndims();

    // check density first to avoid same non-dense src0 and src1 to pass
    // the next check
    bool ok = src0_d.is_dense(true) && src1_d.is_dense(true)
            && dst_d.is_dense(true);
    if (!ok) return false;

    // TODO: fix implementation for tensor with paddings to work with any block
    // size. For now return unimplemented if more than single blocking
    // or `block size > 16`.
    const auto &blk_d = dst_d.blocking_desc();
    if (!dst_d.is_dense()
            && (blk_d.inner_nblks > 1 || blk_d.inner_blks[0] > 16))
        return false;

    const bool is_src_different_layouts = !compare_layouts(src0_d, src1_d);
    const bool different_layouts_allowed
            = is_different_layouts_allowed(src0_d, src1_d);
    if (!conf_.is_i8) {
        const bool has_padding = utils::one_of(true,
                src0_d.nelems(true) != src0_d.nelems(false),
                src1_d.nelems(true) != src1_d.nelems(false),
                dst_d.nelems(true) != dst_d.nelems(false));
        ok = IMPLICATION(has_padding, alg_preserves_zero());
        if (!ok) return false;

        // full tensor operation
        bool same_dims = true;
        const auto &src0_dims = src0_d.dims();
        const auto &src1_dims = src1_d.dims();
        for (int d = 0; d < ndims; d++)
            same_dims = same_dims && src0_dims[d] == src1_dims[d];
        if (same_dims
                && IMPLICATION(
                        is_src_different_layouts, different_layouts_allowed))
            return true;
    } else {
        const dim_t C = ndims >= 2 ? src0_d.dims()[1] : 1;
        const bool has_oc_tail = C != src0_d.padded_dims()[1];
        const bool has_outer_dims_tail = is_src_different_layouts
                && get_outer_dims_product(src0_d.blocking_desc().strides,
                        src0_d.dims(), src0_d.ndims());

        // Disable compare operations when blocked tag with tail.
        // Tail processing is not supported and the vcmps instruction
        // overwrites the output vector.
        if (utils::one_of(desc()->alg_kind, alg_kind::binary_ge,
                    alg_kind::binary_gt, alg_kind::binary_le,
                    alg_kind::binary_lt, alg_kind::binary_eq,
                    alg_kind::binary_ne, alg_kind::binary_prelu)
                && (has_oc_tail || has_outer_dims_tail))
            return false;

        // full tensor operation
        if (src0_d.similar_to(src1_d, true, false, 0)
                || different_layouts_allowed)
            return true;
        // source0 broadcast not supported
        if (!src0_d.similar_to(dst_d, true, false, 0)) return false;
    }
    // broadcast operation
    if (!(is_bcast_allowed(ndims)
                && IMPLICATION(
                        is_src_different_layouts, different_layouts_allowed)))
        return false;

    if (!conf_.is_i8) {
        if (src0_d.is_plain() && src1_d.is_plain()) {
            const auto &bd0 = src0_d.blocking_desc();
            const auto &bd1 = src1_d.blocking_desc();
            // only nspc and ncsp formats are supported for bcast
            return bd0.strides[0]
                    == utils::array_product(
                            src0_d.dims() + 1, src0_d.ndims() - 1)
                    && IMPLICATION(bd0.strides[1] > 1,
                            bd0.strides[1]
                                    == utils::array_product(src0_d.dims() + 2,
                                            src0_d.ndims() - 2))
                    && bd1.strides[0] >= bd1.strides[1];
        }

        // check blocking_desc consistency
        const auto valid_bd = [&](const memory_desc_wrapper &mdw) {
            int blksize = 8;
            if (mayiuse(avx512_core)) blksize = 16;
            const auto &bd = mdw.blocking_desc();

            return bd.inner_nblks == 1 && bd.inner_blks[0] == blksize
                    && bd.inner_idxs[0] == 1;
        };

        return valid_bd(src0_d) && valid_bd(src1_d);
    } else {
        const auto &bd0 = src0_d.blocking_desc();
        const auto &bd1 = src1_d.blocking_desc();
        const auto &bcast_dims = broadcast_dims();
        // disable blocked tag for source1 when W is not broadcast
        return bd0.strides[1] == 1 && bd0.inner_nblks == 0
                && IMPLICATION(
                        bcast_dims[ndims - 1] == 0, bd1.inner_nblks == 0);
    }
}

bool jit_uni_binary_t::post_ops_ok(const primitive_attr_t *attr,
        const memory_desc_wrapper &src0_d, const memory_desc_wrapper &dst_d,
        const bool is_src_different_layouts) {
    using namespace primitive_kind;

    const auto &p = attr->post_ops_;
    const auto is_eltwise = [&](int idx) {
        if (p.entry_[idx].is_eltwise()) {
            const auto alg = p.entry_[idx].eltwise.alg;
            return eltwise_injector::is_alg_supported(alg);
        }
        return false;
    };
    const auto is_binary = [&](int idx) { return p.entry_[idx].is_binary(); };
    const auto is_binary_bf16 = [&](int idx) {
        return is_binary(idx)
                && p.entry_[idx].binary.src1_desc.data_type == data_type::bf16;
    };
    const bool is_avx512_core = mayiuse(avx512_core);
    const bool is_i8 = utils::one_of(dst_d.data_type(), s8, u8);

    for (int i = 0; i < p.len(); i++) {
        if (p.contain(primitive_kind::sum, i)) {
            if (p.entry_[i].sum.zero_point != 0) return false;
            if (i > 0) return false;
            if (src0_d.data_type() != dst_d.data_type()) return false;
        } else if (!(is_eltwise(i) || is_binary(i))
                || ((is_i8 || !is_avx512_core) && is_binary_bf16(i)))
            return false;
    }

    const int vlen = is_avx512_core ? cpu_isa_traits<avx512_core>::vlen
                                    : is_i8 && mayiuse(avx512_common)
                    ? cpu_isa_traits<avx512_common>::vlen
                    : cpu_isa_traits<avx2>::vlen;
    const auto supported_strategies = get_supported_bcast_strategies();
    const bool postops_per_oc_broadcast_exists
            = binary_injector::any_binary_postop_rhs_per_oc_broadcast(
                    p, src0_d, supported_strategies);
    if (postops_per_oc_broadcast_exists && is_src_different_layouts)
        return false;
    const int blksize = vlen / sizeof(float);

    const bool blocked_format = !src0_d.is_plain() && src0_d.is_blocking_desc();

    if (postops_per_oc_broadcast_exists && blocked_format) {
        /*
         * check blocking_desc consistency, currently when among postops exists
         * per_oc broadcast, binary kernel doesn't support situations when blocked
         * format size is smaller then vlen. example: sse41 vlen size is 4 and format
         * is nChw8c - not supported, avx2 vlen size is 8 and format is
         * nChw8c - supported.
         */
        const auto blocking_desc = src0_d.blocking_desc();
        if (blocking_desc.inner_nblks != 1
                || blocking_desc.inner_blks[0] != blksize
                || blocking_desc.inner_idxs[0] != 1)
            return false;
    }

    const dim_t n_dims = src0_d.ndims();
    const dim_t &oc = n_dims >= 2 ? src0_d.dims()[1] : 1;

    /*
     * TODO: Remove limitation supporting tail with blocked format for i8i8
     */
    const bool blocked_tail = p.len() && blocked_format && oc % blksize;

    return binary_injector::binary_args_broadcast_supported(
                   p, src0_d, get_supported_bcast_strategies())
            && IMPLICATION(
                    utils::one_of(src0_d.data_type(), s8, u8), !blocked_tail)
            && IMPLICATION(postops_per_oc_broadcast_exists,
                    binary_injector::all_binary_postop_rhs_per_oc_broadcast(p,
                            src0_d, supported_strategies,
                            [&src0_d](const memory_desc_wrapper &rhs_arg_md) {
                                return IMPLICATION(!mayiuse(avx2),
                                        src0_d.consistent_with(rhs_arg_md)
                                                || src0_d.is_plain());
                            }));
}

binary_kernel_t *create_binary_kernel(
        const jit_uni_binary_t::pd_t *pd, bool tail_kernel) {
    const auto &conf = pd->get_conf();
    if (mayiuse(avx512_core_bf16)) {
        if (conf.is_i8) {
            using kernel_t = jit_uni_binary_kernel_t<avx512_common>;
            return new kernel_t(pd, conf, false);
        } else {
            using kernel_t = jit_uni_binary_kernel_t<avx512_core_bf16>;
            return new kernel_t(pd, conf, tail_kernel);
        }
    } else if (mayiuse(avx512_core)) {
        if (conf.is_i8) {
            using kernel_t = jit_uni_binary_kernel_t<avx512_common>;
            return new kernel_t(pd, conf, false);
        } else {
            using kernel_t = jit_uni_binary_kernel_t<avx512_core>;
            return new kernel_t(pd, conf, tail_kernel);
        }
    } else if (mayiuse(avx512_common) && conf.is_i8) {
        using kernel_t = jit_uni_binary_kernel_t<avx512_common>;
        return new kernel_t(pd, conf, false);
    } else if (mayiuse(avx2)) {
        using kernel_t = jit_uni_binary_kernel_t<avx2>;
        return new kernel_t(pd, conf, tail_kernel && !conf.is_i8);
    } else {
        using kernel_t = jit_uni_binary_kernel_t<sse41>;
        return new kernel_t(pd, conf, tail_kernel && !conf.is_i8);
    }
}

jit_uni_binary_t::jit_uni_binary_t(const pd_t *apd) : primitive_t(apd) {}

status_t jit_uni_binary_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(
            kernel_, create_binary_kernel(pd(), false /*tail_kernel*/)));

    if (utils::one_of(pd()->dst_md(0)->data_type, f32, bf16)) {
        const memory_desc_wrapper src0_d(pd_->src_md(0));
        const auto &simd_w = kernel_->simd_w();
        const auto oc = src0_d.ndims() >= 2 ? src0_d.dims()[1] : 1;

        if (op_t::c_blocked == pd()->get_conf().op_type && oc % simd_w) {
            CHECK(safe_ptr_assign(kernel_tail_,
                    create_binary_kernel(pd(), true /*tail_kernel*/)));
            CHECK(kernel_tail_->create_kernel());
        }
    }

    return kernel_->create_kernel();
}

void jit_uni_binary_t::execute_no_bcast_strategy(const data_t *src0,
        const data_t *src1, data_t *dst, const float *scale0,
        const float *scale1,
        const std::vector<const void *> &post_ops_binary_rhs_arg_vec,
        const bcast_t bcast_type) const {
    const auto kernel = kernel_.get();
    const auto &simd_w = kernel_->simd_w();

    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper src1_d(pd()->src_md(1));
    const memory_desc_wrapper dst_d(pd()->dst_md(0));
    const int src0_type_size = types::data_type_size(src0_d.data_type());
    const int src1_type_size = types::data_type_size(src1_d.data_type());
    const int dst_type_size = types::data_type_size(dst_d.data_type());

    const auto &conf = pd()->get_conf();
    const bool is_src_different_layouts = conf.is_src_different_layouts;

    if (is_src_different_layouts) {
        std::vector<unsigned> indices;

        const dim_t src1_different_layout_stride = conf.src1_stride;
        for (size_t i = 0; i < simd_w; i++)
            indices.push_back(
                    i * src1_different_layout_stride * src1_type_size);

        const dim_t batch = src0_d.dims()[0];
        const dim_t batch_stride = src1_d.blocking_desc().strides[0];
        const dim_t outer_dims = conf.outer_dims;
        const size_t src1_stride_range
                = outer_dims * src1_different_layout_stride * src1_type_size;

        const dim_t nelems_per_aligned_dims
                = src0_d.nelems(true) / (batch * outer_dims);
        const dim_t nelems0_simd = nelems_per_aligned_dims / simd_w;
        const dim_t nelems0_tail = nelems_per_aligned_dims % simd_w;
        const bool has_tail = nelems0_tail > 0;

        const int nthr = dnnl_get_current_num_threads();
        const dim_t thr_per_nelems_group = nstl::min(
                nstl::max(nthr / batch, (dim_t)1), nelems0_simd + has_tail);

        // Compute strategy:
        // Iterate over batch and over outer dims.
        // Divide number of threads by batch size and limiting it by a number
        // of outer_dims nelems to parallel over it when needed.
        parallel_nd(
                batch, thr_per_nelems_group, [&](dim_t b, dim_t nelems_group) {
                    dim_t start = 0, end = 0;
                    balance211(nelems0_simd + has_tail, thr_per_nelems_group,
                            nelems_group, start, end);
                    if (start >= end) return;

                    const bool ithr_does_tail = has_tail
                            && utils::one_of(nelems0_simd + has_tail, end, 0);
                    const dim_t n_simd_to_do
                            = (end - start - ithr_does_tail) * simd_w;
                    const dim_t tail_to_do = ithr_does_tail * nelems0_tail;
                    const size_t batch_off = batch_stride * b;

                    if (nelems0_simd != 0) {
                        start *= outer_dims;
                        end *= outer_dims;
                    }

                    start *= simd_w;
                    jit_binary_call_s p;
                    p.spat_offt_count = (n_simd_to_do + tail_to_do) * outer_dims
                            * dst_type_size;
                    p.src0 = src0 + (start + batch_off) * src0_type_size;
                    p.src1 = src1
                            + (start / outer_dims + batch_off) * src1_type_size;
                    p.dst = dst + (start + batch_off) * dst_type_size;
                    p.indices = &indices[0];
                    p.src1_stride_range = src1_stride_range;
                    p.scales_src0 = scale0;
                    p.scales_src1 = scale1;
                    p.post_ops_binary_rhs_arg_vec
                            = post_ops_binary_rhs_arg_vec.data();
                    (*kernel)(&p);
                });
    } else {
        const dim_t nelems0 = src0_d.nelems(true);
        const dim_t nelems0_simd = nelems0 / simd_w;
        const dim_t nelems0_tail = nelems0 % simd_w;
        const bool has_tail = nelems0_tail > 0;

        const bool point_broadcast = bcast_type == bcast_t::scalar;

        // Compute strategy:
        // Compute number of vectors, divide it equally between all threads.
        // Last one will also handle a tail if present.
        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(nelems0_simd + has_tail, nthr, ithr, start, end);
            if (start >= end) return;

            const bool ithr_does_tail
                    = has_tail && end == nelems0_simd + has_tail;
            const dim_t n_simd_to_do = (end - start - ithr_does_tail) * simd_w;
            const dim_t tail_to_do = ithr_does_tail * nelems0_tail;

            jit_binary_call_s p;
            p.spat_offt_count = (n_simd_to_do + tail_to_do) * dst_type_size;
            p.src0 = src0 + start * simd_w * src0_type_size;
            p.src1 = src1
                    + (point_broadcast ? 0 : (start * simd_w * src1_type_size));
            p.dst = dst + start * simd_w * dst_type_size;
            p.scales_src0 = scale0;
            p.scales_src1 = scale1;
            p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
            (*kernel)(&p);
        });
    }
}

void jit_uni_binary_t::execute_bcast_per_c_strategy(const data_t *src0,
        const data_t *src1, data_t *dst, const float *scale0,
        const float *scale1,
        const std::vector<const void *> &post_ops_binary_rhs_arg_vec,
        const op_t op_type, const bcast_t bcast_type,
        const bool blocked_oc_tail) const {
    const auto kernel = kernel_.get();
    const auto kernel_tail = kernel_tail_.get();
    const auto &simd_w = kernel_->simd_w();

    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper src1_d(pd()->src_md(1));
    const memory_desc_wrapper dst_d(pd()->dst_md(0));
    const int src0_type_size = types::data_type_size(src0_d.data_type());
    const int src1_type_size = types::data_type_size(src1_d.data_type());
    const int dst_type_size = types::data_type_size(dst_d.data_type());
    const auto ndims = src0_d.ndims();
    const auto &dims = src0_d.dims();
    const dim_t MB = dims[0];
    const dim_t C = ndims >= 2 ? dims[1] : 1;
    const dim_t SP = ndims >= 3 ? utils::array_product(dims + 2, ndims - 2) : 1;

    const auto &bcast_dims = pd()->broadcast_dims();
    const bool no_broadcast = bcast_type == bcast_t::none;
    const bool point_broadcast = bcast_type == bcast_t::scalar;

    const dim_t nelems_slice_src0
            = utils::array_product(src0_d.padded_dims() + 1, ndims - 1);
    const dim_t nelems_slice_src1 = no_broadcast
            ? nelems_slice_src0
            : ((bcast_dims[0] == 0) ? utils::array_product(
                       src1_d.padded_dims() + 1, ndims - 1)
                                    : 0);

    if (op_type == op_t::c_blocked) {
        const dim_t C_blocks = std::ceil(
                static_cast<float>(src0_d.padded_dims()[1]) / simd_w);
        // Compute strategy:
        // Each block is individual - parallel over MB and C_blocks safely.

        const std::function<void(jit_binary_call_s *, dim_t)>
                kernel_blocked_no_tail
                = [&](jit_binary_call_s *p, dim_t C_blk) { (*kernel)(p); };
        const std::function<void(jit_binary_call_s *, dim_t)>
                kernel_blocked_tail = [&](jit_binary_call_s *p, dim_t C_blk) {
                    if (C_blk == (C_blocks - 1))
                        (*kernel_tail)(p);
                    else
                        (*kernel)(p);
                };
        const auto &kernel_blocked = blocked_oc_tail ? kernel_blocked_tail
                                                     : kernel_blocked_no_tail;

        parallel_nd(MB, C_blocks, [&](dim_t mb, dim_t C_blk) {
            jit_binary_call_s p;
            p.spat_offt_count = SP * simd_w * dst_type_size;
            const dim_t off = mb * nelems_slice_src0 + C_blk * SP * simd_w;
            p.dst = dst + off * dst_type_size;
            p.src0 = src0 + off * src0_type_size;
            const dim_t src1_off = point_broadcast
                    ? mb * nelems_slice_src1
                    : (no_broadcast ? off
                                    : mb * nelems_slice_src1 + C_blk * simd_w);
            p.src1 = src1 + src1_off * src1_type_size;
            p.oc_l_off = C_blk * simd_w;
            p.scales_src0 = scale0;
            p.scales_src1 = scale1;
            p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
            kernel_blocked(&p, C_blk);
        });
    } else if (op_type == op_t::n_spatial_c) {
        // Compute strategy:
        // Each line of channels is individual, parallel over MB and spatial.

        parallel_nd(MB, SP, [&](dim_t mb, dim_t sp) {
            jit_binary_call_s p;
            p.spat_offt_count = C * dst_type_size;
            const auto off = mb * nelems_slice_src0 + sp * C;
            p.dst = dst + off * dst_type_size;
            p.src0 = src0 + off * src0_type_size;
            const auto src1_off = no_broadcast ? off : mb * nelems_slice_src1;
            p.src1 = src1 + src1_off * src1_type_size;
            p.oc_l_off = 0;
            p.scales_src0 = scale0;
            p.scales_src1 = scale1;
            p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
            (*kernel)(&p);
        });
    } else if (op_type == op_t::n_c_spatial) {
        // Compute strategy:
        // Each line of spatial is individual, parallel over MB and C.

        parallel_nd(MB, C, [&](dim_t mb, dim_t c) {
            jit_binary_call_s p;
            p.spat_offt_count = SP * dst_type_size;
            const auto off = mb * nelems_slice_src0 + c * SP;
            p.dst = dst + off * dst_type_size;
            p.src0 = src0 + off * src0_type_size;
            const dim_t src1_off = point_broadcast
                    ? mb * nelems_slice_src1
                    : (no_broadcast ? off : mb * nelems_slice_src1 + c);
            p.src1 = src1 + src1_off * src1_type_size;
            p.oc_l_off = c;
            p.scales_src0 = scale0;
            p.scales_src1 = scale1;
            p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
            (*kernel)(&p);
        });
    }
}

void jit_uni_binary_t::execute_bcast_per_w_strategy(const data_t *src0,
        const data_t *src1, data_t *dst, const float *scale0,
        const float *scale1,
        const std::vector<const void *> &post_ops_binary_rhs_arg_vec,
        const op_t op_type, const bool blocked_oc_tail) const {
    const auto kernel = kernel_.get();
    const auto kernel_tail = kernel_tail_.get();
    const auto &simd_w = kernel_->simd_w();

    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper src1_d(pd()->src_md(1));
    const memory_desc_wrapper dst_d(pd()->dst_md(0));
    const int src0_type_size = types::data_type_size(src0_d.data_type());
    const int src1_type_size = types::data_type_size(src1_d.data_type());
    const int dst_type_size = types::data_type_size(dst_d.data_type());
    const auto ndims = src0_d.ndims();
    const auto &dims = src0_d.dims();
    const auto &bcast_dims = pd()->broadcast_dims();

    const int not_bcasted_sp_dims = pd()->get_conf().not_bcasted_sp_dims;
    const dim_t MB = dims[0];
    // array product of outer dimensions that are not broadcast
    const dim_t SP_no_bcast = ndims >= 3
            ? utils::array_product(
                    dims + (ndims - not_bcasted_sp_dims), not_bcasted_sp_dims)
            : 1;
    const dim_t C = ndims >= 2 ? dims[1] : 1;
    const dim_t SP = ndims >= 3 ? utils::array_product(dims + 2, ndims - 2) : 1;
    // spatial without dimensions that are not broadcasted by src1
    const dim_t N = SP / SP_no_bcast;

    const dim_t nelems_slice_src0
            = utils::array_product(src0_d.padded_dims() + 1, ndims - 1);

    if (op_type == op_t::c_blocked) {
        const dim_t C_blocks = std::ceil(
                static_cast<float>(src0_d.padded_dims()[1]) / simd_w);
        // Compute strategy:
        // Each line of channels is individual, parallel over MB, C_blocks
        // and spatial (broadcasted and not broadcasted spatial dims
        // separately).

        const std::function<void(jit_binary_call_s *, dim_t)>
                kernel_blocked_no_tail
                = [&](jit_binary_call_s *p, dim_t C_blk) { (*kernel)(p); };
        const std::function<void(jit_binary_call_s *, dim_t)>
                kernel_blocked_tail = [&](jit_binary_call_s *p, dim_t C_blk) {
                    if (C_blk == (C_blocks - 1))
                        (*kernel_tail)(p);
                    else
                        (*kernel)(p);
                };
        const auto &kernel_blocked = blocked_oc_tail ? kernel_blocked_tail
                                                     : kernel_blocked_no_tail;

        parallel_nd(MB, C_blocks, N, SP_no_bcast,
                [&](dim_t mb, dim_t C_blk, dim_t n, dim_t sp) {
                    jit_binary_call_s p;
                    p.spat_offt_count = simd_w * dst_type_size;
                    const auto off = mb * nelems_slice_src0
                            + simd_w * (C_blk * SP + n * SP_no_bcast + sp);
                    p.dst = dst + off * dst_type_size;
                    p.src0 = src0 + off * src0_type_size;
                    // check if mb is broadcast
                    const dim_t src1_off = bcast_dims[0] == 1
                            ? sp * simd_w
                            : (mb * SP_no_bcast + sp) * simd_w;
                    p.src1 = src1 + src1_off * src1_type_size;
                    p.oc_l_off = C_blk * simd_w;
                    p.scales_src0 = scale0;
                    p.scales_src1 = scale1;
                    p.post_ops_binary_rhs_arg_vec
                            = post_ops_binary_rhs_arg_vec.data();
                    kernel_blocked(&p, C_blk);
                });
    } else if (op_type == op_t::n_spatial_c) {
        // Compute strategy:
        // Each line of channels is individual, parallel over MB and spatial
        // (broadcasted and not broadcasted spatial dims separately).

        parallel_nd(MB, N, SP_no_bcast, [&](dim_t mb, dim_t n, dim_t sp) {
            jit_binary_call_s p;
            p.spat_offt_count = C * dst_type_size;
            const auto off
                    = mb * nelems_slice_src0 + n * SP_no_bcast * C + sp * C;
            p.dst = dst + off * dst_type_size;
            p.src0 = src0 + off * src0_type_size;
            const dim_t src1_off
                    = bcast_dims[0] == 1 ? sp : mb * SP_no_bcast + sp;
            p.src1 = src1 + src1_off * src1_type_size;
            p.oc_l_off = 0;
            p.scales_src0 = scale0;
            p.scales_src1 = scale1;
            p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
            (*kernel)(&p);
        });
    } else if (op_type == op_t::n_c_spatial) {
        // Compute strategy:
        // Each line of width is individual, parallel over MB, C and spatial
        // without not broadcasted dims. Use a kernel which broadcasts c_i
        // value into a vector register.

        parallel_nd(MB, C, N, [&](dim_t mb, dim_t c, dim_t n) {
            jit_binary_call_s p;
            p.spat_offt_count = SP_no_bcast * dst_type_size;
            const auto off = mb * nelems_slice_src0 + c * N * SP_no_bcast
                    + n * SP_no_bcast;
            p.dst = dst + off * dst_type_size;
            p.src0 = src0 + off * src0_type_size;
            const dim_t src1_off = bcast_dims[0] == 1 ? 0 : mb * SP_no_bcast;
            p.src1 = src1 + src1_off * src1_type_size;
            p.oc_l_off = c;
            p.scales_src0 = scale0;
            p.scales_src1 = scale1;
            p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
            (*kernel)(&p);
        });
    }
}

status_t jit_uni_binary_t::execute(const exec_ctx_t &ctx) const {
    const auto src0 = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC_0);
    const auto src1 = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC_1);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    const auto &post_ops = pd()->attr()->post_ops_;
    const auto &post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(post_ops, ctx);
    const float *scales[2];
    ASSIGN_INPUT_SCALE_VALUE(scales[0], DNNL_ARG_SRC_0);
    ASSIGN_INPUT_SCALE_VALUE(scales[1], DNNL_ARG_SRC_1);

    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper src1_d(pd()->src_md(1));
    const auto ndims = src0_d.ndims();
    const auto &dims = src0_d.dims();
    const dim_t C = ndims >= 2 ? dims[1] : 0;

    const bool postops_per_oc_broadcast_exists
            = binary_injector::any_binary_postop_rhs_per_oc_broadcast(
                    post_ops, src0_d, get_supported_bcast_strategies());
    const auto &bcast_type = pd()->get_conf().bcast_type;
    const bool point_broadcast = bcast_type == bcast_t::scalar;
    const auto &op_type = pd()->get_conf().op_type;
    const bool with_postops = !post_ops.entry_.empty();
    const auto &simd_w = kernel_->simd_w();
    const bool has_oc_tail = C % simd_w;
    const bool point_broadcast_no_oc_tail = point_broadcast && !has_oc_tail;
    const auto alg = pd()->desc()->alg_kind;
    // Use strategy with kernel_tail for GreaterEqual op with oc_tail and
    // blocked format due to overwriting the vector tail by vcmpps.
    const bool vector_overwrite = utils::one_of(alg, alg_kind::binary_ge,
            alg_kind::binary_gt, alg_kind::binary_le, alg_kind::binary_lt,
            alg_kind::binary_eq, alg_kind::binary_ne, alg_kind::binary_prelu);
    const bool blocked_oc_tail = op_type == op_t::c_blocked && has_oc_tail
            && (with_postops || point_broadcast || bcast_type == bcast_t::per_w
                    || vector_overwrite);

    if ((bcast_type == bcast_t::none || point_broadcast_no_oc_tail)
            && !postops_per_oc_broadcast_exists && !blocked_oc_tail)
        execute_no_bcast_strategy(src0, src1, dst, scales[0], scales[1],
                post_ops_binary_rhs_arg_vec, bcast_type);
    else if (bcast_type == bcast_t::per_w)
        execute_bcast_per_w_strategy(src0, src1, dst, scales[0], scales[1],
                post_ops_binary_rhs_arg_vec, op_type, blocked_oc_tail);
    else
        execute_bcast_per_c_strategy(src0, src1, dst, scales[0], scales[1],
                post_ops_binary_rhs_arg_vec, op_type, bcast_type,
                blocked_oc_tail);

    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
