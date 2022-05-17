/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#ifndef GPU_NVIDIA_CUDNN_DECONVOLUTION_HPP
#define GPU_NVIDIA_CUDNN_DECONVOLUTION_HPP

#include "cudnn.h"

#include "common/c_types_map.hpp"
#include "common/deconvolution_pd.hpp"
#include "common/primitive_iterator.hpp"
#include "gpu/nvidia/cudnn_convolution.hpp"
#include "gpu/nvidia/cudnn_deconvolution_impl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

namespace {
static status_t compute_blocked_format(
        bool with_groups, const memory_desc_t *oi_md, memory_desc_t *io_md) {
    /* Computes blocking for *i*o* format from *o*i* format */

    bool sanity_check_ok = true && oi_md->ndims == io_md->ndims
            && oi_md->format_kind == format_kind::blocked;
    if (!sanity_check_ok) return status::invalid_arguments;

    const blocking_desc_t &oi_blk = oi_md->format_desc.blocking;
    blocking_desc_t io_blk = io_md->format_desc.blocking;

    io_md->format_kind = format_kind::blocked;
    io_blk = oi_blk;

    const int ID_OC = 0 + with_groups;
    const int ID_IC = 1 + with_groups;

    nstl::swap(io_blk.strides[ID_OC], io_blk.strides[ID_IC]);
    for (int i_blk = 0; i_blk < io_blk.inner_nblks; ++i_blk) {
        if (utils::one_of(io_blk.inner_idxs[i_blk], ID_OC, ID_IC)) {
            io_blk.inner_idxs[i_blk]
                    = (io_blk.inner_idxs[i_blk] == ID_OC ? ID_IC : ID_OC);
        }
    }

    return memory_desc_init_by_blocking_desc(*io_md, io_blk);
}

static status_t conv_descr_create(
        const deconvolution_desc_t *dd, convolution_desc_t *cd) {
    using namespace prop_kind;
    alg_kind_t alg_kind = dd->alg_kind == alg_kind::deconvolution_direct
            ? alg_kind::convolution_direct
            : alg_kind::convolution_winograd;

    const memory_desc_t *src_md, *dst_md, *d_weights_d;
    prop_kind_t prop_kind;
    memory_desc_t c_weights_d;
    if (utils::one_of(dd->prop_kind, forward_training, forward_inference)) {
        prop_kind = backward_data;
        src_md = &dd->dst_desc;
        dst_md = &dd->src_desc;
        d_weights_d = &dd->weights_desc;
    } else if (dd->prop_kind == backward_data) {
        prop_kind = forward_training;
        src_md = &dd->diff_dst_desc;
        dst_md = &dd->diff_src_desc;
        d_weights_d = &dd->weights_desc;
    } else {
        prop_kind = dd->prop_kind;
        src_md = &dd->diff_dst_desc;
        dst_md = &dd->src_desc;
        d_weights_d = &dd->diff_weights_desc;
    }

    const bool with_groups = d_weights_d->ndims == src_md->ndims + 1;

    /* create weights desc for convolution */
    c_weights_d = *d_weights_d;

    const int ID_OC = 0 + with_groups;
    const int ID_IC = 1 + with_groups;

    nstl::swap(c_weights_d.dims[ID_OC], c_weights_d.dims[ID_IC]);
    nstl::swap(c_weights_d.padded_dims[ID_OC], c_weights_d.padded_dims[ID_IC]);
    nstl::swap(c_weights_d.padded_offsets[ID_OC],
            c_weights_d.padded_offsets[ID_IC]);

    if (c_weights_d.format_kind != format_kind::any)
        CHECK(compute_blocked_format(with_groups, d_weights_d, &c_weights_d));

    return conv_desc_init(cd, prop_kind, alg_kind, src_md, &c_weights_d,
            prop_kind != backward_weights ? &dd->bias_desc : nullptr, dst_md,
            dd->strides, dd->dilates, dd->padding[0], dd->padding[1]);
}
} // namespace

struct cudnn_deconvolution_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public deconvolution_fwd_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : deconvolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr) {}

        pd_t(const pd_t &other)
            : deconvolution_fwd_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , conv_supports_bias_(other.conv_supports_bias_)
            , dst_tag_(other.dst_tag_) {}

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_deconvolution_fwd_t);

        status_t init_convolution(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;

            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));
            primitive_attr_t conv_attr = *attr();
            dnnl_primitive_desc_iterator it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            while (++it != it.end()) {
                conv_pd_ = *it;
                conv_supports_bias_ = static_cast<convolution_bwd_data_pd_t *>(
                        conv_pd_.get())
                                              ->support_bias();
                bool ref_deconv_supports_bias = true
                        && desc()->accum_data_type == data_type::f32
                        && utils::one_of(desc()->dst_desc.data_type, f32, f16)
                        && IMPLICATION(desc()->src_desc.data_type == f16,
                                memory_desc_matches_one_of_tag(
                                        *conv_pd_->diff_src_md(),
                                        utils::pick(ndims() - 3, ncw, nchw,
                                                ncdhw)));
                bool ok = true
                        && conv_pd_->weights_md()->extra.flags == 0
                        /* deconv reference code can process only f32 bias */
                        && IMPLICATION(with_bias(),
                                conv_supports_bias_
                                        || ref_deconv_supports_bias);
                if (ok) return status::success;
            }
            conv_pd_.reset();
            return status::unimplemented;
        }

        status_t init(engine_t *engine) {
            using namespace format_tag;
            bool ok = true && is_fwd();
            ok = ok
                    && utils::one_of(desc()->alg_kind,
                            alg_kind::deconvolution_direct,
                            alg_kind::deconvolution_winograd);
            ok = ok && attr_.has_default_values();
            ok = ok
                    && (utils::everyone_is(data_type::f32,
                                desc()->src_desc.data_type,
                                desc()->weights_desc.data_type,
                                desc()->dst_desc.data_type)
                            || utils::everyone_is(data_type::f16,
                                    desc()->src_desc.data_type,
                                    desc()->weights_desc.data_type,
                                    desc()->dst_desc.data_type));

            if (ok) {
                CHECK(init_convolution(engine));
                if (weights_md_.format_kind == format_kind::any) {
                    CHECK(compute_blocked_format(with_groups(),
                            conv_pd_->weights_md(), &desc_.weights_desc));
                    weights_md_ = desc_.weights_desc;
                }
                if (src_md_.format_kind == format_kind::any)
                    src_md_ = *conv_pd_->diff_dst_md();
                if (dst_md_.format_kind == format_kind::any)
                    dst_md_ = *conv_pd_->diff_src_md();
                if (bias_md_.format_kind == format_kind::any)
                    CHECK(memory_desc_init_by_tag(bias_md_, x));

                dst_tag_ = memory_desc_matches_one_of_tag(dst_md_,
                        utils::pick(ndims() - 3, ncw, nchw, ncdhw),
                        utils::pick(ndims() - 3, nCw4c, nChw4c, nCdhw4c));
                init_scratchpad();
                return status::success;
            }

            return status::unimplemented;
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }

        std::shared_ptr<primitive_desc_t> conv_pd_;
        bool conv_supports_bias_;
        format_tag_t dst_tag_;
    };

    ~cudnn_deconvolution_fwd_t() {}

    virtual status_t init(engine_t *engine) {
        return pd()->conv_pd_->create_primitive(conv_p_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const {
        using namespace memory_tracking::names;
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
        conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
        conv_args[DNNL_ARG_DIFF_SRC] = args.at(DNNL_ARG_DST);
        if (pd()->with_bias())
            conv_args[DNNL_ARG_BIAS] = args.at(DNNL_ARG_BIAS);
        exec_ctx_t conv_ctx(ctx.stream(), std::move(conv_args));

        nested_scratchpad_t ns(ctx, key_nested, conv_p_);
        conv_ctx.set_scratchpad_grantor(ns.grantor());
        // Executing the convolution kernel
        status_t status = conv_p_->execute(conv_ctx);
        return status;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> conv_p_;
};

struct cudnn_deconvolution_bwd_data_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public deconvolution_bwd_data_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : deconvolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr) {}

        pd_t(const pd_t &other)
            : deconvolution_bwd_data_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        ~pd_t() {}

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_deconvolution_bwd_data_t);

        status_t init_convolution(engine_t *engine) {
            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));
            primitive_attr_t conv_attr = *attr();
            dnnl_primitive_desc_iterator it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            while (++it != it.end()) {
                conv_pd_ = *it;
                return status::success;
            }
            return status::unimplemented;
        }

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && (utils::everyone_is(data_type::f32,
                                desc()->diff_src_desc.data_type,
                                desc()->weights_desc.data_type,
                                desc()->diff_dst_desc.data_type)
                            || utils::everyone_is(data_type::f16,
                                    desc()->weights_desc.data_type,
                                    desc()->diff_dst_desc.data_type))
                    && utils::one_of(desc()->diff_src_desc.data_type,
                            data_type::f16, data_type::f32)
                    && desc()->alg_kind == alg_kind::deconvolution_direct
                    && attr()->has_default_values();

            if (ok) {
                CHECK(init_convolution(engine));
                if (weights_md_.format_kind == format_kind::any) {
                    CHECK(compute_blocked_format(with_groups(),
                            conv_pd_->weights_md(), &desc_.weights_desc));
                    weights_md_ = desc_.weights_desc;
                }
                if (diff_src_md_.format_kind == format_kind::any)
                    diff_src_md_ = *conv_pd_->dst_md();
                if (diff_dst_md_.format_kind == format_kind::any)
                    diff_dst_md_ = *conv_pd_->src_md();
                init_scratchpad();
                return status::success;
            }

            return status::unimplemented;
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }

        std::shared_ptr<primitive_desc_t> conv_pd_;
    };

    ~cudnn_deconvolution_bwd_data_t() {}

    virtual status_t init(engine_t *engine) {
        return pd()->conv_pd_->create_primitive(conv_p_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const {
        using namespace memory_tracking::names;
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_DIFF_DST);
        conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
        conv_args[DNNL_ARG_DST] = args.at(DNNL_ARG_DIFF_SRC);
        if (!types::is_zero_md(pd()->scratchpad_md()))
            conv_args[DNNL_ARG_SCRATCHPAD] = args.at(DNNL_ARG_SCRATCHPAD);
        exec_ctx_t conv_ctx(ctx.stream(), std::move(conv_args));

        nested_scratchpad_t ns(ctx, key_nested, conv_p_);
        conv_ctx.set_scratchpad_grantor(ns.grantor());
        // Executing the convolution kernel
        status_t status = conv_p_->execute(conv_ctx);
        return status;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> conv_p_;
};

struct cudnn_deconvolution_bwd_weights_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public deconvolution_bwd_weights_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : deconvolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr) {}

        pd_t(const pd_t &other)
            : deconvolution_bwd_weights_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        ~pd_t() {}

        DECLARE_COMMON_PD_T(
                "cuda:cudnn:any", cudnn_deconvolution_bwd_weights_t);

        status_t init_convolution(engine_t *engine) {
            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));
            primitive_attr_t conv_attr = *attr();
            dnnl_primitive_desc_iterator it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            while (++it != it.end()) {
                conv_pd_ = *it;
                if (conv_pd_ == nullptr) return status::out_of_memory;
                return status::success;
            }
            return status::unimplemented;
        }

        status_t init(engine_t *engine) {
            using namespace format_tag;
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                    && (utils::everyone_is(data_type::f32,
                                desc()->src_desc.data_type,
                                desc()->diff_weights_desc.data_type,
                                desc()->diff_dst_desc.data_type)
                            || utils::everyone_is(data_type::f16,
                                    desc()->diff_dst_desc.data_type,
                                    desc()->src_desc.data_type))
                    && utils::one_of(
                            desc()->alg_kind, alg_kind::deconvolution_direct)
                    && attr()->has_default_values()
                    && utils::one_of(desc()->diff_weights_desc.data_type,
                            data_type::f16, data_type::f32);
            if (ok) {
                CHECK(init_convolution(engine));
                if (diff_weights_md_.format_kind == format_kind::any) {
                    CHECK(compute_blocked_format(with_groups(),
                            conv_pd_->diff_weights_md(),
                            &desc_.diff_weights_desc));
                    diff_weights_md_ = desc_.diff_weights_desc;
                }
                if (src_md_.format_kind == format_kind::any)
                    src_md_ = *conv_pd_->diff_dst_md();
                if (diff_dst_md_.format_kind == format_kind::any)
                    diff_dst_md_ = *conv_pd_->src_md();
                if (diff_bias_md_.format_kind == format_kind::any)
                    CHECK(memory_desc_init_by_tag(diff_bias_md_, x));
                init_scratchpad();
                return status::success;
            }

            return status::unimplemented;
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }

        std::shared_ptr<primitive_desc_t> conv_pd_;
    };

    ~cudnn_deconvolution_bwd_weights_t() {}

    virtual status_t init(engine_t *engine) {
        if (pd()->with_bias()) {
            if (pd()->ndims() > CUDNN_DIM_MAX) return status::invalid_arguments;

            impl_ = std::make_shared<cudnn_deconvolution_bwd_bias_impl_t>();
            impl_->init(pd()->invariant_dst_md(), pd()->invariant_bia_md());
        }
        return pd()->conv_pd_->create_primitive(conv_p_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const {
        using namespace memory_tracking::names;
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
        conv_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_DIFF_DST);
        conv_args[DNNL_ARG_DIFF_WEIGHTS] = args.at(DNNL_ARG_DIFF_WEIGHTS);
        if (!types::is_zero_md(pd()->scratchpad_md()))
            conv_args[DNNL_ARG_SCRATCHPAD] = args.at(DNNL_ARG_SCRATCHPAD);

        exec_ctx_t conv_ctx(ctx, std::move(conv_args));

        nested_scratchpad_t ns(ctx, key_nested, conv_p_);
        conv_ctx.set_scratchpad_grantor(ns.grantor());
        status_t status = conv_p_->execute(conv_ctx);
        if (status != status::success) return status;

        if (pd()->with_bias()) { return execute_bias(ctx); }
        return status::success;
    }

    status_t execute_bias(const exec_ctx_t &ctx) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> conv_p_;
    std::shared_ptr<cudnn_deconvolution_bwd_bias_impl_t> impl_;
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
