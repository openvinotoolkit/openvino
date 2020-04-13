/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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

#ifndef CPU_REF_DECONVOLUTION_HPP
#define CPU_REF_DECONVOLUTION_HPP

#include <assert.h>
#include <string.h>

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_deconvolution_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "primitive_iterator.hpp"

#include "memory_tracking.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

static status_t compute_blocked_format(bool with_groups,
    const memory_desc_t *oi_md, memory_desc_t *io_md)
{
    using namespace memory_format;
    using namespace types;
    /* Computes blocking for *i*o* format from *o*i* format */
    if (oi_md->ndims != io_md->ndims) return status::invalid_arguments;
    blocking_desc_t oi_blk = oi_md->layout_desc.blocking,
        &io_blk = io_md->layout_desc.blocking;
    io_blk = oi_blk;
    nstl::swap(io_blk.strides[0][0+with_groups], io_blk.strides[0][1+with_groups]);
    nstl::swap(io_blk.strides[1][0+with_groups], io_blk.strides[1][1+with_groups]);
    nstl::swap(io_blk.padding_dims[0+with_groups], io_blk.padding_dims[1+with_groups]);
    nstl::swap(io_blk.offset_padding_to_data[0+with_groups],
         io_blk.offset_padding_to_data[1+with_groups]);
    nstl::swap(io_blk.block_dims[0+with_groups], io_blk.block_dims[1+with_groups]);

    if (is_format_double_blocked(oi_md->format)) {
        memory_format_t fmt;
        switch (oi_md->format) {
        case gOIhw8o16i2o: fmt = gIOhw8i16o2i; break;
        case OIhw8o16i2o: fmt = IOhw8i16o2i; break;
        /* 1x1 formats */
        case IOhw8o16i2o: fmt = OIhw8i16o2i; break;
        case gIOhw8o16i2o: fmt = gOIhw8i16o2i; break;
        case gOIhw8i16o2i: fmt = gIOhw8o16i2o; break;
        case OIhw8i16o2i: fmt = IOhw8o16i2o; break;
        default: return unimplemented;
        }
        io_md->format = fmt;
    } else
        io_md->format = memory_format::blocked;
    return status::success;
}

static status_t conv_descr_create(const deconvolution_desc_t *dd,
        convolution_desc_t *cd)
{
    using namespace prop_kind;
    using namespace memory_format;
    alg_kind_t alg_kind = ( dd->alg_kind == alg_kind::deconvolution_direct
        ? alg_kind::convolution_direct : alg_kind::convolution_winograd );
    prop_kind_t prop_kind;
    const memory_desc_t *src_md, *dst_md;
    memory_desc_t c_weights_d, d_weights_d;
    bool with_groups;
    if ( utils::one_of(dd->prop_kind, forward_training, forward_inference) ) {
        prop_kind = backward_data;
        src_md = &dd->dst_desc;
        dst_md = &dd->src_desc;
        d_weights_d = dd->weights_desc;
    } else if (dd->prop_kind == backward_data) {
        prop_kind = forward_training;
        src_md = &dd->diff_dst_desc;
        dst_md = &dd->diff_src_desc;
        d_weights_d = dd->weights_desc;
    } else {
        prop_kind = dd->prop_kind;
        src_md = &dd->diff_dst_desc;
        dst_md = &dd->src_desc;
        d_weights_d = dd->diff_weights_desc;
    }
    with_groups = d_weights_d.ndims == src_md->ndims + 1;

    /* create weights desc for convolution */
    c_weights_d = d_weights_d;
    nstl::swap(c_weights_d.dims[with_groups + 0], c_weights_d.dims[with_groups + 1]);
    if (c_weights_d.format != any)
    {
        if (utils::one_of(c_weights_d.format, gOIhw4i16o4i, OIhw4i16o4i))
            return unimplemented;
        CHECK(compute_blocked_format(with_groups, &d_weights_d, &c_weights_d));
    }
    return conv_desc_init(cd, prop_kind, alg_kind, src_md, &(c_weights_d),
            (prop_kind != backward_weights ? &(dd->bias_desc) : nullptr),
            dst_md, dd->strides, dd->dilates,
            dd->padding[0], dd->padding[1], dd->padding_kind);
}

struct ref_deconvolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_deconvolution_fwd_pd_t {
        pd_t(engine_t *engine,
                const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr)
        {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_fwd_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , conv_supports_bias_(other.conv_supports_bias_)
        {}

        ~pd_t() { delete conv_pd_; }

        DECLARE_DECONVOLUTION_PD_T(ref_deconvolution_fwd_t);

        status_t init_convolution(){
            using namespace memory_format;
            using namespace types;
            convolution_desc_t cd;
            status_t status;

            status = conv_descr_create(this->desc(), &cd);
            if (status != status::success) return status;

            mkldnn_primitive_desc_iterator it(this->engine_, (op_desc_t *)&cd,
                &(this->attr_), nullptr);
            while (++it != it.end()) {
                conv_pd_ = *it;
                conv_supports_bias_
                        = static_cast<cpu_convolution_bwd_data_pd_t *>(conv_pd_)
                                  ->support_bias();
                bool bias_supported = true
                        && desc()->accum_data_type == data_type::f32
                        && utils::one_of(desc()->dst_desc.data_type,
                                           data_type::f32, data_type::bf16);
                auto wei_fmt = format_normalize(
                        conv_pd_->weights_pd()->desc()->format);
                auto src_fmt = conv_pd_->diff_dst_pd()->desc()->format;

                bool ok = true && (wei_fmt == blocked)
                        && IMPLICATION(desc()->src_desc.data_type
                                           == data_type::bf16,
                                   utils::one_of(src_fmt, nCw16c, nChw16c,
                                               nCdhw16c, ncw, nchw, ncdhw))
                        && IMPLICATION(with_bias(),
                                   conv_supports_bias_ || bias_supported);
                if (ok)
                    return success;
                delete conv_pd_;
            }
            conv_pd_ = nullptr;
            return unimplemented;
        };
        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && utils::one_of(this->desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::one_of(this->desc()->alg_kind,
                        alg_kind::deconvolution_direct,
                        alg_kind::deconvolution_winograd)
                && attr()->post_ops_.has_default_values();

            if (ok) {
                CHECK(init_convolution());
                if (weights_pd_.desc()->format == memory_format::any) {
                    CHECK(compute_blocked_format(with_groups(),
                            conv_pd_->weights_pd()->desc(),
                            &desc_.weights_desc));
                    cpu_memory_pd_t weights(engine_, &desc_.weights_desc);
                    weights_pd_ = weights;
                }
                if (src_pd_.desc()->format == memory_format::any)
                    CHECK(src_pd_.set_format(conv_pd_->diff_dst_pd()->desc()->format));
                if (dst_pd_.desc()->format == memory_format::any)
                    CHECK(dst_pd_.set_format(conv_pd_->diff_src_pd()->desc()->format));
                if (bias_pd_.desc()->format == memory_format::any)
                    CHECK(bias_pd_.set_format(memory_format::x));

                init_scratchpad();
                return status::success;
            }
            else return status::unimplemented;
        }
        primitive_desc_t *conv_pd_;
        bool conv_supports_bias_;

        void init_scratchpad() {
            using namespace memory_format;
            using namespace mkldnn::impl::memory_tracking::names;

            memory_tracking::registrar_t scratchpad
                    = scratchpad_registry().registrar();
            if (desc()->dst_desc.data_type == data_type::bf16
                    && utils::one_of(dst_pd_.desc()->format, ncw, nchw, ncdhw)
                    && with_bias()) {
                const int SP = OW() * OH() * OD();
                const int num_thr
                        = mkldnn_in_parallel() ? 1 : mkldnn_get_max_threads();
                scratchpad.book(key_deconv_dst_bf16_convert_wsp,
                        sizeof(float) * SP * num_thr);
            }
            if (with_bias() && desc()->bias_desc.data_type == data_type::bf16) {
                scratchpad.book(key_conv_bias_bf16_convert_wsp, sizeof(float) * OC());
            }
        }
    };

    ref_deconvolution_fwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs), conv_p_(nullptr) {}

    ~ref_deconvolution_fwd_t() { delete this->conv_p_; }

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference:
            (conv_p_)->execute(e);
            if (pd()->with_bias() && !pd()->conv_supports_bias_) {
                auto dst_t = pd()->desc()->dst_desc.data_type;
                switch (pd()->dst_pd()->desc()->format) {
                /* XXX: current implementation only provides funcitonality. This
                 * needs to be cleaned and optimized. */
                case memory_format::ncw:
                case memory_format::nchw:
                case memory_format::ncdhw:
                    if (dst_t == data_type::bf16)
                        compute_fwd_bias_ncdhw_bf16();
                    else
                        compute_fwd_bias_ncdhw();
                    break;
                case memory_format::nChw8c:
                case memory_format::nCdhw8c:
                    assert(dst_t == data_type::f32);
                    compute_fwd_bias_nCdhwXc<8>();
                    break;
                case memory_format::nCw16c:
                case memory_format::nChw16c:
                case memory_format::nCdhw16c:
                    if (dst_t == data_type::bf16)
                        compute_fwd_bias_nCdhwXc_bf16<16>();
                    else
                        compute_fwd_bias_nCdhwXc<16>();
                    break;
                default:
                    assert(dst_t == data_type::f32);
                    compute_fwd_bias();
                    break;
                }
            }
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    typedef typename prec_traits<data_type::f32>::type f32_data_t;
    typedef typename prec_traits<data_type::bf16>::type bf16_data_t;
    void compute_fwd_bias() const;
    void compute_fwd_bias_ncdhw() const;
    void compute_fwd_bias_ncdhw_bf16() const;
    template <int blksize> void compute_fwd_bias_nCdhwXc() const;
    template <int blksize> void compute_fwd_bias_nCdhwXc_bf16() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    primitive_t *conv_p_;
};

struct ref_deconvolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_deconvolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr)
        {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_bwd_data_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        ~pd_t() { delete conv_pd_; }

        DECLARE_DECONVOLUTION_PD_T(ref_deconvolution_bwd_data_t);

        status_t init_convolution(){
            using namespace memory_format;
            using namespace types;
            convolution_desc_t cd;
            status_t status;

            status = conv_descr_create(this->desc(), &cd);
            if (status != status::success) return status;

             mkldnn_primitive_desc_iterator it(this->engine_, (op_desc_t *)&cd,
                &(this->attr_), nullptr);
             while (++it != it.end()) {
                conv_pd_ = *it;
                if (format_normalize(conv_pd_->weights_pd()->desc()->format)
                        == blocked)
                    return success;
                delete conv_pd_;
            }
            conv_pd_ = nullptr;
            return unimplemented;
        };

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace data_type;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->desc()->prop_kind == backward_data
                && (utils::everyone_is(data_type::f32,
                                this->desc()->weights_desc.data_type,
                                this->desc()->diff_dst_desc.data_type)
                        || utils::everyone_is(data_type::bf16,
                            this->desc()->weights_desc.data_type,
                            this->desc()->diff_dst_desc.data_type))
                && utils::one_of(
                        this->desc()->diff_src_desc.data_type,
                        data_type::f32,
                        data_type::bf16)
                && utils::one_of(this->desc()->alg_kind,
                               alg_kind::deconvolution_direct,
                               alg_kind::deconvolution_winograd);

            if (ok) {
                CHECK(init_convolution());
                if (weights_pd_.desc()->format == memory_format::any)
                {
                    CHECK(compute_blocked_format(with_groups(),
                        conv_pd_->weights_pd()->desc(),
                        &desc_.weights_desc));
                    cpu_memory_pd_t weights(engine_, &desc_.weights_desc);
                    weights_pd_ = weights;
                }
                if (diff_src_pd_.desc()->format == memory_format::any)
                    CHECK(diff_src_pd_.set_format(conv_pd_->dst_pd()->desc()->format));
                if (diff_dst_pd_.desc()->format == memory_format::any)
                    CHECK(diff_dst_pd_.set_format(conv_pd_->src_pd()->desc()->format));
                return status::success;
            }
            else return status::unimplemented;
        }
        primitive_desc_t *conv_pd_;
    };
    ref_deconvolution_bwd_data_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs), conv_p_(nullptr) {}
    ~ref_deconvolution_bwd_data_t() { delete this->conv_p_; }

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::backward_data:
            (conv_p_)->execute(e);
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    primitive_t *conv_p_;
};

struct ref_deconvolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_deconvolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr)
        {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_bwd_weights_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        ~pd_t() { delete conv_pd_; }

        DECLARE_DECONVOLUTION_PD_T(ref_deconvolution_bwd_weights_t);

        status_t init_convolution(){
            using namespace memory_format;
            using namespace types;
            convolution_desc_t cd;
            status_t status;

            status = conv_descr_create(this->desc(), &cd);
            if (status != status::success) return status;

             mkldnn_primitive_desc_iterator it(this->engine_, (op_desc_t *)&cd,
                &(this->attr_), nullptr);
             while (++it != it.end()) {
                conv_pd_ = *it;
                auto wei_fmt = conv_pd_->diff_weights_pd()->desc()->format;
                auto diff_dst_fmt = conv_pd_->src_pd()->desc()->format;
                bool ok = true && format_normalize(wei_fmt) == blocked
                        && !is_format_double_blocked(wei_fmt)
                        && IMPLICATION(desc()->src_desc.data_type
                                           == data_type::bf16,
                                   utils::one_of(diff_dst_fmt, nCw16c, nChw16c,
                                               nCdhw16c, ncw, nchw, ncdhw));
                if (ok)
                    return success;
                delete conv_pd_;
            }
            conv_pd_ = nullptr;
            return unimplemented;
        };

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->desc()->prop_kind == backward_weights
                && (utils::everyone_is(data_type::f32,
                        this->desc()->src_desc.data_type,
                        this->desc()->diff_dst_desc.data_type)
                    || (utils::everyone_is(data_type::bf16,
                            this->desc()->src_desc.data_type,
                            this->desc()->diff_dst_desc.data_type)
                        && utils::one_of(
                            this->desc()->diff_weights_desc.data_type,
                                                     data_type::f32,
                                                     data_type::bf16)))
                && utils::one_of(this->desc()->alg_kind,
                        alg_kind::deconvolution_direct,
                        alg_kind::deconvolution_winograd)
                && this->attr()->has_default_values()
                && IMPLICATION(this->with_bias(),
                        (utils::one_of(this->desc()->diff_bias_desc.data_type,
                                        data_type::f32, data_type::bf16))
                        && utils::one_of(this->desc()->diff_dst_desc.data_type,
                            data_type::f32, data_type::bf16));
            if (ok) {
                CHECK(init_convolution());
                if (diff_weights_pd_.desc()->format == memory_format::any)
                {
                    CHECK(compute_blocked_format(with_groups(),
                        conv_pd_->diff_weights_pd()->desc(),
                        &desc_.diff_weights_desc));
                    cpu_memory_pd_t weights(engine_, &desc_.diff_weights_desc);
                    diff_weights_pd_ = weights;
                }
                if (src_pd_.desc()->format == memory_format::any)
                    CHECK(src_pd_.set_format(conv_pd_->diff_dst_pd()->desc()->format));
                if (diff_dst_pd_.desc()->format == memory_format::any)
                    CHECK(diff_dst_pd_.set_format(conv_pd_->src_pd()->desc()->format));
                if (diff_bias_pd_.desc()->format == memory_format::any)
                    CHECK(diff_bias_pd_.set_format(memory_format::x));

                init_scratchpad();
                return status::success;
            }
            else return status::unimplemented;
        }
        primitive_desc_t *conv_pd_;

        void init_scratchpad() {
            using namespace memory_format;
            using namespace mkldnn::impl::memory_tracking::names;
            memory_tracking::registrar_t scratchpad
                    = scratchpad_registry().registrar();
            if (desc()->diff_dst_desc.data_type == data_type::bf16
                    && utils::one_of(
                               diff_dst_pd_.desc()->format, ncw, nchw, ncdhw)
                    && with_bias()) {
                const int SP = OW() * OH() * OD();
                const int num_thr
                        = mkldnn_in_parallel() ? 1 : mkldnn_get_max_threads();
                memory_tracking::registrar_t scratchpad
                        = scratchpad_registry().registrar();
                scratchpad.book(key_deconv_dst_bf16_convert_wsp,
                        sizeof(float) * SP * num_thr);
            }
            if (with_bias() && desc()->diff_bias_desc.data_type == data_type::bf16) {
                scratchpad.book(key_conv_bias_bf16_convert_wsp, sizeof(float) * OC());
            }
        }
    };

    ref_deconvolution_bwd_weights_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs), conv_p_(nullptr) {}

    ~ref_deconvolution_bwd_weights_t() { delete this->conv_p_; }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::backward_weights:
            (conv_p_)->execute(e);
            if (pd()->with_bias()) {
                auto dst_t = pd()->desc()->diff_dst_desc.data_type;
                switch (pd()->diff_dst_pd()->desc()->format) {
                /* XXX: current implementation only provides funcitonality. This
                 * needs to be cleaned and optimized. */
                case memory_format::ncw:
                case memory_format::nchw:
                case memory_format::ncdhw:
                    if (dst_t == data_type::bf16)
                        compute_bwd_bias_ncdhw_bf16();
                    else
                        compute_bwd_bias_ncdhw();
                    break;
                case memory_format::nChw8c:
                    assert(dst_t == data_type::f32);
                    compute_bwd_bias_nCdhwXc<8>();
                    break;
                case memory_format::nCw16c:
                case memory_format::nChw16c:
                case memory_format::nCdhw16c:
                    if (dst_t == data_type::bf16)
                        compute_bwd_bias_nCdhwXc_bf16<16>();
                    else
                        compute_bwd_bias_nCdhwXc<16>();
                    break;
                default:
                    assert(dst_t == data_type::f32);
                    compute_bwd_bias();
                    break;
                }
            }
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    typedef typename prec_traits<data_type::f32>::type f32_data_t;
    typedef typename prec_traits<data_type::bf16>::type bf16_data_t;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    primitive_t *conv_p_;
    void compute_bwd_bias() const;
    void compute_bwd_bias_ncdhw() const;
    void compute_bwd_bias_ncdhw_bf16() const;
    template <int blksize> void compute_bwd_bias_nCdhwXc() const;
    template <int blksize> void compute_bwd_bias_nCdhwXc_bf16() const;
};

}
}
}
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
