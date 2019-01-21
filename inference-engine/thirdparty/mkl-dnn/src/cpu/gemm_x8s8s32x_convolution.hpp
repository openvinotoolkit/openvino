/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef GEMM_X8S8S32X_CONVOLUTION_HPP
#define GEMM_X8S8S32X_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_primitive_conf.hpp"
#include "gemm_convolution_utils.hpp"

#include "gemm/os_blas.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <bool with_relu, data_type_t src_type, data_type_t dst_type>
struct _gemm_x8s8s32x_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine, const typename pd_t::base_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, attr,
                    hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T("gemm:blas",
                _gemm_x8s8s32x_convolution_fwd_t<with_relu, src_type, dst_type>);

        virtual status_t init() override {
            using namespace data_type;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
#if !USE_MKL_IGEMM
                && false
#endif
                && this->set_default_params() == status::success
                && utils::one_of(this->cdesc_().prop_kind,
                        prop_kind::forward_training,
                        prop_kind::forward_inference)
                && this->cdesc_().alg_kind == alg_kind::convolution_direct
                && !this->has_zero_dim_memory()
                && this->cdesc_().src_desc.data_type == src_type
                && this->cdesc_().dst_desc.data_type == dst_type
                && this->cdesc_().weights_desc.data_type == s8
                && IMPLICATION(this->with_bias(), utils::one_of(
                            this->cdesc_().bias_desc.data_type, f32, s32, s8,
                            u8))
                && this->cdesc_().accum_data_type == data_type::s32
                && utils::everyone_is(nhwc, this->src_pd_.desc()->format,
                        this->dst_pd_.desc()->format)
                && this->weights_pd_.desc()->format == (this->with_groups()
                        ? ((src_type == data_type::s8) ? hwigo_s8s8 : hwigo)
                        : ((src_type == data_type::s8) ? hwio_s8s8 : hwio))
                && this->is_gemm_conv_format();

            return ok ? status::success : status::unimplemented;
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            bool is_sign_input =
                    (this->cdesc_().src_desc.data_type == data_type::s8);
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(nhwc));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(nhwc));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(this->with_groups()
                            ? ((is_sign_input) ? hwigo_s8s8 : hwigo)
                            : ((is_sign_input) ? hwio_s8s8 : hwio)));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }

        virtual bool is_gemm_conv_format() const {
            using namespace mkldnn::impl::primitive_kind;
            bool ok = true;
            auto const &po = this->attr()->post_ops_;
            switch (po.len_) {
            case 0: break;
            case 1: ok = ok
                    && (po.entry_[0].is_relu() || po.contain(sum, 0));
                break;
            case 2: ok = ok
                    && (po.contain(sum, 0) && po.entry_[1].is_relu());
                break;
            default: ok = false;
            }
            return ok;
        }
    };

    _gemm_x8s8s32x_convolution_fwd_t(const pd_t *pd, const input_vector &inputs,
           const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
        , scratchpad_(nullptr)
    {
        jit_gemm_convolution_utils::init_conf(conf_.jcp_,
            *conf_.cdesc(), conf_.src_pd(), conf_.weights_pd(0),
            conf_.dst_pd(), mkldnn_get_max_threads(), with_relu, conf_.negative_slope());

        size_t col_size = (size_t)conf_.jcp_.im2col_sz * sizeof(src_data_t);
        size_t acc_size = (size_t)conf_.jcp_.os * conf_.jcp_.oc
                            * sizeof(acc_data_t);
        size_t size = col_size + acc_size;

        jit_gemm_convolution_utils::prepare_scratchpad(this->conf_.jcp_,
                &this->scratchpad_, size, this->conf_.jcp_.nthr);
    }

    ~_gemm_x8s8s32x_convolution_fwd_t() {
        delete this->scratchpad_;
    };

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src_base, const wei_data_t *wei_base,
            const char *bia_base, dst_data_t *dst_base,
            char *scratchpad);
    pd_t conf_;
    scratchpad_t *scratchpad_;
    int nthr_;
};

template <data_type_t dst_type>
struct _gemm_u8s8s32x_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t{
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
        {}

        DECLARE_COMMON_PD_T("gemm:blas",
                _gemm_u8s8s32x_convolution_bwd_data_t<dst_type>);

        virtual status_t init() override {
            using namespace data_type;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
#if !USE_MKL_IGEMM
                && false
#endif
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == prop_kind::backward_data
                && this->desc()->alg_kind == alg_kind::convolution_direct
                && !this->has_zero_dim_memory()
                && this->desc()->diff_src_desc.data_type == dst_type
                && this->desc()->diff_dst_desc.data_type == u8
                && this->desc()->weights_desc.data_type == s8
                && IMPLICATION(this->with_bias(), utils::one_of(
                            this->desc()->bias_desc.data_type, f32, s32, s8,
                            u8))
                && this->desc()->accum_data_type == data_type::s32
                && utils::everyone_is(nhwc, this->diff_src_pd_.desc()->format,
                        this->diff_dst_pd_.desc()->format)
                && this->weights_pd_.desc()->format == (this->with_groups()
                        ? hwigo : hwio)
                && attr()->post_ops_.has_default_values();

            return ok ? status::success : status::unimplemented;
        }

        virtual bool support_bias() const override { return true; }

        jit_gemm_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(nhwc));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(nhwc));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(this->with_groups()
                            ? hwigo : hwio));
            if (bias_pd_.desc()->format == any)
                CHECK(bias_pd_.set_format(x));
             return status::success;
        }
    };

    _gemm_u8s8s32x_convolution_bwd_data_t(const pd_t *pd, const input_vector &inputs,
           const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
        , scratchpad_(nullptr)
    {
        jit_gemm_convolution_utils::init_conf(conf_.jcp_,
            *conf_.desc(), conf_.diff_src_pd(), conf_.weights_pd(0),
            conf_.diff_dst_pd(), mkldnn_get_max_threads());

        size_t col_size = (size_t)conf_.jcp_.im2col_sz * sizeof(acc_data_t);
        size_t acc_size = (size_t)conf_.jcp_.is * conf_.jcp_.ic
                            * sizeof(acc_data_t);
        size_t size = col_size + acc_size;

        jit_gemm_convolution_utils::prepare_scratchpad(this->conf_.jcp_,
                &this->scratchpad_, size, this->conf_.jcp_.nthr);
    }

    ~_gemm_u8s8s32x_convolution_bwd_data_t() {
        delete this->scratchpad_;
    };

    typedef typename prec_traits<data_type::u8>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type diff_src_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual void execute(event_t *e) {
        execute_backward_data();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data();
    void execute_backward_data_thr(const int ithr, const int nthr,
            const diff_dst_data_t *diff_dst_base, const wei_data_t *wei_base,
            const char *bia_base, diff_src_data_t *diff_src_base,
            char *scratchpad);
    pd_t conf_;
    scratchpad_t *scratchpad_;
};

}
}
}

#endif
