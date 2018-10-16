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

#ifndef CPU_JIT_AVX512_CORE_CONVOLUTION_WINOGRAD_HPP
#define CPU_JIT_AVX512_CORE_CONVOLUTION_WINOGRAD_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "scratchpad.hpp"

#include "jit_avx512_core_conv_winograd_kernel_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace winograd {

struct winograd_scratchpad_avx512_core_t {
    public:
        winograd_scratchpad_avx512_core_t(const jit_conv_winograd_conf_t &jcp)
        {
            get_scratchpad_size_(jcp);
            allocate_scratchpad_(jcp);
        }

        ~winograd_scratchpad_avx512_core_t() {
            if (scratchpad_ != nullptr)
                delete scratchpad_;
        }

        char *U_ptr() {
            /* buffer for wei transform U*/
            return scratchpad_->get() + U_offset_;
        }

        char *V_ptr() {
            /* buffer for src transform V*/
            return scratchpad_->get() + V_offset_;
        }

        char *M_ptr() {
            /* buffer for dst transform M*/
            return scratchpad_->get() + M_offset_;
        }

        char *bias_ptr() {
            /* buffer for bias update in bwdw*/
            return scratchpad_->get() + bias_offset_;
        }

        int num_threads(){
            return nthreads_;
        }

    private:
        inline void get_scratchpad_size_(const jit_conv_winograd_conf_t &jcp) {
            nthreads_ = omp_get_max_threads();

            U_sz_ = size_t(alpha) * alpha * jcp.ic * jcp.oc * sizeof(float);
            V_sz_ = size_t(alpha) * alpha * jcp.mb * jcp.ic
                           * jcp.itiles * jcp.jtiles
                           * sizeof(float);
            M_sz_ = size_t(alpha) * alpha * jcp.mb * jcp.oc
                           * jcp.itiles * jcp.jtiles
                           * sizeof(float);

            switch (jcp.sched_policy) {
            case WSCHED_DATA_W_SGD:
                V_sz_ = nthreads_ * alpha * alpha
                    * jcp.nb_tile_block_ur * jcp.tile_block_ur
                    * jcp.ic * sizeof(float);
                M_sz_ = nthreads_* alpha * alpha
                    * jcp.nb_tile_block_ur * jcp.tile_block_ur
                    * jcp.oc * sizeof(float);
                break;
            case WSCHED_WEI_SDGtWo:
                U_sz_ = nthreads_
                    * (alpha * alpha * jcp.oc * (jcp.ic / jcp.nb_ic)
                      + jcp.ic * jcp.oc * jcp.kh * jcp.kw)
                    * sizeof(float);
                M_sz_ = nthreads_ * alpha * alpha
                        * (jcp.ntiles / jcp.tile_block)
                        * (jcp.oc / jcp.nb_oc) * sizeof(float);
                V_sz_ = nthreads_ * alpha * alpha
                        * (jcp.ntiles / jcp.tile_block)
                        * (jcp.ic / jcp.nb_ic)
                        * sizeof(float);
                bias_sz_ = nthreads_ * jcp.oc * sizeof(float);
                break;
            case WSCHED_WEI_S_D_Giot_W:
                U_sz_ = (nthreads_ + 1) * alpha * alpha * jcp.ic * jcp.oc
                      * sizeof(float);
                M_sz_ = size_t(alpha) * alpha * jcp.oc * jcp.ntiles * sizeof(float);
                V_sz_ = size_t(alpha) * alpha * jcp.ic * jcp.ntiles * sizeof(float);
                bias_sz_ = nthreads_ * jcp.oc * sizeof(float);
                break;
            default:
                break;
            }
        }

        inline void allocate_scratchpad_(const jit_conv_winograd_conf_t &jcp) {
            const size_t page_size = PAGE_2M;
            U_offset_ = 0;
            V_offset_ = utils::rnd_up(U_sz_, page_size);
            M_offset_ = V_offset_ + utils::rnd_up(V_sz_, page_size);
            scratchpad_sz_ = M_offset_ + M_sz_;
            if (bias_sz_) {
                bias_offset_ = M_offset_ + utils::rnd_up(M_sz_, page_size);
                scratchpad_sz_ = bias_offset_ + bias_sz_;
            }
            scratchpad_ = create_scratchpad(scratchpad_sz_);
        }

        scratchpad_t *scratchpad_;
        size_t nthreads_;
        size_t scratchpad_sz_ = 0, U_sz_ = 0, V_sz_ = 0, M_sz_ = 0,
               bias_sz_ = 0;
        size_t U_offset_ = 0;
        size_t V_offset_ = 0;
        size_t M_offset_ = 0;
        size_t bias_offset_ = 0;
};
}

template <bool is_fwd>
struct _jit_avx512_core_convolution_winograd_t {

    _jit_avx512_core_convolution_winograd_t(
            const jit_conv_winograd_conf_t &jcp, const primitive_attr_t *attr)
        : kernel_(nullptr), scratchpad_(nullptr), attr_(attr) {
            kernel_ =  new _jit_avx512_core_conv_winograd_data_kernel_f32(jcp);
            scratchpad_ = new winograd::winograd_scratchpad_avx512_core_t(jcp);
        }

    ~_jit_avx512_core_convolution_winograd_t() {
        delete kernel_;
        delete scratchpad_;
    };

    protected:
        void weight_transform_data(const jit_conv_winograd_conf_t &jcp,
            float *wp, float *twp);
        void input_transform_data(int image,
            const jit_conv_winograd_conf_t &jcp,
            float *inp, float *tinp);
        void input_transform_tileblock_data(int tile_block,
            const jit_conv_winograd_conf_t &jcp,
            float *inp, float *tinp);
        void output_transform_data(int image,
            const jit_conv_winograd_conf_t &jcp,
            const post_ops_t &p_ops, float *toutp, float *pout_b, float *bias);
        void output_transform_tileblock_data(int tile_block,
            const jit_conv_winograd_conf_t &jcp, const post_ops_t &p_ops,
            float *toutp, float *outp, float *bias);
        void _execute_data_W_S_G_D(const int MB, float *inp_ptr, float *out_ptr,
                float *wei_ptr, float *bias_ptr = NULL);
        void _execute_data_W_SGD(const int MB, float *inp_ptr, float *out_ptr,
                float *wei_ptr, float *bias_ptr = NULL);
        _jit_avx512_core_conv_winograd_data_kernel_f32 *kernel_;
        // Buffer required to store transforms in the frequency domain
        winograd::winograd_scratchpad_avx512_core_t *scratchpad_;
        const primitive_attr_t *attr_;
};

template <bool with_relu>
struct _jit_avx512_core_convolution_winograd_fwd_t
     : _jit_avx512_core_convolution_winograd_t<true>
     , public cpu_primitive_t
    {
    struct pd_t : public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine, const typename pd_t::base_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, attr,
                    hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_wino:", avx512_core, ""),
                _jit_avx512_core_convolution_winograd_fwd_t<with_relu>);

        virtual status_t init() override
        {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true && this->set_default_params() == status::success
                    && utils::one_of(this->cdesc_().prop_kind, forward_training,
                               forward_inference)
                    && this->cdesc_().alg_kind == alg_kind::convolution_winograd
                    && utils::everyone_is(data_type::f32,
                               this->cdesc_().src_desc.data_type,
                               this->cdesc_().weights_desc.data_type,
                               this->cdesc_().dst_desc.data_type)
                    && utils::implication(this->with_bias(), data_type::f32
                                       == this->cdesc_().bias_desc.data_type);
            if (!ok)
                return status::unimplemented;

            return jit_avx512_core_conv_winograd_fwd_kernel_f32::init_conf(
                    jcp_, this->cdesc_(), *this->src_pd_.desc(),
                    *this->weights_pd_.desc(), *this->dst_pd_.desc(),
                    *this->attr(), with_relu, this->negative_slope());
        }

        jit_conv_winograd_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override
        {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(nChw16c));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(nChw16c));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(
                        this->with_groups() ? gOIhw16i16o : OIhw16i16o));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }
    };

    _jit_avx512_core_convolution_winograd_fwd_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs)
        : _jit_avx512_core_convolution_winograd_t<true>(pd->jcp_, pd->attr())
        , cpu_primitive_t(&conf_, inputs, outputs)
        , conf_(*pd) {}

    ~_jit_avx512_core_convolution_winograd_fwd_t(){};

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        float *src = (float *)this->input_memory(0);
        float *dst = (float *)this->memory();
        float *weights = (float *)this->input_memory(1);
        float *bias = (float *)this->input_memory(2);

        switch ((conf_.jcp_).sched_policy) {
        case WSCHED_DATA_W_S_G_D:
            this->_execute_data_W_S_G_D(conf_.MB(), src, dst, weights, bias);
            break;
        case WSCHED_DATA_W_SGD:
            this->_execute_data_W_SGD(conf_.MB(), src, dst, weights, bias);
            break;
        default:
            break;
        }
        e->set_state(event_t::ready);
    }

private:
    pd_t conf_;
};

using jit_avx512_core_convolution_winograd_fwd_t
        = _jit_avx512_core_convolution_winograd_fwd_t<false>;
using jit_avx512_core_convolution_winograd_relu_t
        = _jit_avx512_core_convolution_winograd_fwd_t<true>;

struct jit_avx512_core_convolution_winograd_bwd_data_t
        : _jit_avx512_core_convolution_winograd_t<false>,
        public cpu_primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_wino:", avx512_core, ""),
                jit_avx512_core_convolution_winograd_bwd_data_t);

        virtual status_t init() override
        {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true && this->set_default_params() == status::success
                    && utils::one_of(this->desc()->prop_kind, backward_data)
                    && this->desc()->alg_kind == alg_kind::convolution_winograd
                    && utils::everyone_is(data_type::f32,
                               this->desc()->diff_src_desc.data_type,
                               this->desc()->weights_desc.data_type,
                               this->desc()->diff_dst_desc.data_type);
            if (!ok)
                return status::unimplemented;

            return jit_avx512_core_conv_winograd_bwd_data_kernel_f32::
                    init_conf(jcp_, *this->desc(), *this->diff_src_pd_.desc(),
                            *this->weights_pd_.desc(),
                            *this->diff_dst_pd_.desc());
        }

        jit_conv_winograd_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override
        {
            using namespace memory_format;

            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(nChw16c));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(nChw16c));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(
                        this->with_groups() ? gOIhw16i16o : OIhw16i16o));
            return status::success;
        }
    };

    jit_avx512_core_convolution_winograd_bwd_data_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs)
        : _jit_avx512_core_convolution_winograd_t<false>(pd->jcp_, pd->attr())
        , cpu_primitive_t(&conf_, inputs, outputs)
        , conf_(*pd) {}

    ~jit_avx512_core_convolution_winograd_bwd_data_t(){};

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        float *diff_dst = (float *)this->input_memory(0);
        float *diff_src = (float *)this->memory();
        float *weights = (float *)this->input_memory(1);

        if (conf_.desc()->prop_kind == prop_kind::backward_data) {
            switch ((conf_.jcp_).sched_policy) {
            case WSCHED_DATA_W_S_G_D:
                this->_execute_data_W_S_G_D(conf_.MB(), diff_dst, diff_src, weights, NULL);
                break;

            case WSCHED_DATA_W_SGD:
                this->_execute_data_W_SGD(conf_.MB(), diff_dst, diff_src, weights, NULL);
                break;

            default:
                break;
            }
        } else {
            assert(!"invalid prop_kind");
        }

        e->set_state(event_t::ready);
    }

private:
    pd_t conf_;
};

struct jit_avx512_core_convolution_winograd_bwd_weights_t
        : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr,
                    hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_wino:", avx512_core, ""),
                jit_avx512_core_convolution_winograd_bwd_weights_t);

        virtual status_t init() override
        {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true && this->set_default_params() == status::success
                    && utils::one_of(this->desc()->prop_kind, backward_weights)
                    && this->desc()->alg_kind == alg_kind::convolution_winograd
                    && utils::everyone_is(data_type::f32,
                               this->desc()->src_desc.data_type,
                               this->desc()->diff_dst_desc.data_type,
                               this->desc()->diff_weights_desc.data_type);
            if (!ok)
                return status::unimplemented;

            return jit_avx512_core_conv_winograd_bwd_weights_kernel_f32::
                    init_conf(jcp_, *this->desc(), *this->src_pd_.desc(),
                            *this->diff_dst_pd_.desc(),
                            *this->diff_weights_pd_.desc());
        }

        jit_conv_winograd_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override
        {
            using namespace memory_format;

            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(nChw16c));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(nChw16c));
            if (this->diff_weights_pd_.desc()->format == any)
                CHECK(this->diff_weights_pd_.set_format(
                        this->with_groups() ? gOIhw16i16o : OIhw16i16o));
            if (diff_bias_pd_.desc()->format == any)
                CHECK(diff_bias_pd_.set_format(x));
            return status::success;
        }
    };

    jit_avx512_core_convolution_winograd_bwd_weights_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs)
        , conf_(*pd)
        , kernel_(nullptr)
        , scratchpad_(nullptr)
    {
        auto jcp = conf_.jcp_;
        kernel_ = new jit_avx512_core_conv_winograd_bwd_weights_kernel_f32(
                jcp);
        scratchpad_ = new winograd::winograd_scratchpad_avx512_core_t(jcp);
    }

    ~jit_avx512_core_convolution_winograd_bwd_weights_t()
    {
        delete kernel_;
        delete scratchpad_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        if (conf_.desc()->prop_kind == prop_kind::backward_weights) {
            const auto &jcp = kernel_->jcp;
            switch (jcp.sched_policy) {
            case WSCHED_WEI_SDGtWo:
                _execute_backward_weights_SDGtWo();
                break;
            case WSCHED_WEI_S_D_Giot_W:
                _execute_backward_weights_S_D_Giot_W();
                break;
            default:
                assert(jcp.sched_policy != WSCHED_INVALID);
                break;
            }
        }
        else
            assert(!"invalid prop_kind");
        e->set_state(event_t::ready);
    }

private:
    void _execute_backward_weights_SDGtWo();
    void _execute_backward_weights_S_D_Giot_W();

    pd_t conf_;
    jit_avx512_core_conv_winograd_bwd_weights_kernel_f32 *kernel_;

    // Buffer required to store transforms in the frequency domain
    winograd::winograd_scratchpad_avx512_core_t *scratchpad_;
};
}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
