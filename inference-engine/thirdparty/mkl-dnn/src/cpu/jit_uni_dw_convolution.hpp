/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef CPU_JIT_UNI_DW_CONVOLUTION_HPP
#define CPU_JIT_UNI_DW_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_dw_conv_kernel_f32.hpp"
#include "cpu_reducer.hpp"
#include "cpu_barrier.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa, bool with_relu>
struct _jit_uni_dw_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine, const typename pd_t::base_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, attr,
                hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_dw:", isa, ""),
                _jit_uni_dw_convolution_fwd_t<isa, with_relu>);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->cdesc_().prop_kind, forward_training,
                        forward_inference)
                && this->cdesc_().alg_kind == alg_kind::convolution_direct
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::f32,
                        this->cdesc_().src_desc.data_type,
                        this->cdesc_().weights_desc.data_type,
                        this->cdesc_().dst_desc.data_type)
                && IMPLICATION(this->with_bias(),
                        data_type::f32 == this->cdesc_().bias_desc.data_type);

            if (!ok) return status::unimplemented;

            return jit_uni_dw_conv_fwd_kernel_f32<isa>::init_conf(jcp_,
                        this->cdesc_(),
                        this->src_pd_.desc(), *this->weights_pd_.desc(),
                        *this->dst_pd_.desc(), *this->attr(),
                        with_relu, this->negative_slope());
        }

        jit_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            auto desired_act_fmt = isa == avx512_common ? nChw16c : nChw8c;
            auto desired_wei_fmt = isa == avx512_common ? Goihw16g : Goihw8g;

            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(desired_act_fmt));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(desired_act_fmt));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(desired_wei_fmt));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }
    };

    _jit_uni_dw_convolution_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
        , padded_bias_(nullptr) {
        kernel_ = new jit_uni_dw_conv_fwd_kernel_f32<isa>(conf_.jcp_, *conf_.attr());
        if (conf_.want_padded_bias()) {
            padded_bias_ = (float *)malloc(sizeof(float) * conf_.jcp_.oc, 64);
            for (int c = conf_.jcp_.oc_without_padding; c < conf_.jcp_.oc; ++c)
                padded_bias_[c] = 0;
        }
    }

    ~_jit_uni_dw_convolution_fwd_t() {
        delete kernel_;
        free(padded_bias_);
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    jit_uni_dw_conv_fwd_kernel_f32<isa> *kernel_;
    float *padded_bias_;
};

using jit_avx512_common_dw_convolution_fwd_t =
    _jit_uni_dw_convolution_fwd_t<avx512_common, false>;
using jit_avx2_dw_convolution_fwd_t =
    _jit_uni_dw_convolution_fwd_t<avx2, false>;
using jit_sse42_dw_convolution_fwd_t =
    _jit_uni_dw_convolution_fwd_t<sse42, false>;

using jit_avx512_common_dw_convolution_relu_t =
    _jit_uni_dw_convolution_fwd_t<avx512_common, true>;
using jit_avx2_dw_convolution_relu_t =
    _jit_uni_dw_convolution_fwd_t<avx2, true>;
using jit_sse42_dw_convolution_relu_t =
    _jit_uni_dw_convolution_fwd_t<sse42, true>;

template <cpu_isa_t isa>
struct _jit_uni_dw_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
        {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_dw:", isa, ""),
                _jit_uni_dw_convolution_bwd_data_t);

        virtual status_t init() override {
            using namespace prop_kind;

            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, backward,
                        backward_data)
                && this->desc()->alg_kind == alg_kind::convolution_direct
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::f32,
                        this->desc()->diff_src_desc.data_type,
                        this->desc()->weights_desc.data_type,
                        this->desc()->diff_dst_desc.data_type);

            if (!ok) return status::unimplemented;

            return jit_uni_dw_conv_bwd_data_kernel_f32<isa>::init_conf(jcp_,
                        *this->desc(), *this->diff_src_pd_.desc(),
                        *this->weights_pd_.desc(), *this->diff_dst_pd_.desc());
        }

        jit_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {

            using namespace memory_format;
            auto desired_act_fmt = isa == avx512_common ? nChw16c : nChw8c;
            auto desired_wei_fmt = isa == avx512_common ? Goihw16g : Goihw8g;

            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(desired_act_fmt));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(desired_act_fmt));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(desired_wei_fmt));

            return status::success;
        }
    };

    _jit_uni_dw_convolution_bwd_data_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    { kernel_ = new jit_uni_dw_conv_bwd_data_kernel_f32<isa>(conf_.jcp_); }
    ~_jit_uni_dw_convolution_bwd_data_t() { delete kernel_; };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        switch (conf_.desc()->prop_kind) {
        case prop_kind::backward_data:
            execute_backward_data();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data();
    pd_t conf_;
    jit_uni_dw_conv_bwd_data_kernel_f32<isa> *kernel_;
};

using jit_avx512_common_dw_convolution_bwd_data_t =
    _jit_uni_dw_convolution_bwd_data_t<avx512_common>;
using jit_avx2_dw_convolution_bwd_data_t =
    _jit_uni_dw_convolution_bwd_data_t<avx2>;
using jit_sse42_dw_convolution_bwd_data_t =
    _jit_uni_dw_convolution_bwd_data_t<sse42>;

template <cpu_isa_t isa>
struct _jit_uni_dw_convolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_dw:", isa, ""),
                _jit_uni_dw_convolution_bwd_weights_t<isa>);

        virtual status_t init() override {
            using namespace prop_kind;

            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == prop_kind::backward_weights
                && this->desc()->alg_kind == alg_kind::convolution_direct
                && utils::everyone_is(data_type::f32,
                        this->desc()->src_desc.data_type,
                        this->desc()->diff_weights_desc.data_type,
                        this->desc()->diff_dst_desc.data_type);

            if (!ok) return status::unimplemented;

            return jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::init_conf(jcp_,
                        *this->desc(), *this->src_pd_.desc(),
                        *this->diff_weights_pd_.desc(), *this->diff_dst_pd_.desc());
        }

        jit_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {

            using namespace memory_format;
            auto desired_act_fmt = isa == avx512_common ? nChw16c : nChw8c;
            auto desired_wei_fmt = isa == avx512_common ? Goihw16g : Goihw8g;

            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(desired_act_fmt));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(desired_act_fmt));
            if (this->diff_weights_pd_.desc()->format == any)
                CHECK(this->diff_weights_pd_.set_format(desired_wei_fmt));
            if (this->diff_bias_pd_.desc()->format == any)
                CHECK(this->diff_bias_pd_.set_format(x));

            return status::success;
        }
    };

    _jit_uni_dw_convolution_bwd_weights_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs);
    ~_jit_uni_dw_convolution_bwd_weights_t() {
        delete kernel_;
        if (acc_ker_)
            delete acc_ker_;

        free(ws_reduction_);
        free(bias_reduction_);
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        execute_backward_weights();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_weights();

    pd_t conf_;
    jit_uni_dw_conv_bwd_weights_kernel_f32<isa> *kernel_;

    data_t *ws_reduction_ = nullptr;
    data_t *bias_reduction_ = nullptr;

    /* Used when executing a parallel reduction */
    cpu_accumulator_1d_t<data_type::f32> *acc_ker_ = nullptr;
    simple_barrier::ctx_t reduction_bctx_;

    /* For parallel implementation details see '.cpp' file in the
     * backwards-by-wights section. */
    int nthr_, nthr_g_, nthr_mb_;

    inline bool do_parallel_reduction(){
        return false;
    }
};

using jit_avx512_common_dw_convolution_bwd_weights_t =
    _jit_uni_dw_convolution_bwd_weights_t<avx512_common>;
using jit_avx2_dw_convolution_bwd_weights_t =
    _jit_uni_dw_convolution_bwd_weights_t<avx2>;
using jit_sse42_dw_convolution_bwd_weights_t =
    _jit_uni_dw_convolution_bwd_weights_t<sse42>;

}
}
}

#endif
