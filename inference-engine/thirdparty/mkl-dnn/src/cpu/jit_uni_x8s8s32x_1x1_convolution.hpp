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

#ifndef CPU_JIT_UNI_X8S8S32X_1x1_CONVOLUTION_HPP
#define CPU_JIT_UNI_X8S8S32X_1x1_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_reducer.hpp"
#include "jit_uni_x8s8s32x_1x1_conv_kernel.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa, bool with_relu, impl::data_type_t src_type, impl::data_type_t dst_type>
struct _jit_uni_x8s8s32x_1x1_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine, const typename pd_t::base_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, attr,
                hint_fwd_pd)
            , jcp_({}) {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_1x1:", isa, ""),
            _jit_uni_x8s8s32x_1x1_convolution_fwd_t<isa, with_relu, src_type, dst_type>);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->cdesc_().prop_kind, forward_training,
                        forward_inference)
                && this->cdesc_().alg_kind == alg_kind::convolution_direct
                && this->cdesc_().src_desc.data_type == data_type::u8
                && this->cdesc_().dst_desc.data_type == dst_type
                && this->cdesc_().weights_desc.data_type == data_type::s8
                && IMPLICATION(this->with_bias(), utils::one_of(
                   this->cdesc_().bias_desc.data_type, data_type::f32,
                   data_type::s32, data_type::s8, data_type::u8))
                && this->cdesc_().accum_data_type == data_type::s32;
            if (!ok) return status::unimplemented;

            return jit_uni_x8s8s32x_1x1_conv_fwd_kernel<isa>::init_conf(jcp_,
                        this->cdesc_(),
                        this->src_pd_.desc(), *this->weights_pd_.desc(),
                        *this->dst_pd_.desc(), *this->bias_pd_.desc(),
                        *this->attr(), with_relu, this->negative_slope());
        }

        jit_1x1_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            auto desired_act_fmt = nhwc;

            auto desired_wei_fmt = OhIw8o4i;
            auto desired_gr_wei_fmt = gOhIw8o4i;

            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(desired_act_fmt));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(desired_act_fmt));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(this->with_groups() ? desired_gr_wei_fmt : desired_wei_fmt));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }
    };

    _jit_uni_x8s8s32x_1x1_convolution_fwd_t(const pd_t *pd, const
                                            input_vector &inputs,
                                            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
        , kernel_(nullptr), ws_(nullptr)
    {
        kernel_ = new jit_uni_x8s8s32x_1x1_conv_fwd_kernel<isa>(conf_.jcp_, *conf_.attr());
        const int nthreads = mkldnn_get_max_threads();
        ws_per_thread_ = conf_.jcp_.ow * conf_.jcp_.nb_oh_blocking_max * conf_.jcp_.oc_block;
        ws_ = (acc_data_t*)malloc(nthreads * ws_per_thread_ * sizeof(acc_data_t), 64);
    }
    ~_jit_uni_x8s8s32x_1x1_convolution_fwd_t() {
        delete kernel_;
        free(ws_);
    }

    typedef typename prec_traits<data_type::u8>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    jit_uni_x8s8s32x_1x1_conv_fwd_kernel<isa> *kernel_;

    /* reduction to unit stride */
    size_t ws_per_thread_;
    acc_data_t *ws_;
};

template <impl::data_type_t src_type, impl::data_type_t dst_type>
using jit_avx2_x8s8s32x_1x1_convolution_fwd_t = _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, false, src_type, dst_type>;
template <impl::data_type_t src_type, impl::data_type_t dst_type>
using jit_sse42_x8s8s32x_1x1_convolution_fwd_t = _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, false, src_type, dst_type>;
template <impl::data_type_t src_type, impl::data_type_t dst_type>
using jit_avx2_x8s8s32x_1x1_convolution_relu_t = _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, true, src_type, dst_type>;
template <impl::data_type_t src_type, impl::data_type_t dst_type>
using jit_sse42_x8s8s32x_1x1_convolution_relu_t = _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, true, src_type, dst_type>;

}
}
}

#endif
