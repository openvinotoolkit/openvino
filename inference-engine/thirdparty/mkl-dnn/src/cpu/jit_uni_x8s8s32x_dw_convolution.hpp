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

#ifndef CPU_JIT_UNI_X8S8S32X_DW_CONVOLUTION_HPP
#define CPU_JIT_UNI_X8S8S32X_DW_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_generator.hpp"
#include "jit_uni_x8s8s32x_dw_conv_kernel.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa, impl::data_type_t src_type, impl::data_type_t dst_type>
struct _jit_uni_x8s8s32x_dw_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr,
                hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_dw:", isa, ""),
                _jit_uni_x8s8s32x_dw_convolution_fwd_t<isa, src_type, dst_type>);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, forward_training,
                        forward_inference)
                && this->desc()->alg_kind == alg_kind::convolution_direct
                && this->desc()->dst_desc.data_type == dst_type
                && IMPLICATION(this->with_bias(), utils::one_of(
                    this->desc()->bias_desc.data_type, data_type::f32,
                    data_type::s32, data_type::s8, data_type::u8))
                && this->desc()->accum_data_type == data_type::s32;
            if (!ok) return status::unimplemented;

            status_t sts = jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>::init_conf(jcp_,
                        *this->desc(),
                        *this->src_pd_.desc(), *this->weights_pd_.desc(),
                        *this->dst_pd_.desc(), *this->bias_pd_.desc(),
                        *this->attr());
            if (sts != status::success) return sts;

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>::init_scratchpad(scratchpad, jcp_, *this->attr());

            return status::success;
        }

        jit_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            auto desired_act_fmt = (ndims() == 5) ? ndhwc : nhwc;
            auto desired_wei_fmt = (ndims() == 5) ? isa == avx512_common ? Goidhw16g : Goidhw8g
                                                  : isa == avx512_common ? Goihw16g : Goihw8g;

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

    _jit_uni_x8s8s32x_dw_convolution_fwd_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
    {
        kernel_ = new jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>(pd()->jcp_, *pd()->attr());
    }

    ~_jit_uni_x8s8s32x_dw_convolution_fwd_t() { delete kernel_; };

    typedef typename prec_traits<data_type::u8>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const ;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa> *kernel_;
};

template <impl::data_type_t src_type, impl::data_type_t dst_type>
using jit_avx2_x8s8s32x_dw_convolution_fwd_t = _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, src_type, dst_type>;
template <impl::data_type_t src_type, impl::data_type_t dst_type>
using jit_sse42_x8s8s32x_dw_convolution_fwd_t = _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, src_type, dst_type>;

}
}
}

#endif
