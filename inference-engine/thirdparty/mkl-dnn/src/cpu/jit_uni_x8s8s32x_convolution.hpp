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

#ifndef CPU_JIT_UNI_X8S8S32X_CONVOLUTION_HPP
#define CPU_JIT_UNI_X8S8S32X_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_reducer.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_x8s8s32x_conv_kernel.hpp"
#include "jit_generator.hpp"
#include "mkldnn_thread.hpp"
#include "jit_uni_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa, impl::data_type_t src_type, impl::data_type_t dst_type>
struct _jit_uni_x8s8s32x_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_(), jcp_dw_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                _jit_uni_x8s8s32x_convolution_fwd_t<isa, src_type, dst_type>);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && utils::one_of(this->desc()->prop_kind, forward_training,
                        forward_inference)
                && this->desc()->alg_kind == alg_kind::convolution_direct
                && IMPLICATION(this->with_bias(), utils::one_of(
                    this->desc()->bias_desc.data_type, data_type::f32,
                    data_type::s32, data_type::s8, data_type::u8))
                && this->desc()->accum_data_type == data_type::s32
                && this->desc()->src_desc.data_type == src_type
                && this->desc()->dst_desc.data_type == dst_type;
            if (!ok) return status::unimplemented;

            status_t sts = jit_uni_x8s8s32x_conv_fwd_kernel<isa>::init_conf(jcp_, *this->desc(),
                    this->src_pd_, this->weights_pd_,
                    this->dst_pd_, this->bias_pd_, *this->attr());
            if (sts != status::success) return sts;

            if (jcp_.with_dw_conv) {
                status_t sts_dw = jit_uni_dw_conv_row_f32<isa>::init_conf(jcp_, jcp_dw_, *this->attr());
                if (sts_dw != status::success) return sts_dw;
            }

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_x8s8s32x_conv_fwd_kernel<isa>::init_scratchpad(scratchpad, jcp_, jcp_dw_, *this->attr());

            return status::success;
        }

        jit_conv_conf_t jcp_;
        jit_conv_conf_t jcp_dw_;
    };

    _jit_uni_x8s8s32x_convolution_fwd_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {
        kernel_ = new jit_uni_x8s8s32x_conv_fwd_kernel<isa>(pd()->jcp_, pd()->jcp_dw_, *pd()->attr());

        if (pd()->jcp_.with_dw_conv) {
            kernel_dw_ = new jit_uni_dw_conv_row_f32<isa>(pd()->jcp_dw_, *pd()->attr(), pd()->jcp_dw_.oc);
        }
    }

    ~_jit_uni_x8s8s32x_convolution_fwd_t() {
        delete kernel_;

        if (pd()->jcp_.with_dw_conv) {
            delete kernel_dw_;
        }
    };

    typedef typename prec_traits<data_type::u8>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<data_type::f32>::type bia_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual void execute(event_t *e) const {
        if (pd()->jcp_.with_dw_conv)
            execute_forward_with_dw_conv();
        else
            execute_forward();

        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    void execute_forward_with_dw_conv() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_uni_x8s8s32x_conv_fwd_kernel<isa> *kernel_;
    jit_uni_dw_conv_row_f32<isa> *kernel_dw_;
};

template <impl::data_type_t src_type, impl::data_type_t dst_type>
using jit_avx2_x8s8s32x_convolution_fwd_t = _jit_uni_x8s8s32x_convolution_fwd_t<avx2, src_type, dst_type>;

template <impl::data_type_t src_type, impl::data_type_t dst_type>
using jit_sse42_x8s8s32x_convolution_fwd_t = _jit_uni_x8s8s32x_convolution_fwd_t<sse42, src_type, dst_type>;

}
}
}

#endif
