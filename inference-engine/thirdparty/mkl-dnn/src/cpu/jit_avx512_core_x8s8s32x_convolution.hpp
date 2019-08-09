/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef CPU_JIT_AVX512_CORE_X8S8S32X_CONVOLUTION_HPP
#define CPU_JIT_AVX512_CORE_X8S8S32X_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "cpu_convolution_pd.hpp"

#include "jit_avx512_core_x8s8s32x_conv_kernel.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type, impl::data_type_t dst_type>
struct jit_avx512_core_x8s8s32x_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
        {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_int8:", avx512_core, ""),
                jit_avx512_core_x8s8s32x_convolution_fwd_t<src_type, dst_type>);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                    && utils::one_of(this->desc()->prop_kind, forward_training,
                               forward_inference)
                    && utils::one_of(this->desc()->alg_kind,
                            alg_kind::convolution_auto,
                            alg_kind::convolution_direct)
                    && !this->has_zero_dim_memory()
                    && this->desc()->src_desc.data_type == src_type
                    && this->desc()->dst_desc.data_type == dst_type
                    && IMPLICATION(this->with_bias(), utils::one_of(
                            this->desc()->bias_desc.data_type, data_type::f32,
                            data_type::s32, data_type::s8, data_type::u8))
                    && this->desc()->accum_data_type == data_type::s32;
            if (!ok) return status::unimplemented;

            status_t status = jit_avx512_core_x8s8s32x_fwd_kernel::init_conf(
                    jcp_, *this->desc(), this->src_pd_, this->weights_pd_,
                    this->dst_pd_,this->bias_pd_, *this->attr(),
                    mkldnn_get_max_threads());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_core_x8s8s32x_fwd_kernel::init_scratchpad(scratchpad,
                    jcp_, *this->attr());

            if (status == status::success
                    && this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status;
        }

        jit_conv_conf_t jcp_;
    };

    jit_avx512_core_x8s8s32x_convolution_fwd_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
    {
        kernel_ = new jit_avx512_core_x8s8s32x_fwd_kernel(pd()->jcp_,
                    *pd()->attr());
    }

    ~jit_avx512_core_x8s8s32x_convolution_fwd_t() { delete kernel_; }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual void execute(event_t *e) const
    {
        const auto &jcp = pd()->jcp_;
        if (pd()->ndims() == 3)
            execute_forward_1d();
        else if (jcp.is_depthwise)
            execute_forward_2d_dw();
        else
            execute_forward_2d();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward_1d() const;
    void execute_forward_2d() const;
    void execute_forward_2d_dw() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx512_core_x8s8s32x_fwd_kernel *kernel_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
