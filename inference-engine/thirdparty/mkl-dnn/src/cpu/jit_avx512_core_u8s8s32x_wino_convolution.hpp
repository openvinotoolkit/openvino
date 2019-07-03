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

#ifndef CPU_JIT_AVX512_CORE_U8S8S32X_WINO_CONVOLUTION_HPP
#define CPU_JIT_AVX512_CORE_U8S8S32X_WINO_CONVOLUTION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_primitive_conf.hpp"

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t;
struct jit_avx512_core_u8s8s32x_wino_conv_src_trans_t;
struct jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t;

template <data_type_t dst_data_type>
struct jit_avx512_core_u8s8s32x_wino_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            :  cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
        {}
        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_int8_wino:", avx512_core, ""),
                jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<dst_data_type>);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind,
                                    forward_training, forward_inference)
                && utils::one_of(this->desc()->alg_kind,
                           alg_kind::convolution_auto,
                           alg_kind::convolution_winograd)
                && !this->has_zero_dim_memory()
                && this->desc()->src_desc.data_type == data_type::u8
                && this->desc()->dst_desc.data_type == dst_data_type
                && this->desc()->weights_desc.data_type == data_type::s8
                && IMPLICATION(this->with_bias(),
                    utils::one_of(this->desc()->bias_desc.data_type,
                                                data_type::f32, data_type::s32,
                                                data_type::s8, data_type::u8))
                && this->desc()->accum_data_type == data_type::s32;

            if (!ok) return status::unimplemented;

            status_t status = jit_conf();
            if (status != status::success) return status;

            init_scratchpad();

            if (status == status::success
                    && this->desc()->alg_kind == alg_kind::convolution_auto)
                this->set_alg_kind(alg_kind::convolution_winograd);
            return status;
        }

        jit_conv_conf_2x3_wino_t jcp_;

    protected:
        status_t jit_conf();
        void init_scratchpad();

        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(nhwc));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(nhwc));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }
    };

    typedef typename prec_traits<data_type::u8>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;
    typedef typename prec_traits<dst_data_type>::type dst_data_t;

    jit_avx512_core_u8s8s32x_wino_convolution_fwd_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs);
    ~jit_avx512_core_u8s8s32x_wino_convolution_fwd_t();

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    const float *adjust_oscales(const memory_tracking::grantor_t &scratchpad)
        const;
    void execute_forward() const;
    void execute_forward_small_mb() const;
    void execute_forward_mbN() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t *kernel_;
    jit_avx512_core_u8s8s32x_wino_conv_src_trans_t *src_trans_;
    jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t *dst_trans_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
