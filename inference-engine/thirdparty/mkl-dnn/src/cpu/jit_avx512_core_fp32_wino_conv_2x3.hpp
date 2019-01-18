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

#ifndef CPU_JIT_AVX512_CORE_FP32_WINO_CONV_2x3_HPP
#define CPU_JIT_AVX512_CORE_FP32_WINO_CONV_2x3_HPP

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

struct jit_avx512_core_fp32_wino_conv_2x3_fwd_ker_t;
struct jit_avx512_core_fp32_wino_conv_2x3_src_trans_t;
struct jit_avx512_core_fp32_wino_conv_2x3_dst_trans_t;

template <bool with_relu>
struct _jit_avx512_core_fp32_wino_conv_2x3_fwd_t : public cpu_primitive_t {
    struct pd_t : public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine, const typename pd_t::base_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            :  _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, attr,
            hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_fp32_wino_2x3:", avx512_core, ""),
                _jit_avx512_core_fp32_wino_conv_2x3_fwd_t<with_relu>);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true && this->set_default_params() == status::success
                    && utils::one_of(this->cdesc_().prop_kind, forward_inference)
                    && this->cdesc_().alg_kind == alg_kind::convolution_winograd
                    && this->cdesc_().src_desc.data_type == data_type::f32
                    && this->cdesc_().dst_desc.data_type == data_type::f32
                    && this->cdesc_().weights_desc.data_type == data_type::f32
                    && IMPLICATION(this->with_bias(),
                               utils::one_of(this->cdesc_().bias_desc.data_type,
                                       data_type::f32));
            if (!ok)
                return status::unimplemented;

            memory_desc_t expect_wei_md = *(this->weights_pd_.desc());
            status_t jit_conf_result = jit_conf(expect_wei_md);
            if (jit_conf_result == success) {
                cpu_memory_t::pd_t new_weights_pd(this->engine_, &expect_wei_md);
                if (this->weights_pd_.desc()->format == any)
                    this->weights_pd_ = new_weights_pd;
                if (!this->weights_pd_.is_equal(&new_weights_pd))
                    return status::unimplemented;
            }
            return jit_conf_result;
        }

        jit_conv_conf_2x3_wino_t jcp_;

    protected:
        status_t jit_conf(memory_desc_t& expect_wei_md);

        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(nChw16c));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(nChw16c));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }
    };

    _jit_avx512_core_fp32_wino_conv_2x3_fwd_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs);
    
    ~_jit_avx512_core_fp32_wino_conv_2x3_fwd_t();

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    void execute_forward_small_mb();
    void execute_forward_mbN();
    pd_t conf_;

    jit_avx512_core_fp32_wino_conv_2x3_fwd_ker_t *kernel_;
    jit_avx512_core_fp32_wino_conv_2x3_src_trans_t *src_trans_;
    jit_avx512_core_fp32_wino_conv_2x3_dst_trans_t *dst_trans_;

    size_t size_wino_wei;
    size_t size_wino_src;
    size_t size_wino_dst;

    const float *wino_wei_;
    const float *dst_bias_;

    float *wino_src_;
    float *wino_dst_;
    float *padded_bias_;
};

using jit_avx512_core_fp32_wino_conv_2x3_fwd_t =
    _jit_avx512_core_fp32_wino_conv_2x3_fwd_t<false>;

using jit_avx512_core_fp32_wino_convolution_relu_t =
    _jit_avx512_core_fp32_wino_conv_2x3_fwd_t<true>;
}
}
}

#endif
