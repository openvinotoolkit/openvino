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

struct jit_avx512_core_fp32_wino_conv_2x3_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_fp32_wino_2x3:", avx512_core, ""),
                jit_avx512_core_fp32_wino_conv_2x3_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true && this->set_default_params() == status::success
                    && utils::one_of(this->desc()->prop_kind, forward_inference)
                    && utils::one_of(this->desc()->alg_kind,
                               alg_kind::convolution_auto,
                               alg_kind::convolution_winograd)
                    && this->desc()->src_desc.data_type == data_type::f32
                    && this->desc()->dst_desc.data_type == data_type::f32
                    && this->desc()->weights_desc.data_type == data_type::f32
                    && IMPLICATION(this->with_bias(),
                               utils::one_of(this->desc()->bias_desc.data_type,
                                       data_type::f32));
            if (!ok)
                return status::unimplemented;

            memory_desc_t expect_wei_md = *(this->weights_pd_.desc());
            status_t jit_conf_result = jit_conf(expect_wei_md);
            if (jit_conf_result != success) return jit_conf_result;

            cpu_memory_t::pd_t new_weights_pd(this->engine_, &expect_wei_md);
            if (this->weights_pd_.desc()->format == any)
                this->weights_pd_ = new_weights_pd;
            if (!this->weights_pd_.is_equal(&new_weights_pd))
                return unimplemented;

            init_scratchpad();

            if (this->desc()->alg_kind == alg_kind::convolution_auto)
               CHECK(this->set_alg_kind(alg_kind::convolution_winograd));

            return success;
        }

        jit_conv_conf_2x3_wino_t jcp_;

    protected:
        status_t jit_conf(memory_desc_t& expect_wei_md);

        void init_scratchpad() {
            using namespace memory_tracking::names;

            auto scratchpad = this->scratchpad_registry().registrar();

            int wino_size_offset = (jcp_.yb / 2) * (jcp_.xb / 2) + jcp_.xb;

            size_t V_sz = (size_t)jcp_.ic * 16 * wino_size_offset * jcp_.nthr;
            scratchpad.book(key_wino_V, sizeof(float) * V_sz, PAGE_4K);

            size_t M_sz = (size_t)jcp_.oc * 16 * wino_size_offset * jcp_.nthr;
            scratchpad.book(key_wino_M, sizeof(float) * M_sz, PAGE_4K);

            if (wants_padded_bias()) {
                assert(jcp_.ngroups == 1);
                scratchpad.book(key_conv_padded_bias, sizeof(float) * jcp_.oc);
            }
        }

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

    jit_avx512_core_fp32_wino_conv_2x3_fwd_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs);

    ~jit_avx512_core_fp32_wino_conv_2x3_fwd_t();

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    void execute_forward_small_mb() const;
    void execute_forward_mbN() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx512_core_fp32_wino_conv_2x3_fwd_ker_t *kernel_;
    jit_avx512_core_fp32_wino_conv_2x3_src_trans_t *src_trans_;
    jit_avx512_core_fp32_wino_conv_2x3_dst_trans_t *dst_trans_;
};

}
}
}

#endif
