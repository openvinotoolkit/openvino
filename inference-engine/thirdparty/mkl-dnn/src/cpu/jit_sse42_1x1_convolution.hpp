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

#ifndef CPU_JIT_SSE42_1x1_CONVOLUTION_HPP
#define CPU_JIT_SSE42_1x1_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_sse42_1x1_conv_kernel_f32.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"
#include "jit_uni_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_sse42_1x1_convolution_fwd_t: public cpu_primitive_t {
    // TODO: (Roma) Code duplication duplication! Remove with templates
    //              (maybe...)!
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_(), jcp_dw_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_1x1:", sse42, ""),
                jit_sse42_1x1_convolution_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::one_of(this->desc()->alg_kind,
                           alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::f32,
                        this->desc()->src_desc.data_type,
                        this->desc()->weights_desc.data_type,
                        this->desc()->dst_desc.data_type)
                && IMPLICATION(this->with_bias(),
                        data_type::f32 == this->desc()->bias_desc.data_type);
            if (!ok) return status::unimplemented;

            status_t sts_1x1 = jit_sse42_1x1_conv_kernel_f32::init_conf(jcp_,
                    *this->desc(),
                    *this->src_pd_.desc(), *this->weights_pd_.desc(),
                    *this->dst_pd_.desc(), *this->attr());
            if (sts_1x1 != status::success) return sts_1x1;

            if (jcp_.with_dw_conv) {
                status_t sts_dw = jit_uni_dw_conv_row_f32<sse42>::init_conf(jcp_, jcp_dw_, *this->attr());
                if (sts_dw != status::success) return sts_dw;
            }

            auto scratchpad = scratchpad_registry().registrar();
            jit_sse42_1x1_conv_kernel_f32::init_scratchpad(scratchpad, jcp_, jcp_dw_);

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_;
        jit_conv_conf_t jcp_dw_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(utils::pick(this->ndims() - 3,
                    nCw8c, nChw8c)));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(utils::pick(this->ndims() - 3,
                    nCw8c, nChw8c)));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(this->with_groups()
                    ? utils::pick(this->ndims() - 3, gOIw8i8o, gOIhw8i8o)
                    : utils::pick(this->ndims() - 3, OIw8i8o, OIhw8i8o)));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }
    };

    jit_sse42_1x1_convolution_fwd_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
    {
        kernel_ = new jit_sse42_1x1_conv_kernel_f32(pd()->jcp_, pd()->jcp_dw_, *pd()->attr());

        if (pd()->jcp_.with_dw_conv) {
            kernel_dw_ = new jit_uni_dw_conv_row_f32<sse42>(pd()->jcp_dw_, *pd()->attr(), pd()->jcp_dw_.ch_block);
        }
    }

    ~jit_sse42_1x1_convolution_fwd_t() {
        delete kernel_;

        if (pd()->jcp_.with_dw_conv) {
            delete kernel_dw_;
        }
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

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

    jit_sse42_1x1_conv_kernel_f32 *kernel_;
    jit_uni_dw_conv_row_f32<sse42> *kernel_dw_;
};

}
}
}

#endif
