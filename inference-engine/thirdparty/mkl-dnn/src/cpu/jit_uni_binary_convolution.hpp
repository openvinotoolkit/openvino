/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef CPU_JIT_UNI_BINARY_CONVOLUTION_HPP
#define CPU_JIT_UNI_BINARY_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_binary_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_reducer.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_bin_conv_kernel.hpp"
#include "mkldnn_thread.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_binary_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public _cpu_binary_convolution_fwd_pd_t {
        pd_t(engine_t *engine,
                const binary_convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_binary_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_(), jcp_dw_conv() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_binary_convolution_fwd_t<isa>);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->cdesc_().prop_kind, forward_training, forward_inference)
                && this->cdesc_().alg_kind == alg_kind::binary_convolution_direct
                && utils::everyone_is(data_type::bin,
                        this->cdesc_().src_desc.data_type,
                        this->cdesc_().weights_desc.data_type)
                && utils::one_of(this->cdesc_().dst_desc.data_type,
                        memory::data_type::f32,
                        memory::data_type::bin);
            if (!ok) return status::unimplemented;

            status_t sts = jit_uni_bin_conv_fwd_kernel<isa>::init_conf(jcp_, *this->desc(),
                    *this->src_pd_.desc(), *this->weights_pd_.desc(),
                    *this->dst_pd_.desc(), *this->attr());
            if (sts != status::success) return sts;

            if (jcp_.with_dw_conv) {
                status_t sts_dw = jit_uni_dw_conv_row_f32<isa>::init_conf(jcp_, jcp_dw_conv, *this->attr());
                if (sts_dw != status::success) return sts_dw;
            }

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_bin_conv_fwd_kernel<isa>::init_scratchpad(scratchpad, jcp_, jcp_dw_conv);

            return status::success;
        }

        jit_bin_conv_conf_t jcp_;
        jit_conv_conf_t jcp_dw_conv;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;

            auto desired_weights_format = isa == avx512_common ? OhIw16o32i : OhIw8o32i;

            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(nhwc));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(nhwc));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(desired_weights_format));
            return status::success;
        }
    };

    jit_uni_binary_convolution_fwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {
        kernel_ = new jit_uni_bin_conv_fwd_kernel<isa>(pd()->jcp_, pd()->jcp_dw_conv, *pd()->attr());

        if (pd()->jcp_.with_dw_conv) {
            dw_conv_kernel_ = new jit_uni_dw_conv_row_f32<isa>(pd()->jcp_dw_conv, *pd()->attr(), pd()->jcp_dw_conv.oc);
        }
    }

    ~jit_uni_binary_convolution_fwd_t() {
        delete kernel_;

        if (pd()->jcp_.with_dw_conv) {
            delete dw_conv_kernel_;
        }
    };

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

    jit_uni_bin_conv_fwd_kernel<isa> *kernel_;
    /* fuse with dw conv */
    jit_uni_dw_conv_row_f32<isa> *dw_conv_kernel_;
};

}
}
}

#endif
