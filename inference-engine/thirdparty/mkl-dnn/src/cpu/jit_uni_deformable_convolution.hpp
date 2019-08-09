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

#ifndef CPU_JIT_UNI_DEFORMABLE_CONVOLUTION_HPP
#define CPU_JIT_UNI_DEFORMABLE_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_engine.hpp"
#include "cpu_reducer.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_def_conv_kernel_f32.hpp"
#include "jit_generator.hpp"
#include "mkldnn_thread.hpp"
#include "cpu_deformable_convolution_pd.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_deformable_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public _cpu_deformable_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const deformable_convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_deformable_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_deformable_convolution_fwd_t<isa>);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && utils::one_of(this->desc()->prop_kind, forward_training, forward_inference)
                && this->desc()->alg_kind == alg_kind::deformable_convolution_direct
                && this->desc()->src_descs[0].data_type == data_type::f32
                && this->desc()->src_descs[1].data_type == data_type::f32
                && this->desc()->weights_desc.data_type == data_type::f32
                && IMPLICATION(this->with_bias(), this->desc()->bias_desc.data_type == data_type::f32)
                && this->desc()->dst_desc.data_type == data_type::f32;
            if (!ok) return status::unimplemented;

            status_t sts = jit_uni_def_conv_fwd_kernel_f32<isa>::init_conf(jcp_, *this->desc(),
                    this->src_pds_[0], this->src_pds_[1], this->weights_pd_,
                    this->dst_pd_, this->bias_pd_, *this->attr());
            if (sts != status::success) return sts;

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_def_conv_fwd_kernel_f32<isa>::init_scratchpad(scratchpad, jcp_, *this->attr());

            return status::success;
        }

        jit_def_conv_conf_t jcp_;
    };

    jit_uni_deformable_convolution_fwd_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {
        kernel_ = new jit_uni_def_conv_fwd_kernel_f32<isa>(pd()->jcp_, *pd()->attr());
    }

    ~jit_uni_deformable_convolution_fwd_t() {
        delete kernel_;
    };

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_uni_def_conv_fwd_kernel_f32<isa> *kernel_;
};

}
}
}

#endif
